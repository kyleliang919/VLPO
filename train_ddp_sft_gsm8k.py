# train_ddp_sft_gsm8k.py
import os
import math
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

# ----------------------------
# Utilities
# ----------------------------
def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------------
# Data / Formatting
# ----------------------------
INSTRUCTION_TEMPLATE = (
    "You are a math tutor. Solve the problem step by step.\n\n"
    "Question:\n{question}\n\nAnswer:\n{answer}\n"
)

def format_example(ex):
    # GSM8K 'main' split has fields 'question' and 'answer' (CoT + final).
    # For simplicity, train to reproduce full "Answer" (including reasoning).
    return INSTRUCTION_TEMPLATE.format(question=ex["question"], answer=ex["answer"])

@dataclass
class Collator:
    tokenizer: AutoTokenizer
    max_len: int
    def __call__(self, batch: List[Dict]):
        texts = [b["text"] for b in batch]
        toks = self.tokenizer(
            texts,
            max_length=self.max_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        # Simple SFT: predict every token
        toks["labels"] = toks["input_ids"].clone()
        return toks

# ----------------------------
# Training
# ----------------------------
def train(args):
    local_rank = setup_ddp()
    device = torch.device("cuda", local_rank)
    seed_everything(args.seed)

    if is_main_process():
        wandb.init(project=os.getenv("WANDB_PROJECT", "ddp-sft"), config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        # Safety: set pad_token if missing
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map=None,
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Load GSM8K (train only). Use a tiny streaming or full load depending on memory.
    ds = load_dataset("gsm8k", "main", split="train")  # 7473 examples
    ds = ds.shuffle(seed=args.seed)
    ds = ds.map(lambda ex: {"text": format_example(ex)}, remove_columns=ds.column_names)

    sampler = DistributedSampler(ds, shuffle=True, drop_last=True)
    collate = Collator(tokenizer, args.max_len)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate,
        pin_memory=True,
        num_workers=4,
    )

    # Optimizer / Scheduler
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    total_steps = (len(dl) * args.epochs) // max(1, args.grad_accum)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=not torch.cuda.is_bf16_supported())

    # ----------------------------
    # Loop
    # ----------------------------
    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        running_loss = 0.0

        for step, batch in enumerate(dl):
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                # Logging (main process only)
                if is_main_process() and (global_step % args.log_every == 0):
                    # Average running loss across processes
                    loss_detached = outputs.loss.detach().float()
                    dist.all_reduce(loss_detached, op=dist.ReduceOp.SUM)
                    loss_avg = (loss_detached / dist.get_world_size()).item()
                    lr = scheduler.get_last_lr()[0]
                    wandb.log({"train/loss": loss_avg, "train/lr": lr, "step": global_step})

            running_loss += outputs.loss.detach().float().item()

        # End-of-epoch logging
        if is_main_process():
            epoch_loss = running_loss / len(dl)
            ppl = math.exp(epoch_loss)
            wandb.log({"epoch": epoch, "epoch/loss": epoch_loss, "epoch/ppl": ppl, "step": global_step})

        # (Optional) Save at epoch end (main process only)
        if is_main_process() and args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f"epoch{epoch}")
            model.module.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    cleanup_ddp()

# ----------------------------
# Args
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="Replace with your Qwen3-0.6B base if desired.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
