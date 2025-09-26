# train_ddp_vlpo_gsm8k.py
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

def format_example(ex, tokenizer):
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": ex["question"]},
        {"role": "assistant", "content": ex["answer"]},
        {"role": "user", "content": "Provide a detailed, step-by-step reasoning process for the above answer, don't repeat yourself."},
        ],
        tokenize=False,
        add_generation_prompt=True,  # adds the assistant header/start token(s)
    )

    # How many tokens belong to the prompt/prefix (to be ignored in loss)?
    prompt_len = len(
        tokenizer(prefix, add_special_tokens=False).input_ids
    )


    return {"text": prefix,
            "prompt_len": prompt_len,
            "question":ex["question"],
            "answer":ex["answer"], 
            }

@dataclass
class Collator:
    tokenizer: AutoTokenizer
    max_len: int

    def __call__(self, batch: List[Dict]):
        texts = [b["text"] for b in batch]
        prompt_lens = [b["prompt_len"] for b in batch]
        toks = self.tokenizer(
            texts,
            max_length=self.max_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        toks["question"] = [b["question"] for b in batch]
        toks["answer"] = [b["answer"] for b in batch]
        return toks

def tokenize_batch(batch, thoughts, tokenizer, device):
    prefix = [tokenizer.apply_chat_template(
        [{"role": "user", "content": q}],
        tokenize=False,
        add_generation_prompt=True,  # adds the assistant header/start token(s)
    ) for q in batch["question"]]

    # How many tokens belong to the prompt/prefix (to be ignored in loss)?
    prompt_lens = [len(
        tokenizer(p, add_special_tokens=False).input_ids
    ) for p in prefix]

    texts = [p + a for p, a in zip(prefix, [t + "\n\n" + a for t, a in zip(thoughts, batch["answer"])])]

    toks = tokenizer(
        texts,
        max_length=args.max_len,
        truncation=True,
        padding=True,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = toks["input_ids"]
    labels = input_ids.clone()

    # Mask padding tokens (optional but recommended)
    if "attention_mask" in toks:
        labels[toks["attention_mask"] == 0] = -100
    else:
        # Fallback: mask pad_token_id if attention_mask isn't returned
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            labels[input_ids == pad_id] = -100

    pad_start_idx = (labels == -100).sum(-1)
    # Mask the prompt part per-sample so only assistant tokens are trained
    seq_len = input_ids.size(1)
    thought_starts = []
    for i, (p_len, pad_start) in enumerate(zip(prompt_lens, pad_start_idx)):
        start = min(pad_start + p_len, seq_len) # handle truncation of long prompts
        labels[i, :start] = -100
        thought_starts.append(start)

    toks["input_ids"] = toks["input_ids"].to(device)
    toks["labels"] = labels.to(device)
    return toks, torch.stack(thought_starts) # return tokens

# ----------------------------
# Training
# ----------------------------
def train(args):
    local_rank = setup_ddp()
    device = torch.device("cuda", local_rank)
    seed_everything(args.seed)

    if is_main_process():
        wandb.init(project=os.getenv("WANDB_PROJECT", "ddp-sft"), config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, padding_side="left")
    if tokenizer.pad_token is None:
        # Safety: set pad_token if missing
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map=None,
    ).to(device)

    z_model = AutoModelForCausalLM.from_pretrained(
        args.latent_model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map=None,
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    z_model = DDP(z_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Load GSM8K (train only). Use a tiny streaming or full load depending on memory.
    ds = load_dataset("gsm8k", "main", split="train")  # 7473 examples
    ds = ds.shuffle(seed=args.seed)
    ds = ds.map(lambda ex: format_example(ex, tokenizer), remove_columns=ds.column_names)

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

    grouped = [
        {"params": [p for n, p in z_model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in z_model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    z_optimizer = torch.optim.AdamW(grouped, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    total_steps = (len(dl) * args.epochs) // max(1, args.grad_accum)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    z_scheduler = get_linear_schedule_with_warmup(z_optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=not torch.cuda.is_bf16_supported())
    z_scaler = torch.cuda.amp.GradScaler(enabled=not torch.cuda.is_bf16_supported())
    # ----------------------------
    # Loop
    # ----------------------------
    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        running_loss = 0.0

        for step, batch in enumerate(dl):
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            with torch.no_grad():
                gen = z_model.module.generate(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    max_new_tokens=args.max_len,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
                z_attention_mask = gen!=tokenizer.pad_token_id
                thought_start = input_ids.shape[1]
                thought_toks = gen[:, thought_start:]
                thought_toks_len = thought_toks.shape[-1]
            thoughts = tokenizer.batch_decode(thought_toks)
            # attaching the generated thoughts to the original answers        
            new_batch, thought_starts = tokenize_batch(batch, thoughts, tokenizer, device)
            # next token prediction to optimize the current policy for longer generation (E step)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                outputs = model(**new_batch)
                loss = outputs.loss / args.grad_accum

            scaler.scale(loss).backward()


            # compute log
            z_output = z_model(gen, attention_mask = z_attention_mask)
            with torch.no_grad():
                thought_logits_idx = torch.arange(thought_toks_len, device = device).unsqueeze(0) + thought_starts.to(device)
                thought_logits = torch.gather(outputs.logits.detach(), dim = 1, index = thought_logits_idx.unsqueeze(-1).expand(-1, -1, outputs.logits.shape[-1]))
                p_logprob = torch.gather(thought_logits.softmax(-1), dim = -1, index = thought_toks.unsqueeze(-1)).log()
                
            q_prob = torch.gather(z_output.logits.softmax(-1)[:,thought_start:], dim =-1, index = gen[:,thought_start:].unsqueeze(-1))
            q_logprob = q_prob.log()
            z_loss = (q_prob * (p_logprob - q_logprob) * z_attention_mask[:,thought_start:]).sum()
            z_scaler.scale(z_loss).backward()
            
            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_
                scaler.step(optimizer)
                scaler.update()
                z_scaler.unscale_
                z_scaler.step(z_optimizer)
                z_scaler.update()

                optimizer.zero_grad(set_to_none=True)
                z_optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                z_scheduler.step()
                global_step += 1

                # Logging (main process only)
                if is_main_process() and (global_step % args.log_every == 0):
                    # Average running loss across processes
                    loss_detached = outputs.loss.detach().float()
                    dist.all_reduce(loss_detached, op=dist.ReduceOp.SUM)
                    loss_avg = (loss_detached / dist.get_world_size()).item()
                    z_loss_detached = z_output.loss.detach().float()
                    dist.all_reduce(z_loss_detached, op=dist.ReduceOp.SUM)
                    z_loss_avg = (z_loss_detached / dist.get_world_size()).item()
                    lr = scheduler.get_last_lr()[0]
                    wandb.log({"train/loss": loss_avg, "train/lr": lr, "step": global_step, "train/z_loss": z_loss_avg})
                    

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
    p.add_argument("--model_id", type=str, default="Qwen/Qwen3-0.6B",
                   help="Replace with your favorite huggingface model")
    p.add_argument("--latent_model_id", type=str, default="Qwen/Qwen3-0.6B",
                   help="Replace with your favorite huggingface model")
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
