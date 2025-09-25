# train_ddp_grpo_gsm8k.py
import os, math, argparse, random, re
from dataclasses import dataclass
from typing import List, Dict, Tuple

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
)

# ----------------------------
# DDP utils
# ----------------------------
def is_main() -> bool:
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

def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------------
# Data / prompts / reward
# ----------------------------

def make_prompt(q: str) -> str:
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": ex["question"]}],
        tokenize=False,
        add_generation_prompt=True,  # adds the assistant header/start token(s)
    )
    return prefix

ANS_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")

def extract_gold_ans(answer_field: str):
    m = ANS_RE.search(answer_field)
    if m:
        return m.group(1).strip()
    return None

def extract_pred_ans(generated_text: str):
    # Prefer GSM8K “#### number” if present; else last number fallback
    m = ANS_RE.search(generated_text)
    if m:
        return m.group(1).strip()
    # fallback: last number in text
    nums = re.findall(r"-?\d+(?:\.\d+)?", generated_text)
    return nums[-1] if nums else None

def reward_correct(pred: str, gold: str) -> float:
    if pred is None or gold is None:
        return 0.0
    # strict numeric equality; you can add tolerance if desired
    try:
        return 1.0 if float(pred) == float(gold) else 0.0
    except:
        return 1.0 if pred == gold else 0.0

# ----------------------------
# Collation (prompts only)
# ----------------------------
@dataclass
class PromptBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    gold_final: List[str]  # for reward

class PromptCollator:
    def __init__(self, tokenizer, max_prompt_len: int):
        self.tok = tokenizer
        self.max_prompt_len = max_prompt_len

    def __call__(self, batch: List[Dict]) -> PromptBatch:
        prompts = [make_prompt(b["question"]) for b in batch]
        enc = self.tok(
            prompts,
            max_length=self.max_prompt_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        golds = [b["gold_final"] for b in batch]
        return PromptBatch(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            gold_final=golds,
        )

# ----------------------------
# Logprob utils
# ----------------------------
@torch.no_grad()
def concat_and_positions(prompt_ids: torch.Tensor, gen_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # full = [prompt ...] + [gen ...]; we will score logp of generated tokens
    B = prompt_ids.size(0)
    full = torch.cat([prompt_ids, gen_ids], dim=1)
    # positions within 'full' that correspond to generated tokens
    gen_pos = torch.arange(gen_ids.size(1), device=gen_ids.device)[None, :] + prompt_ids.size(1)
    gen_pos = gen_pos.expand(B, -1)  # [B, Tgen]
    return full, gen_pos

def gather_token_logps(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    # logits: [B, T, V], target_ids: [B, T]
    logprobs = torch.log_softmax(logits, dim=-1)
    return torch.gather(logprobs, -1, target_ids.unsqueeze(-1)).squeeze(-1)  # [B, T]

@torch.no_grad()
def sequence_logprob(model, input_ids: torch.Tensor, attn: torch.Tensor, gen_ids: torch.Tensor) -> torch.Tensor:
    """
    Returns sum of token log-probs for the generated segment.
    """
    full, gen_pos = concat_and_positions(input_ids, gen_ids)
    attn_full = torch.ones_like(full).to(full.device)
    out = model(input_ids=full, attention_mask=attn_full)
    # next-token logits (shift by 1)
    logits = out.logits[:, :-1, :]
    targets = full[:, 1:]
    # keep only rows/cols for generated segment
    # gen positions in 'full' predict tokens at same indices (targets shifted already)
    gen_pos_m1 = gen_pos - 1  # since logits are for next token
    # collect per-token logp for generated tokens
    bidx = torch.arange(full.size(0), device=full.device)[:, None]
    gen_targets = targets[bidx, gen_pos]
    gen_logits = logits[bidx, gen_pos_m1]
    token_logps = gather_token_logps(gen_logits, gen_targets)  # [B, Tgen]
    return token_logps.sum(dim=1)  # [B]

# ----------------------------
# GRPO step
# ----------------------------
def grpo_step(
    policy, ref, tokenizer, batch: PromptBatch, args, device
) -> Dict[str, float]:
    """
    1) Sample K completions per prompt from current policy.
    2) Compute rewards → group baseline advantages.
    3) Estimate KL(policy||ref) via summed token-level log-probs.
    4) Loss = -E[A * logp_policy] + beta * E[ (logp_policy - logp_ref) ].
    """
    policy.train()

    # 1) repeat prompts K times and sample
    B = batch.input_ids.size(0)
    K = args.group_size
    rep_input = batch.input_ids.to(device).repeat_interleave(K, dim=0)
    rep_attn  = batch.attention_mask.to(device).repeat_interleave(K, dim=0)

    with torch.no_grad():
        gen = policy.module.generate(  # DDP wrapped
            input_ids=rep_input,
            attention_mask=rep_attn,
            max_new_tokens=args.max_gen_len,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    # slice out generated tokens (everything after each original prompt length)
    prompt_lengths = rep_attn.sum(dim=1)  # [B*K]
    max_len_gen = gen.size(1) - prompt_lengths.max().item()
    # build gen_ids tensor per row
    gen_ids = []
    for i in range(gen.size(0)):
        pL = prompt_lengths[i].item()
        gen_ids.append(gen[i, pL : pL + args.max_gen_len])
    # pad to equal length
    gen_ids = torch.nn.utils.rnn.pad_sequence(
        [g for g in gen_ids], batch_first=True, padding_value=tokenizer.pad_token_id
    ).to(device)

    # 2) rewards (correctness)
    # decode each sample to extract predicted answer
    texts = tokenizer.batch_decode(gen[:, rep_input.size(1):], skip_special_tokens=True)
    golds = sum([[g]*K for g in batch.gold_final], [])  # length B*K
    rewards = torch.tensor(
        [reward_correct(extract_pred_ans(t), gold) for t, gold in zip(texts, golds)],
        device=device, dtype=torch.float32
    )  # [B*K]

    # group-relative baseline: A = r - mean_group(r)
    rewards = rewards.view(B, K)
    group_mean = rewards.mean(dim=1, keepdim=True)
    adv = rewards - group_mean
    # optional: normalize within group to unit variance (stabilizes)
    if args.group_norm:
        std = adv.std(dim=1, keepdim=True) + 1e-8
        adv = adv / std
    adv = adv.view(B*K)

    # 3) compute summed log-probs under policy and reference for generated tokens
    with torch.no_grad():
        logp_ref = sequence_logprob(ref, rep_input, rep_attn, gen_ids)  # [B*K]
    logp_pol = sequence_logprob(policy.module, rep_input, rep_attn, gen_ids)  # [B*K]

    # 4) loss
    # policy gradient surrogate: - A * logp_pol  (treat A as stop-grad)
    # KL penalty (per-sample): beta * (logp_pol - logp_ref)
    loss_vec = -(adv.detach() * logp_pol) + args.beta_kl * (logp_pol - logp_ref)
    loss = loss_vec.mean()

    # backprop
    loss.backward()

    with torch.no_grad():
        # metrics
        kl = (logp_pol - logp_ref).mean()
        ent = (-logp_pol).mean()  # rough proxy
        acc = (rewards.view(-1) > 0.5).float().mean()
        return {
            "loss": loss.detach().float().item(),
            "reward_mean": rewards.mean().item(),
            "reward_acc": acc.item(),
            "kl_mean": kl.item(),
            "neg_logp_mean": ent.item(),
        }

# ----------------------------
# Main training
# ----------------------------
def train(args):
    local_rank = setup_ddp()
    device = torch.device("cuda", local_rank)
    seed_all(args.seed)

    if is_main():
        wandb.init(project=os.getenv("WANDB_PROJECT", "ddp-grpo"), config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # policy (trainable) and reference (frozen)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    policy = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
    ref    = AutoModelForCausalLM.from_pretrained(args.ref_id,   torch_dtype=dtype).to(device)
    for p in ref.parameters():
        p.requires_grad_(False)
    policy = DDP(policy, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # dataset: GSM8K train → (question, gold_final answer)
    raw = load_dataset("gsm8k", "main", split="train")  # 7.4k
    ds = raw.map(
        lambda ex: {"gold_final": extract_gold_ans(ex["answer"])},
        remove_columns=[c for c in raw.column_names if c not in ["question", "answer"]],
    ).filter(lambda ex: ex["gold_final"] is not None)

    sampler = DistributedSampler(ds, shuffle=True, drop_last=True)
    collate = PromptCollator(tokenizer, args.max_prompt_len)
    dl = DataLoader(ds, batch_size=args.batch_size, sampler=sampler, num_workers=2, collate_fn=collate, pin_memory=True)

    # optimizer
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"]
    grouped = [
        {"params": [p for n, p in policy.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in policy.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optim = torch.optim.AdamW(grouped, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype==torch.float16))

    global_step = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(dl):
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=dtype):
                metrics = grpo_step(policy, ref, tokenizer, batch, args, device)

            # gradient step (no scheduler for minimalism)
            scaler.unscale_
            scaler.step(optim)
            scaler.update()

            global_step += 1

            # reduce metrics across processes
            if is_main() and (global_step % args.log_every == 0):
                # gather means
                tensor_metrics = {k: torch.tensor(v, device=device, dtype=torch.float32) for k, v in metrics.items()}
                for k, t in tensor_metrics.items():
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    tensor_metrics[k] = (t / dist.get_world_size()).item()
                wandb.log({**tensor_metrics, "step": global_step, "epoch": epoch})

        # save end of epoch (rank 0)
        if is_main() and args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            out = os.path.join(args.save_dir, f"epoch{epoch}")
            policy.module.save_pretrained(out)
            tokenizer.save_pretrained(out)

    cleanup_ddp()

# ----------------------------
# Args
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--ref_id",   type=str, default="Qwen/Qwen3-0.6B",
                   help="Frozen reference; often the initial policy checkpoint.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16, help="number of prompts per step")
    p.add_argument("--group_size", type=int, default=4, help="K completions per prompt")
    p.add_argument("--max_prompt_len", type=int, default=512)
    p.add_argument("--max_gen_len", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--beta_kl", type=float, default=0.02)
    p.add_argument("--group_norm", action="store_true", help="normalize advantages within each group")
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="grpo_ckpts")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
