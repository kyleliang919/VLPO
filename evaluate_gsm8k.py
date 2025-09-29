#!/usr/bin/env python3
# gsm8k_verify.py
# Usage examples:
#   python gsm8k_verify.py --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct --split test --max_samples 50 --n_samples 5 --temperature 0.7 --top_p 0.95
#   python gsm8k_verify.py --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --manual --n_samples 3
#   python gsm8k_verify.py --load_preds preds.jsonl  # verify previously saved per-sample predictions (flat JSONL)

import argparse
import dataclasses
import json
import math
import re
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Iterable

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ======================================================================================
# Verifier: parse numeric answers and compare
# ======================================================================================

_NUM_RE = re.compile(
    r"""
    (?P<sign>[-+])?
    (?:
        (?P<mixed>\d+)\s+(?P<mixed_num>\d+)\/(?P<mixed_den>\d+) |   # mixed fraction: 3 1/2
        (?P<fnum>\d+)\/(?P<fden>\d+) |                              # simple fraction: 1/2
        (?P<num>(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)               # decimal/int with commas
    )
    """,
    re.VERBOSE,
)

def _strip_units(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)
    s = re.sub(r"[€$£¥]", "", s)
    s = s.replace(",", "")
    return s.strip()

def _to_float(num_str: str) -> Optional[float]:
    m = _NUM_RE.fullmatch(num_str.strip())
    if not m:
        return None
    sign = -1.0 if (m.group("sign") == "-") else 1.0
    if m.group("mixed"):
        whole = float(m.group("mixed"))
        n = float(m.group("mixed_num"))
        d = float(m.group("mixed_den"))
        return sign * (whole + n / d)
    if m.group("fnum"):
        n = float(m.group("fnum"))
        d = float(m.group("fden"))
        return sign * (n / d)
    if m.group("num"):
        return sign * float(m.group("num"))
    return None

def _extract_final_candidate(text: str) -> Optional[str]:
    if text is None:
        return None
    s = text.strip()

    m = re.search(r"####\s*(.+)", s)  # GSM8K convention
    if m:
        cand = m.group(1).strip().splitlines()[0].strip()
        return cand

    boxed = re.findall(r"\\boxed\{([^}]*)\}", s)
    if boxed:
        return boxed[-1].strip()

    nums = list(_NUM_RE.finditer(s))
    if nums:
        return nums[-1].group(0).strip()

    return None

def _normalize_percent(s: str) -> Tuple[str, bool]:
    s = s.strip()
    is_percent = s.endswith("%")
    if is_percent:
        s = s[:-1]
    return s.strip(), is_percent

def canonicalize_answer(raw: str) -> Optional[float]:
    if raw is None:
        return None
    stripped = _strip_units(raw)
    stripped, is_percent = _normalize_percent(stripped)
    val = _to_float(stripped)
    if val is None:
        return None
    # Keep percent as a number in percent units (e.g., "12%" -> 12.0)
    return val

def extract_numeric_answer(text: str) -> Optional[float]:
    cand = _extract_final_candidate(text or "")
    return canonicalize_answer(cand)

def verify_gsm8k(gold_answer_text: str,
                 pred_text: str,
                 atol: float = 1e-4,
                 rtol: float = 1e-3) -> Dict[str, object]:
    """Return dict: correct, gold_value, pred_value, gold_raw, pred_raw, reason"""
    # Prefer gold after '####'
    m = re.search(r"####\s*(.+)", (gold_answer_text or "").strip())
    if m:
        gold_line = m.group(1).strip().splitlines()[0].strip()
    else:
        gold_line = _extract_final_candidate(gold_answer_text or "")

    pred_line = _extract_final_candidate(pred_text or "")

    gold_val = canonicalize_answer(gold_line)
    pred_val = canonicalize_answer(pred_line)

    if gold_val is None:
        return {
            "correct": False,
            "gold_value": None,
            "pred_value": pred_val,
            "gold_raw": gold_line,
            "pred_raw": pred_line,
            "reason": "Could not parse gold numeric value.",
        }
    if pred_val is None:
        return {
            "correct": False,
            "gold_value": gold_val,
            "pred_value": None,
            "gold_raw": gold_line,
            "pred_raw": pred_line,
            "reason": "Could not parse predicted numeric value.",
        }

    diff = abs(gold_val - pred_val)
    tol = max(atol, rtol * abs(gold_val))
    correct = diff <= tol
    return {
        "correct": bool(correct),
        "gold_value": float(gold_val),
        "pred_value": float(pred_val),
        "gold_raw": gold_line,
        "pred_raw": pred_line,
        "reason": f"|gold - pred| = {diff:.6g} <= tol {tol:.6g}" if correct
                  else f"|gold - pred| = {diff:.6g} > tol {tol:.6g}",
    }

# ======================================================================================
# Inference / data structures
# ======================================================================================

def default_prompt(q: str) -> str:
    return (
        "You are a careful math tutor. Solve the problem step by step, "
        "and put your final numeric answer after '####' on its own line.\n\n"
        f"Question: {q}\nAnswer:"
    )

@dataclass
class ExamplePred:
    # Flat per-sample record; id ties to dataset example index
    id: int                  # dataset row id
    sample_ix: int           # which sample for this problem (0..n_samples-1)
    question: str
    gold: str
    pred: str
    verdict: Dict[str, object]

def _batched(iterable: Iterable[int], n: int):
    arr = list(iterable)
    for i in range(0, len(arr), n):
        yield arr[i:i+n]

def generate_predictions(model_name_or_path: str,
                         split: str,
                         max_samples: Optional[int],
                         batch_size: int,
                         device: str,
                         dtype: str,
                         temperature: float,
                         top_p: float,
                         max_new_tokens: int,
                         seed: int,
                         n_samples: int) -> List[ExamplePred]:
    ds = load_dataset("gsm8k", "main", split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=getattr(torch, dtype),
        device_map="auto" if device == "auto" else None,
    )
    if device != "auto":
        model.to(device)
    model.eval()

    prompts = [default_prompt(q) for q in ds["question"]]
    preds: List[ExamplePred] = []

    # To encourage diversity across samples, vary seed each round
    for s in range(n_samples):
        # set a deterministic generator per round
        gen_device = model.device if device == "auto" else torch.device(device)
        g = torch.Generator(device=gen_device).manual_seed(seed + s)

        for idxs in _batched(range(len(prompts)), batch_size):
            batch_prompts = [prompts[i] for i in idxs]
            enc = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            enc = {k: v.to(model.device if device == "auto" else device) for k, v in enc.items()}

            with torch.no_grad():
                gen_out = model.generate(
                    **enc,
                    do_sample=True if (temperature is None or temperature > 0) else True,  # force sampling across n_samples
                    temperature=temperature if (temperature is not None and temperature > 0) else 0.7,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    generator=g,
                )

            # Decode only the newly generated continuation
            out_texts = []
            for j, i in enumerate(idxs):
                prompt_len = enc["input_ids"][j].shape[-1]
                out_ids = gen_out[j][prompt_len:]
                out_texts.append(tokenizer.decode(out_ids, skip_special_tokens=True))

            for j, i in enumerate(idxs):
                gold = ds[i]["answer"]
                pred_text = out_texts[j]
                verdict = verify_gsm8k(gold, pred_text)
                preds.append(ExamplePred(
                    id=int(i),
                    sample_ix=int(s),
                    question=ds[i]["question"],
                    gold=gold,
                    pred=pred_text,
                    verdict=verdict,
                ))

    return preds

# ======================================================================================
# Save / load, aggregation, manual check
# ======================================================================================

def save_preds(path: str, preds: List[ExamplePred]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p in preds:
            row = {
                "id": p.id,
                "sample_ix": p.sample_ix,
                "question": p.question,
                "gold": p.gold,
                "pred": p.pred,
                "verdict": p.verdict,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def load_preds(path: str) -> List[ExamplePred]:
    out: List[ExamplePred] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            out.append(ExamplePred(
                id=o.get("id", -1),
                sample_ix=o.get("sample_ix", 0),
                question=o.get("question", ""),
                gold=o.get("gold", ""),
                pred=o.get("pred", ""),
                verdict=o.get("verdict", {}),
            ))
    return out

def summarize_sample_accuracy(preds: List[ExamplePred]) -> Dict[str, float]:
    total = len(preds)
    auto_correct = sum(1 for p in preds if p.verdict.get("correct") is True)
    return {
        "n_samples": total,
        "sample_correct": auto_correct,
        "sample_acc_%": (auto_correct / total) * 100.0 if total else 0.0,
    }

def aggregate_by_problem(preds: List[ExamplePred]) -> Dict[int, Dict[str, object]]:
    """Group flat per-sample predictions by problem id and compute any-correct."""
    by_id: Dict[int, List[ExamplePred]] = {}
    for p in preds:
        by_id.setdefault(p.id, []).append(p)

    agg: Dict[int, Dict[str, object]] = {}
    for pid, lst in by_id.items():
        any_correct = any(x.verdict.get("correct") is True for x in lst)
        # also record which sample indices were correct
        correct_samples = [x.sample_ix for x in lst if x.verdict.get("correct") is True]
        agg[pid] = {
            "n_samples": len(lst),
            "any_correct": any_correct,
            "correct_sample_ixs": correct_samples,
        }
    return agg

def summarize_any_correct(preds: List[ExamplePred]) -> Dict[str, float]:
    agg = aggregate_by_problem(preds)
    n_problems = len(agg)
    n_any = sum(1 for v in agg.values() if v["any_correct"])
    return {
        "n_problems": n_problems,
        "problems_any_correct": n_any,
        "any_correct_acc_%": (n_any / n_problems) * 100.0 if n_problems else 0.0,
    }

def interactive_manual_check(preds: List[ExamplePred]) -> None:
    """
    Walk through samples in the terminal; allow user to override correctness.
    Commands:
      y = mark correct
      n = mark incorrect
      s = skip (keep auto verdict)
      q = quit
    """
    print("\nEntering manual-check mode. Commands: [y]=correct, [n]=incorrect, [s]=skip, [q]=quit.\n")
    for p in preds:
        v = p.verdict
        print("=" * 80)
        print(f"[Problem ID {p.id}] Sample {p.sample_ix}")
        print(f"QUESTION:\n{p.question}\n")
        print("GOLD (tail):")
        print((p.gold or "").splitlines()[-5:])
        print("\nMODEL PREDICTION:\n", p.pred.strip(), "\n", sep="")
        print(f"AUTO VERDICT: {v.get('correct')} | gold={v.get('gold_value')} (raw='{v.get('gold_raw')}') | "
              f"pred={v.get('pred_value')} (raw='{v.get('pred_raw')}') | {v.get('reason')}")
        while True:
            cmd = input("[y/n/s/q] > ").strip().lower()
            if cmd in ("y", "n", "s", "q"):
                break
        if cmd == "q":
            print("Exiting manual-check.")
            break
        if cmd == "y":
            p.verdict["correct"] = True
            p.verdict["reason"] = "Manually overridden: correct."
        elif cmd == "n":
            p.verdict["correct"] = False
            p.verdict["reason"] = "Manually overridden: incorrect."
        # 's' => keep auto verdict

# ======================================================================================
# CLI
# ======================================================================================

def parse_args():
    ap = argparse.ArgumentParser(description="GSM8K verification with HuggingFace model, multi-sample, and manual checking.")
    ap.add_argument("--model_name_or_path", type=str, default=None,
                    help="HF model to run generation. If omitted, use --load_preds to verify existing predictions.")
    ap.add_argument("--split", type=str, default="test", choices=["train", "test", "validation", "main", "test_scratch"],
                    help="GSM8K split; commonly 'train' or 'test'.")
    ap.add_argument("--max_samples", type=int, default=None, help="Limit number of problems.")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", type=str, default="auto", help="'auto', 'cuda', 'cpu', or 'cuda:0', etc.")
    ap.add_argument("--dtype", type=str, default="bfloat16", help="torch dtype: float16, bfloat16, float32")
    ap.add_argument("--temperature", type=float, default=0.7, help=">0 to sample (recommended for multi-sample).")
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_samples", type=int, default=1, help="Number of samples per problem (>=1).")

    ap.add_argument("--save_preds", type=str, default=None, help="Path to save per-sample predictions JSONL.")
    ap.add_argument("--load_preds", type=str, default=None, help="Verify from an existing per-sample JSONL.")
    ap.add_argument("--manual", action="store_true", help="Enable interactive manual checking loop.")
    return ap.parse_args()

def main():
    args = parse_args()

    if args.load_preds and args.model_name_or_path:
        print("Both --load_preds and --model_name_or_path provided. "
              "I'll verify loaded predictions and ignore generation.", file=sys.stderr)

    if args.load_preds:
        preds = load_preds(args.load_preds)
    elif args.model_name_or_path:
        preds = generate_predictions(
            model_name_or_path=args.model_name_or_path,
            split=args.split,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            device=args.device,
            dtype=args.dtype,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            n_samples=args.n_samples,
        )
        if args.save_preds:
            save_preds(args.save_preds, preds)
    else:
        print("Error: provide --model_name_or_path to generate or --load_preds to verify.", file=sys.stderr)
        sys.exit(1)

    # Auto summary (before manual overrides)
    sample_stats = summarize_sample_accuracy(preds)
    any_stats = summarize_any_correct(preds)
    print("\nAuto summary (before manual overrides):")
    print(json.dumps({**sample_stats, **any_stats}, indent=2))

    # Optional manual check override
    if args.manual:
        interactive_manual_check(preds)
        print("\nAfter manual overrides:")
        sample_stats = summarize_sample_accuracy(preds)
        any_stats = summarize_any_correct(preds)
        print(json.dumps({**sample_stats, **any_stats}, indent=2))

        # Re-save if paths provided
        if args.load_preds:
            save_preds(args.load_preds, preds)
            print(f"Saved updated predictions to: {args.load_preds}")
        elif args.save_preds:
            save_preds(args.save_preds, preds)
            print(f"Saved predictions to: {args.save_preds}")

if __name__ == "__main__":
    main()
