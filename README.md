# VLPO
Variational Latent Policy Optimization

# Example
## SFT Baseline
```python
export WANDB_PROJECT=gsm8k-ddp
torchrun --nproc_per_node=4 train_ddp_sft_gsm8k.py \
  --model_id Qwen/Qwen2.5-0.5B-Instruct \
  --epochs 1 --batch_size 4 --lr 2e-5 --max_len 1024
```

## GRPO Baseline
```python
export WANDB_PROJECT=gsm8k-grpo-ddp
torchrun --nproc_per_node=4 train_ddp_grpo_gsm8k.py \
  --model_id Qwen/Qwen2.5-0.5B-Instruct \
  --ref_id   Qwen/Qwen2.5-0.5B-Instruct \
  --epochs 1 --batch_size 16 --group_size 4 --lr 1e-6 \
  --max_prompt_len 512 --max_gen_len 128 --beta_kl 0.02
```

## VLPO
```python
export WANDB_PROJECT=gsm8k-grpo-ddp
torchrun --nproc_per_node=4 train_ddp_grpo_gsm8k.py \
  --model_id Qwen/Qwen2.5-0.5B-Instruct \
  --ref_id   Qwen/Qwen2.5-0.5B-Instruct \
  --epochs 1 --batch_size 16 --group_size 4 --lr 1e-6 \
  --max_prompt_len 512 --max_gen_len 128 --beta_kl 0.02
```