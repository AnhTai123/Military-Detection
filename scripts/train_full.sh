#!/usr/bin/env bash
# Full training run — 3000 steps, 1 GPU, no SLURM, no flash_attn.
# Run from: /home/aiplatform/workspace/Eagle/Embodied
# Usage:  bash scripts/train_full.sh
#
# Memory profile (A100 80GB, 3B model, bf16, freeze_backbone=True):
#   batch=1, grad_accum=8  → ~30-40 GB  (safe, effective batch = 8)
#   batch=2, grad_accum=4  → ~45-60 GB  (tight, may OOM with long sequences)
#   batch=4, grad_accum=2  → ~65-75 GB  (risky)
# Keeping batch=1, grad_accum=8 for safety (same effective batch size as debug ×2).

set -euo pipefail

cd /home/aiplatform/workspace/Eagle/Embodied

mkdir -p work_dirs/locany_military_all_full

echo "[INFO] Starting full training (3000 steps)..."
CUDA_VISIBLE_DEVICES=0 LAUNCHER=pytorch torchrun \
  --standalone \
  --nproc_per_node=1 \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path nvidia/LocateAnything-3B \
  --meta_path ./locany_recipe/military_all_classes_recipe.json \
  --output_dir work_dirs/locany_military_all_full \
  --max_steps 3000 \
  --learning_rate 2e-5 \
  --bf16 True \
  --block_size 6 \
  --attn_implementation sdpa \
  --causal_attn False \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 3 \
  --weight_decay 0.01 \
  --warmup_steps 100 \
  --lr_scheduler_type cosine \
  --max_grad_norm 1.0 \
  --logging_steps 10 \
  --packing_buffer_size 16 \
  --max_seq_length 4096 \
  --max_num_tokens_per_sample 4096 \
  --max_num_tokens 4096 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --deepspeed deepspeed_configs/zero_stage2_config.json \
  --report_to tensorboard \
  --mlp_connector_layers 2 \
  2>&1 | tee work_dirs/locany_military_all_full/training_log.txt

echo "[INFO] Full training finished."
