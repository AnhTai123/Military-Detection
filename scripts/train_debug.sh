#!/usr/bin/env bash
# Debug training run — 100 steps, 1 GPU, no SLURM, no flash_attn.
# Run from: /home/aiplatform/workspace/Eagle/Embodied
# Usage:  bash scripts/train_debug.sh [port]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PORT="${1:-29510}"

cd /home/aiplatform/workspace/Eagle/Embodied

# 1. Apply flash_attn → sdpa patches (idempotent)
echo "[INFO] Applying flash_attn patches..."
python "${REPO_DIR}/scripts/patch_flash_attn.py" \
  --root /home/aiplatform/workspace/Eagle/Embodied
echo "[INFO] Patches applied."

# 2. Convert dataset (idempotent — skips if outputs already exist)
JSONL_TRAIN=/home/aiplatform/workspace/Eagle/Embodied/locany_military_data_all/train_all_classes.jsonl
if [ ! -f "${JSONL_TRAIN}" ]; then
  echo "[INFO] Converting COCO dataset → JSONL..."
  python "${REPO_DIR}/convert_coco_to_locany_all_classes.py"
else
  echo "[INFO] JSONL dataset already exists, skipping conversion."
fi

# 3. Copy recipe if not present
RECIPE_DST=/home/aiplatform/workspace/Eagle/Embodied/locany_recipe/military_all_classes_recipe.json
if [ ! -f "${RECIPE_DST}" ]; then
  mkdir -p "$(dirname "${RECIPE_DST}")"
  cp "${REPO_DIR}/locany_recipe/military_all_classes_recipe.json" "${RECIPE_DST}"
  echo "[INFO] Recipe copied → ${RECIPE_DST}"
fi

# 4. Copy deepspeed config if not present
DS_DST=/home/aiplatform/workspace/Eagle/Embodied/deepspeed_configs/zero_stage2_config.json
if [ ! -f "${DS_DST}" ]; then
  mkdir -p "$(dirname "${DS_DST}")"
  cp "${REPO_DIR}/deepspeed_configs/zero_stage2_config.json" "${DS_DST}"
  echo "[INFO] DeepSpeed config copied → ${DS_DST}"
fi

# 5. Run debug training
mkdir -p /home/aiplatform/workspace/Eagle/Embodied/work_dirs/locany_military_all_debug

echo "[INFO] Starting debug training (100 steps)..."
CUDA_VISIBLE_DEVICES=0 LAUNCHER=pytorch torchrun \
  --standalone \
  --nproc_per_node=1 \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path nvidia/LocateAnything-3B \
  --meta_path ./locany_recipe/military_all_classes_recipe.json \
  --output_dir work_dirs/locany_military_all_debug \
  --max_steps 100 \
  --learning_rate 2e-5 \
  --bf16 True \
  --block_size 6 \
  --attn_implementation sdpa \
  --causal_attn False \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 2 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --save_strategy steps \
  --save_steps 50 \
  --save_total_limit 2 \
  --weight_decay 0.01 \
  --warmup_steps 10 \
  --lr_scheduler_type cosine \
  --max_grad_norm 1.0 \
  --logging_steps 1 \
  --packing_buffer_size 8 \
  --max_seq_length 2048 \
  --max_num_tokens_per_sample 2048 \
  --max_num_tokens 2048 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --deepspeed deepspeed_configs/zero_stage2_config.json \
  --report_to tensorboard \
  --mlp_connector_layers 2 \
  2>&1 | tee work_dirs/locany_military_all_debug/training_log.txt

echo "[INFO] Debug training finished."
