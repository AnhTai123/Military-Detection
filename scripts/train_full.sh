#!/usr/bin/env bash
# Full training run — 3000 steps, 1 GPU, no SLURM, LoRA.
# Run from anywhere — script auto-navigates to Eagle/Embodied.
# Usage:  bash scripts/train_full.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EAGLE_DIR=/home/aiplatform/workspace/Eagle/Embodied

cd "${EAGLE_DIR}"

# 1. Sync recipe + deepspeed config — skip if same file
RECIPE_DST="${EAGLE_DIR}/locany_recipe/military_all_classes_recipe.json"
mkdir -p "$(dirname "${RECIPE_DST}")"
if [ "${REPO_DIR}/locany_recipe/military_all_classes_recipe.json" != "${RECIPE_DST}" ]; then
  cp "${REPO_DIR}/locany_recipe/military_all_classes_recipe.json" "${RECIPE_DST}"
fi

DS_DST="${EAGLE_DIR}/deepspeed_configs/zero_stage2_config.json"
mkdir -p "$(dirname "${DS_DST}")"
if [ "${REPO_DIR}/deepspeed_configs/zero_stage2_config.json" != "${DS_DST}" ]; then
  cp "${REPO_DIR}/deepspeed_configs/zero_stage2_config.json" "${DS_DST}"
fi

# 2. Apply LoRA config
echo "[INFO] Applying LoRA config..."
python - <<'PYEOF'
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("nvidia/LocateAnything-3B", trust_remote_code=True)
cfg.use_llm_lora = 64
cfg.use_backbone_lora = 64
cfg.save_pretrained("/tmp/locateanything_lora_config")
print("[INFO] LoRA config saved.")
PYEOF

# 3. Run full training
OUT_DIR="${EAGLE_DIR}/work_dirs/locany_military_all_full"
mkdir -p "${OUT_DIR}"

# Clear stale done.txt — the script exits immediately if it exists.
rm -f "${OUT_DIR}/done.txt"

echo "[INFO] Starting full training (3000 steps, LoRA)..."
LAUNCHER=pytorch CUDA_VISIBLE_DEVICES=0 torchrun \
  --standalone \
  --nproc_per_node=1 \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path nvidia/LocateAnything-3B \
  --meta_path ./locany_recipe/military_all_classes_recipe.json \
  --output_dir "${OUT_DIR}" \
  --do_train True \
  --max_steps 3000 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --bf16 True \
  --block_size 6 \
  --attn_implementation sdpa \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_seq_length 8192 \
  --save_steps 500 \
  --logging_steps 10 \
  --report_to tensorboard \
  --grad_checkpoint True \
  --deepspeed deepspeed_configs/zero_stage2_config.json \
  2>&1 | tee "${OUT_DIR}/training_log.txt"

echo "[INFO] Full training finished."
