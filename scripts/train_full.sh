#!/usr/bin/env bash
# Full training run — 3000 steps, 1 GPU, no SLURM, LoRA.
# Run from anywhere — script auto-navigates to Eagle/Embodied.
# Usage:  bash scripts/train_full.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EAGLE_DIR=/home/aiplatform/workspace/Eagle/Embodied

cd "${EAGLE_DIR}"

# 0. Attention backend: prefer flash-attn (avoids the vision-encoder attention
#    matrix that SDPA materializes -> OOM). Fall back to sdpa if unavailable.
echo "[INFO] Ensuring flash-attn is available..."
if python -c "import flash_attn" 2>/dev/null; then
  ATTN_IMPL=flash_attention_2
  python "${REPO_DIR}/scripts/patch_flash_attn.py" --root "${EAGLE_DIR}" --revert || true
elif pip install flash-attn --no-build-isolation 2>&1 | tail -5 && python -c "import flash_attn" 2>/dev/null; then
  ATTN_IMPL=flash_attention_2
  python "${REPO_DIR}/scripts/patch_flash_attn.py" --root "${EAGLE_DIR}" --revert || true
else
  echo "[WARN] flash-attn unavailable — falling back to sdpa."
  ATTN_IMPL=sdpa
  python "${REPO_DIR}/scripts/patch_flash_attn.py" --root "${EAGLE_DIR}"
fi
echo "[INFO] Attention backend: ${ATTN_IMPL}"

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

# 1b. Create a stub nvcc so DeepSpeed can read the CUDA version without nvcc
#     being actually installed. DS_BUILD_OPS=0 ensures nothing gets compiled.
NVCC_STUB=/home/aiplatform/.conda/envs/locateanything/bin/nvcc
if [ ! -f "${NVCC_STUB}" ]; then
  echo "[INFO] Creating stub nvcc for DeepSpeed version check..."
  cat > "${NVCC_STUB}" <<'NVCCEOF'
#!/bin/bash
echo "Cuda compilation tools, release 11.8, V11.8.89"
NVCCEOF
  chmod +x "${NVCC_STUB}"
fi

# 2. Build a COMPLETE local model dir with LoRA enabled in config.
#    (weights symlinked from HF cache + patched config.json). Required so
#    from_pretrained actually honors use_llm_lora / use_backbone_lora.
LORA_MODEL_DIR=/tmp/LocateAnything-3B-lora
echo "[INFO] Building LoRA model dir..."
python "${REPO_DIR}/scripts/build_lora_model_dir.py" \
  --base nvidia/LocateAnything-3B \
  --out "${LORA_MODEL_DIR}" \
  --llm_lora 64 \
  --backbone_lora 0

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
  --model_name_or_path "${LORA_MODEL_DIR}" \
  --meta_path ./locany_recipe/military_all_classes_recipe.json \
  --output_dir "${OUT_DIR}" \
  --do_train True \
  --max_steps 3000 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --bf16 True \
  --block_size 6 \
  --attn_implementation "${ATTN_IMPL}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_seq_length 8192 \
  --save_steps 500 \
  --logging_steps 10 \
  --report_to tensorboard \
  --grad_checkpoint True \
  --freeze_backbone True \
  --optim adamw_torch \
  2>&1 | tee "${OUT_DIR}/training_log.txt"

echo "[INFO] Full training finished."
