#!/usr/bin/env bash
# Debug training run — 100 steps, 1 GPU, no SLURM.
# Run from anywhere — script auto-navigates to Eagle/Embodied.
# Usage:  bash scripts/train_debug.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EAGLE_DIR=/home/aiplatform/workspace/Eagle/Embodied

cd "${EAGLE_DIR}"

# 1. Attention backend: prefer flash-attn (avoids the 33k×33k attention matrix
#    that SDPA materializes in the vision encoder -> 36GiB OOM). Fall back to
#    sdpa + reduced vision tokens only if flash-attn can't be installed.
echo "[INFO] Ensuring flash-attn is available..."
if python -c "import flash_attn" 2>/dev/null; then
  echo "[INFO] flash-attn already installed."
  ATTN_IMPL=flash_attention_2
  python "${REPO_DIR}/scripts/patch_flash_attn.py" --root "${EAGLE_DIR}" --revert || true
elif pip install flash-attn --no-build-isolation 2>&1 | tail -5 && python -c "import flash_attn" 2>/dev/null; then
  echo "[INFO] flash-attn installed successfully."
  ATTN_IMPL=flash_attention_2
  python "${REPO_DIR}/scripts/patch_flash_attn.py" --root "${EAGLE_DIR}" --revert || true
else
  echo "[WARN] flash-attn unavailable — falling back to sdpa + reduced vision tokens."
  ATTN_IMPL=sdpa
  python "${REPO_DIR}/scripts/patch_flash_attn.py" --root "${EAGLE_DIR}"
fi
echo "[INFO] Attention backend: ${ATTN_IMPL}"

# 2. Convert dataset (idempotent)
JSONL_TRAIN="${EAGLE_DIR}/locany_military_data_all/train_all_classes.jsonl"
if [ ! -f "${JSONL_TRAIN}" ]; then
  echo "[INFO] Converting COCO dataset → JSONL..."
  python "${REPO_DIR}/convert_coco_to_locany_all_classes.py"
else
  echo "[INFO] JSONL already exists, skipping conversion."
fi

# 3. Always sync recipe (format may have changed) — skip if same file
RECIPE_DST="${EAGLE_DIR}/locany_recipe/military_all_classes_recipe.json"
mkdir -p "$(dirname "${RECIPE_DST}")"
if [ "${REPO_DIR}/locany_recipe/military_all_classes_recipe.json" != "${RECIPE_DST}" ]; then
  cp "${REPO_DIR}/locany_recipe/military_all_classes_recipe.json" "${RECIPE_DST}"
fi
echo "[INFO] Recipe ready → ${RECIPE_DST}"

# 4. Sync deepspeed config — skip if same file
DS_DST="${EAGLE_DIR}/deepspeed_configs/zero_stage2_config.json"
mkdir -p "$(dirname "${DS_DST}")"
if [ "${REPO_DIR}/deepspeed_configs/zero_stage2_config.json" != "${DS_DST}" ]; then
  cp "${REPO_DIR}/deepspeed_configs/zero_stage2_config.json" "${DS_DST}"
fi
echo "[INFO] DeepSpeed config ready."

# 5. Create a stub nvcc so DeepSpeed can read the CUDA version without nvcc
#    being actually installed. DS_BUILD_OPS=0 ensures nothing gets compiled.
NVCC_STUB=/home/aiplatform/.conda/envs/locateanything/bin/nvcc
if [ ! -f "${NVCC_STUB}" ]; then
  echo "[INFO] Creating stub nvcc for DeepSpeed version check..."
  cat > "${NVCC_STUB}" <<'NVCCEOF'
#!/bin/bash
echo "Cuda compilation tools, release 11.8, V11.8.89"
NVCCEOF
  chmod +x "${NVCC_STUB}"
fi

# 6. Build a COMPLETE local model dir with LoRA enabled in config.
#    (weights symlinked from HF cache + patched config.json). Pointing
#    --model_name_or_path at this dir makes from_pretrained actually honor
#    use_llm_lora / use_backbone_lora — saving only a /tmp config.json does NOT,
#    because the weights are missing and the base model gets loaded instead.
LORA_MODEL_DIR=/tmp/LocateAnything-3B-lora
echo "[INFO] Building LoRA model dir..."
python "${REPO_DIR}/scripts/build_lora_model_dir.py" \
  --base nvidia/LocateAnything-3B \
  --out "${LORA_MODEL_DIR}" \
  --llm_lora 64 \
  --backbone_lora 0

# 6. Run debug training
OUT_DIR="${EAGLE_DIR}/work_dirs/locany_military_all_debug"
mkdir -p "${OUT_DIR}"

# IMPORTANT: the training script exits immediately if done.txt exists in
# output_dir (the check runs right after init_dist, before logging is even
# configured, so the "Training done" message is swallowed). A previous run
# that finished with 0 steps leaves done.txt behind and blocks all reruns.
rm -f "${OUT_DIR}/done.txt"
echo "[INFO] Cleared stale done.txt (if any)."

echo "[INFO] Starting debug training (100 steps, LoRA)..."
CUDA_HOME=/home/aiplatform/.conda/envs/locateanything \
  LAUNCHER=pytorch CUDA_VISIBLE_DEVICES=0 DS_BUILD_OPS=0 DS_SKIP_CUDA_CHECK=1 torchrun \
  --standalone \
  --nproc_per_node=1 \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path "${LORA_MODEL_DIR}" \
  --meta_path ./locany_recipe/military_all_classes_recipe.json \
  --output_dir "${OUT_DIR}" \
  --do_train True \
  --max_steps 100 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --bf16 True \
  --block_size 6 \
  --attn_implementation "${ATTN_IMPL}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_seq_length 8192 \
  --save_steps 50 \
  --logging_steps 10 \
  --report_to tensorboard \
  --grad_checkpoint True \
  --freeze_backbone True \
  2>&1 | tee "${OUT_DIR}/training_log.txt"

echo "[INFO] Debug training finished."
