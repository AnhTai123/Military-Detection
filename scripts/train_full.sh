#!/usr/bin/env bash
# FULL training run — STEPS env (default 500), 1 GPU, no SLURM.
# Run from anywhere — script auto-navigates to Eagle/Embodied.
# Usage:  STEPS=500 bash scripts/train_full.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EAGLE_DIR=/home/aiplatform/workspace/Eagle/Embodied

cd "${EAGLE_DIR}"

# 1. Attention backend: prefer flash-attn (avoids 33k×33k SDPA matrix → OOM).
#    Override with FORCE_SDPA=1 if the flash text path misbehaves
#    (e.g. absurd multi-TiB torch.gather allocation).
echo "[INFO] Checking flash-attn..."
if [ "${FORCE_SDPA:-0}" = "1" ]; then
  echo "[INFO] FORCE_SDPA=1 — using sdpa even though flash-attn may exist."
  ATTN_IMPL=sdpa
  python "${REPO_DIR}/scripts/patch_flash_attn.py" --root "${EAGLE_DIR}"
elif python -c "import flash_attn" 2>/dev/null; then
  echo "[INFO] flash-attn available."
  ATTN_IMPL=flash_attention_2
  python "${REPO_DIR}/scripts/patch_flash_attn.py" --root "${EAGLE_DIR}" --revert || true
else
  echo "[WARN] flash-attn not found — falling back to sdpa."
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

# 3. Sync recipe — skip if same file
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

# 5. Build a complete local model dir. IMPORTANT: use_llm_lora must be 0 here!
#    Config-based LoRA wraps PEFT inside __init__, renaming all LLM params ->
#    from_pretrained can't match checkpoint keys -> whole LLM random-init -> NaN.
#    LoRA is applied AFTER loading via patch_wrap_lora.py (step 6e) instead.
LORA_MODEL_DIR=/tmp/LocateAnything-3B-lora
echo "[INFO] Building model dir (LoRA wrapped after load, not in config)..."
python "${REPO_DIR}/scripts/build_lora_model_dir.py" \
  --base nvidia/LocateAnything-3B \
  --out "${LORA_MODEL_DIR}" \
  --llm_lora 0 \
  --backbone_lora 0

# 5b. Sanitize NaN/Inf in safetensors files before training.
#     The LocateAnything-3B checkpoint ships with NaN in norm weights (likely
#     a cast artefact). Fix them on disk so the model loads with clean weights.
echo "[INFO] Sanitizing model weights (NaN/Inf -> 1.0 for norms, 0.0 otherwise)..."
python "${REPO_DIR}/scripts/sanitize_model_weights.py" \
  --model_dir "${LORA_MODEL_DIR}"

# 6. Patch MoonViT: replace sdpa_attention with a per-segment implementation.
#    Mathematically identical to the original block-diagonal mask (packing),
#    but never materialises the N×N matrix -> no OOM, no special kernels.
#    The script auto-restores from .orig_bak first, so it is idempotent.
echo "[INFO] Patching MoonViT sdpa_attention (per-segment)..."
python "${REPO_DIR}/scripts/patch_sdpa_segments.py" --root "${EAGLE_DIR}"

# 6b. Patch LocateAnything: clone input_embeds before in-place scatter.
#     Frozen embeddings (LoRA) + grad checkpointing make input_embeds a leaf
#     requiring grad -> in-place write raises RuntimeError without the clone.
echo "[INFO] Patching input_embeds clone..."
python "${REPO_DIR}/scripts/patch_clone_embeds.py" --root "${EAGLE_DIR}"

# 6c. Patch SDPA packing masks: force diagonal visibility so no row is
#     fully -inf (fully-masked rows make softmax produce NaN -> loss=nan).
echo "[INFO] Patching SDPA mask diagonal..."
python "${REPO_DIR}/scripts/patch_mask_diag.py" --root "${EAGLE_DIR}"

# 6e. Wrap LLM LoRA AFTER from_pretrained so checkpoint keys match during load.
echo "[INFO] Patching wrap_llm_lora after model load..."
python "${REPO_DIR}/scripts/patch_wrap_lora.py" --root "${EAGLE_DIR}" --rank 64 --alpha 128

# 6f. Fix flash_attention_2 text path: 4D packing mask causes _upad_input to
#     misread kv_seq_len as seq_len^2 -> torch.gather allocates 4870 GiB OOM.
#     Patch passes None mask to _upad_input when mask is 4D float.
echo "[INFO] Patching Qwen2 flash attention _upad_input..."
python "${REPO_DIR}/scripts/patch_flash_qwen2.py" --root "${EAGLE_DIR}"

# 6d. (Diagnostic probe disabled — root cause found: NaN in RMSNorm weights,
#      now fixed by the sanitize block inside patch_clone_embeds.py. Re-enabling
#      the probe would restore a stale backup and wipe the sanitize patch.)
# python "${REPO_DIR}/scripts/patch_layer_probe.py" --root "${EAGLE_DIR}"

# 7. Run debug training
OUT_DIR="${EAGLE_DIR}/work_dirs/locany_military_all_full"
mkdir -p "${OUT_DIR}"

# The training script exits immediately if done.txt exists in output_dir.
rm -f "${OUT_DIR}/done.txt"
echo "[INFO] Cleared stale done.txt (if any)."

echo "[INFO] Starting FULL training (${STEPS:-500} steps, LoRA)..."
# Speed knobs (override via env):
#   ACCUM=4    -> 4 fwd/bwd per step instead of 8 (~2x faster/step)
#   NO_CKPT=1  -> disable gradient checkpointing (~30-40% faster,
#                 higher memory; revert to NO_CKPT=0 if OOM)
ACCUM="${ACCUM:-4}"
GRAD_CKPT=True
if [ "${NO_CKPT:-0}" = "1" ]; then GRAD_CKPT=False; fi
echo "[INFO] gradient_accumulation_steps=${ACCUM}  grad_checkpoint=${GRAD_CKPT}"
# Start checkpoint watcher in background: keeps checkpoint-best (lowest loss)
# alongside checkpoint-last (save_total_limit=1 rolling checkpoint).
python "${REPO_DIR}/scripts/watch_best_ckpt.py" \
  --out_dir "${OUT_DIR}" --poll_secs 30 &
WATCHER_PID=$!
echo "[INFO] Checkpoint watcher started (pid=${WATCHER_PID})"

LAUNCHER=pytorch CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun \
  --standalone \
  --nproc_per_node=1 \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path "${LORA_MODEL_DIR}" \
  --meta_path ./locany_recipe/military_all_classes_recipe.json \
  --output_dir "${OUT_DIR}" \
  --do_train True \
  --max_steps "${STEPS:-500}" \
  --learning_rate 2e-5 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --bf16 True \
  --block_size 6 \
  --attn_implementation "${ATTN_IMPL}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "${ACCUM}" \
  --max_seq_length 8192 \
  --save_steps 50 \
  --save_total_limit 1 \
  --logging_steps 10 \
  --report_to tensorboard \
  --gradient_checkpointing "${GRAD_CKPT}" \
  --freeze_backbone True \
  --optim adamw_torch \
  2>&1 | tee "${OUT_DIR}/training_log.txt"

wait "${WATCHER_PID}" 2>/dev/null || true

echo "[INFO] Final checkpoints in ${OUT_DIR}:"
ls -d "${OUT_DIR}"/checkpoint-* 2>/dev/null | xargs -I{} du -sh {} 2>/dev/null
echo "[INFO] Full training finished."
