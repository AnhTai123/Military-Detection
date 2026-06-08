#!/usr/bin/env bash
# Eval checkpoint fine-tuned LocateAnything trên test set.
# Chạy sau khi train xong.
#
# Usage:
#   bash scripts/eval_locateanything_checkpoint.sh \
#       /path/to/checkpoint_dir   \   # thư mục chứa checkpoint (có config.json)
#       test                           # split: test | val | train

set -euo pipefail

CKPT_DIR="${1:?Cần truyền path checkpoint}"
SPLIT="${2:-test}"

ANN="/home/aiplatform/workspace/research/research_res/merged_dataset/labels/${SPLIT}_gdino.json"
IMG_ROOT="/home/aiplatform/workspace/research/research_res/merged_dataset/images"
OUT_DIR="/home/aiplatform/workspace/outputs_locateanything_finetuned_${SPLIT}"

echo "[INFO] Checkpoint : ${CKPT_DIR}"
echo "[INFO] Split      : ${SPLIT}"
echo "[INFO] Output     : ${OUT_DIR}"

conda activate locateanything

# dùng script inference đã có, chỉ đổi MODEL_ID sang checkpoint fine-tuned
python infer_locateanything_military.py \
    --ann        "${ANN}" \
    --img-root   "${IMG_ROOT}" \
    --out-dir    "${OUT_DIR}" \
    --model-id   "${CKPT_DIR}" \
    --device     cuda:0 \
    --mode       hybrid \
    --max-new-tokens 512

echo "[INFO] Eval done → ${OUT_DIR}/eval_report.md"
