# Hướng dẫn chi tiết — Fine-tune LocateAnything-3B (10 Military Classes)

## Tổng quan luồng

```
COCO JSON (train/val/test)
        │
        ▼  (Bước 2)
JSONL all-classes  ──────────────────────────────────┐
        │                                            │
        ▼  (Bước 3)                                  │
  Recipe JSON                                        │
        │                                            │
        ▼  (Bước 4 → 5)                              │
  Fine-tune LocateAnything-3B                        │
        │                                            │
        ▼  (Bước 6)                                  │
  Checkpoint fine-tuned  ←────────────────────────── ┘
        │
        ▼  (Bước 7)
  Eval trên test set → so sánh với base model
```

---

## Môi trường

| Thứ | Giá trị |
|---|---|
| Conda env | `locateanything` |
| Repo Eagle | `/home/aiplatform/workspace/Eagle/Embodied` |
| GPU | A100 80GB |
| Model gốc | `nvidia/LocateAnything-3B` |

---

## Bước 0 — Copy scripts từ repo này vào Eagle

```bash
EAGLE=/home/aiplatform/workspace/Eagle/Embodied
REPO=/path/to/Military-Detection   # thư mục clone repo này

# scripts
cp $REPO/convert_coco_to_locany_all_classes.py  $EAGLE/
cp $REPO/scripts/verify_jsonl.py                $EAGLE/scripts/
cp $REPO/scripts/patch_flash_attn.py            $EAGLE/scripts/
cp $REPO/scripts/train_debug.sh                 $EAGLE/scripts/
cp $REPO/scripts/train_full.sh                  $EAGLE/scripts/

# recipe + deepspeed config
cp -r $REPO/locany_recipe      $EAGLE/
cp -r $REPO/deepspeed_configs  $EAGLE/   # bỏ qua nếu đã có

# inference + eval
cp $REPO/infer_locateanything_military.py $EAGLE/
```

---

## Bước 1 — Kiểm tra môi trường

```bash
conda activate locateanything
cd /home/aiplatform/workspace/Eagle/Embodied

# kiểm tra Python packages cần thiết
python -c "import torch; print('torch', torch.__version__)"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import deepspeed; print('deepspeed', deepspeed.__version__)"

# kiểm tra GPU
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# kiểm tra training script tồn tại
ls eaglevl/train/locany_finetune_magi_stream.py
```

---

## Bước 2 — Convert COCO → JSONL

LocateAnything không đọc trực tiếp COCO JSON. Cần convert sang JSONL
theo format `conversations` với bbox normalized [0, 1000].

### Format JSONL (1 dòng = 1 ảnh)

```json
{
  "image": "abc123.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nZumwalt class destroyer . Admiral Gorshkov class frigate . F-22 Raptor fighter jet . F-35 Lightning II fighter jet . BM-30 Smerch multiple rocket launcher . M2 Bradley infantry fighting vehicle . BTR-90 armored personnel carrier . HIMARS rocket artillery launcher . M1 Abrams main battle tank . T-90 main battle tank ."
    },
    {
      "from": "gpt",
      "value": "<ref> F-22 Raptor fighter jet </ref><box><412><100><890><530></box><ref> M1 Abrams main battle tank </ref><box><10><600><450><950></box>"
    }
  ]
}
```

- Prompt = toàn bộ 10 class, nối bằng ` . `
- Mỗi ảnh → 1 sample (dù ảnh có nhiều object)
- COCO bbox `[x, y, w, h]` → `[x1, y1, x2, y2]` normalized [0, 1000]

### Chạy convert

```bash
cd /home/aiplatform/workspace/Eagle/Embodied
conda activate locateanything

python convert_coco_to_locany_all_classes.py
```

Xuất ra:
```
locany_military_data_all/
├── train_all_classes.jsonl   ← dùng để train
├── val_all_classes.jsonl     ← dùng để monitor
└── test_all_classes.jsonl
```

### Kiểm tra JSONL sau khi convert

```bash
python scripts/verify_jsonl.py \
    --jsonl locany_military_data_all/train_all_classes.jsonl \
    --img-root /home/aiplatform/workspace/research/research_res/merged_dataset/images \
    --n-show 3
```

Output mong đợi:
```
[RESULT] OK — dữ liệu sẵn sàng để train!
```

Nếu có warning `ảnh không tồn tại` → kiểm tra lại `IMAGE_ROOT` trong `convert_coco_to_locany_all_classes.py`.

---

## Bước 3 — Kiểm tra Recipe

File `locany_recipe/military_all_classes_recipe.json`:

```json
{
  "military_detection_all_classes": {
    "annotation": "/home/aiplatform/workspace/Eagle/Embodied/locany_military_data_all/train_all_classes.jsonl",
    "root": "/home/aiplatform/workspace/research/research_res/merged_dataset/images",
    "repeat_time": 1.0,
    "data_augment": true
  }
}
```

Kiểm tra 2 đường dẫn trong file này có đúng không:
```bash
ls /home/aiplatform/workspace/Eagle/Embodied/locany_military_data_all/train_all_classes.jsonl
ls /home/aiplatform/workspace/research/research_res/merged_dataset/images | head -5
```

---

## Bước 4 — Patch flash_attn → sdpa

Script tự động tìm và sửa tất cả chỗ hard-code `flash_attention_2`
trong repo Eagle, đồng thời wrap `import flash_attn` trong try/except.

```bash
# Xem trước (dry run)
python scripts/patch_flash_attn.py --dry-run

# Áp dụng thật
python scripts/patch_flash_attn.py
```

---

## Bước 5 — Train

### 5a. Debug 100 steps (chạy trước để kiểm tra không lỗi)

```bash
conda activate locateanything
cd /home/aiplatform/workspace/Eagle/Embodied

mkdir -p work_dirs/locany_military_all_debug

CUDA_VISIBLE_DEVICES=0 torchrun \
  --standalone \
  --nproc_per_node=1 \
  eaglevl/train/locany_finetune_magi_stream.py \
  --launcher pytorch \
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
```

**Kiểm tra debug thành công:**
```bash
# Phải thấy dòng "Step X/100 | loss: ..."
grep "loss" work_dirs/locany_military_all_debug/training_log.txt | tail -5

# Checkpoint phải xuất hiện
ls work_dirs/locany_military_all_debug/
```

### 5b. Full training 3000 steps

Chỉ chạy sau khi debug 100 steps OK:

```bash
mkdir -p work_dirs/locany_military_all_full

CUDA_VISIBLE_DEVICES=0 torchrun \
  --standalone \
  --nproc_per_node=1 \
  eaglevl/train/locany_finetune_magi_stream.py \
  --launcher pytorch \
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
  --num_train_epochs 10 \
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
  --max_seq_length 2048 \
  --max_num_tokens_per_sample 2048 \
  --max_num_tokens 2048 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --deepspeed deepspeed_configs/zero_stage2_config.json \
  --report_to tensorboard \
  --mlp_connector_layers 2 \
  2>&1 | tee work_dirs/locany_military_all_full/training_log.txt
```

**Monitor training:**
```bash
# Theo dõi loss giảm không
tail -f work_dirs/locany_military_all_full/training_log.txt

# Tensorboard
tensorboard --logdir work_dirs/locany_military_all_full --port 6007
```

**Giải thích các tham số quan trọng:**

| Tham số | Giá trị | Lý do |
|---|---|---|
| `--freeze_backbone True` | đóng băng vision encoder | Tiết kiệm VRAM ~30-40%, giữ feature ảnh tốt |
| `--freeze_llm False` | train LLM | LLM cần học cách output `<ref>...<box>` |
| `--attn_implementation sdpa` | không dùng flash_attn | Tránh cài thêm package |
| `--grad_checkpoint True` | gradient checkpointing | Giảm VRAM activation ~40% |
| `--per_device_train_batch_size 1` | batch 1 | An toàn với A100 80GB |
| `--gradient_accumulation_steps 8` | effective batch = 8 | Bù lại batch nhỏ |
| `--learning_rate 2e-5` | lr vừa phải | Không quá lớn (overfit) |

---

## Bước 6 — Theo dõi training

### Loss curve
```bash
# Loss phải giảm đều từ step 1 đến cuối
grep "loss" work_dirs/locany_military_all_full/training_log.txt | \
    awk '{print NR, $NF}' | \
    head -50
```

### Kiểm tra checkpoint
```bash
ls -lh work_dirs/locany_military_all_full/
# Phải thấy: checkpoint-500/, checkpoint-1000/, ...
```

### Dự kiến loss

| Giai đoạn | Loss dự kiến |
|---|---|
| Step 1–100 (warmup) | 2.0 – 4.0 |
| Step 500 | 1.0 – 2.0 |
| Step 1500 | 0.5 – 1.5 |
| Step 3000 | 0.3 – 1.0 |

Nếu loss không giảm sau 500 steps → tăng learning rate lên `5e-5`.

---

## Bước 7 — Eval checkpoint fine-tuned

### Chạy inference + eval trên test set

```bash
conda activate locateanything
cd /home/aiplatform/workspace/Eagle/Embodied

CKPT=work_dirs/locany_military_all_full/checkpoint-3000  # hoặc checkpoint mới nhất

python infer_locateanything_military.py \
    --model-id  "$CKPT" \
    --ann       /home/aiplatform/workspace/research/research_res/merged_dataset/labels/test_gdino.json \
    --img-root  /home/aiplatform/workspace/research/research_res/merged_dataset/images \
    --out-dir   /home/aiplatform/workspace/outputs_locateanything_finetuned \
    --device    cuda:0 \
    --mode      hybrid
```

### So sánh base model vs fine-tuned

```bash
# Base model (đã chạy trước đó)
cat /home/aiplatform/workspace/outputs_locateanything_military/eval_report.md

# Fine-tuned
cat /home/aiplatform/workspace/outputs_locateanything_finetuned/eval_report.md
```

Cột quan trọng để so sánh: **AP@0.50** và **Recall per class**.

---

## Bước 8 — Dùng trong tracking pipeline

Sau khi fine-tune xong và kết quả tốt, cắm vào tracking pipeline:

```python
# tracking_pipeline/vlm/locateanything_finetuned.py
# (viết adapter theo mẫu qwen_vl.py — đã có sẵn trong repo)

from tracking_pipeline.vlm.base import BaseVLM

class LocateAnythingFinetuned(BaseVLM):
    def __init__(self, checkpoint_path, device="cuda"):
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        self.tokenizer  = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        self.processor  = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True,
                                                         min_pixels=128*28*28, max_pixels=512*28*28)
        self.model      = AutoModel.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True).to(device).eval()
        self.device     = device

    def classify(self, crop):
        # ... (giống run_one_image trong infer_locateanything_military.py)
        pass
```

---

## Lỗi thường gặp và cách sửa

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `KeyError: 'SLURM_PROCID'` | Script dùng SLURM launcher | Thêm `--launcher pytorch` ✅ đã có |
| `ImportError: flash_attn` | flash_attn chưa cài | Chạy `python scripts/patch_flash_attn.py` |
| `Address already in use` | Port bị chiếm | Dùng `--standalone` ✅ đã có |
| `CUDA out of memory` | VRAM không đủ | Giảm `--packing_buffer_size 4`, tắt `--data_augment` |
| Loss không giảm | LR quá nhỏ hoặc dữ liệu sai | Tăng LR lên `5e-5`, chạy `verify_jsonl.py` |
| `No boxes parsed` khi eval | Format output sai | Xem raw answer trong JSON output, kiểm tra `parse_boxes()` |

---

## Bộ nhớ dự kiến (A100 80GB)

| Bước | VRAM |
|---|---|
| Load model | ~8 GB |
| Debug 100 steps | ~28–34 GB ✅ |
| Full training 3000 steps | ~30–40 GB ✅ |
| Inference / eval | ~12–16 GB ✅ |

---

## Checklist

- [ ] Bước 0: Copy scripts vào Eagle repo
- [ ] Bước 1: Kiểm tra môi trường (torch, transformers, deepspeed)
- [ ] Bước 2: Convert COCO → JSONL (`python convert_coco_to_locany_all_classes.py`)
- [ ] Bước 2: Verify JSONL (`python scripts/verify_jsonl.py ...`)
- [ ] Bước 3: Kiểm tra Recipe JSON đúng đường dẫn
- [ ] Bước 4: Patch flash_attn (`python scripts/patch_flash_attn.py`)
- [ ] Bước 5a: Debug 100 steps → không lỗi
- [ ] Bước 5b: Full training 3000 steps → loss giảm đều
- [ ] Bước 7: Eval fine-tuned vs base model → AP@0.50 tăng
- [ ] Bước 8: Tích hợp vào tracking pipeline
