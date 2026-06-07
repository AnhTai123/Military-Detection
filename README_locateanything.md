# Fine-tune LocateAnything-3B — Military Detection (All-Classes)

## Tổng quan

Fine-tune `nvidia/LocateAnything-3B` trên dataset 10 class quân sự theo kiểu **all-classes prompt**:
- Mỗi ảnh → 1 sample JSONL, prompt chứa toàn bộ 10 class.
- Output chứa tất cả object xuất hiện trong ảnh theo format `<ref>class</ref><box>`.
- Dùng `torchrun`, launcher `pytorch`, attention backend `sdpa` (không cần `flash_attn`).

---

## Cấu trúc file

```
Military-Detection/
├── convert_coco_to_locany_all_classes.py   # Convert COCO → JSONL
├── locany_recipe/
│   └── military_all_classes_recipe.json    # Training recipe
├── deepspeed_configs/
│   └── zero_stage2_config.json             # DeepSpeed ZeRO-2 config
└── scripts/
    ├── patch_flash_attn.py                 # Patch Eagle repo: flash_attn → sdpa
    ├── train_debug.sh                      # Debug run: 100 steps
    └── train_full.sh                       # Full run: 3000 steps
```

---

## Bước 1: Copy files vào Eagle repo

```bash
EAGLE=/home/aiplatform/workspace/Eagle/Embodied
REPO=/path/to/this/Military-Detection  # thư mục clone repo này

cp $REPO/convert_coco_to_locany_all_classes.py $EAGLE/
cp -r $REPO/locany_recipe $EAGLE/
cp -r $REPO/deepspeed_configs $EAGLE/   # nếu chưa có
cp -r $REPO/scripts $EAGLE/
```

---

## Bước 2: Patch flash_attn → sdpa

```bash
cd $EAGLE
conda activate locateanything

# Xem những file sẽ bị sửa (dry run)
python scripts/patch_flash_attn.py --dry-run

# Áp dụng patch thật
python scripts/patch_flash_attn.py
```

Patch tự động:
- Thay `attn_implementation="flash_attention_2"` → `"sdpa"` trong toàn bộ repo.
- Wrap các `import flash_attn` / `from flash_attn import ...` ở module-level vào `try/except ImportError`.

---

## Bước 3: Convert dataset

```bash
cd $EAGLE
conda activate locateanything
python convert_coco_to_locany_all_classes.py
```

Output:
```
locany_military_data_all/
├── train_all_classes.jsonl
├── val_all_classes.jsonl
└── test_all_classes.jsonl
```

Format mỗi dòng JSONL:
```json
{
  "image": "filename.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nZumwalt class destroyer . Admiral Gorshkov class frigate . ... ."
    },
    {
      "from": "gpt",
      "value": "<ref> F-22 Raptor fighter jet </ref><box><412><100><890><530></box><ref> M1 Abrams main battle tank </ref><box><10><600><450><950></box>"
    }
  ]
}
```

---

## Bước 4: Debug training (100 steps)

```bash
cd $EAGLE
conda activate locateanything
bash scripts/train_debug.sh
```

Log: `work_dirs/locany_military_all_debug/training_log.txt`

---

## Bước 5: Full training (3000 steps)

Sau khi debug 100 step chạy OK:

```bash
cd $EAGLE
conda activate locateanything
bash scripts/train_full.sh
```

---

## Xử lý lỗi thường gặp

### `KeyError: 'SLURM_PROCID'`
→ Thêm `--launcher pytorch` vào lệnh train (đã có trong script).

### `ImportError: flash_attn`
→ Chạy `python scripts/patch_flash_attn.py` để patch repo.
→ Đảm bảo `--attn_implementation sdpa` trong lệnh train (đã có trong script).

### `Address already in use` (port conflict)
→ Script dùng `--standalone` của torchrun, tự chọn port tự do.
  Nếu vẫn lỗi: `bash scripts/train_debug.sh 29520` (truyền port khác).

---

## 10 Military Classes

| # | Class name |
|---|-----------|
| 1 | Zumwalt class destroyer |
| 2 | Admiral Gorshkov class frigate |
| 3 | F-22 Raptor fighter jet |
| 4 | F-35 Lightning II fighter jet |
| 5 | BM-30 Smerch multiple rocket launcher |
| 6 | M2 Bradley infantry fighting vehicle |
| 7 | BTR-90 armored personnel carrier |
| 8 | HIMARS rocket artillery launcher |
| 9 | M1 Abrams main battle tank |
| 10 | T-90 main battle tank |
