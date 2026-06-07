# Cải thiện fine-tune GroundingDINO Swin-B (Military, 10 class)

Toàn bộ file ở đây tạo MỚI, **không sửa config cũ**. Copy sang repo mmdetection:

```bash
MMDET=/home/aiplatform/workspace/mmdetection
cp mmdet_improvements/tools/*.py                       $MMDET/tools/
cp mmdet_improvements/configs/grounding_dino/*.py      $MMDET/configs/grounding_dino/
```

---

## Quy trình đề xuất (theo thứ tự)

### Bước 0 — Phân tích dataset (A)
```bash
cd /home/aiplatform/workspace/mmdetection
conda activate mmdet_gdino

python tools/analyze_military_dataset.py \
  --data-root /home/aiplatform/workspace/research/research_res/merged_dataset \
  --out-dir work_dirs/dataset_analysis
```
Xuất: `class_distribution.csv`, `bbox_anomalies.csv`, `dataset_report.md`.

**Dùng để:** xác nhận class imbalance (M1 Abrams/Bradley/F-22 có ít bbox?),
phát hiện annotation lỗi, kiểm tra phân bố train vs test có lệch không. Đây là
nguyên nhân #1 khiến test thấp hơn valid nhiều.

### Bước 1 — Phân tích lỗi model v1 (B)
```bash
python tools/error_analysis_military.py \
  configs/grounding_dino/grounding_dino_swin-b_finetune_military.py \
  work_dirs/gdino_military_swinb/best_coco_bbox_mAP_epoch_17.pth \
  --ann /home/aiplatform/workspace/research/research_res/merged_dataset/labels/test_gdino.json \
  --img-prefix /home/aiplatform/workspace/research/research_res/merged_dataset/images \
  --out-dir work_dirs/gdino_military_swinb/error_analysis \
  --score-thr 0.25
```
Xuất: `confusion_matrix.png`, `confusion_matrix_raw.csv`, `test_classwise_ap30.csv`,
`test_eval_log.txt`, và folder ảnh lỗi `error_analysis/<Class>/<error_type>/`,
`error_analysis/F22_F35_confusion/`.

**Dùng để:** xem M1 Abrams miss (false_negative) hay sai class (wrong_class);
xem F-22 có thực sự bị đẩy sang F-35 không; xem Gorshkov false positive nằm ở đâu.

### Bước 2 — Oversample class yếu (C.1)
```bash
python tools/make_oversampled_train.py \
  --in-json  /home/aiplatform/workspace/research/research_res/merged_dataset/labels/train_gdino.json \
  --out-json /home/aiplatform/workspace/research/research_res/merged_dataset/labels/train_gdino_oversampled.json
```
Lặp ảnh chứa M1 Abrams (x3), Bradley (x3), F-22 (x2), BTR-90 (x2), T-90 (x2).
Điều chỉnh hệ số trong `WEAK_CLASS_REPEAT` dựa trên report bước 0.

### Bước 3 — Train v2 (D)
```bash
python tools/train.py \
  configs/grounding_dino/grounding_dino_swin-b_finetune_military_v2.py \
  --work-dir work_dirs/gdino_military_swinb_v2
```

### Bước 4 — Eval valid
```bash
python tools/test.py \
  configs/grounding_dino/grounding_dino_swin-b_finetune_military_v2.py \
  work_dirs/gdino_military_swinb_v2/best_coco_bbox_mAP_epoch_XX.pth \
  --out work_dirs/gdino_military_swinb_v2/val_results.pkl \
  --cfg-options test_evaluator.classwise=True \
                test_dataloader.dataset.ann_file=labels/val_gdino.json \
                test_evaluator.ann_file=/home/aiplatform/workspace/research/research_res/merged_dataset/labels/val_gdino.json
```

### Bước 5 — Eval test + error analysis v2
```bash
# COCO mAP trên test
python tools/test.py \
  configs/grounding_dino/grounding_dino_swin-b_finetune_military_v2.py \
  work_dirs/gdino_military_swinb_v2/best_coco_bbox_mAP_epoch_XX.pth \
  --out work_dirs/gdino_military_swinb_v2/test_results.pkl \
  --cfg-options test_evaluator.classwise=True \
  2>&1 | tee work_dirs/gdino_military_swinb_v2/test_eval_log.txt

# confusion matrix + hard examples + per-class AP
python tools/error_analysis_military.py \
  configs/grounding_dino/grounding_dino_swin-b_finetune_military_v2.py \
  work_dirs/gdino_military_swinb_v2/best_coco_bbox_mAP_epoch_XX.pth \
  --ann /home/aiplatform/workspace/research/research_res/merged_dataset/labels/test_gdino.json \
  --img-prefix /home/aiplatform/workspace/research/research_res/merged_dataset/images \
  --out-dir work_dirs/gdino_military_swinb_v2/error_analysis \
  --score-thr 0.25
```

---

## Vì sao từng thay đổi giúp cải thiện

| Thay đổi | Vấn đề giải quyết | Cơ chế |
|---|---|---|
| **Oversample class yếu** | M1 Abrams/Bradley recall thấp | Tăng số lần model thấy class hiếm → học feature tốt hơn, giảm bias về class nhiều |
| **LR nhỏ hơn (5e-5, thử 1e-5)** | Overfit (valid cao, test thấp) | Bước cập nhật nhỏ → ít memorize train, generalize tốt hơn |
| **Backbone/text LR x0.1** | Mất feature pretrained | Giữ feature Swin + BERT đã học từ data lớn, chỉ tinh chỉnh nhẹ |
| **Multi-scale resize** | Object kích thước đa dạng | Model robust với scale → xe nhỏ/máy bay lớn đều detect được |
| **Crop nhẹ, allow_negative_crop=False** | Crop mạnh làm mất object | Giữ object quân sự nguyên vẹn, bbox không lệch |
| **RandomFlip** | Thiếu đa dạng tư thế | Tăng dữ liệu rẻ, an toàn (object quân sự đối xứng trái-phải tốt) |
| **save_best coco/bbox_mAP + early stop** | Chọn sai epoch | Luôn giữ checkpoint tốt nhất theo val, tránh chọn epoch overfit |
| **metainfo full name khớp COCO** | Sai class / nhầm prompt | GroundingDINO match text-visual → tên đúng tuyệt đối là bắt buộc |
| **load_from best v1** | Train lại từ đầu tốn thời gian | Warm start từ checkpoint tốt → hội tụ nhanh hơn |

### Riêng F-22 ↔ F-35
Hai máy bay tàng hình rất giống nhau. Hướng xử lý:
1. Xem `error_analysis/F22_F35_confusion/` để biết model nhầm kiểu gì.
2. Oversample F-22 (đã làm, x2).
3. Nếu vẫn nhầm: kiểm tra annotation có gán nhầm nhãn không (rất hay gặp với 2 class này).

### Riêng Gorshkov false positive (precision thấp)
Recall cao nhưng precision thấp = model "vẽ" Gorshkov ở chỗ không có.
1. Xem `error_analysis/Gorshkov/false_positive/`.
2. Tăng score_thr khi inference (0.25 → 0.3-0.4) để lọc FP.
3. Kiểm tra train có ảnh tàu khác bị gán nhãn Gorshkov không.

---

## Lưu ý đánh giá (E)

- **Luôn eval val và test cùng pipeline** (test_pipeline trong config) để so sánh công bằng.
- **Không báo COCO mAP cho folder chỉ có label ảnh** (vd `selected_best_25`). Folder
  không có bbox annotation → chỉ là **image-level evaluation**, không phải COCO mAP.
- File `test_classwise_ap30.csv` từ `error_analysis_military.py` cho AP@30/50/75/mAP@50:95
  per-class — dùng để so sánh trực tiếp v1 vs v2.

---

## Tinh chỉnh thêm nếu v2 chưa đủ

1. **LR 1e-5**: sửa `base_lr = 1e-5` trong config v2.
2. **Tăng oversample**: chỉnh `WEAK_CLASS_REPEAT` (vd M1 Abrams x4).
3. **Bật EarlyStoppingHook**: bỏ comment `custom_hooks` trong config.
4. **Tăng score_thr inference**: giảm false positive (Gorshkov).
5. **Kiểm tra lại annotation** các class hay nhầm (F-22/F-35, nhóm xe mặt đất) —
   nếu nhãn sai thì không augmentation/LR nào cứu được.
