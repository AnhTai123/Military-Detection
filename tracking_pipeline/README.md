# Tracking + VLM Classification Pipeline

Pipeline tracking quân sự theo sơ đồ:

```
Ảnh ──► detection + tracking ──► object {class, id, đã phân loại?, class_vlm}
                                          │
                               ┌──────────┴───────────┐
                               │  đã phân loại chưa?    │
                               └──────────┬───────────┘
                          chưa            │            rồi
                           ▼              │             ▼
                      VLM phân loại       │         dùng cache
                      update class_vlm ───┘         (kết thúc)
```

## Ý tưởng cốt lõi

**VLM chỉ chạy 1 lần cho mỗi `track_id`.** Sau khi 1 object đã được VLM phân loại
(`classified=True`, có `class_vlm`), các frame sau **dùng lại cache** thay vì gọi
VLM. Điều này tiết kiệm rất nhiều compute trong video dài.

## Lựa chọn component (đã chọn sẵn cho ổn định + nhẹ)

| Thành phần | Lựa chọn | Lý do |
|---|---|---|
| **Detector** | GroundingDINO fine-tuned | Bạn đã train thành công; qua interface trừu tượng |
| **Tracker** | ByteTrack (`supervision`) | Nhẹ, không cần GPU thêm, gán id ổn định |
| **VLM** | **CLIP zero-shot** (mặc định) | Chỉ ~150MB, rất nhanh — LocateAnything quá nặng & chưa train xong |
| **VLM** (tùy chọn) | Qwen2.5-VL-3B | Chính xác hơn nhưng nặng hơn |

> Khi LocateAnything-3B train xong, chỉ cần viết 1 adapter trong `vlm/`
> (theo mẫu `qwen_vl.py`) và đổi `--vlm`. Pipeline không cần sửa.

## Cài đặt

```bash
pip install -r tracking_pipeline/requirements.txt
```

## Chạy

### Video → video annotated + JSON
```bash
python -m tracking_pipeline.run \
  --source input.mp4 \
  --detector-path /path/to/grounding_dino_finetuned \
  --vlm clip \
  --output-video out.mp4 \
  --output-json out.json
```

### Folder ảnh (sequence) → video + JSON
```bash
python -m tracking_pipeline.run \
  --source /path/to/frames_folder \
  --detector-path /path/to/grounding_dino_finetuned \
  --output-video out.mp4 \
  --output-json out.json
```

### Dùng Qwen2.5-VL thay CLIP
```bash
python -m tracking_pipeline.run \
  --source input.mp4 \
  --detector-path /path/to/grounding_dino_finetuned \
  --vlm qwen \
  --output-video out.mp4
```

## Tham số quan trọng

| Flag | Mặc định | Ý nghĩa |
|---|---|---|
| `--min-vlm-score` | 0.0 | VLM score thấp hơn ngưỡng → chưa mark classified, thử lại frame sau |
| `--reclassify-after N` | None | Cho phép phân loại lại sau N frame (None = chỉ 1 lần) |
| `--box-threshold` | 0.3 | Ngưỡng detection của GroundingDINO |
| `--device` | cuda | `cuda` hoặc `cpu` |

## Cấu trúc

```
tracking_pipeline/
├── run.py                  # entry point
├── core/
│   ├── types.py            # TrackedObject, Detection, FrameResult
│   ├── classes.py          # 10 military classes
│   └── pipeline.py         # logic chính (decision: đã phân loại chưa?)
├── detectors/
│   ├── base.py             # BaseDetector
│   └── grounding_dino.py   # GroundingDINO adapter
├── trackers/
│   └── bytetrack.py        # ByteTrack wrapper
└── vlm/
    ├── base.py             # BaseVLM
    ├── clip_classifier.py  # CLIP (mặc định, nhẹ)
    └── qwen_vl.py          # Qwen2.5-VL (tùy chọn) + mẫu cho LocateAnything
```

## Output JSON

Mỗi frame chứa list objects:
```json
{
  "frame": 12,
  "name": "frame_000012",
  "objects": [
    {
      "track_id": 3,
      "bbox_xyxy": [120.5, 80.0, 410.2, 360.1],
      "det_class": "M1 Abrams main battle tank",
      "det_score": 0.87,
      "classified": true,
      "class_vlm": "M1 Abrams main battle tank",
      "vlm_score": 0.93,
      "final_class": "M1 Abrams main battle tank"
    }
  ]
}
```

Cuối run sẽ in thống kê số lần gọi VLM thực tế vs nếu gọi mỗi frame
(thể hiện hiệu quả caching).
