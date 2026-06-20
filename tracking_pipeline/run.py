"""
Entry point cho tracking + VLM pipeline.

Hỗ trợ:
  - input video (.mp4, ...) → xuất video annotated + JSON
  - input folder ảnh (sequence) → xuất video annotated + JSON

Ví dụ:
    python -m tracking_pipeline.run \
        --source video.mp4 \
        --detector-path /path/to/grounding_dino_finetuned \
        --output-video out.mp4 \
        --output-json out.json

Mặc định VLM = CLIP (nhẹ). Đổi sang Qwen bằng --vlm qwen.
"""

import argparse
import json
import os
from typing import List, Optional

import cv2
import numpy as np

from .core.classes import MILITARY_CLASSES
from .core.pipeline import TrackingVLMPipeline
from .trackers.bytetrack import ByteTrackWrapper


# ── màu cố định cho mỗi class (BGR) ─────────────────────────────────────────────
def _class_colors(n: int):
    rng = np.random.RandomState(42)
    return [tuple(int(c) for c in rng.randint(0, 255, 3)) for _ in range(n)]


COLORS = _class_colors(len(MILITARY_CLASSES))
CLS_TO_IDX = {c: i for i, c in enumerate(MILITARY_CLASSES)}


def draw(frame, objects):
    for obj in objects:
        x1, y1, x2, y2 = [int(v) for v in obj.bbox]
        color = COLORS[CLS_TO_IDX.get(obj.final_class, 0)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        tag = "✓vlm" if obj.classified else "det"
        label = f"#{obj.track_id} {obj.final_class} [{tag}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def build_detector(args):
    if args.detector == "grounding_dino":
        from .detectors.grounding_dino import GroundingDINODetector
        return GroundingDINODetector(
            model_path=args.detector_path,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=args.device,
        )
    if args.detector == "yolo":
        from .detectors.yolo import YOLODetector
        return YOLODetector(
            model_path=args.detector_path,
            conf_threshold=args.box_threshold,
            iou_threshold=args.iou_threshold,
            device=args.device,
            imgsz=args.imgsz,
        )
    raise ValueError(f"Unknown detector: {args.detector}")


def build_vlm(args):
    if args.vlm == "clip":
        from .vlm.clip_classifier import CLIPClassifier
        return CLIPClassifier(device=args.device)
    if args.vlm == "siglip":
        from .vlm.siglip_classifier import SigLIPClassifier
        return SigLIPClassifier(device=args.device)
    if args.vlm == "dinov2":
        from .vlm.dinov2_classifier import DINOv2Classifier
        if not args.vlm_path:
            raise ValueError("--vlm dinov2 cần --vlm-path tới file head .pt "
                             "(train bằng tracking_pipeline.train_dino_head)")
        return DINOv2Classifier(head_path=args.vlm_path, device=args.device)
    if args.vlm == "grounding_dino":
        from .vlm.grounding_dino import GroundingDINOVLM
        return GroundingDINOVLM(
            model_path=args.vlm_path,
            box_threshold=args.vlm_box_threshold,
            text_threshold=args.vlm_text_threshold,
            device=args.device,
        )
    if args.vlm == "qwen":
        from .vlm.qwen_vl import QwenVLClassifier
        return QwenVLClassifier(model_path=args.vlm_path, device=args.device)
    raise ValueError(f"Unknown vlm: {args.vlm}")


def iter_frames(source: str):
    """Yield (frame_idx, image_bgr). Hỗ trợ video file hoặc folder ảnh."""
    if os.path.isdir(source):
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        files = sorted(
            f for f in os.listdir(source) if f.lower().endswith(exts)
        )
        for i, fn in enumerate(files):
            img = cv2.imread(os.path.join(source, fn))
            if img is not None:
                yield i, img, fn
    else:
        cap = cv2.VideoCapture(source)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame, f"frame_{i:06d}"
            i += 1
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video file hoặc folder ảnh")
    ap.add_argument("--detector", default="yolo",
                    choices=["yolo", "grounding_dino"])
    ap.add_argument("--detector-path", required=True,
                    help="YOLO: path to .pt file | GDino: HF model path")
    ap.add_argument("--vlm", default="dinov2",
                    choices=["clip", "siglip", "dinov2", "grounding_dino", "qwen"])
    ap.add_argument("--vlm-path", default=None,
                    help="VLM model path (GDino fine-tuned / Qwen). "
                         "Không cần nếu --vlm clip")
    ap.add_argument("--device", default="cuda")
    # detector thresholds
    ap.add_argument("--box-threshold", type=float, default=0.25,
                    help="YOLO conf / GDino box threshold")
    ap.add_argument("--text-threshold", type=float, default=0.25,
                    help="GDino text threshold (chỉ dùng khi detector=grounding_dino)")
    ap.add_argument("--iou-threshold", type=float, default=0.45,
                    help="NMS IoU threshold (YOLO)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="Input size cho YOLO")
    # VLM thresholds
    ap.add_argument("--vlm-box-threshold", type=float, default=0.1,
                    help="GDino box threshold khi dùng làm VLM (crop đã tight)")
    ap.add_argument("--vlm-text-threshold", type=float, default=0.1,
                    help="GDino text threshold khi dùng làm VLM")
    ap.add_argument("--min-vlm-score", type=float, default=0.0)
    ap.add_argument("--reclassify-after", type=int, default=None,
                    help="phân loại lại sau N frame (mặc định: chỉ 1 lần)")
    ap.add_argument("--output-video", default=None)
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    detector = build_detector(args)
    tracker = ByteTrackWrapper(classes=MILITARY_CLASSES, frame_rate=args.fps)
    vlm = build_vlm(args)

    pipeline = TrackingVLMPipeline(
        detector=detector,
        tracker=tracker,
        vlm=vlm,
        min_vlm_score=args.min_vlm_score,
        reclassify_after=args.reclassify_after,
    )

    writer = None
    json_results = []
    total_vlm_calls = 0
    total_frames = 0

    for frame_idx, frame, name in iter_frames(args.source):
        result = pipeline.process_frame(frame, frame_idx)
        total_vlm_calls += result.vlm_calls
        total_frames += 1

        if args.output_video:
            vis = draw(frame.copy(), result.objects)
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(
                    args.output_video,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps, (w, h),
                )
            writer.write(vis)

        if args.output_json is not None:
            json_results.append({
                "frame": frame_idx,
                "name": name,
                "objects": [
                    {
                        "track_id": o.track_id,
                        "bbox_xyxy": [round(v, 1) for v in o.bbox],
                        "det_class": o.det_class,
                        "det_score": round(o.det_score, 3),
                        "classified": o.classified,
                        "class_vlm": o.class_vlm,
                        "vlm_score": round(o.vlm_score, 3) if o.vlm_score else None,
                        "final_class": o.final_class,
                    }
                    for o in result.objects
                ],
            })

    if writer is not None:
        writer.release()

    if args.output_json is not None:
        with open(args.output_json, "w") as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)

    # thống kê hiệu quả caching
    n_tracks = len(pipeline.objects)
    print(f"\n── Thống kê ──")
    print(f"Tổng frame          : {total_frames}")
    print(f"Tổng track id        : {n_tracks}")
    print(f"Tổng lần gọi VLM     : {total_vlm_calls}")
    if total_frames and n_tracks:
        naive = total_frames * n_tracks  # nếu gọi VLM mỗi frame mỗi object
        print(f"Nếu gọi VLM mỗi frame: ~{naive}  → tiết kiệm "
              f"{100 * (1 - total_vlm_calls / max(naive, 1)):.1f}%")
    if args.output_video:
        print(f"Video : {args.output_video}")
    if args.output_json:
        print(f"JSON  : {args.output_json}")


if __name__ == "__main__":
    main()
