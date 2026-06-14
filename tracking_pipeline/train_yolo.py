"""
Train YOLO detector cho 10 class quân sự — input là dataset COCO format.

Pipeline:
  1. Chuyển COCO JSON (train/val/test) -> YOLO format (labels .txt + data.yaml)
  2. Train YOLO (ultralytics)

Class order khớp TUYỆT ĐỐI với tracking_pipeline/core/classes.py MILITARY_CLASSES
để detector trả về đúng tên class mà tracker/VLM mong đợi.

Chạy:
    # chỉ convert:
    python -m tracking_pipeline.train_yolo --convert-only

    # convert + train:
    python -m tracking_pipeline.train_yolo \
        --model yolo11m.pt --epochs 100 --imgsz 640 --batch 16

Sau khi train xong, weights tốt nhất ở:
    runs/detect/military_yolo/weights/best.pt
dùng đường dẫn đó cho tracking_pipeline.run --detector-path
"""

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

from .core.classes import MILITARY_CLASSES


# ── paths mặc định (sửa nếu dataset của bạn ở chỗ khác) ──────────────────────────
DEFAULT_SPLITS = {
    "train": "/home/aiplatform/workspace/research/research_res/merged_dataset/labels/train_gdino.json",
    "val":   "/home/aiplatform/workspace/research/research_res/merged_dataset/labels/val_gdino.json",
    "test":  "/home/aiplatform/workspace/research/research_res/merged_dataset/labels/test_gdino.json",
}
DEFAULT_IMG_ROOT = "/home/aiplatform/workspace/research/research_res/merged_dataset/images"
DEFAULT_OUT      = "/home/aiplatform/workspace/yolo_military_dataset"

# map tên ngắn (nếu dataset dùng) -> tên chuẩn
SHORT_TO_FULL = {
    "bm30smerch":  "BM-30 Smerch multiple rocket launcher",
    "bradley":     "M2 Bradley infantry fighting vehicle",
    "btr90":       "BTR-90 armored personnel carrier",
    "f22":         "F-22 Raptor fighter jet",
    "f35":         "F-35 Lightning II fighter jet",
    "gorkskov":    "Admiral Gorshkov class frigate",
    "gorshkov":    "Admiral Gorshkov class frigate",
    "m142_himars": "HIMARS rocket artillery launcher",
    "himars":      "HIMARS rocket artillery launcher",
    "m1_abrams":   "M1 Abrams main battle tank",
    "t90":         "T-90 main battle tank",
    "zumwalt":     "Zumwalt class destroyer",
}

CLS_TO_IDX = {c: i for i, c in enumerate(MILITARY_CLASSES)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def canonical(name: str):
    """Đưa tên category về 1 trong MILITARY_CLASSES, hoặc None nếu không khớp."""
    if name in CLS_TO_IDX:
        return name
    if name in SHORT_TO_FULL:
        return SHORT_TO_FULL[name]
    low = name.lower().strip()
    if low in SHORT_TO_FULL:
        return SHORT_TO_FULL[low]
    for c in MILITARY_CLASSES:
        if c.lower() == low:
            return c
    return None


def build_img_lookup(img_root: Path):
    """basename -> full path, hỗ trợ ảnh nằm trong subfolder theo class."""
    lookup = {}
    for p in img_root.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            lookup[p.name] = p
    return lookup


def convert_split(coco_path, img_root, out_root, split, img_lookup):
    """COCO JSON -> YOLO labels + symlink ảnh. Trả về số ảnh đã ghi."""
    with open(coco_path, encoding="utf-8") as f:
        coco = json.load(f)

    cat_id_to_idx = {}
    skipped_cats = []
    for c in coco["categories"]:
        canon = canonical(c["name"])
        if canon is not None:
            cat_id_to_idx[c["id"]] = CLS_TO_IDX[canon]
        else:
            skipped_cats.append(c["name"])
    if skipped_cats:
        print(f"  [WARN] Bỏ qua category lạ: {skipped_cats}")

    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] in cat_id_to_idx:
            anns_by_img[ann["image_id"]].append(ann)

    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    missing = 0
    for img in coco["images"]:
        fname = img["file_name"]
        src = img_root / fname
        if not src.exists():
            src = img_lookup.get(Path(fname).name)
        if not src or not src.exists():
            missing += 1
            continue

        W, H = img["width"], img["height"]
        anns = anns_by_img.get(img["id"], [])
        if not anns:
            continue

        # symlink ảnh (tên phẳng để tránh trùng) + file label cùng stem
        stem = Path(fname).name
        dst_img = img_dir / stem
        if not dst_img.exists():
            try:
                dst_img.symlink_to(src.resolve())
            except FileExistsError:
                pass
            except OSError:
                shutil.copy(src, dst_img)

        lines = []
        for a in anns:
            x, y, w, h = a["bbox"]               # COCO: x,y,w,h (pixel, top-left)
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            # clamp [0,1]
            cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
            nw, nh = min(max(nw, 0), 1), min(max(nh, 0), 1)
            cls_idx = cat_id_to_idx[a["category_id"]]
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        (lbl_dir / f"{Path(stem).stem}.txt").write_text("\n".join(lines))
        written += 1

    print(f"  [{split}] ghi {written} ảnh"
          + (f"  (thiếu {missing} ảnh)" if missing else ""))
    return written


def write_data_yaml(out_root: Path):
    yaml_path = out_root / "data.yaml"
    lines = [
        f"path: {out_root}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(MILITARY_CLASSES)}",
        "names:",
    ]
    for i, c in enumerate(MILITARY_CLASSES):
        lines.append(f"  {i}: {c}")
    yaml_path.write_text("\n".join(lines) + "\n")
    print(f"  data.yaml -> {yaml_path}")
    return yaml_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-root", default=DEFAULT_IMG_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="Thư mục xuất YOLO dataset")
    ap.add_argument("--train-json", default=DEFAULT_SPLITS["train"])
    ap.add_argument("--val-json",   default=DEFAULT_SPLITS["val"])
    ap.add_argument("--test-json",  default=DEFAULT_SPLITS["test"])
    ap.add_argument("--convert-only", action="store_true",
                    help="Chỉ convert COCO->YOLO, không train")
    # training args
    ap.add_argument("--model", default="yolo11m.pt",
                    help="YOLO base (yolo11n/s/m/l/x.pt hoặc yolov8...)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="military_yolo")
    args = ap.parse_args()

    out_root = Path(args.out)
    img_root = Path(args.img_root)
    img_lookup = build_img_lookup(img_root)
    print(f"Tìm thấy {len(img_lookup)} ảnh trong {img_root}")

    print("\n[1] Convert COCO -> YOLO ...")
    splits = {"train": args.train_json, "val": args.val_json, "test": args.test_json}
    for split, jp in splits.items():
        if jp and os.path.exists(jp):
            print(f"  {split}: {jp}")
            convert_split(jp, img_root, out_root, split, img_lookup)
        else:
            print(f"  [SKIP] {split}: không có file {jp}")

    yaml_path = write_data_yaml(out_root)

    if args.convert_only:
        print("\nXong convert. Bỏ qua train (--convert-only).")
        return

    print("\n[2] Train YOLO ...")
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=20,
    )
    print("\nXong. Weights tốt nhất: runs/detect/"
          f"{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
