"""
Train linear head cho DINOv2 classifier (giải pháp VLM chính của pipeline).

Ý tưởng: đóng băng DINOv2, crop tất cả GT box trong dataset COCO, trích feature
DINOv2, rồi train 1 lớp linear 10 class. Nhẹ, nhanh (~vài phút), chính xác cao.

Chạy:
    python -m tracking_pipeline.train_dino_head \
        --train-json .../train_gdino.json \
        --val-json   .../val_gdino.json \
        --img-root   .../images \
        --dino-model facebook/dinov2-small \
        --out dino_head.pt

Sau đó dùng trong pipeline:
    python -m tracking_pipeline.run ... --vlm dinov2 --vlm-path dino_head.pt
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from .core.classes import MILITARY_CLASSES

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


def canonical(name):
    if name in CLS_TO_IDX:
        return name
    low = name.lower().strip()
    if name in SHORT_TO_FULL:
        return SHORT_TO_FULL[name]
    if low in SHORT_TO_FULL:
        return SHORT_TO_FULL[low]
    for c in MILITARY_CLASSES:
        if c.lower() == low:
            return c
    return None


def build_img_lookup(img_root):
    lookup = {}
    for p in Path(img_root).rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            lookup[p.name] = p
    return lookup


def extract_split_features(coco_path, img_root, img_lookup, backbone,
                           processor, device, torch, pad=0.05):
    """Crop GT box -> feature DINOv2. Trả về (X: NxD, y: N)."""
    from PIL import Image
    import cv2
    from .vlm.dinov2_classifier import DINOv2Classifier

    with open(coco_path, encoding="utf-8") as f:
        coco = json.load(f)

    cat_id_to_idx = {}
    for c in coco["categories"]:
        canon = canonical(c["name"])
        if canon is not None:
            cat_id_to_idx[c["id"]] = CLS_TO_IDX[canon]

    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] in cat_id_to_idx:
            anns_by_img[ann["image_id"]].append(ann)

    feats, labels = [], []
    n_imgs = len(coco["images"])
    for k, img in enumerate(coco["images"]):
        anns = anns_by_img.get(img["id"], [])
        if not anns:
            continue
        fname = img["file_name"]
        src = Path(img_root) / fname
        if not src.exists():
            src = img_lookup.get(Path(fname).name)
        if not src or not Path(src).exists():
            continue

        bgr = cv2.imread(str(src))
        if bgr is None:
            continue
        H, W = bgr.shape[:2]

        for a in anns:
            x, y, w, h = a["bbox"]
            px, py = w * pad, h * pad
            x1 = max(0, int(x - px)); y1 = max(0, int(y - py))
            x2 = min(W, int(x + w + px)); y2 = min(H, int(y + h + py))
            crop = bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            pil = Image.fromarray(crop[:, :, ::-1])
            feat = DINOv2Classifier.extract_feature(
                backbone, processor, pil, device, torch
            )
            feats.append(feat.cpu().numpy()[0])
            labels.append(cat_id_to_idx[a["category_id"]])

        if (k + 1) % 50 == 0:
            print(f"    {k+1}/{n_imgs} ảnh, {len(feats)} crop")

    if not feats:
        return np.zeros((0, 1)), np.zeros((0,), dtype=int)
    return np.stack(feats), np.array(labels, dtype=int)


def main():
    import torch
    from transformers import AutoImageProcessor, AutoModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json",
                    default="/home/aiplatform/workspace/research/research_res/"
                            "merged_dataset/labels/train_gdino.json")
    ap.add_argument("--val-json",
                    default="/home/aiplatform/workspace/research/research_res/"
                            "merged_dataset/labels/val_gdino.json")
    ap.add_argument("--img-root",
                    default="/home/aiplatform/workspace/research/research_res/"
                            "merged_dataset/images")
    ap.add_argument("--dino-model", default="facebook/dinov2-small")
    ap.add_argument("--out", default="dino_head.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Backbone: {args.dino_model}")

    processor = AutoImageProcessor.from_pretrained(args.dino_model)
    backbone = AutoModel.from_pretrained(args.dino_model).to(device).eval()
    img_lookup = build_img_lookup(args.img_root)
    print(f"Tìm thấy {len(img_lookup)} ảnh.")

    print("\n[1] Trích feature TRAIN ...")
    Xtr, ytr = extract_split_features(
        args.train_json, args.img_root, img_lookup,
        backbone, processor, device, torch)
    print(f"  train: {Xtr.shape[0]} crop, dim={Xtr.shape[1] if Xtr.size else 0}")

    Xva, yva = np.zeros((0, 1)), np.zeros((0,), dtype=int)
    if args.val_json and os.path.exists(args.val_json):
        print("[1b] Trích feature VAL ...")
        Xva, yva = extract_split_features(
            args.val_json, args.img_root, img_lookup,
            backbone, processor, device, torch)
        print(f"  val: {Xva.shape[0]} crop")

    if Xtr.shape[0] == 0:
        print("Không có crop nào — kiểm tra lại đường dẫn dataset!")
        return

    feat_dim = Xtr.shape[1]
    n_cls = len(MILITARY_CLASSES)

    print("\n[2] Train linear head ...")
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    head = torch.nn.Linear(feat_dim, n_cls).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        head.train()
        opt.zero_grad()
        out = head(Xtr_t)
        loss = lossf(out, ytr_t)
        loss.backward()
        opt.step()
        if (ep + 1) % 50 == 0 or ep == 0:
            head.eval()
            with torch.no_grad():
                tr_acc = (head(Xtr_t).argmax(1) == ytr_t).float().mean().item()
                msg = f"  epoch {ep+1}: loss={loss.item():.4f} train_acc={tr_acc:.3f}"
                if Xva.shape[0] > 0:
                    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
                    yva_t = torch.tensor(yva, dtype=torch.long, device=device)
                    va_acc = (head(Xva_t).argmax(1) == yva_t).float().mean().item()
                    msg += f" val_acc={va_acc:.3f}"
                print(msg)

    # đánh giá per-class trên val
    if Xva.shape[0] > 0:
        head.eval()
        with torch.no_grad():
            pred = head(torch.tensor(Xva, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
        print("\n[3] Độ chính xác per-class (val):")
        for i, c in enumerate(MILITARY_CLASSES):
            mask = yva == i
            if mask.sum() == 0:
                continue
            acc = (pred[mask] == i).mean()
            print(f"  {c:<40} {acc:.3f}  (n={int(mask.sum())})")

    torch.save({
        "head_state": head.state_dict(),
        "classes": MILITARY_CLASSES,
        "dino_model": args.dino_model,
        "feat_dim": feat_dim,
    }, args.out)
    print(f"\nĐã lưu head -> {args.out}")
    print(f"Dùng:  --vlm dinov2 --vlm-path {args.out}")


if __name__ == "__main__":
    main()
