"""
Phân tích lỗi GroundingDINO trên test/val.

Chạy inference checkpoint → so với ground truth → xuất:
  - confusion_matrix.png + confusion_matrix_raw.csv
  - per-class AP@30/50/75 + mAP@50:95  (test_classwise_ap30.csv)
  - hard examples mỗi class: false_negative / false_positive / wrong_class / low_iou
  - copy ảnh lỗi vào error_analysis/<Class>/<error_type>/

Dùng mmdet runner để load model + config (đảm bảo pipeline khớp eval thật).

Chạy:
    python tools/error_analysis_military.py \
        configs/grounding_dino/grounding_dino_swin-b_finetune_military_v2.py \
        work_dirs/gdino_military_swinb/best_coco_bbox_mAP_epoch_17.pth \
        --ann /home/aiplatform/workspace/research/research_res/merged_dataset/labels/test_gdino.json \
        --img-prefix /home/aiplatform/workspace/research/research_res/merged_dataset/images \
        --out-dir work_dirs/gdino_military_swinb/error_analysis \
        --score-thr 0.25
"""

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict

import numpy as np


MILITARY_CLASSES = [
    "Zumwalt class destroyer",
    "Admiral Gorshkov class frigate",
    "F-22 Raptor fighter jet",
    "F-35 Lightning II fighter jet",
    "BM-30 Smerch multiple rocket launcher",
    "M2 Bradley infantry fighting vehicle",
    "BTR-90 armored personnel carrier",
    "HIMARS rocket artillery launcher",
    "M1 Abrams main battle tank",
    "T-90 main battle tank",
]

# tên folder an toàn (không khoảng trắng) cho error_analysis
SAFE_NAME = {
    "Zumwalt class destroyer": "Zumwalt",
    "Admiral Gorshkov class frigate": "Gorshkov",
    "F-22 Raptor fighter jet": "F22",
    "F-35 Lightning II fighter jet": "F35",
    "BM-30 Smerch multiple rocket launcher": "BM30",
    "M2 Bradley infantry fighting vehicle": "Bradley",
    "BTR-90 armored personnel carrier": "BTR90",
    "HIMARS rocket artillery launcher": "HIMARS",
    "M1 Abrams main battle tank": "M1_Abrams",
    "T-90 main battle tank": "T90",
}


def iou_xywh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ── inference ───────────────────────────────────────────────────────────────────
def run_inference(config, checkpoint, ann_file, img_prefix, score_thr, device):
    """Trả về dict image_id → list[(bbox_xywh, class_idx, score)]."""
    from mmdet.apis import init_detector, inference_detector

    model = init_detector(config, checkpoint, device=device)

    with open(ann_file) as f:
        coco = json.load(f)

    # map category_id (coco) → index trong MILITARY_CLASSES theo name
    catid_to_idx = {}
    for c in coco["categories"]:
        if c["name"] in MILITARY_CLASSES:
            catid_to_idx[c["id"]] = MILITARY_CLASSES.index(c["name"])

    preds = {}
    for i, img in enumerate(coco["images"]):
        img_path = os.path.join(img_prefix, img["file_name"])
        if not os.path.exists(img_path):
            preds[img["id"]] = []
            continue
        result = inference_detector(model, img_path)
        inst = result.pred_instances
        bboxes = inst.bboxes.cpu().numpy()      # xyxy
        scores = inst.scores.cpu().numpy()
        labels = inst.labels.cpu().numpy()

        dets = []
        for b, s, l in zip(bboxes, scores, labels):
            if s < score_thr:
                continue
            x1, y1, x2, y2 = b
            dets.append(((float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                         int(l), float(s)))
        preds[img["id"]] = dets
        if (i + 1) % 50 == 0:
            print(f"  inferred {i+1}/{len(coco['images'])}")

    return preds, coco, catid_to_idx


# ── ground truth ─────────────────────────────────────────────────────────────────
def build_gt(coco, catid_to_idx):
    gt = defaultdict(list)  # image_id → list[(bbox_xywh, class_idx)]
    for ann in coco["annotations"]:
        cidx = catid_to_idx.get(ann["category_id"])
        if cidx is None:
            continue
        gt[ann["image_id"]].append((tuple(ann["bbox"]), cidx))
    return gt


# ── matching + error categorization ──────────────────────────────────────────────
def categorize(preds, gt, iou_thr=0.5):
    """
    Trả về:
        confusion[gt_idx][pred_idx] += 1   (+ background column = len(classes))
        errors: list dict {image_id, type, gt_class, pred_class, iou}
    """
    n = len(MILITARY_CLASSES)
    BG = n  # background index
    confusion = np.zeros((n + 1, n + 1), dtype=int)  # rows=gt, cols=pred
    errors = []

    for img_id in set(list(preds.keys()) + list(gt.keys())):
        gts = list(gt.get(img_id, []))
        dts = sorted(preds.get(img_id, []), key=lambda d: -d[2])  # by score

        gt_matched = [False] * len(gts)

        for (dbox, dcls, dscore) in dts:
            best_iou, best_j = 0.0, -1
            for j, (gbox, gcls) in enumerate(gts):
                if gt_matched[j]:
                    continue
                iou = iou_xywh(dbox, gbox)
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_iou >= iou_thr and best_j >= 0:
                gt_matched[best_j] = True
                gcls = gts[best_j][1]
                confusion[gcls][dcls] += 1
                if gcls != dcls:
                    errors.append({
                        "image_id": img_id, "type": "wrong_class",
                        "gt_class": gcls, "pred_class": dcls, "iou": best_iou,
                    })
            elif best_j >= 0 and best_iou > 0.1:
                # có overlap nhưng IoU thấp
                errors.append({
                    "image_id": img_id, "type": "low_iou",
                    "gt_class": gts[best_j][1], "pred_class": dcls, "iou": best_iou,
                })
                confusion[BG][dcls] += 1  # tính như false positive
            else:
                # false positive (không khớp gt nào)
                confusion[BG][dcls] += 1
                errors.append({
                    "image_id": img_id, "type": "false_positive",
                    "gt_class": -1, "pred_class": dcls, "iou": 0.0,
                })

        # gt không được match → false negative (miss)
        for j, matched in enumerate(gt_matched):
            if not matched:
                gcls = gts[j][1]
                confusion[gcls][BG] += 1
                errors.append({
                    "image_id": img_id, "type": "false_negative",
                    "gt_class": gcls, "pred_class": -1, "iou": 0.0,
                })

    return confusion, errors


# ── per-class AP ─────────────────────────────────────────────────────────────────
def compute_ap(preds, gt, iou_thr):
    """AP cho từng class tại 1 ngưỡng IoU (11-point ~ area)."""
    aps = {}
    for cidx in range(len(MILITARY_CLASSES)):
        scored = []  # (score, is_tp)
        n_gt = 0
        gt_per_img = {}
        for img_id, gts in gt.items():
            boxes = [g[0] for g in gts if g[1] == cidx]
            gt_per_img[img_id] = [(b, False) for b in boxes]
            n_gt += len(boxes)

        for img_id, dts in preds.items():
            cand = sorted(
                [(d[0], d[2]) for d in dts if d[1] == cidx],
                key=lambda x: -x[1],
            )
            gbs = gt_per_img.get(img_id, [])
            for dbox, dscore in cand:
                best_iou, best_j = 0.0, -1
                for j, (gbox, used) in enumerate(gbs):
                    if used:
                        continue
                    iou = iou_xywh(dbox, gbox)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= iou_thr and best_j >= 0:
                    gbs[best_j] = (gbs[best_j][0], True)
                    scored.append((dscore, 1))
                else:
                    scored.append((dscore, 0))

        if n_gt == 0:
            aps[cidx] = float("nan")
            continue
        scored.sort(key=lambda x: -x[0])
        tp = np.cumsum([s[1] for s in scored])
        fp = np.cumsum([1 - s[1] for s in scored])
        recall = tp / n_gt
        precision = tp / np.maximum(tp + fp, 1e-9)
        # VOC-style area under PR
        ap = 0.0
        for t in np.linspace(0, 1, 101):
            p = precision[recall >= t].max() if np.any(recall >= t) else 0
            ap += p / 101
        aps[cidx] = float(ap)
    return aps


# ── plotting ─────────────────────────────────────────────────────────────────────
def plot_confusion(confusion, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [SAFE_NAME[c] for c in MILITARY_CLASSES] + ["background"]
    # normalize theo hàng (recall view)
    cm = confusion.astype(float)
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix (row-normalized)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if confusion[i][j] > 0:
                ax.text(j, i, str(confusion[i][j]), ha="center", va="center",
                        color="white" if norm[i][j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--ann", required=True)
    ap.add_argument("--img-prefix", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--score-thr", type=float, default=0.25)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--copy-errors", action="store_true", default=True)
    ap.add_argument("--max-copy-per-class", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Running inference...")
    preds, coco, catid_to_idx = run_inference(
        args.config, args.checkpoint, args.ann,
        args.img_prefix, args.score_thr, args.device,
    )
    gt = build_gt(coco, catid_to_idx)
    img_by_id = {im["id"]: im for im in coco["images"]}

    # save raw predictions + gt
    with open(os.path.join(args.out_dir, "predictions.json"), "w") as f:
        json.dump({str(k): v for k, v in preds.items()}, f)

    print("Categorizing errors...")
    confusion, errors = categorize(preds, gt, args.iou_thr)

    # confusion matrix outputs
    np.savetxt(
        os.path.join(args.out_dir, "confusion_matrix_raw.csv"),
        confusion, fmt="%d", delimiter=",",
        header=",".join([SAFE_NAME[c] for c in MILITARY_CLASSES] + ["background"]),
        comments="",
    )
    print(f"  → confusion_matrix_raw.csv")
    plot_confusion(confusion, os.path.join(args.out_dir, "confusion_matrix.png"))

    # per-class AP
    print("Computing per-class AP...")
    ap30 = compute_ap(preds, gt, 0.30)
    ap50 = compute_ap(preds, gt, 0.50)
    ap75 = compute_ap(preds, gt, 0.75)
    map5095 = {}
    for cidx in range(len(MILITARY_CLASSES)):
        vals = [compute_ap(preds, gt, t)[cidx]
                for t in np.arange(0.5, 1.0, 0.05)]
        vals = [v for v in vals if not np.isnan(v)]
        map5095[cidx] = float(np.mean(vals)) if vals else float("nan")

    csv_path = os.path.join(args.out_dir, "test_classwise_ap30.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "AP@30", "AP@50", "AP@75", "mAP@50:95"])
        for cidx, cname in enumerate(MILITARY_CLASSES):
            w.writerow([
                cname,
                round(ap30[cidx], 4), round(ap50[cidx], 4),
                round(ap75[cidx], 4), round(map5095[cidx], 4),
            ])
        # mean
        def mean_valid(d):
            v = [x for x in d.values() if not np.isnan(x)]
            return round(float(np.mean(v)), 4) if v else 0.0
        w.writerow(["MEAN", mean_valid(ap30), mean_valid(ap50),
                    mean_valid(ap75), mean_valid(map5095)])
    print(f"  → {csv_path}")

    # error log
    err_by_class = defaultdict(lambda: defaultdict(list))
    for e in errors:
        cls_idx = e["gt_class"] if e["gt_class"] >= 0 else e["pred_class"]
        if cls_idx < 0:
            continue
        cname = MILITARY_CLASSES[cls_idx]
        err_by_class[cname][e["type"]].append(e)

    with open(os.path.join(args.out_dir, "errors.json"), "w") as f:
        json.dump(errors, f, indent=2)

    # copy hard example images
    if args.copy_errors:
        print("Copying error images...")
        for cname, types in err_by_class.items():
            safe = SAFE_NAME[cname]
            for etype, elist in types.items():
                dst_dir = os.path.join(args.out_dir, safe, etype)
                os.makedirs(dst_dir, exist_ok=True)
                seen = set()
                for e in elist[: args.max_copy_per_class]:
                    img = img_by_id.get(e["image_id"])
                    if not img or e["image_id"] in seen:
                        continue
                    seen.add(e["image_id"])
                    src = os.path.join(args.img_prefix, img["file_name"])
                    if os.path.exists(src):
                        shutil.copy(src, os.path.join(
                            dst_dir, os.path.basename(img["file_name"])))

        # F-22 ↔ F-35 confusion folder riêng
        conf_dir = os.path.join(args.out_dir, "F22_F35_confusion")
        os.makedirs(conf_dir, exist_ok=True)
        f22, f35 = MILITARY_CLASSES.index("F-22 Raptor fighter jet"), \
            MILITARY_CLASSES.index("F-35 Lightning II fighter jet")
        for e in errors:
            if e["type"] == "wrong_class" and {e["gt_class"], e["pred_class"]} == {f22, f35}:
                img = img_by_id.get(e["image_id"])
                if img:
                    src = os.path.join(args.img_prefix, img["file_name"])
                    if os.path.exists(src):
                        shutil.copy(src, os.path.join(
                            conf_dir, os.path.basename(img["file_name"])))

    # summary log
    log_path = os.path.join(args.out_dir, "test_eval_log.txt")
    with open(log_path, "w") as f:
        f.write(f"Config: {args.config}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Ann: {args.ann}\n")
        f.write(f"Score thr: {args.score_thr}  IoU thr: {args.iou_thr}\n")
        f.write(f"Images: {len(coco['images'])}\n\n")
        f.write("Error counts per class:\n")
        for cname in MILITARY_CLASSES:
            t = err_by_class.get(cname, {})
            f.write(f"  {cname}: FN={len(t.get('false_negative',[]))} "
                    f"FP={len(t.get('false_positive',[]))} "
                    f"wrong_class={len(t.get('wrong_class',[]))} "
                    f"low_iou={len(t.get('low_iou',[]))}\n")
    print(f"  → {log_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
