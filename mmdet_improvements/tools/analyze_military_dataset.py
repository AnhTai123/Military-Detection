"""
Phân tích dataset military COCO (train/val/test).

Xuất:
  - Số ảnh + số bbox mỗi class mỗi split
  - Thống kê bbox area / aspect ratio / small-medium-large (COCO định nghĩa)
  - Ảnh thiếu annotation
  - Class imbalance
  - Ảnh có nhiều object
  - Annotation bất thường (bbox quá nhỏ, vượt biên ảnh, w/h <= 0)
  - Report CSV + Markdown

Chạy:
    python tools/analyze_military_dataset.py \
        --data-root /home/aiplatform/workspace/research/research_res/merged_dataset \
        --out-dir work_dirs/dataset_analysis
"""

import argparse
import csv
import json
import os
from collections import defaultdict

# COCO area thresholds: small < 32^2, medium < 96^2, large >= 96^2
SMALL_AREA = 32 ** 2
MEDIUM_AREA = 96 ** 2

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


def load_coco(path):
    with open(path) as f:
        return json.load(f)


def analyze_split(coco, split_name):
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_by_id = {im["id"]: im for im in coco["images"]}

    n_images = len(coco["images"])
    bbox_count = defaultdict(int)
    img_with_class = defaultdict(set)
    areas = defaultdict(list)
    aspect_ratios = defaultdict(list)
    size_bucket = defaultdict(lambda: {"small": 0, "medium": 0, "large": 0})
    anns_per_image = defaultdict(int)
    anomalies = []

    images_with_ann = set()

    for ann in coco["annotations"]:
        cid = ann["category_id"]
        cname = cat_id_to_name.get(cid, f"UNKNOWN_{cid}")
        img_id = ann["image_id"]
        images_with_ann.add(img_id)
        anns_per_image[img_id] += 1

        x, y, w, h = ann["bbox"]
        area = w * h

        # anomaly checks
        img = img_by_id.get(img_id, {})
        iw, ih = img.get("width", 0), img.get("height", 0)
        reasons = []
        if w <= 0 or h <= 0:
            reasons.append("w/h<=0")
        if area < 4:
            reasons.append("area<4px")
        if iw and ih and (x < -1 or y < -1 or x + w > iw + 1 or y + h > ih + 1):
            reasons.append("out_of_bounds")
        if reasons:
            anomalies.append({
                "split": split_name,
                "image_id": img_id,
                "file_name": img.get("file_name", ""),
                "class": cname,
                "bbox": [x, y, w, h],
                "img_wh": [iw, ih],
                "reasons": ";".join(reasons),
            })

        bbox_count[cname] += 1
        img_with_class[cname].add(img_id)
        areas[cname].append(area)
        if h > 0:
            aspect_ratios[cname].append(w / h)
        if area < SMALL_AREA:
            size_bucket[cname]["small"] += 1
        elif area < MEDIUM_AREA:
            size_bucket[cname]["medium"] += 1
        else:
            size_bucket[cname]["large"] += 1

    images_without_ann = [
        img_by_id[i]["file_name"]
        for i in img_by_id if i not in images_with_ann
    ]

    # ảnh nhiều object (top theo số ann)
    multi_obj = sorted(anns_per_image.items(), key=lambda x: -x[1])[:20]
    multi_obj = [
        {"file_name": img_by_id[i]["file_name"], "num_objects": n}
        for i, n in multi_obj if n > 1
    ]

    return {
        "split": split_name,
        "n_images": n_images,
        "n_images_with_ann": len(images_with_ann),
        "n_images_without_ann": len(images_without_ann),
        "images_without_ann": images_without_ann,
        "bbox_count": dict(bbox_count),
        "n_images_per_class": {k: len(v) for k, v in img_with_class.items()},
        "areas": areas,
        "aspect_ratios": aspect_ratios,
        "size_bucket": {k: dict(v) for k, v in size_bucket.items()},
        "multi_obj": multi_obj,
        "anomalies": anomalies,
    }


def _mean(lst):
    return sum(lst) / len(lst) if lst else 0.0


def write_csv(stats_list, out_dir):
    # per-class per-split summary
    path = os.path.join(out_dir, "class_distribution.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "split", "class", "n_bbox", "n_images",
            "mean_area", "mean_aspect_ratio",
            "small", "medium", "large",
        ])
        for st in stats_list:
            for cname in MILITARY_CLASSES:
                bucket = st["size_bucket"].get(cname, {})
                w.writerow([
                    st["split"], cname,
                    st["bbox_count"].get(cname, 0),
                    st["n_images_per_class"].get(cname, 0),
                    round(_mean(st["areas"].get(cname, [])), 1),
                    round(_mean(st["aspect_ratios"].get(cname, [])), 3),
                    bucket.get("small", 0),
                    bucket.get("medium", 0),
                    bucket.get("large", 0),
                ])
    print(f"  → {path}")

    # anomalies
    apath = os.path.join(out_dir, "bbox_anomalies.csv")
    with open(apath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "file_name", "class", "bbox", "img_wh", "reasons"])
        for st in stats_list:
            for a in st["anomalies"]:
                w.writerow([
                    a["split"], a["file_name"], a["class"],
                    a["bbox"], a["img_wh"], a["reasons"],
                ])
    print(f"  → {apath}")


def write_markdown(stats_list, out_dir):
    path = os.path.join(out_dir, "dataset_report.md")
    lines = ["# Dataset Analysis Report\n"]

    # overview table
    lines.append("## Tổng quan splits\n")
    lines.append("| Split | #Images | #Images có ann | #Images thiếu ann |")
    lines.append("|---|---|---|---|")
    for st in stats_list:
        lines.append(
            f"| {st['split']} | {st['n_images']} | "
            f"{st['n_images_with_ann']} | {st['n_images_without_ann']} |"
        )
    lines.append("")

    # per-class bbox count across splits
    lines.append("## Số bbox mỗi class theo split\n")
    header = "| Class | " + " | ".join(st["split"] for st in stats_list) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(stats_list) + 1))
    for cname in MILITARY_CLASSES:
        row = f"| {cname} | "
        row += " | ".join(
            str(st["bbox_count"].get(cname, 0)) for st in stats_list
        )
        row += " |"
        lines.append(row)
    lines.append("")

    # class imbalance (dựa trên train)
    train = next((s for s in stats_list if s["split"] == "train"), stats_list[0])
    counts = {c: train["bbox_count"].get(c, 0) for c in MILITARY_CLASSES}
    mx = max(counts.values()) if counts else 1
    lines.append("## Class imbalance (theo train, ratio so với class nhiều nhất)\n")
    lines.append("| Class | #bbox train | ratio | gợi ý oversample |")
    lines.append("|---|---|---|---|")
    for c in MILITARY_CLASSES:
        ratio = counts[c] / mx if mx else 0
        factor = max(1, round(mx / counts[c])) if counts[c] else "N/A"
        flag = "⚠️ yếu" if ratio < 0.6 else ""
        lines.append(f"| {c} | {counts[c]} | {ratio:.2f} | x{factor} {flag} |")
    lines.append("")

    # size distribution train
    lines.append("## Phân bố kích thước bbox (train)\n")
    lines.append("| Class | small | medium | large |")
    lines.append("|---|---|---|---|")
    for c in MILITARY_CLASSES:
        b = train["size_bucket"].get(c, {})
        lines.append(
            f"| {c} | {b.get('small',0)} | {b.get('medium',0)} | {b.get('large',0)} |"
        )
    lines.append("")

    # anomalies summary
    total_anom = sum(len(s["anomalies"]) for s in stats_list)
    lines.append(f"## Annotation bất thường: {total_anom}\n")
    lines.append("Chi tiết trong `bbox_anomalies.csv`.\n")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  → {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", default="work_dirs/dataset_analysis")
    ap.add_argument("--train-json", default="labels/train_gdino.json")
    ap.add_argument("--val-json", default="labels/val_gdino.json")
    ap.add_argument("--test-json", default="labels/test_gdino.json")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    splits = {
        "train": os.path.join(args.data_root, args.train_json),
        "val": os.path.join(args.data_root, args.val_json),
        "test": os.path.join(args.data_root, args.test_json),
    }

    stats_list = []
    for name, path in splits.items():
        if not os.path.exists(path):
            print(f"[WARN] không tìm thấy {path}, bỏ qua")
            continue
        print(f"[{name}] {path}")
        coco = load_coco(path)
        stats_list.append(analyze_split(coco, name))

    print("\nWriting reports...")
    write_csv(stats_list, args.out_dir)
    write_markdown(stats_list, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
