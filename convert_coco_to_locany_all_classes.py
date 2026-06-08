"""
Convert COCO-format annotations to LocateAnything all-classes JSONL format.

Each image produces ONE sample containing ALL objects present in that image.
Prompt lists all 10 military classes; output contains only the classes found.

COCO bbox [x, y, w, h] → LocateAnything normalized [0, 1000] (x1, y1, x2, y2).
"""

import json
import os
from collections import defaultdict

# ── paths ──────────────────────────────────────────────────────────────────────
SPLITS = {
    "train": "/home/aiplatform/workspace/research/research_res/merged_dataset/labels/train_gdino.json",
    "val":   "/home/aiplatform/workspace/research/research_res/merged_dataset/labels/val_gdino.json",
    "test":  "/home/aiplatform/workspace/research/research_res/merged_dataset/labels/test_gdino.json",
}
IMAGE_ROOT = "/home/aiplatform/workspace/research/research_res/merged_dataset/images"
OUTPUT_DIR = "/home/aiplatform/workspace/Eagle/Embodied/locany_military_data_all"

# ── canonical 10 classes ───────────────────────────────────────────────────────
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
VALID_CLASSES = set(MILITARY_CLASSES)

# Prompt presented to the model for every sample
ALL_CLASSES_PROMPT = " . ".join(MILITARY_CLASSES) + " ."


def coco_bbox_to_locany(bbox, img_w, img_h):
    """[x, y, w, h] → normalized [x1, y1, x2, y2] in [0, 1000]."""
    x, y, w, h = bbox
    x1 = round(x / img_w * 1000)
    y1 = round(y / img_h * 1000)
    x2 = round((x + w) / img_w * 1000)
    y2 = round((y + h) / img_h * 1000)
    # clamp to [0, 1000]
    x1, x2 = max(0, min(1000, x1)), max(0, min(1000, x2))
    y1, y2 = max(0, min(1000, y1)), max(0, min(1000, y2))
    return x1, y1, x2, y2


def build_output_string(objects):
    """
    objects: list of (class_name, (x1, y1, x2, y2))
    Returns LocateAnything output string with <ref>/<box> tokens.
    """
    parts = []
    for cls, (x1, y1, x2, y2) in objects:
        parts.append(f"<ref> {cls} </ref><box><{x1}><{y1}><{x2}><{y2}></box>")
    return "".join(parts)


def convert_split(coco_path, out_path):
    with open(coco_path) as f:
        coco = json.load(f)

    # build category id → name map, validating against known classes
    cat_id_to_name = {}
    skipped_cats = []
    for cat in coco["categories"]:
        name = cat["name"]
        if name in VALID_CLASSES:
            cat_id_to_name[cat["id"]] = name
        else:
            skipped_cats.append(name)
    if skipped_cats:
        print(f"  [WARN] Skipped unknown categories: {skipped_cats}")

    # build image id → image info
    img_id_to_info = {img["id"]: img for img in coco["images"]}

    # group annotations by image id
    anns_by_image = defaultdict(list)
    skipped_anns = 0
    for ann in coco["annotations"]:
        if ann["category_id"] in cat_id_to_name:
            anns_by_image[ann["image_id"]].append(ann)
        else:
            skipped_anns += 1

    if skipped_anns:
        print(f"  [WARN] Skipped {skipped_anns} annotations with unknown category")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    written = 0
    no_ann = 0

    with open(out_path, "w") as fout:
        for img_info in coco["images"]:
            img_id = img_info["id"]
            anns = anns_by_image.get(img_id, [])

            if not anns:
                no_ann += 1
                continue

            img_w = img_info["width"]
            img_h = img_info["height"]
            file_name = img_info["file_name"]

            # build sorted list of objects (sort by class name for determinism)
            objects = []
            for ann in anns:
                cls = cat_id_to_name[ann["category_id"]]
                coords = coco_bbox_to_locany(ann["bbox"], img_w, img_h)
                objects.append((cls, coords))
            objects.sort(key=lambda x: x[0])

            output_str = build_output_string(objects)

            sample = {
                "image": file_name,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image-1>\n{ALL_CLASSES_PROMPT}"
                    },
                    {
                        "from": "gpt",
                        "value": output_str
                    }
                ]
            }

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

    print(f"  Written: {written} samples  |  Skipped (no valid ann): {no_ann}")
    return written


def main():
    print(f"Output dir: {OUTPUT_DIR}")
    for split, coco_path in SPLITS.items():
        out_path = os.path.join(OUTPUT_DIR, f"{split}_all_classes.jsonl")
        print(f"\n[{split.upper()}] {coco_path}")
        n = convert_split(coco_path, out_path)
        print(f"  → {out_path}  ({n} samples)")
    print("\nDone.")


if __name__ == "__main__":
    main()
