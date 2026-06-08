"""
Kiểm tra JSONL data trước khi train LocateAnything.

Chạy:
    python scripts/verify_jsonl.py \
        --jsonl /home/aiplatform/workspace/Eagle/Embodied/locany_military_data_all/train_all_classes.jsonl \
        --img-root /home/aiplatform/workspace/research/research_res/merged_dataset/images \
        --n-show 3
"""

import argparse
import json
import os
import re
from pathlib import Path
from collections import defaultdict


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

BOX_PAT = re.compile(r"<ref>(.*?)</ref><box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


def check_jsonl(jsonl_path, img_root, n_show=3):
    path = Path(jsonl_path)
    if not path.exists():
        print(f"[FAIL] File không tồn tại: {jsonl_path}")
        return False

    img_root = Path(img_root)
    errors   = []
    warnings = []
    class_count   = defaultdict(int)
    img_missing   = 0
    n_total       = 0
    n_no_box      = 0
    n_shown       = 0

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            n_total += 1

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i+1}: JSON parse error — {e}")
                continue

            # ── kiểm tra keys ───────────────────────────────────────────
            if "image" not in sample:
                errors.append(f"Line {i+1}: thiếu key 'image'")
                continue
            if "conversations" not in sample:
                errors.append(f"Line {i+1}: thiếu key 'conversations'")
                continue

            convs = sample["conversations"]
            if len(convs) < 2:
                errors.append(f"Line {i+1}: conversations phải có ≥ 2 turn")
                continue

            # ── kiểm tra ảnh tồn tại ───────────────────────────────────
            img_file = img_root / sample["image"]
            if not img_file.exists():
                img_file2 = img_root / Path(sample["image"]).name
                if not img_file2.exists():
                    img_missing += 1
                    if img_missing <= 5:
                        warnings.append(f"Line {i+1}: ảnh không tồn tại — {sample['image']}")

            # ── kiểm tra human turn có <image> ─────────────────────────
            human_val = convs[0].get("value", "")
            if "<image-1>" not in human_val and "<image>" not in human_val:
                errors.append(f"Line {i+1}: human turn thiếu <image-1> token")

            # ── kiểm tra gpt turn có <box> ──────────────────────────────
            gpt_val = convs[1].get("value", "")
            boxes = BOX_PAT.findall(gpt_val)
            if not boxes:
                n_no_box += 1
                if n_no_box <= 3:
                    warnings.append(f"Line {i+1}: không parse được box nào — {gpt_val[:80]}")

            # ── đếm class ──────────────────────────────────────────────
            for (cls, x1, y1, x2, y2) in boxes:
                cls = cls.strip()
                class_count[cls] += 1
                if cls not in MILITARY_CLASSES:
                    warnings.append(f"Line {i+1}: class không nằm trong 10 class: '{cls}'")

            # ── bbox sanity: tọa độ [0,1000] ───────────────────────────
            for (cls, x1, y1, x2, y2) in boxes:
                coords = [int(x1), int(y1), int(x2), int(y2)]
                if any(c < 0 or c > 1000 for c in coords):
                    warnings.append(f"Line {i+1}: tọa độ ngoài [0,1000]: {coords}")
                if int(x2) <= int(x1) or int(y2) <= int(y1):
                    warnings.append(f"Line {i+1}: bbox đảo (x2≤x1 hoặc y2≤y1): {coords}")

            # ── in vài sample đầu ───────────────────────────────────────
            if n_shown < n_show:
                print(f"\n── Sample {i+1} ──")
                print(f"  image      : {sample['image']}")
                print(f"  human      : {human_val[:80]}...")
                print(f"  gpt output : {gpt_val[:120]}")
                print(f"  n_boxes    : {len(boxes)}")
                n_shown += 1

    print(f"\n{'='*60}")
    print(f"JSONL: {jsonl_path}")
    print(f"{'='*60}")
    print(f"  Tổng sample      : {n_total}")
    print(f"  Ảnh không tìm thấy: {img_missing}")
    print(f"  Sample không có box: {n_no_box}")

    print(f"\n  Phân bố class (số bbox):")
    for cls in MILITARY_CLASSES:
        cnt = class_count.get(cls, 0)
        bar = "█" * min(40, cnt // max(1, max(class_count.values()) // 40))
        flag = "⚠️" if cnt == 0 else ""
        print(f"    {cls:<40} {cnt:>5}  {bar} {flag}")
    unknown = {k: v for k, v in class_count.items() if k not in MILITARY_CLASSES}
    if unknown:
        print(f"\n  Class không hợp lệ: {unknown}")

    if warnings:
        print(f"\n  Warnings ({len(warnings)}):")
        for w in warnings[:10]:
            print(f"    ⚠️  {w}")
        if len(warnings) > 10:
            print(f"    ... và {len(warnings)-10} warning khác")

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"    ❌ {e}")
        print("\n[RESULT] FAIL — có lỗi cần sửa trước khi train.")
        return False

    if img_missing > 0 or n_no_box > 0:
        print("\n[RESULT] WARNING — dữ liệu có vấn đề nhỏ, kiểm tra lại trước khi train.")
    else:
        print("\n[RESULT] OK — dữ liệu sẵn sàng để train!")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl",     required=True)
    ap.add_argument("--img-root",  required=True)
    ap.add_argument("--n-show",    type=int, default=3,
                    help="Số sample in ra để xem thử")
    args = ap.parse_args()
    check_jsonl(args.jsonl, args.img_root, args.n_show)


if __name__ == "__main__":
    main()
