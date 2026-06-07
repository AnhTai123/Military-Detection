"""
Tạo train json mới có oversample ảnh chứa class yếu (M1 Abrams, Bradley, F-22, ...).

Cách làm: với mỗi ảnh, tính "repeat factor" = max factor của các class yếu
xuất hiện trong ảnh đó. Nhân bản image + annotation (id mới) theo factor.

An toàn: KHÔNG sửa file gốc. Xuất file mới train_gdino_oversampled.json.

Chạy:
    python tools/make_oversampled_train.py \
        --in-json /home/aiplatform/workspace/research/research_res/merged_dataset/labels/train_gdino.json \
        --out-json /home/aiplatform/workspace/research/research_res/merged_dataset/labels/train_gdino_oversampled.json
"""

import argparse
import json
from collections import defaultdict

# class yếu → số lần lặp (điều chỉnh theo report dataset/error analysis)
WEAK_CLASS_REPEAT = {
    "M1 Abrams main battle tank": 3,
    "M2 Bradley infantry fighting vehicle": 3,
    "F-22 Raptor fighter jet": 2,
    "BTR-90 armored personnel carrier": 2,
    "T-90 main battle tank": 2,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-json", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    with open(args.in_json) as f:
        coco = json.load(f)

    catid_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    # repeat factor mỗi ảnh = max factor của các class yếu trong ảnh
    def img_factor(img_id):
        factor = 1
        for ann in anns_by_img.get(img_id, []):
            name = catid_to_name.get(ann["category_id"])
            factor = max(factor, WEAK_CLASS_REPEAT.get(name, 1))
        return factor

    new_images = []
    new_anns = []
    next_img_id = max((im["id"] for im in coco["images"]), default=0) + 1
    next_ann_id = max((a["id"] for a in coco["annotations"]), default=0) + 1

    orig_img = orig_ann = dup_img = dup_ann = 0

    for im in coco["images"]:
        factor = img_factor(im["id"])
        # bản gốc
        new_images.append(im)
        orig_img += 1
        for ann in anns_by_img.get(im["id"], []):
            new_anns.append(ann)
            orig_ann += 1
        # bản nhân thêm (factor - 1 lần)
        for _ in range(factor - 1):
            new_im = dict(im)
            new_im["id"] = next_img_id
            new_images.append(new_im)
            dup_img += 1
            for ann in anns_by_img.get(im["id"], []):
                new_ann = dict(ann)
                new_ann["id"] = next_ann_id
                new_ann["image_id"] = next_img_id
                new_anns.append(new_ann)
                next_ann_id += 1
                dup_ann += 1
            next_img_id += 1

    out = dict(coco)
    out["images"] = new_images
    out["annotations"] = new_anns

    with open(args.out_json, "w") as f:
        json.dump(out, f)

    print(f"Original: {orig_img} imgs, {orig_ann} anns")
    print(f"Added   : {dup_img} imgs, {dup_ann} anns")
    print(f"Total   : {len(new_images)} imgs, {len(new_anns)} anns")
    # report số bbox mỗi class sau oversample
    cnt = defaultdict(int)
    for a in new_anns:
        cnt[catid_to_name.get(a["category_id"])] += 1
    print("\nClass bbox count sau oversample:")
    for name, n in sorted(cnt.items(), key=lambda x: -x[1]):
        print(f"  {name}: {n}")
    print(f"\n→ {args.out_json}")


if __name__ == "__main__":
    main()
