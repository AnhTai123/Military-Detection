"""
Inference LocateAnything-3B trên dataset quân sự 10 class (COCO format).
Tính định lượng: Precision/Recall/F1 + COCO mAP + Confusion matrix.

Chạy:
    python infer_locateanything_military.py \
        --ann  /home/aiplatform/workspace/research/research_res/merged_dataset/labels/test_gdino.json \
        --img-root /home/aiplatform/workspace/research/research_res/merged_dataset/images \
        --out-dir  /home/aiplatform/workspace/outputs_locateanything_military \
        [--limit 20]          # test nhanh với 20 ảnh đầu
        [--skip-existing]     # tiếp tục nếu bị ngắt

Kết quả xuất:
    summary.csv
    eval_report.md            (P/R/F1 per class + mAP table)
    confusion_matrix.png
    coco_eval_raw.txt
    <image_stem>_vis.jpg      (ảnh vẽ bbox)
    <image_stem>.json         (raw output mỗi ảnh)
"""

import argparse
import contextlib
import csv
import gc
import io
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer


# ══════════════════════════════════════════════════════════════════════════════
# 10 MILITARY CLASSES — khớp tuyệt đối với COCO categories trong merged_dataset
# ══════════════════════════════════════════════════════════════════════════════
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

SHORT_NAMES = {
    "Zumwalt class destroyer":              "Zumwalt",
    "Admiral Gorshkov class frigate":       "Gorshkov",
    "F-22 Raptor fighter jet":              "F-22",
    "F-35 Lightning II fighter jet":        "F-35",
    "BM-30 Smerch multiple rocket launcher": "BM-30",
    "M2 Bradley infantry fighting vehicle": "Bradley",
    "BTR-90 armored personnel carrier":     "BTR-90",
    "HIMARS rocket artillery launcher":     "HIMARS",
    "M1 Abrams main battle tank":           "M1 Abrams",
    "T-90 main battle tank":                "T-90",
}

MODEL_ID   = "nvidia/LocateAnything-3B"

# Map short category names (from selected_best_25) → full class names
SHORT_TO_FULL = {
    "bm30smerch":  "BM-30 Smerch multiple rocket launcher",
    "bradley":     "M2 Bradley infantry fighting vehicle",
    "btr90":       "BTR-90 armored personnel carrier",
    "f22":         "F-22 Raptor fighter jet",
    "f35":         "F-35 Lightning II fighter jet",
    "gorkskov":    "Admiral Gorshkov class frigate",
    "m142_himars": "HIMARS rocket artillery launcher",
    "m1_abrams":   "M1 Abrams main battle tank",
    "t90":         "T-90 main battle tank",
    "zumwalt":     "Zumwalt class destroyer",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Prompt all-classes: toàn bộ 10 class trong 1 lần hỏi
_CLASSES_STR  = " . ".join(MILITARY_CLASSES) + " ."
PROMPT        = f"Locate all the instances that match any of the following descriptions: {_CLASSES_STR}"


# ══════════════════════════════════════════════════════════════════════════════
# ARGS
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ann",
                   default="/home/aiplatform/workspace/research/research_res/"
                           "merged_dataset/labels/test_gdino.json",
                   help="COCO annotation JSON (test/val/train)")
    p.add_argument("--img-root",
                   default="/home/aiplatform/workspace/research/research_res/"
                           "merged_dataset/images",
                   help="Folder chứa ảnh (flat, không phân cấp class)")
    p.add_argument("--out-dir",
                   default="/home/aiplatform/workspace/outputs_locateanything_military")
    p.add_argument("--model-id",
                   default=MODEL_ID,
                   help="Model ID hoặc path checkpoint fine-tuned")
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--attn",        default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"],
                   help="Attention backend cho inference (mặc định sdpa)")
    p.add_argument("--mode",        default="hybrid",
                   choices=["fast", "hybrid", "slow"])
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--limit",       type=int,   default=None,
                   help="Giới hạn số ảnh để test nhanh, ví dụ: --limit 20")
    p.add_argument("--skip-existing", action="store_true",
                   help="Bỏ qua ảnh đã có file JSON output (tiếp tục khi bị ngắt)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# LOAD COCO — trả về list item để chạy inference
# ══════════════════════════════════════════════════════════════════════════════
def load_coco_items(ann_file, img_root, limit=None):
    """
    Mỗi item = 1 ảnh:
        image_path, image_id, width, height,
        gt_classes (list tên class GT),
        gt_boxes   (list [x1,y1,x2,y2] xyxy pixel)
    """
    with open(ann_file, encoding="utf-8") as f:
        coco = json.load(f)

    # category_id → tên class chuẩn (hỗ trợ cả tên đầy đủ và tên ngắn)
    cat_id_to_name = {}
    for c in coco["categories"]:
        name = c["name"]
        if name in MILITARY_CLASSES:
            cat_id_to_name[c["id"]] = name
        elif name in SHORT_TO_FULL:
            cat_id_to_name[c["id"]] = SHORT_TO_FULL[name]

    # group annotations by image_id
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] in cat_id_to_name:
            anns_by_img[ann["image_id"]].append(ann)

    img_root = Path(img_root)
    items = []

    # Build lookup: basename → full path (for subdirectory layouts)
    img_lookup = {}
    for p in img_root.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            img_lookup[p.name] = p

    for img in coco["images"]:
        fname = img["file_name"]
        img_path = img_root / fname
        if not img_path.exists():
            img_path = img_lookup.get(Path(fname).name)
        if not img_path or not img_path.exists():
            continue

        anns = anns_by_img.get(img["id"], [])
        if not anns:
            continue  # bỏ ảnh không có GT annotation

        gt_classes, gt_boxes = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            gt_classes.append(cat_id_to_name[a["category_id"]])
            gt_boxes.append([x, y, x + w, y + h])

        items.append({
            "image_path": img_path,
            "image_id":   img["id"],
            "file_name":  img["file_name"],
            "width":      img["width"],
            "height":     img["height"],
            "gt_classes": gt_classes,
            "gt_boxes":   gt_boxes,
        })

    if limit:
        items = items[:limit]

    return items, cat_id_to_name


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
def parse_boxes(answer, img_w, img_h):
    """Parse <ref>class</ref><box><x1><y1><x2><y2></box> → pixel coords."""
    results = []
    pat = re.compile(
        r"<ref>\s*(.*?)\s*</ref>\s*<box><(\d+)><(\d+)><(\d+)><(\d+)></box>",
        re.DOTALL)
    for m in pat.finditer(answer):
        label = m.group(1).strip()
        x1, y1, x2, y2 = [int(v) for v in m.groups()[1:]]
        results.append({
            "label": label,
            "x1": x1 / 1000.0 * img_w,
            "y1": y1 / 1000.0 * img_h,
            "x2": x2 / 1000.0 * img_w,
            "y2": y2 / 1000.0 * img_h,
        })
    # fallback: box không có ref
    if not results:
        for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
            x1, y1, x2, y2 = [int(v) for v in m.groups()]
            results.append({
                "label": "unknown",
                "x1": x1 / 1000.0 * img_w, "y1": y1 / 1000.0 * img_h,
                "x2": x2 / 1000.0 * img_w, "y2": y2 / 1000.0 * img_h,
            })
    return results


def match_label_to_class(label):
    """Khớp label text từ <ref> tag → tên class chuẩn."""
    label_l = label.strip().lower()
    # exact match
    for cls in MILITARY_CLASSES:
        if cls.lower() == label_l:
            return cls
    # substring match
    for cls in MILITARY_CLASSES:
        if label_l in cls.lower() or cls.lower() in label_l:
            return cls
    # token overlap
    label_tokens = set(label_l.split())
    best, best_score = None, 0
    for cls in MILITARY_CLASSES:
        overlap = len(set(cls.lower().split()) & label_tokens)
        if overlap > best_score:
            best, best_score = cls, overlap
    return best if best_score > 0 else None


def run_one_image(model, tokenizer, processor, item, device, mode,
                  max_new_tokens, temperature):
    img_pil = Image.open(item["image_path"]).convert("RGB")
    w, h = img_pil.size

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img_pil},
            {"type": "text",  "text":  PROMPT},
        ],
    }]

    text = processor.py_apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    proc_images, proc_videos = processor.process_vision_info(messages)
    img_pil.close()
    del img_pil, messages

    inputs = processor(text=[text], images=proc_images, videos=proc_videos,
                       return_tensors="pt").to(device)
    del text, proc_images, proc_videos

    dtype = torch.bfloat16 if "cuda" in device else torch.float32
    pixel_values   = inputs["pixel_values"].to(dtype)
    input_ids      = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    image_grid_hws = inputs.get("image_grid_hws", None)
    del inputs

    t0 = time.perf_counter()
    with torch.inference_mode():
        response = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_hws=image_grid_hws,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=mode,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=False,
        )
    latency = time.perf_counter() - t0

    del pixel_values, input_ids, attention_mask, image_grid_hws
    if "cuda" in device:
        torch.cuda.empty_cache()

    answer = response[0] if isinstance(response, tuple) else response
    raw_dets = parse_boxes(answer, w, h)

    # match label → canonical class name
    detections = []
    for d in raw_dets:
        cls = match_label_to_class(d["label"])
        if cls:
            d["class"] = cls
            detections.append(d)

    return {"answer": answer, "latency": latency,
            "width": w, "height": h, "detections": detections}


def draw_boxes(image_path, detections, gt_boxes, gt_classes, out_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return
    h, w = img.shape[:2]

    # vẽ GT (xanh lam)
    for box, cls in zip(gt_boxes, gt_classes):
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.putText(img, f"GT:{SHORT_NAMES.get(cls, cls)}", (x1, max(15, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1, cv2.LINE_AA)

    # vẽ prediction (xanh lá)
    for det in detections:
        x1, y1 = int(max(0, det["x1"])), int(max(0, det["y1"]))
        x2, y2 = int(min(w-1, det["x2"])), int(min(h-1, det["y2"]))
        label = SHORT_NAMES.get(det.get("class", ""), det["label"])[:20]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, max(0, y1-th-6)), (x1+tw+4, y1), (0, 220, 0), -1)
        cv2.putText(img, label, (x1+2, max(14, y1-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def evaluate(all_results, ann_file, out_dir, iou_thrs=(0.3, 0.5, 0.75)):
    """
    all_results: list dict {item, detections}
    Xuất:
      - Per-class TP/FP/FN → Precision/Recall/F1 tại mỗi IoU threshold
      - COCO mAP (pycocotools) nếu có
      - Confusion matrix (image-level: GT class vs predicted class nhiều nhất)
      - Markdown report
    """
    print("\n" + "=" * 70)
    print("ĐỊNH LƯỢNG — LocateAnything-3B  (10 military classes)")
    print("=" * 70)

    cls_idx = {c: i for i, c in enumerate(MILITARY_CLASSES)}
    n_cls   = len(MILITARY_CLASSES)

    # ── Box-level TP/FP/FN ───────────────────────────────────────────────────
    iou_metrics = {}
    for iou_thr in iou_thrs:
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)

        for r in all_results:
            gt_cls   = r["item"]["gt_classes"]   # list str
            gt_boxes = r["item"]["gt_boxes"]       # list [x1y1x2y2]
            dets     = r["detections"]             # list {class, x1,y1,x2,y2}

            gt_matched = [False] * len(gt_boxes)

            for det in dets:
                pred_cls = det.get("class")
                if pred_cls is None:
                    continue
                pred_box = [det["x1"], det["y1"], det["x2"], det["y2"]]
                best_iou, best_j = 0.0, -1
                for j, (gc, gb) in enumerate(zip(gt_cls, gt_boxes)):
                    if gt_matched[j] or gc != pred_cls:
                        continue
                    v = iou_xyxy(pred_box, gb)
                    if v > best_iou:
                        best_iou, best_j = v, j
                if best_iou >= iou_thr and best_j >= 0:
                    gt_matched[best_j] = True
                    tp[pred_cls] += 1
                else:
                    fp[pred_cls] += 1

            for j, gc in enumerate(gt_cls):
                if not gt_matched[j]:
                    fn[gc] += 1

        iou_metrics[iou_thr] = {"tp": tp, "fp": fp, "fn": fn}

    # ── Image-level confusion matrix ─────────────────────────────────────────
    # Row = GT class chính của ảnh (class nhiều GT box nhất)
    # Col = class predict nhiều nhất trong ảnh; "NoDet" nếu không detect gì
    NODET = "NoDet"
    labels   = MILITARY_CLASSES + [NODET]
    lbl_idx  = {n: i for i, n in enumerate(labels)}
    cm       = np.zeros((len(labels), len(labels)), dtype=int)

    for r in all_results:
        # GT class chính = class xuất hiện nhiều nhất
        from collections import Counter
        gt_counter = Counter(r["item"]["gt_classes"])
        gt_cls_main = gt_counter.most_common(1)[0][0]

        dets = r["detections"]
        if dets:
            pred_counter = Counter(d.get("class") for d in dets if d.get("class"))
            pred_cls_main = pred_counter.most_common(1)[0][0] if pred_counter else NODET
        else:
            pred_cls_main = NODET

        gi = lbl_idx.get(gt_cls_main, lbl_idx[NODET])
        pi = lbl_idx.get(pred_cls_main, lbl_idx[NODET])
        cm[gi, pi] += 1

    # vẽ confusion matrix
    _plot_confusion(
        cm, [SHORT_NAMES.get(c, c) for c in MILITARY_CLASSES] + [NODET],
        out_dir / "confusion_matrix.png",
    )

    # ── COCO mAP (pycocotools) ────────────────────────────────────────────────
    coco_summary = {}
    coco_per_cls = {}
    coco_raw_txt = ""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        with open(ann_file) as f:
            raw = json.load(f)
        cat_id_to_name = {c["id"]: c["name"] for c in raw["categories"]
                          if c["name"] in MILITARY_CLASSES}
        name_to_cat_id = {v: k for k, v in cat_id_to_name.items()}
        img_fname_to_id = {img["file_name"]: img["id"] for img in raw["images"]}
        # also map by basename
        img_fname_to_id.update(
            {Path(img["file_name"]).name: img["id"] for img in raw["images"]})

        dt_list = []
        for r in all_results:
            img_id = img_fname_to_id.get(
                r["item"]["file_name"],
                img_fname_to_id.get(Path(r["item"]["file_name"]).name))
            if img_id is None:
                continue
            for det in r["detections"]:
                cls = det.get("class")
                cat_id = name_to_cat_id.get(cls)
                if cat_id is None:
                    continue
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                dt_list.append({
                    "image_id":    img_id,
                    "category_id": cat_id,
                    "bbox":        [x1, y1, x2-x1, y2-y1],
                    "score":       1.0,  # LocateAnything không ra confidence
                })

        if dt_list:
            coco_gt = COCO(str(ann_file))
            coco_dt = coco_gt.loadRes(dt_list)
            cat_ids_sorted = sorted(cat_id_to_name.keys())
            lines = []

            iou_arr = np.array(sorted(iou_thrs), dtype=np.float64)
            ev = COCOeval(coco_gt, coco_dt, "bbox")
            ev.params.iouThrs = iou_arr
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ev.evaluate(); ev.accumulate(); ev.summarize()
            lines.append(f"=== Custom IoU {list(iou_thrs)} ===\n" + buf.getvalue())

            prec = ev.eval["precision"]
            for t_idx, iou in enumerate(iou_arr):
                key = f"AP@{iou:.2f}"
                coco_per_cls[key] = {}
                for k_idx, cat_id in enumerate(cat_ids_sorted):
                    p = prec[t_idx, :, k_idx, 0, -1]
                    p = p[p > -1]
                    coco_per_cls[key][cat_id_to_name[cat_id]] = float(np.mean(p)) if len(p) else 0.0
                coco_summary[key] = float(np.mean(list(coco_per_cls[key].values())))

            # standard mAP@0.50:0.95
            ev2 = COCOeval(coco_gt, coco_dt, "bbox")
            buf2 = io.StringIO()
            with contextlib.redirect_stdout(buf2):
                ev2.evaluate(); ev2.accumulate(); ev2.summarize()
            lines.append("=== Standard mAP@0.50:0.95 ===\n" + buf2.getvalue())
            prec2 = ev2.eval["precision"]
            key95 = "mAP@0.50:0.95"
            coco_per_cls[key95] = {}
            for k_idx, cat_id in enumerate(cat_ids_sorted):
                p = prec2[:, :, k_idx, 0, -1]; p = p[p > -1]
                coco_per_cls[key95][cat_id_to_name[cat_id]] = float(np.mean(p)) if len(p) else 0.0
            coco_summary[key95] = float(np.mean(list(coco_per_cls[key95].values())))

            coco_raw_txt = "\n\n".join(lines)
            print(coco_raw_txt)
            (out_dir / "coco_eval_raw.txt").write_text(coco_raw_txt, encoding="utf-8")
        else:
            print("[COCO] Không có detection hợp lệ để tính mAP.")
    except ImportError:
        print("[COCO] pycocotools chưa cài — bỏ qua mAP. Cài: pip install pycocotools")
    except Exception as e:
        print(f"[COCO] Lỗi: {e}")

    # ── Markdown report ───────────────────────────────────────────────────────
    lines = []
    a = lines.append
    a("# LocateAnything-3B — Evaluation Report (10 Military Classes)\n")
    a(f"- **Model**: `{MODEL_ID}`")
    a(f"- **Annotation**: `{ann_file}`")
    a(f"- **Images evaluated**: {len(all_results)}")
    a(f"- **Prompt**: all 10 classes trong 1 prompt")
    a(f"- **Note**: LocateAnything không output confidence → dùng score=1.0 cho mAP\n")

    # Image-level accuracy (chỉ 10 class, bỏ hàng/cột NoDet để tính acc)
    total = cm[:n_cls, :n_cls].sum()
    correct = sum(cm[i, i] for i in range(n_cls))
    acc = correct / total if total else 0.0
    a(f"## Image-level Accuracy (10 classes)\n")
    a(f"**Overall**: {acc:.4f} ({acc*100:.2f}%)\n")

    # Per-class P/R/F1 tại mỗi IoU
    for iou_thr in iou_thrs:
        m = iou_metrics[iou_thr]
        a(f"## Box-level Metrics @ IoU ≥ {iou_thr:.2f}\n")
        a("| Class | TP | FP | FN | Precision | Recall | F1 |")
        a("|---|---|---|---|---|---|---|")
        ps, rs, f1s = [], [], []
        for cls in MILITARY_CLASSES:
            t  = m["tp"].get(cls, 0)
            fp = m["fp"].get(cls, 0)
            fn = m["fn"].get(cls, 0)
            pr = t / (t + fp + 1e-9)
            rc = t / (t + fn + 1e-9)
            f1 = 2*pr*rc / (pr+rc+1e-9)
            ps.append(pr); rs.append(rc); f1s.append(f1)
            short = SHORT_NAMES[cls]
            a(f"| {short} | {t} | {fp} | {fn} | {pr:.3f} | {rc:.3f} | {f1:.3f} |")
        a(f"| **Mean** | | | | **{np.mean(ps):.3f}** | **{np.mean(rs):.3f}** | **{np.mean(f1s):.3f}** |")
        a("")

    # COCO mAP table
    if coco_per_cls:
        iou_keys = [k for k in ["AP@0.30", "AP@0.50", "AP@0.75", "mAP@0.50:0.95"]
                    if k in coco_per_cls]
        a("## COCO mAP (score=1.0)\n")
        a("| Metric | Value |"); a("|---|---|")
        for k in iou_keys:
            a(f"| {k} | **{coco_summary[k]:.4f}** |")
        a("")
        a("### Per-class AP\n")
        a("| Class | " + " | ".join(iou_keys) + " |")
        a("|---|" + "---|" * len(iou_keys))
        for cls in MILITARY_CLASSES:
            short = SHORT_NAMES[cls]
            vals  = " | ".join(f"{coco_per_cls[k].get(cls, 0.0):.3f}" for k in iou_keys)
            a(f"| {short} | {vals} |")
        means = " | ".join(
            f"**{np.mean(list(coco_per_cls[k].values())):.3f}**" for k in iou_keys)
        a(f"| **Mean** | {means} |")
        a("")

    # Confusion matrix note
    a("## Confusion Matrix (image-level)\n")
    a(f"Xem file `confusion_matrix.png` trong output dir.\n")
    a("**Đọc**: hàng = GT class, cột = class model predict nhiều nhất trong ảnh.\n")

    # Kết luận nên dùng hay không
    a("## Kết luận — Nên dùng LocateAnything?\n")
    ref_iou = 0.5
    m = iou_metrics.get(ref_iou, {})
    weak = []
    strong = []
    for cls in MILITARY_CLASSES:
        t  = m.get("tp", {}).get(cls, 0)
        fn = m.get("fn", {}).get(cls, 0)
        fp = m.get("fp", {}).get(cls, 0)
        rc = t / (t + fn + 1e-9)
        pr = t / (t + fp + 1e-9)
        if rc < 0.4 or pr < 0.4:
            weak.append(f"{SHORT_NAMES[cls]} (Prec={pr:.2f}, Rec={rc:.2f})")
        elif rc >= 0.7 and pr >= 0.7:
            strong.append(f"{SHORT_NAMES[cls]} (Prec={pr:.2f}, Rec={rc:.2f})")
    if strong:
        a(f"**Class hoạt động tốt** (P≥0.7, R≥0.7 @ IoU≥0.5): {', '.join(strong)}")
    if weak:
        a(f"**Class yếu** (P<0.4 hoặc R<0.4 @ IoU≥0.5): {', '.join(weak)}")
    overall_map = coco_summary.get("AP@0.50", None)
    if overall_map is not None:
        verdict = ("✅ Đủ dùng làm VLM classifier trong pipeline tracking"
                   if overall_map >= 0.4
                   else "⚠️ Chưa đủ — cần fine-tune LocateAnything trước khi dùng")
        a(f"\n**AP@0.50 = {overall_map:.4f}** → {verdict}")
    a("")

    report_path = out_dir / "eval_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Terminal summary
    print("\n" + "=" * 70)
    print("EVAL SUMMARY")
    print("=" * 70)
    print(f"Image-level accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    if coco_summary:
        for k, v in coco_summary.items():
            print(f"  {k:<20}: {v:.4f}")
    print(f"\nPer-class @ IoU≥0.50:")
    print(f"  {'Class':<30} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    m05 = iou_metrics.get(0.5, {"tp": {}, "fp": {}, "fn": {}})
    for cls in MILITARY_CLASSES:
        t  = m05["tp"].get(cls, 0)
        fp = m05["fp"].get(cls, 0)
        fn = m05["fn"].get(cls, 0)
        pr = t/(t+fp+1e-9); rc = t/(t+fn+1e-9)
        f1 = 2*pr*rc/(pr+rc+1e-9)
        print(f"  {SHORT_NAMES[cls]:<30} {pr:>6.3f} {rc:>6.3f} {f1:>6.3f}")
    print("=" * 70)
    print(f"Report   : {report_path}")
    print(f"Confusion: {out_dir / 'confusion_matrix.png'}")


def _plot_confusion(cm, labels, out_path):
    n = len(labels)
    row_sum = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm.astype(float), row_sum,
                    out=np.zeros_like(cm, dtype=float),
                    where=row_sum != 0) * 100

    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    im = ax.imshow(pct, cmap="YlGnBu", vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, label="Row %")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")
    ax.set_title("LocateAnything — Confusion Matrix (image-level)", fontsize=12, fontweight="bold")
    for i in range(n):
        for j in range(n):
            v = pct[i, j]
            if v < 1:
                continue
            ax.text(j, i, f"{v:.0f}%\n({cm[i,j]})", ha="center", va="center",
                    color="white" if v >= 50 else "black", fontsize=7, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    out_dir  = Path(args.out_dir)
    ann_file = Path(args.ann)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LocateAnything-3B — Military 10-class inference")
    print(f"  Ann      : {ann_file}")
    print(f"  Img root : {args.img_root}")
    print(f"  Out dir  : {out_dir}")
    print(f"  Device   : {args.device}")
    print(f"  Limit    : {args.limit or 'all'}")
    print("=" * 70)

    items, _ = load_coco_items(ann_file, args.img_root, args.limit)
    print(f"Loaded {len(items)} images with GT annotations.\n")

    if not items:
        print("Không tìm thấy ảnh nào có GT annotation trong dataset!")
        return

    # ── Load model ────────────────────────────────────────────────────────────
    dtype = torch.bfloat16 if "cuda" in args.device else torch.float32
    model_id = args.model_id
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True,
        min_pixels=128*28*28, max_pixels=512*28*28)
    # NOTE: dùng sdpa cho inference. Checkpoint fine-tuned lưu config với
    # _attn_implementation=flash_attention_2, nhưng đường generate (MoonViT +
    # packing) báo lỗi với flash khi sinh từng token -> ép sdpa cho ổn định.
    for attn in (args.attn, "sdpa", "eager"):
        try:
            model = AutoModel.from_pretrained(
                model_id, torch_dtype=dtype, trust_remote_code=True,
                attn_implementation=attn,
            ).to(args.device).eval()
            print(f"Model loaded [{attn}]")
            break
        except Exception as e:
            print(f"[WARN] attn={attn} failed: {e}")
    else:
        raise RuntimeError("Could not load model with any attn_implementation")

    # ── Inference loop ────────────────────────────────────────────────────────
    all_results = []
    summary_rows = []

    for idx, item in enumerate(items, 1):
        stem      = Path(item["file_name"]).stem
        json_path = out_dir / f"{stem}.json"
        vis_path  = out_dir / f"{stem}_vis.jpg"

        print(f"\n[{idx}/{len(items)}] {item['file_name']}")
        print(f"  GT: {list(set(item['gt_classes']))}")

        # skip nếu đã có
        if args.skip_existing and json_path.exists():
            print(f"  SKIP (đã có output)")
            try:
                with open(json_path, encoding="utf-8") as f:
                    saved = json.load(f)
                all_results.append({"item": item, "detections": saved.get("detections", [])})
            except Exception:
                pass
            continue

        try:
            result = run_one_image(
                model, tokenizer, processor, item,
                device=args.device, mode=args.mode,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            dets = result["detections"]
            print(f"  Detected: {len(dets)} boxes  |  {[d.get('class','?') for d in dets]}")
            print(f"  Latency : {result['latency']:.2f}s")

            # lưu JSON
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "file_name":  item["file_name"],
                    "gt_classes": item["gt_classes"],
                    "prompt":     PROMPT,
                    "answer":     result["answer"],
                    "latency":    result["latency"],
                    "detections": dets,
                }, f, indent=2, ensure_ascii=False)

            draw_boxes(item["image_path"], dets,
                       item["gt_boxes"], item["gt_classes"], vis_path)

            all_results.append({"item": item, "detections": dets})
            summary_rows.append({
                "file_name":  item["file_name"],
                "gt_classes": ";".join(item["gt_classes"]),
                "n_gt":       len(item["gt_boxes"]),
                "n_det":      len(dets),
                "pred_classes": ";".join(d.get("class","?") for d in dets),
                "latency_sec":  f"{result['latency']:.3f}",
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({"item": item, "detections": []})
            summary_rows.append({
                "file_name": item["file_name"],
                "gt_classes": ";".join(item["gt_classes"]),
                "n_gt": len(item["gt_boxes"]),
                "n_det": 0,
                "pred_classes": f"ERROR:{e}",
                "latency_sec": "",
            })
        finally:
            gc.collect()
            if "cuda" in args.device:
                torch.cuda.empty_cache()

    # ── Lưu summary CSV ───────────────────────────────────────────────────────
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file_name","gt_classes","n_gt","n_det","pred_classes","latency_sec"])
        writer.writeheader(); writer.writerows(summary_rows)
    print(f"\nSummary CSV: {csv_path}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    if all_results:
        evaluate(all_results, ann_file, out_dir, iou_thrs=(0.3, 0.5, 0.75))

    print(f"\nDone. Outputs: {out_dir}")


if __name__ == "__main__":
    main()
