"""
Build a LOCAL LocateAnything model directory with LoRA enabled in config.

Why: the training script loads the model via from_pretrained(model_name_or_path).
If we only edit a stray config.json in /tmp (without the weights), the base model
is loaded and use_llm_lora / use_backbone_lora are silently ignored -> OOM.

This script materializes a real, complete model dir:
  - symlinks every file from the downloaded HF snapshot (weights, tokenizer, etc.)
  - writes a patched config.json with use_llm_lora / use_backbone_lora set

Then point training at it:
    --model_name_or_path /tmp/LocateAnything-3B-lora

Matches the article's config-based LoRA loading exactly, but for from_pretrained(path).
"""

import argparse
import json
import os
import shutil

from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="nvidia/LocateAnything-3B")
    ap.add_argument("--out", default="/tmp/LocateAnything-3B-lora")
    ap.add_argument("--llm_lora", type=int, default=64)
    ap.add_argument("--backbone_lora", type=int, default=64)
    args = ap.parse_args()

    # 1. Ensure the full snapshot (weights + code + tokenizer) is on disk.
    print(f"[INFO] Resolving snapshot for {args.base} ...")
    snap = snapshot_download(repo_id=args.base)
    print(f"[INFO] Snapshot at: {snap}")

    # 2. Recreate output dir as symlinks to every snapshot file.
    if os.path.islink(args.out) or os.path.isfile(args.out):
        os.remove(args.out)
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    for name in os.listdir(snap):
        src = os.path.join(snap, name)
        dst = os.path.join(args.out, name)
        # config.json is written separately (patched) below.
        if name == "config.json":
            continue
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.realpath(src), dst)

    # 3. Patch config.json -> enable LoRA, write real file (not symlink).
    with open(os.path.join(snap, "config.json")) as f:
        cfg = json.load(f)
    cfg["use_llm_lora"] = args.llm_lora
    cfg["use_backbone_lora"] = args.backbone_lora
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[INFO] LoRA model dir ready -> {args.out}")
    print(f"[INFO]   use_llm_lora={args.llm_lora}  use_backbone_lora={args.backbone_lora}")
    print(f"[INFO] Point training at: --model_name_or_path {args.out}")


if __name__ == "__main__":
    main()
