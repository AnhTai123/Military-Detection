"""
Sanitize NaN/Inf values in a model directory's safetensors files.

Norm weights (*norm*.weight, *.norm.weight) get NaN/Inf replaced with 1.0
(RMSNorm identity scale). All other tensors get 0.0.

Run once before training:
    python scripts/sanitize_model_weights.py --model_dir /tmp/LocateAnything-3B-lora
"""

import argparse
import os
import sys

import torch


def sanitize_dir(model_dir: str):
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError:
        print("[ERROR] safetensors not installed.", file=sys.stderr)
        sys.exit(1)

    st_files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    if not st_files:
        print(f"[WARN] No safetensors files in {model_dir}")
        return

    total_fixed = 0
    for fname in sorted(st_files):
        path = os.path.join(model_dir, fname)
        # Resolve symlink so we can write a real file (not overwrite the cache).
        real_path = os.path.realpath(path)
        tensors = {}
        metadata = {}
        with safe_open(real_path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

        fixed_in_file = 0
        for key, t in tensors.items():
            bad = torch.isnan(t) | torch.isinf(t)
            n = int(bad.sum().item())
            if n > 0:
                fill = 1.0 if "norm" in key.lower() else 0.0
                t[bad] = fill
                tensors[key] = t
                fixed_in_file += n
                print(f"  [FIX] {fname}::{key}  {n} bad values -> {fill}")

        if fixed_in_file > 0:
            # Write sanitized copy next to the symlink (in the lora model dir).
            out_path = path  # path IS in the lora dir (may itself be a symlink target)
            if os.path.islink(path):
                # Replace symlink with a real file containing the sanitized tensors.
                os.remove(path)
                out_path = path  # same name, now a real file
            save_file(tensors, out_path, metadata=metadata)
            print(f"  [SAVED] {out_path}  ({fixed_in_file} elements fixed)")
            total_fixed += fixed_in_file
        else:
            print(f"  [OK] {fname}  (clean)")

    print(f"[SANITIZE] Total fixed across all files: {total_fixed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="/tmp/LocateAnything-3B-lora")
    args = ap.parse_args()
    sanitize_dir(args.model_dir)


if __name__ == "__main__":
    main()
