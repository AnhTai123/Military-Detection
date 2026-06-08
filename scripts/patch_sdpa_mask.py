"""
Patch moon_vit/modeling_vit.py to drop the explicit attention_mask passed
to F.scaled_dot_product_attention.

When an explicit 4-D attention_mask is provided, PyTorch's SDPA is forced
to use the math (materialise-full-matrix) backend — allocating O(N²) VRAM.
Dropping it lets PyTorch pick the memory-efficient backend automatically,
which processes attention in tiles and uses O(N) memory.

Vision encoders use bidirectional (non-causal) self-attention, so passing
attention_mask=None is correct and does not change model behaviour.
"""

import os
import re
import shutil
import sys

TARGET_REL = "eaglevl/model/moon_vit/modeling_vit.py"
REPO_ROOT_DEFAULT = "/home/aiplatform/workspace/Eagle/Embodied"

# Match the sdpa_attention call that passes attention_mask
# attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, ...)
PATTERN = re.compile(
    r"(F\.scaled_dot_product_attention\s*\([^)]*?)attention_mask([^)]*?\))",
    re.MULTILINE | re.DOTALL,
)

REPLACEMENT = r"\1None\2"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=REPO_ROOT_DEFAULT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    target = os.path.join(args.root, TARGET_REL)
    if not os.path.isfile(target):
        print(f"[ERROR] Not found: {target}", file=sys.stderr)
        sys.exit(1)

    with open(target, encoding="utf-8") as f:
        src = f.read()

    dst, n = PATTERN.subn(REPLACEMENT, src)
    if n == 0:
        print(f"[INFO] No match found — already patched or pattern changed.")
        return

    print(f"[PATCH] {TARGET_REL}: replaced attention_mask → None in {n} SDPA call(s)")
    if not args.dry_run:
        bak = target + ".orig_bak"
        if not os.path.exists(bak):
            shutil.copyfile(target, bak)
        with open(target, "w", encoding="utf-8") as f:
            f.write(dst)
        print(f"[INFO] Backup: {bak}")
    else:
        print("[DRY RUN] No files written.")


if __name__ == "__main__":
    main()
