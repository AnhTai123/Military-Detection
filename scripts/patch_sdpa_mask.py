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
TORCH_IMPORT = "import torch\n"
REPO_ROOT_DEFAULT = "/home/aiplatform/workspace/Eagle/Embodied"

# Match the sdpa_attention call that passes attention_mask
# attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, ...)
PATTERN = re.compile(
    r"([ \t]*)(attn_output\s*=\s*F\.scaled_dot_product_attention\s*\([^)]*?attention_mask[^)]*?\))",
    re.MULTILINE | re.DOTALL,
)


CHUNK_ATTN_FN = '''
def _chunked_attention(q, k, v, chunk_size=512):
    """Tiled attention: O(chunk*N) memory instead of O(N^2)."""
    import math
    B, H, N, D = q.shape
    scale = math.sqrt(D)
    orig_dtype = q.dtype
    q, k, v = q.float(), k.float(), v.float()
    out = torch.zeros(B, H, N, D, dtype=torch.float32, device=q.device)
    for i in range(0, N, chunk_size):
        qi = q[:, :, i:i + chunk_size]
        scores = torch.matmul(qi, k.transpose(-2, -1)) / scale
        attn = torch.softmax(scores, dim=-1)
        out[:, :, i:i + chunk_size] = torch.matmul(attn, v)
    return out.to(orig_dtype)
'''


def _replace(m):
    indent = m.group(1)
    return (
        f"{CHUNK_ATTN_FN}\n"
        f"{indent}attn_output = _chunked_attention(q, k, v)"
    )


REPLACEMENT = _replace


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

    dst = PATTERN.sub(REPLACEMENT, src)
    n = len(PATTERN.findall(src))
    # Ensure torch is imported (needed for sdp_kernel context manager)
    if n > 0 and "import torch" not in dst.split("\n")[:10]:
        dst = TORCH_IMPORT + dst
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
