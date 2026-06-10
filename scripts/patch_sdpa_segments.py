"""
Replace MoonViT's sdpa_attention with a per-segment implementation.

The original builds a (1, N, N) boolean block-diagonal attention_mask from
q_cu_seqlens (sequence packing: each packed sample may only attend within
itself). Passing that explicit mask forces PyTorch SDPA onto the math
backend, materialising the full N x N matrix -> OOM for long packed
sequences (N ~ 33k => hundreds of GiB).

Computing attention separately per cu_seqlens segment is mathematically
IDENTICAL to the block-diagonal mask, but each segment is only a few
thousand tokens, so memory stays small and no mask is needed at all.

Run:  python scripts/patch_sdpa_segments.py [--root ...] [--revert]
"""

import argparse
import os
import shutil
import sys

TARGET_REL = "eaglevl/model/moon_vit/modeling_vit.py"
REPO_ROOT_DEFAULT = "/home/aiplatform/workspace/Eagle/Embodied"

START_MARK = "def sdpa_attention("
END_MARK = "\ndef eager_attention("

NEW_FUNC = '''def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu_seqlens: Optional[torch.Tensor] = None,
    k_cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """SDPA attention computed per packed segment.

    Equivalent to the original block-diagonal attention_mask built from
    q_cu_seqlens, but never materialises the full N x N matrix: each
    packed sample attends only within its own [start, end) range.
    """
    seq_length = q.shape[0]
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    attn_output = torch.empty_like(q)
    for i in range(1, len(q_cu_seqlens)):
        s = q_cu_seqlens[i - 1]
        e = q_cu_seqlens[i]
        attn_output[:, s:e] = F.scaled_dot_product_attention(
            q[:, s:e].unsqueeze(0),
            k[:, s:e].unsqueeze(0),
            v[:, s:e].unsqueeze(0),
            dropout_p=0.0,
        ).squeeze(0)
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    return attn_output

'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=REPO_ROOT_DEFAULT)
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    target = os.path.join(args.root, TARGET_REL)
    bak = target + ".orig_bak"

    if args.revert:
        if os.path.exists(bak):
            shutil.copyfile(bak, target)
            print(f"[REVERT] Restored {TARGET_REL} from backup.")
        else:
            print("[REVERT] No backup found — nothing to do.")
        return

    if not os.path.isfile(target):
        print(f"[ERROR] Not found: {target}", file=sys.stderr)
        sys.exit(1)

    # Always start from the pristine original if a backup exists,
    # so this patch is idempotent and never stacks on broken edits.
    if os.path.exists(bak):
        shutil.copyfile(bak, target)

    with open(target, encoding="utf-8") as f:
        src = f.read()

    start = src.find(START_MARK)
    end = src.find(END_MARK)
    if start == -1 or end == -1 or end <= start:
        print("[ERROR] Could not locate sdpa_attention function boundaries.",
              file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(bak):
        shutil.copyfile(target, bak)

    dst = src[:start] + NEW_FUNC + src[end + 1:]
    with open(target, "w", encoding="utf-8") as f:
        f.write(dst)

    print(f"[PATCH] Replaced sdpa_attention with per-segment implementation "
          f"in {TARGET_REL}")
    print(f"[INFO] Backup: {bak}")


if __name__ == "__main__":
    main()
