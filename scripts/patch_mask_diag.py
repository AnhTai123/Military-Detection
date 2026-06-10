"""
Patch mask_sdpa_utils.py: keep the diagonal visible in 4D SDPA masks.

create_mtp_packing_mask_4d / create_block_diff_mask_by_pe_4d build float
masks with -inf for hidden positions. If any row ends up fully -inf
(e.g. padding or pack-boundary tokens that may attend to nothing),
softmax over all--inf produces NaN, which propagates through every layer
-> "Non-finite loss detected before backward: loss=nan".

Forcing mask[i, i] = 0.0 (every token may attend itself) is mathematically
harmless and guarantees no fully-masked rows.

Run:  python scripts/patch_mask_diag.py [--root ...] [--revert]
"""

import argparse
import os
import shutil
import sys

TARGET_REL = "eaglevl/model/locany/mask_sdpa_utils.py"
REPO_ROOT_DEFAULT = "/home/aiplatform/workspace/Eagle/Embodied"

REPLACEMENTS = [
    # create_mtp_packing_mask_4d: attention_mask shape [seq_len, seq_len]
    (
        "    return attention_mask.unsqueeze(0).unsqueeze(0)",
        "    _diag = torch.arange(seq_len, device=device)\n"
        "    attention_mask[_diag, _diag] = 0.0\n"
        "    attention_mask = torch.nan_to_num(attention_mask, nan=0.0, posinf=0.0, neginf=-65504.0)\n"
        "    return attention_mask.unsqueeze(0).unsqueeze(0)",
    ),
    # create_block_diff_mask_by_pe_4d: customized_mask and final_mask shape [B, seq_len, seq_len]
    (
        "    return customized_mask.unsqueeze(1).to(device=device), final_mask.unsqueeze(1).to(device=device)",
        "    _diag = torch.arange(seq_len, device=device)\n"
        "    customized_mask[:, _diag, _diag] = 0.0\n"
        "    final_mask[:, _diag, _diag] = 0.0\n"
        "    customized_mask = torch.nan_to_num(customized_mask, nan=0.0, posinf=0.0, neginf=-65504.0)\n"
        "    final_mask = torch.nan_to_num(final_mask, nan=0.0, posinf=0.0, neginf=-65504.0)\n"
        "    return customized_mask.unsqueeze(1).to(device=device), final_mask.unsqueeze(1).to(device=device)",
    ),
]


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

    # Idempotent: start from pristine original if a backup exists.
    if os.path.exists(bak):
        shutil.copyfile(bak, target)

    with open(target, encoding="utf-8") as f:
        src = f.read()

    n = 0
    for old, new in REPLACEMENTS:
        if old in src:
            src = src.replace(old, new, 1)
            n += 1

    if n == 0:
        print("[ERROR] No return statements matched — file layout changed?",
              file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(bak):
        shutil.copyfile(target, bak)

    with open(target, "w", encoding="utf-8") as f:
        f.write(src)

    print(f"[PATCH] Forced diagonal visibility in {n} mask function(s) "
          f"in {TARGET_REL}")
    print(f"[INFO] Backup: {bak}")


if __name__ == "__main__":
    main()
