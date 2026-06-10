"""
Patch modeling_qwen2.py: fix flash_attention_2 with 4D packing mask.

_upad_input expects a 2D bool mask [B, seq_len] to locate non-padding tokens.
The MTP packing code passes a 4D float mask [B, 1, seq_len, seq_len], which
makes _upad_input read kv_seq_len as seq_len^2 (8192^2 = 67M) and the
torch.gather inside flash_attn's index_first_axis tries to allocate 4870 GiB.

Fix: insert a guard at the TOP of _upad_input: if the mask is 4D, replace it
with an all-ones 2D bool mask sized from key_layer (packed sequences have no
padding; flash-attn handles packing via cu_seqlens/position_ids).

This is a line-based patch: it finds the `def _upad_input(` line, detects the
body indentation from the following line, and inserts the guard right after
the def line — no fragile multi-line string matching.

Run:  python scripts/patch_flash_qwen2.py [--root ...] [--revert]
"""

import argparse
import os
import shutil
import sys

TARGET_REL = "eaglevl/model/locany/modeling_qwen2.py"
REPO_ROOT_DEFAULT = "/home/aiplatform/workspace/Eagle/Embodied"

GUARD_LINES = [
    "# [PATCH flash-qwen2] 4D packing mask makes kv_seq_len = seq_len^2",
    "# -> torch.gather OOM (4870 GiB). Packed sequences have no padding,",
    "# so treat every token as valid (2D all-ones bool mask).",
    "if attention_mask is not None and attention_mask.dim() == 4:",
    "    attention_mask = torch.ones(",
    "        key_layer.shape[0], key_layer.shape[1],",
    "        dtype=torch.bool, device=key_layer.device,",
    "    )",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=REPO_ROOT_DEFAULT)
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    target = os.path.join(args.root, TARGET_REL)
    bak = target + ".flashfix_bak"

    if args.revert:
        if os.path.exists(bak):
            shutil.copyfile(bak, target)
            print(f"[REVERT] Restored {TARGET_REL}.")
        else:
            print("[REVERT] No backup found.")
        return

    if not os.path.isfile(target):
        print(f"[ERROR] Not found: {target}", file=sys.stderr)
        sys.exit(1)

    # Idempotent: restore pristine file from backup first.
    if os.path.exists(bak):
        shutil.copyfile(bak, target)

    with open(target, encoding="utf-8") as f:
        lines = f.readlines()

    # Find the def _upad_input line and where its signature ends (the line
    # whose stripped content ends with ':').
    def_idx = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("def _upad_input("):
            def_idx = i
            break
    if def_idx is None:
        print("[ERROR] def _upad_input not found.", file=sys.stderr)
        sys.exit(1)

    sig_end = def_idx
    while sig_end < len(lines) and not lines[sig_end].rstrip().endswith(":"):
        sig_end += 1
    if sig_end >= len(lines):
        print("[ERROR] Could not find end of _upad_input signature.",
              file=sys.stderr)
        sys.exit(1)

    # Detect body indentation from the first non-empty line after the signature.
    body_indent = None
    for j in range(sig_end + 1, len(lines)):
        stripped = lines[j].strip()
        if stripped:
            body_indent = lines[j][: len(lines[j]) - len(lines[j].lstrip())]
            break
    if body_indent is None:
        print("[ERROR] Empty _upad_input body?", file=sys.stderr)
        sys.exit(1)

    guard = "".join(f"{body_indent}{g}\n" for g in GUARD_LINES)

    if not os.path.exists(bak):
        shutil.copyfile(target, bak)

    new_lines = lines[: sig_end + 1] + [guard] + lines[sig_end + 1:]
    with open(target, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    # Syntax check the patched file.
    import ast
    try:
        ast.parse("".join(new_lines))
    except SyntaxError as e:
        shutil.copyfile(bak, target)
        print(f"[ERROR] Patched file failed syntax check ({e}); reverted.",
              file=sys.stderr)
        sys.exit(1)

    print(f"[PATCH] Inserted 4D-mask guard at top of _upad_input in {TARGET_REL}")
    print(f"[INFO] Backup: {bak}")


if __name__ == "__main__":
    main()
