"""
Patch modeling_locateanything.py: clone input_embeds before in-place scatter.

With LoRA the LLM embedding layer is frozen, so input_embeds comes out as a
leaf tensor; gradient checkpointing's make_inputs_require_grad hook then sets
requires_grad=True on it. Writing `input_embeds[selected] = ...` in-place on
a leaf that requires grad raises:
    RuntimeError: a view of a leaf Variable that requires grad is being used
    in an in-place operation.

Inserting `input_embeds = input_embeds.clone()` immediately before each
scatter makes it a non-leaf (clone is an autograd op), so the in-place
write is legal and gradients still flow to vit_embeds.

Run:  python scripts/patch_clone_embeds.py [--root ...] [--revert]
"""

import argparse
import os
import re
import shutil
import sys

TARGET_REL = "eaglevl/model/locany/modeling_locateanything.py"
REPO_ROOT_DEFAULT = "/home/aiplatform/workspace/Eagle/Embodied"

ASSIGN_RE = re.compile(
    r"^(?P<indent>[ \t]+)(?P<line>input_embeds\[selected\] = input_embeds\[selected\][^\n]*)$",
    re.MULTILINE,
)


def _replace(m):
    indent = m.group("indent")
    line = m.group("line")
    probe = (
        f"{indent}if torch.isnan(vit_embeds).any():\n"
        f"{indent}    print('[NAN-PROBE] NaN in vit_embeds (vision encoder output)!', flush=True)\n"
        f"{indent}if torch.isnan(input_embeds).any():\n"
        f"{indent}    print('[NAN-PROBE] NaN in input_embeds (LLM embeddings)!', flush=True)\n"
    )
    return f"{probe}{indent}input_embeds = input_embeds.clone()\n{indent}{line}"


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

    # Idempotent: always start from pristine original if backup exists.
    if os.path.exists(bak):
        shutil.copyfile(bak, target)

    with open(target, encoding="utf-8") as f:
        src = f.read()

    dst, n = ASSIGN_RE.subn(_replace, src)
    if n == 0:
        print("[ERROR] No `input_embeds[selected] = ...` assignment found.",
              file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(bak):
        shutil.copyfile(target, bak)

    with open(target, "w", encoding="utf-8") as f:
        f.write(dst)

    print(f"[PATCH] Inserted input_embeds.clone() before {n} scatter site(s) "
          f"in {TARGET_REL}")
    print(f"[INFO] Backup: {bak}")


if __name__ == "__main__":
    main()
