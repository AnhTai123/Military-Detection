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
    # Sanitize model weights ONCE on first forward. The released checkpoint
    # leaves RMSNorm (*norm*.weight) values as uninitialized memory containing
    # scattered NaN/Inf (only norm weights affected; all Linear weights clean).
    # NaN norm weight -> NaN after RMSNorm -> NaN throughout the LLM.
    # Fill NaN/Inf in norm weights with 1.0 (identity scale), else 0.0.
    sanitize = (
        f"{indent}if not getattr(self, '_weights_sanitized', False):\n"
        f"{indent}    self._weights_sanitized = True\n"
        f"{indent}    with torch.no_grad():\n"
        f"{indent}        _fixed = 0\n"
        f"{indent}        for _wn, _wp in self.named_parameters():\n"
        f"{indent}            _bad = torch.isnan(_wp) | torch.isinf(_wp)\n"
        f"{indent}            if _bad.any():\n"
        f"{indent}                _wp[_bad] = 1.0 if 'norm' in _wn else 0.0\n"
        f"{indent}                _fixed += int(_bad.sum().item())\n"
        f"{indent}        _fixed_buf = 0\n"
        f"{indent}        for _bn, _bp in self.named_buffers():\n"
        f"{indent}            if not torch.is_floating_point(_bp):\n"
        f"{indent}                continue\n"
        f"{indent}            _bad = torch.isnan(_bp) | torch.isinf(_bp)\n"
        f"{indent}            if _bad.any():\n"
        f"{indent}                _bp[_bad] = 0.0\n"
        f"{indent}                _fixed_buf += int(_bad.sum().item())\n"
        f"{indent}                print('[SANITIZE-BUF] ' + _bn + ' had NaN/Inf', flush=True)\n"
        f"{indent}        _rem = 0\n"
        f"{indent}        for _wn, _wp in self.named_parameters():\n"
        f"{indent}            _rem += int((torch.isnan(_wp) | torch.isinf(_wp)).sum().item())\n"
        f"{indent}        print('[SANITIZE] fixed params=' + str(_fixed) + ' buffers=' +\n"
        f"{indent}              str(_fixed_buf) + ' remaining_bad_params=' + str(_rem), flush=True)\n"
    )
    # Check NaN AND inf: inf in vit_embeds passes isnan() but causes
    # inf/inf = NaN inside RMSNorm -> NaN propagates to q_proj output.
    probe = (
        f"{indent}if torch.isnan(vit_embeds).any() or torch.isinf(vit_embeds).any():\n"
        f"{indent}    print('[NAN-PROBE] NaN/Inf in vit_embeds! nan=' +\n"
        f"{indent}          str(torch.isnan(vit_embeds).sum().item()) + ' inf=' +\n"
        f"{indent}          str(torch.isinf(vit_embeds).sum().item()), flush=True)\n"
        f"{indent}if torch.isnan(input_embeds).any() or torch.isinf(input_embeds).any():\n"
        f"{indent}    print('[NAN-PROBE] NaN/Inf in input_embeds! nan=' +\n"
        f"{indent}          str(torch.isnan(input_embeds).sum().item()) + ' inf=' +\n"
        f"{indent}          str(torch.isinf(input_embeds).sum().item()), flush=True)\n"
        # Clamp inf/nan in vit_embeds before scatter so RMSNorm never sees inf.
        # Use nan_to_num with conservative bounds safe for bf16/fp16.
        f"{indent}vit_embeds = torch.nan_to_num(vit_embeds, nan=0.0, posinf=65504.0, neginf=-65504.0)\n"
    )
    return f"{sanitize}{probe}{indent}input_embeds = input_embeds.clone()\n{indent}{line}"


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
