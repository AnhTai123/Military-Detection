"""
Patch modeling_locateanything.py: register forward hooks that pinpoint the
FIRST sub-module whose output contains NaN.

We have confirmed vit_embeds and input_embeds are clean, yet the final
hidden_states carry NaN. Forcing the SDPA mask diagonal did not help, so the
"fully-masked row -> softmax NaN" theory is likely wrong. Instead of guessing,
this patch instruments every sub-module of the model with a forward hook.

Forward hooks fire in execution order and a child fires before its parent, so
the FIRST line printed as `[NAN-LAYER] <name>` is the exact leaf module that
first introduced a NaN. That tells us whether it is an attention block, an MLP,
a norm, or a LoRA adapter -- and which layer index.

Run:  python scripts/patch_layer_probe.py [--root ...] [--revert]
"""

import argparse
import os
import re
import shutil
import sys

TARGET_REL = "eaglevl/model/locany/modeling_locateanything.py"
REPO_ROOT_DEFAULT = "/home/aiplatform/workspace/Eagle/Embodied"

# Anchor: the scatter line that the clone-patch also targets. We insert the
# hook-registration block right before it (self is the top-level model here).
ANCHOR_RE = re.compile(
    r"^(?P<indent>[ \t]+)(?P<line>input_embeds\[selected\] = input_embeds\[selected\][^\n]*)$",
    re.MULTILINE,
)

PROBE_TEMPLATE = (
    "{i}if not getattr(self, '_nan_probe_hooks', False):\n"
    "{i}    self._nan_probe_hooks = True\n"
    "{i}    _bad_w = 0\n"
    "{i}    for _wn, _wp in self.named_parameters():\n"
    "{i}        if torch.isnan(_wp).any() or torch.isinf(_wp).any():\n"
    "{i}            _bad_w += 1\n"
    "{i}            print('[NAN-WEIGHT] ' + _wn + ' nan=' +\n"
    "{i}                  str(torch.isnan(_wp).sum().item()) + ' inf=' +\n"
    "{i}                  str(torch.isinf(_wp).sum().item()), flush=True)\n"
    "{i}    print('[NAN-PROBE] weights with NaN/Inf: ' + str(_bad_w), flush=True)\n"
    "{i}    def _nan_probe_mk(_name):\n"
    "{i}        def _nan_probe_hook(_mod, _inp, _out):\n"
    "{i}            _t = _out[0] if isinstance(_out, (tuple, list)) and len(_out) else _out\n"
    "{i}            if torch.is_tensor(_t) and torch.isnan(_t).any():\n"
    "{i}                _it = _inp[0] if isinstance(_inp, (tuple, list)) and len(_inp) else _inp\n"
    "{i}                _in_nan = torch.is_tensor(_it) and torch.isnan(_it).any().item()\n"
    "{i}                _in_inf = torch.is_tensor(_it) and torch.isinf(_it).any().item()\n"
    "{i}                print('[NAN-LAYER] first NaN at: ' + _name +\n"
    "{i}                      ' | input_nan=' + str(_in_nan) + ' input_inf=' + str(_in_inf), flush=True)\n"
    "{i}        return _nan_probe_hook\n"
    "{i}    for _pn, _pm in self.named_modules():\n"
    "{i}        _pm.register_forward_hook(_nan_probe_mk(_pn))\n"
    "{i}    print('[NAN-PROBE] registered forward hooks on all submodules', flush=True)\n"
)


def _replace(m):
    indent = m.group("indent")
    line = m.group("line")
    probe = PROBE_TEMPLATE.format(i=indent)
    return f"{probe}{indent}{line}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=REPO_ROOT_DEFAULT)
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    target = os.path.join(args.root, TARGET_REL)
    bak = target + ".probe_bak"

    if args.revert:
        if os.path.exists(bak):
            shutil.copyfile(bak, target)
            print(f"[REVERT] Restored {TARGET_REL} from probe backup.")
        else:
            print("[REVERT] No probe backup found — nothing to do.")
        return

    if not os.path.isfile(target):
        print(f"[ERROR] Not found: {target}", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(bak):
        shutil.copyfile(bak, target)

    with open(target, encoding="utf-8") as f:
        src = f.read()

    dst, n = ANCHOR_RE.subn(_replace, src)
    if n == 0:
        print("[ERROR] Anchor `input_embeds[selected] = ...` not found.",
              file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(bak):
        shutil.copyfile(target, bak)

    with open(target, "w", encoding="utf-8") as f:
        f.write(dst)

    print(f"[PATCH] Inserted NaN-layer forward hooks before {n} site(s) "
          f"in {TARGET_REL}")
    print(f"[INFO] Backup: {bak}")


if __name__ == "__main__":
    main()
