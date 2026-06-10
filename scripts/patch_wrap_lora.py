"""
Patch locany_finetune_magi_stream.py: wrap LLM LoRA AFTER from_pretrained.

ROOT CAUSE of loss=nan: setting use_llm_lora=64 in config.json makes the
model wrap PEFT/LoRA inside __init__, renaming every LLM parameter to
language_model.base_model.model.model.layers...; from_pretrained then can't
match the checkpoint keys (language_model.model.layers...) so the ENTIRE LLM
is left randomly initialized ("Some weights ... were not used" warning) ->
garbage weights -> NaN.

Correct order: load the model with use_llm_lora=0 (keys match, weights load),
THEN call model.wrap_llm_lora(r=..., lora_alpha=...).

This script finds the `model = ....from_pretrained(...)` call in the training
script and inserts the wrap right after it.

Run:  python scripts/patch_wrap_lora.py [--root ...] [--rank 64] [--revert]
"""

import argparse
import os
import re
import shutil
import sys

TARGET_REL = "eaglevl/train/locany_finetune_magi_stream.py"
REPO_ROOT_DEFAULT = "/home/aiplatform/workspace/Eagle/Embodied"

WRAP_TEMPLATE = """
{i}# [PATCH wrap-lora] Apply LLM LoRA AFTER weights are loaded so checkpoint
{i}# keys match. (Config-based use_llm_lora wraps in __init__ -> keys mismatch
{i}# -> whole LLM left random-init -> loss=nan.)
{i}if hasattr(model, 'wrap_llm_lora'):
{i}    model.wrap_llm_lora(r={rank}, lora_alpha={alpha})
{i}    print('[WRAP-LORA] wrapped LLM LoRA after load: r={rank} alpha={alpha}', flush=True)
{i}else:
{i}    print('[WRAP-LORA][ERROR] model has no wrap_llm_lora method!', flush=True)
"""


def find_from_pretrained_block(src):
    """Locate `model = ...from_pretrained(` and the matching close paren.

    Returns (insert_pos, indent) — position right after the full call's
    closing paren (end of that line), or (None, None).
    """
    m = re.search(r"^([ \t]*)model\s*=\s*[\w.]+\.from_pretrained\s*\(",
                  src, re.MULTILINE)
    if not m:
        return None, None
    indent = m.group(1)
    # paren matching from the opening "("
    i = src.index("(", m.start())
    depth = 0
    while i < len(src):
        c = src[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    # advance to end of line
    nl = src.find("\n", i)
    return (nl + 1 if nl != -1 else len(src)), indent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=REPO_ROOT_DEFAULT)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    target = os.path.join(args.root, TARGET_REL)
    bak = target + ".wraplora_bak"

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

    if os.path.exists(bak):
        shutil.copyfile(bak, target)

    with open(target, encoding="utf-8") as f:
        src = f.read()

    pos, indent = find_from_pretrained_block(src)
    if pos is None:
        print("[ERROR] Could not find `model = ...from_pretrained(` in "
              f"{TARGET_REL}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(bak):
        shutil.copyfile(target, bak)

    wrap = WRAP_TEMPLATE.format(i=indent, rank=args.rank, alpha=args.alpha)
    dst = src[:pos] + wrap + src[pos:]
    with open(target, "w", encoding="utf-8") as f:
        f.write(dst)

    print(f"[PATCH] Inserted wrap_llm_lora(r={args.rank}) after "
          f"from_pretrained in {TARGET_REL}")
    print(f"[INFO] Backup: {bak}")


if __name__ == "__main__":
    main()
