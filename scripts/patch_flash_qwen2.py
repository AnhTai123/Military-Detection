"""
Patch modeling_qwen2.py: fix flash_attention_2 with 4D packing mask.

When attn_implementation=flash_attention_2, _flash_attention_forward calls
_upad_input(attention_mask=...). That function expects a 2D bool mask
[B, seq_len] to locate non-padding tokens. But the MTP packing code passes
a 4D float mask [B, 1, seq_len, seq_len] with -inf/0 values.

_upad_input misreads kv_seq_len as seq_len*seq_len (8192^2 = 67M), then
torch.gather tries to allocate 67M * heads * head_dim bytes -> 4870 GiB OOM.

Fix: in _flash_attention_forward, if attention_mask is 4D, replace it with
None before calling _upad_input. Flash-attn uses cu_seqlens from position_ids
for packing boundaries and does not need an explicit mask.

Run:  python scripts/patch_flash_qwen2.py [--root ...] [--revert]
"""

import argparse
import os
import re
import shutil
import sys

TARGET_REL = "eaglevl/model/locany/modeling_qwen2.py"
REPO_ROOT_DEFAULT = "/home/aiplatform/workspace/Eagle/Embodied"

# Find the start of _flash_attention_forward and insert the mask fix.
START_MARK = "    def _flash_attention_forward("
INSERT_AFTER = re.compile(
    r"(    def _flash_attention_forward\([^)]*\):)\n",
    re.DOTALL
)

# We insert right after the first line of _flash_attention_forward body
# (after the def line and any docstring/first statement).
BODY_START = re.compile(
    r"(    def _flash_attention_forward\(.*?\n)([ \t]+)",
    re.DOTALL
)

FIX_CODE = (
    "    def _flash_attention_forward("
)

REPLACEMENTS = [
    (
        "        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(",
        "        # [PATCH flash-qwen2] 4D packing mask -> None so _upad_input\n"
        "        # reads kv_seq_len correctly (not seq_len^2 -> 4870 GiB OOM).\n"
        "        _attn_mask_for_upad = attention_mask\n"
        "        if attention_mask is not None and attention_mask.dim() == 4:\n"
        "            _attn_mask_for_upad = None\n"
        "        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(",
    ),
]

# Also fix the attention_mask reference inside the _upad_input call
UPAD_CALL_FIX = [
    (
        "        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(\n"
        "            query_states, key_states, value_states, attention_mask, query_length\n"
        "        )",
        "        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(\n"
        "            query_states, key_states, value_states, _attn_mask_for_upad, query_length\n"
        "        )",
    ),
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

    if os.path.exists(bak):
        shutil.copyfile(bak, target)

    with open(target, encoding="utf-8") as f:
        src = f.read()

    n = 0
    for old, new in REPLACEMENTS + UPAD_CALL_FIX:
        if old in src:
            src = src.replace(old, new, 1)
            n += 1

    if n == 0:
        # Try just the first replacement (upad_input call might vary)
        print("[WARN] Exact match failed, trying broader search...", file=sys.stderr)
        # Find _upad_input call and patch attention_mask arg
        pattern = re.compile(
            r"([ \t]+)(query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self\._upad_input\(\s*"
            r"query_states, key_states, value_states, )(attention_mask)(, query_length\s*\))",
            re.MULTILINE,
        )
        def _fix_upad(m):
            i = m.group(1)
            prefix = m.group(2)
            suffix = m.group(4)
            guard = (
                f"{i}# [PATCH flash-qwen2] 4D packing mask causes _upad_input to read\n"
                f"{i}# kv_seq_len = seq_len^2 -> torch.gather OOM 4870 GiB. Pass None.\n"
                f"{i}_attn_mask_4d_fix = attention_mask\n"
                f"{i}if attention_mask is not None and attention_mask.dim() == 4:\n"
                f"{i}    _attn_mask_4d_fix = None\n"
            )
            return f"{guard}{i}{prefix}_attn_mask_4d_fix{suffix}"

        dst, n = pattern.subn(_fix_upad, src)
        if n == 0:
            print("[ERROR] Could not find _upad_input call in "
                  f"{TARGET_REL}. File layout may have changed.", file=sys.stderr)
            sys.exit(1)
        src = dst

    if not os.path.exists(bak):
        shutil.copyfile(target, bak)

    with open(target, "w", encoding="utf-8") as f:
        f.write(src)

    print(f"[PATCH] Fixed 4D mask -> None before _upad_input in {TARGET_REL}")
    print(f"[INFO] Backup: {bak}")


if __name__ == "__main__":
    main()
