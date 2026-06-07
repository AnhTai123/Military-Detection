"""
Patch the Eagle/LocateAnything repo to remove hard-coded flash_attn dependencies.

Changes made:
  1. Replace attn_implementation="flash_attention_2" → "sdpa" everywhere.
  2. Wrap unconditional `from flash_attn import ...` / `import flash_attn` so they
     only execute when flash_attn is actually requested, not when using sdpa.
  3. Report every file touched.

Run from the Embodied repo root:
    python scripts/patch_flash_attn.py [--root /path/to/Embodied] [--dry-run]
"""

import argparse
import os
import re
import sys

REPO_ROOT_DEFAULT = "/home/aiplatform/workspace/Eagle/Embodied"

# ── regex patterns ─────────────────────────────────────────────────────────────

# Pattern 1: attn_implementation="flash_attention_2"  (or single quotes)
FLASH_ATTN_IMPL_RE = re.compile(
    r"""(attn_implementation\s*=\s*["'])flash_attention_2(["'])""",
    re.MULTILINE,
)

# Pattern 2: bare `import flash_attn` at module level (not inside a try/if)
BARE_IMPORT_FLASH_ATTN_RE = re.compile(
    r"^(import flash_attn(?:\.\S*)?\s*)$",
    re.MULTILINE,
)

# Pattern 3: bare `from flash_attn import ...` at module level
BARE_FROM_FLASH_ATTN_RE = re.compile(
    r"^(from flash_attn(?:\.\S*)? import .+)$",
    re.MULTILINE,
)

# Guard wrapper template (inline, so indentation stays correct)
GUARD_TEMPLATE = (
    "try:\n"
    "    {line}\n"
    "    _FLASH_ATTN_AVAILABLE = True\n"
    "except ImportError:\n"
    "    _FLASH_ATTN_AVAILABLE = False\n"
)


def patch_content(src: str) -> tuple[str, list[str]]:
    """Return (patched_src, list_of_change_descriptions)."""
    changes = []
    dst = src

    # 1. Replace flash_attention_2 → sdpa
    new_dst, n = FLASH_ATTN_IMPL_RE.subn(r"\1sdpa\2", dst)
    if n:
        changes.append(f"  replaced {n}x attn_implementation='flash_attention_2' → 'sdpa'")
        dst = new_dst

    # 2. Guard bare `import flash_attn*`
    def guard_import(m):
        line = m.group(1).rstrip()
        guarded = GUARD_TEMPLATE.format(line=line)
        changes.append(f"  guarded bare import: {line.strip()!r}")
        return guarded

    # Only patch lines NOT already inside a try block (simple heuristic:
    # check there's no 4-space indent, i.e. it's at module level).
    def guard_if_toplevel(pattern, text):
        def _replace(m):
            line = m.group(1)
            # skip if it's already indented (inside try/if/def)
            if line.startswith((" ", "\t")):
                return line
            return guard_import(m)
        return pattern.sub(_replace, text)

    dst = guard_if_toplevel(BARE_IMPORT_FLASH_ATTN_RE, dst)
    dst = guard_if_toplevel(BARE_FROM_FLASH_ATTN_RE, dst)

    return dst, changes


def iter_py_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        # skip hidden dirs and common non-source dirs
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d not in ("__pycache__", ".git", "build", "dist")
        ]
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=REPO_ROOT_DEFAULT)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change but do not write files")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"ERROR: repo root not found: {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning: {root}  (dry_run={args.dry_run})")
    total_files = 0
    patched_files = 0

    for fpath in sorted(iter_py_files(root)):
        total_files += 1
        with open(fpath, encoding="utf-8", errors="replace") as f:
            src = f.read()

        dst, changes = patch_content(src)
        if not changes:
            continue

        patched_files += 1
        rel = os.path.relpath(fpath, root)
        print(f"\n[PATCH] {rel}")
        for c in changes:
            print(c)

        if not args.dry_run:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(dst)

    print(f"\n{'(DRY RUN) ' if args.dry_run else ''}Patched {patched_files}/{total_files} files.")


if __name__ == "__main__":
    main()
