# tools/anchor_apply.py
#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Ensure the backend is in the path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from backend.registry.anchors import (
    find_anchor,
    apply_anchor,
)  # backend.repo.anchors in prompt, but registry.anchors is correct


def main():
    ap = argparse.ArgumentParser(description="Apply a named anchor to a file.")
    ap.add_argument("--file", required=True)
    ap.add_argument("--anchor", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--payload-file")
    g.add_argument("--payload")
    args = ap.parse_args()

    target = Path(args.file)
    if not target.exists():
        print(f"ERROR: file not found: {target}", file=sys.stderr)
        sys.exit(2)

    anc = find_anchor(args.anchor)
    if not anc:
        print(f"ERROR: unknown anchor: {args.anchor}", file=sys.stderr)
        sys.exit(2)

    text = target.read_text(encoding="utf-8")
    payload = (
        Path(args.payload_file).read_text(encoding="utf-8")
        if args.payload_file
        else args.payload
    )
    new_text, _ = apply_anchor(text, anc, payload or "")
    if new_text and not new_text.endswith("\n"):
        new_text += "\n"
    print(new_text, end="")


if __name__ == "__main__":
    main()
