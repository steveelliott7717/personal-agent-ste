# tools/anchor_apply.py
import argparse
import sys
from pathlib import Path


# Ensure the backend is in the path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from backend.registry.anchors import find_anchor, apply_anchor


def main():
    parser = argparse.ArgumentParser(
        description="Apply a built-in anchor to a file and print the result."
    )
    parser.add_argument("--file", required=True, help="Path to the input file.")
    parser.add_argument(
        "--anchor", required=True, help="Name of the built-in anchor to apply."
    )
    parser.add_argument(
        "--payload-file",
        required=True,
        help="Path to the file containing the payload to insert.",
    )

    args = parser.parse_args()

    file_path = Path(args.file)
    payload_path = Path(args.payload_file)

    if not file_path.is_file():
        print(f"Error: Input file not found at {file_path}", file=sys.stderr)
        sys.exit(1)

    if not payload_path.is_file():
        print(f"Error: Payload file not found at {payload_path}", file=sys.stderr)
        sys.exit(1)

    anchor = find_anchor(args.anchor)
    if not anchor:
        print(f"Error: Anchor '{args.anchor}' not found in built-ins.", file=sys.stderr)
        sys.exit(1)

    original_text = file_path.read_text(encoding="utf-8")
    payload_text = payload_path.read_text(encoding="utf-8")

    new_text, changed = apply_anchor(original_text, anchor, payload_text)
    if changed:
        print(f"-- Anchor '{args.anchor}' applied to {file_path}. --", file=sys.stderr)
    else:
        print(
            f"-- Anchor '{args.anchor}' was a no-op on {file_path}. --", file=sys.stderr
        )

    print(new_text, end="")


if __name__ == "__main__":
    main()
