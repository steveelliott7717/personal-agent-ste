#!/usr/bin/env python3
"""
Idempotently ensure RequestLoggingMiddleware is wired in backend/api.py.

- Adds: from backend.logging_utils import RequestLoggingMiddleware
  after the first "from fastapi import ... FastAPI" (or after the top import block)
- Adds: app.add_middleware(RequestLoggingMiddleware)
  after the first "app = FastAPI("

Usage:
  python tools/ensure_request_logging.py            # modify file if needed
  python tools/ensure_request_logging.py --check    # only report, no changes
"""
import argparse
import re
from pathlib import Path
from typing import Tuple

TARGET = Path("backend/api.py")
IMPORT_LINE = "from backend.logging_utils import RequestLoggingMiddleware"
MW_LINE = "app.add_middleware(RequestLoggingMiddleware)"

FASTAPI_IMPORT_RX = re.compile(r"^\s*from\s+fastapi\s+import\s+.*\bFastAPI\b")
APP_CTOR_RX = re.compile(r"^\s*app\s*=\s*FastAPI\s*\(")


def ensure_import(text: str) -> Tuple[str, bool]:
    if IMPORT_LINE in text:
        return text, False

    lines = text.splitlines(True)
    out = []
    inserted = False

    # Insert after first FastAPI import if present
    for ln in lines:
        out.append(ln)
        if not inserted and FASTAPI_IMPORT_RX.match(ln):
            out.append(IMPORT_LINE + "\n")
            inserted = True

    if inserted:
        new_text = "".join(out)
        return (new_text if new_text.endswith("\n") else new_text + "\n"), True

    # Fallback: insert at end of top import block
    # (scan until the first non-import, non-blank line)
    i = 0
    while i < len(lines):
        s = lines[i].lstrip()
        if s.startswith("import ") or s.startswith("from ") or s == "\n":
            i += 1
            continue
        break
    lines.insert(i, IMPORT_LINE + "\n")
    new_text = "".join(lines)
    return (new_text if new_text.endswith("\n") else new_text + "\n"), True


def ensure_middleware(text: str) -> Tuple[str, bool]:
    if MW_LINE in text:
        return text, False

    lines = text.splitlines(True)
    out = []
    inserted = False

    for ln in lines:
        out.append(ln)
        if not inserted and APP_CTOR_RX.match(ln):
            out.append(MW_LINE + "\n")
            inserted = True

    new_text = "".join(out)
    if not new_text.endswith("\n"):
        new_text += "\n"
    return new_text, inserted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only report, do not write")
    args = ap.parse_args()

    if not TARGET.exists():
        print(f"[error] {TARGET} not found")
        raise SystemExit(2)

    original = (
        TARGET.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    )
    # step 1: ensure import
    t1, imp_added = ensure_import(original)
    # step 2: ensure middleware
    t2, mw_added = ensure_middleware(t1)

    if not imp_added and not mw_added:
        print("[ok] backend/api.py already contains import + middleware")
        raise SystemExit(0)

    if args.check:
        print(f"[would-change] import_added={imp_added}, middleware_added={mw_added}")
        raise SystemExit(1)

    TARGET.write_text(t2, encoding="utf-8")
    print(
        f"[ok] updated backend/api.py (import_added={imp_added}, middleware_added={mw_added})"
    )


if __name__ == "__main__":
    main()
