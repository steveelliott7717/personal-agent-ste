# tools/test_files_validate.py
from __future__ import annotations

import sys, re

_BEGIN = re.compile(r"^BEGIN_FILE\s+(.+)$")
_END = re.compile(r"^END_FILE\s*$")


def is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def validate(text: str) -> tuple[bool, str]:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    if not is_ascii(t):
        return False, "Non-ASCII characters found"
    i = 0
    n = len(t)
    lines = t.split("\n")
    L = len(lines)
    out = []
    first_seen = False
    while i < L:
        line = lines[i]
        if not first_seen and line.strip() == "":
            i += 1
            continue
        first_seen = True
        m = _BEGIN.match(line)
        if not m:
            return False, f"Expected 'BEGIN_FILE <path>' at line {i+1}"
        path = m.group(1).strip()
        if not path:
            return False, f"Empty path at line {i+1}"
        i += 1
        body = []
        while i < L and not _END.match(lines[i]):
            body.append(lines[i])
            i += 1
        if i >= L:
            return False, f"Missing END_FILE for path: {path}"
        # END_FILE line at i
        file_text = "\n".join(body)
        if "\r" in file_text:
            return False, f"{path}: CR characters found; must be LF only"
        if not is_ascii(file_text):
            return False, f"{path}: non-ASCII characters found"
        if not file_text.endswith("\n"):
            return False, f"{path}: file must end with exactly one LF"
        i += 1  # move past END_FILE
    return True, "ok"


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python tools/test_files_validate.py <files.out>", file=sys.stderr)
        return 2
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        text = f.read()
    ok, msg = validate(text)
    if ok:
        print("VALID:", msg)
        return 0
    else:
        print("INVALID:", msg, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
