# backend/utils/patch_sanitizer.py
from __future__ import annotations

import re
from typing import Tuple, Optional

def _to_lf(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")

def _strip_bom(s: str) -> str:
    return s.lstrip("\ufeff")

def _strip_code_fences(s: str) -> str:
    # Remove common markdown fences like ``` or ```diff
    lines = []
    for line in _to_lf(s).split("\n"):
        if line.strip().startswith("```"):
            continue
        lines.append(line)
    return "\n".join(lines)

def _ascii_only(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False

_DIFF_HEADER_RE = re.compile(r"^diff --git a/(?P<a>.+) b/(?P<b>.+)$")
_TRIPLE_RE = re.compile(r"^\\-\\-\\-\\s+(a/.+|/dev/null)$")
_PLUSPLUS_RE = re.compile(r"^\\+\\+\\+\\s+b/.+$")
_HUNK_RE = re.compile(r"^@@\\s+-(\\d+)(?:,(\\d+))?\\s+\\+(\\d+)(?:,(\\d+))?\\s+@@")

def validate_patch_structure(patch_text: str, path_prefix: Optional[str] = None) -> Tuple[bool, str]:
    """
    Lightweight structural validation for a unified diff.
    Ensures ASCII+LF, header order, and that b/<path> starts with path_prefix (if given).
    """
    t = _to_lf(_strip_bom(patch_text))
    if not _ascii_only(t):
        return False, "Non-ASCII characters detected"
    if "\r" in t:
        return False, "CR characters detected; must be LF only"
    if not t.endswith("\n"):
        return False, "Patch must end with exactly one LF"

    lines = t.split("\n")
    i = 0
    saw_any = False
    seen_paths = set()

    while i < len(lines) and lines[i].strip() == "":
        i += 1
    if i >= len(lines) or not lines[i].startswith("diff --git a/"):
        return False, "First non-empty line must start with 'diff --git a/'"

    while i < len(lines):
        if lines[i].startswith("diff --git a/"):
            saw_any = True
            m = _DIFF_HEADER_RE.match(lines[i])
            if not m:
                return False, f"Malformed diff header: {lines[i]}"
            a_path = m.group("a")
            b_path = m.group("b")
            if path_prefix and not b_path.startswith(path_prefix):
                return False, f"b/{b_path} must start with {path_prefix}"
            if b_path in seen_paths:
                return False, f"Duplicate file section for {b_path}"
            seen_paths.add(b_path)

            i += 1
            # optional index line
            if i < len(lines) and lines[i].startswith("index "):
                i += 1

            if i >= len(lines) or not _TRIPLE_RE.match(lines[i]):
                return False, f"Expected '--- a/<path>' or '--- /dev/null' at line {i+1}"
            i += 1

            if i >= len(lines) or not _PLUSPLUS_RE.match(lines[i]):
                return False, f"Expected '+++ b/<path>' at line {i+1}"
            i += 1

            # Expect at least one hunk
            if i >= len(lines) or not _HUNK_RE.match(lines[i]):
                return False, f"Expected hunk header @@ at line {i+1}"

            while i < len(lines) and not lines[i].startswith("diff --git a/"):
                # hunk header or body
                if lines[i].startswith("@@ "):
                    if not _HUNK_RE.match(lines[i]):
                        return False, f"Malformed hunk header at line {i+1}"
                else:
                    if lines[i] and lines[i][0] not in (" ", "+", "-", "\\"):
                        return False, f"Invalid hunk body line prefix at line {i+1}"
                i += 1
        else:
            return False, f"Unexpected line outside a file section at {i+1}: {lines[i]}"
    if not saw_any:
        return False, "No file sections found"
    return True, "ok"

def sanitize_patch(patch_text: str) -> Tuple[str, list[str]]:
    """
    Applies minimal, safe normalizations:
    - Strip BOM
    - Remove markdown code fences
    - Convert to LF
    - Ensure exactly one trailing LF
    Returns (sanitized_text, warnings)
    """
    warnings = []
    t = _strip_bom(patch_text)
    t2 = _strip_code_fences(t)
    if t != t2:
        warnings.append("Removed markdown code fences")
    t3 = _to_lf(t2)
    if t3.endswith("\n\n"):
        # collapse multiple trailing LFs to one
        t3 = t3.rstrip("\n") + "\\n"
        warnings.append("Collapsed extra trailing newlines")
    elif not t3.endswith("\\n"):
        t3 = t3 + "\\n"
        warnings.append("Appended trailing LF")
    return t3, warnings
