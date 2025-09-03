# backend/utils/patch_linter.py
from __future__ import annotations
import re
from typing import Dict, List, Optional

_DIFF_HEADER = re.compile(r"^diff --git a/(.+?) b/\1$", re.MULTILINE)
_FILE_HEADER = re.compile(r"^(?:---|\+\+\+) (.+)$", re.MULTILINE)
FENCE_RE = re.compile(r"```")


def _paths_from_patch(patch: str) -> List[str]:
    return _DIFF_HEADER.findall(patch or "")


def _has_crlf(patch: str) -> bool:
    # Detect any CRLF (Windows) sequences
    return "\r\n" in patch


def _outside_prefix(paths: List[str], prefix: Optional[str]) -> List[str]:
    if not prefix:
        return []
    pref = prefix.strip("/")
    bad = []
    for p in paths:
        if not (p == pref or p.startswith(pref + "/")):
            bad.append(p)
    return bad


def lint_patch(
    *,
    patch: str,
    path_prefix: Optional[str],
    required: Optional[List[str]] = None,
    forbidden: Optional[List[str]] = None,
    max_files: Optional[int] = 40,
    max_bytes: Optional[int] = 250_000,
) -> Dict[str, object]:
    """
    Lightweight checks to catch common patch problems. Returns:
      {
        "ok": bool,
        "issues": [str, ...],
        "summary": str,
        "files": [str, ...]
      }
    """
    issues: List[str] = []
    files = _paths_from_patch(patch)
    text = patch or ""

    if not text.strip():
        issues.append("Empty patch body.")
    if not text.startswith("diff --git") and not text.startswith("--- "):
        issues.append(
            "Patch does not start with a unified diff header (expected 'diff --git' or '---/+++')."
        )
    if FENCE_RE.search(text):
        issues.append(
            "Patch contains markdown code fences (```), which will break git apply."
        )
    if _has_crlf(text):
        issues.append("Patch contains CRLF newlines; convert to LF before applying.")
    if max_files is not None and len(files) > max_files:
        issues.append(f"Patch touches too many files ({len(files)} > {max_files}).")
    if max_bytes is not None and len(text.encode("utf-8")) > max_bytes:
        issues.append("Patch is very large; consider splitting or reviewing carefully.")

    # Path-prefix fence
    bad = _outside_prefix(files, path_prefix)
    if bad:
        issues.append(
            f"Patch modifies files outside path_prefix '{path_prefix}': {', '.join(sorted(set(bad)))}"
        )

    # Custom caller-provided checks
    for pat in required or []:
        if re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE) is None:
            issues.append(f"Required pattern not found: /{pat}/")
    for pat in forbidden or []:
        if re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE):
            issues.append(f"Forbidden pattern present: /{pat}/")

    ok = len(issues) == 0
    summary = "OK" if ok else (issues[0] if issues else "Issues found")
    return {"ok": ok, "issues": issues, "summary": summary, "files": files}


def make_followup_prompt(
    *, task: str, issues: List[str], path_prefix: Optional[str]
) -> str:
    bullet = "\n".join(f"- {i}" for i in issues) or "- (no issues)"
    return (
        "Role: Disposable Project Orchestrator — SMALL FOLLOW-UP PATCH ONLY.\n"
        f"Path prefix: {path_prefix or '(as before)'}\n\n"
        "Fix strictly and minimally the following problems detected in the last patch:\n"
        f"{bullet}\n\n"
        "Constraints:\n"
        "- Keep changes minimal and limited to the existing files unless creation is required.\n"
        "- Do NOT revert prior good changes; only address the items above.\n"
        "- Return ONLY a unified diff (no fences or prose), LF newlines, must apply with: git apply --whitespace=fix.\n\n"
        "Original task (for context):\n"
        f"{task.strip()}\n"
    )
