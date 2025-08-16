# backend/rms.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import os

# test f
from backend.services.supabase_service import supabase

# In backend/rms.py
import re

_CODEBLOCK_RE = re.compile(r"(?s)```(?:diff|patch)?\s*(?P<body>.*?)```", re.IGNORECASE)
_CBLOCK_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_ELLIPSIS_RE = re.compile(r"(?m)^\s*\.\.\.\s*$")
# Optional envelope (won't be used if you stick to raw diffs)
_ENVELOPE_RE = re.compile(
    r"(?ms)^\s*BEGIN_PATCH\s*\n(?P<body>.*?)(?:\n)?\s*END_PATCH\s*$"
)
_DIFF_START_RE = re.compile(r"(?m)^\s*diff --git\s+a/")


def _sanitize_patch_text(text: str) -> str:
    if not text:
        return ""
    s = text.replace("\r\n", "\n").replace("\r", "\n")

    # 1) Strip optional BEGIN/END envelope
    m = _ENVELOPE_RE.search(s)
    if m:
        s = m.group("body")

    # 2) Strip optional fenced block
    m = _CODEBLOCK_RE.search(s)
    if m:
        s = m.group("body")

    # 3) Remove C-style blocks and ellipsis-only lines
    s = _CBLOCK_RE.sub("", s)
    s = _ELLIPSIS_RE.sub("", s)

    # 4) Trim & allow leading whitespace/newlines
    return s.lstrip()


def _looks_like_unified_diff(s: str) -> bool:
    return bool(_DIFF_START_RE.search(s))


def _extract_patch_from_text(text: str) -> str | None:
    cleaned = _sanitize_patch_text(text)
    if not cleaned or not _looks_like_unified_diff(cleaned):
        return None
    return cleaned


def _choose_dims(embedding: List[float]) -> int:
    """
    Decide which RPC (1024 vs 1536) to call.
    - If RMS_FORCE_DIMS is set to "1024" or "1536", use that.
    - Else infer from the vector length; default to 1536 if unknown.
    """
    if not isinstance(embedding, list) or not embedding:
        raise ValueError("embedding must be a non-empty list[float]")
    forced = os.getenv("RMS_FORCE_DIMS", "").strip() or ""
    if forced.isdigit():
        n = int(forced)
        if n in (1024, 1536):
            return n
    return len(embedding) if len(embedding) in (1024, 1536) else 1536


# ---- Low-level raw search: calls the Postgres RPC directly ----
def repo_search_raw(params: Dict[str, Any], *, dims: int) -> List[Dict[str, Any]]:
    """
    params expects:
      {
        "q": <embedding list[float]>,
        "repo": <str>,
        "branch_in": <str>,
        "prefix": <str | None>,
        "match_count": <int>
      }
    dims: 1536 or 1024 (decides which RPC to call)
    """
    vec = params.get("q")
    repo = params.get("repo")
    branch = params.get("branch_in")
    prefix = params.get("prefix")
    k = params.get("match_count", 8)

    if not isinstance(vec, list) or not vec:
        raise ValueError("repo_search_raw: 'q' must be a non-empty list[float]")

    fn = "repo_search_1536" if int(dims) == 1536 else "repo_search_1024"

    # RPC expects these names exactly (to avoid ambiguous overload selection)
    rpc_args = {
        "repo_in": repo,
        "branch_in": branch,
        "prefix_in": prefix,  # may be None/null -> SQL handles it
        "query_embedding": vec,
        "match_count": int(k),
    }

    res = supabase.rpc(fn, rpc_args).execute()
    rows = getattr(res, "data", None) or []
    return rows


# ---- Friendly wrapper: chooses dims & passes normalized args ----
def repo_search(
    embedding: List[float],
    *,
    repo: str,
    branch: str,
    k: int = 8,
    prefix: Optional[str] = None,
) -> List[Dict[str, Any]]:
    dims = _choose_dims(embedding)
    params = {
        "q": embedding,
        "repo": repo,
        "branch_in": branch,
        "prefix": prefix,
        "match_count": int(k),
    }
    return repo_search_raw(params, dims=dims)
