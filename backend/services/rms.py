# backend/rms.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import os

from backend.services.supabase_service import supabase

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

    # RPC expects these names exactly
    rpc_args = {
        "repo_in": repo,
        "branch_in": branch,
        "prefix_in": prefix,
        "query_embedding": vec,
        "match_count": int(k),
    }

    res = supabase.rpc(fn, rpc_args).execute()
    rows = getattr(res, "data", None) or []
    # Rows should already contain fields our UI expects; just pass through
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
    if not isinstance(embedding, list) or not embedding:
        raise ValueError("repo_search: embedding must be list[float]")

    dims_env = os.getenv("RMS_FORCE_DIMS", "").strip()
    dims = int(dims_env) if dims_env.isdigit() else len(embedding)
    if dims not in (1024, 1536):
        # default to 1536 unless explicitly overridden
        dims = 1536

    params = {
        "q": embedding,
        "repo": repo,
        "branch_in": branch,
        "prefix": prefix,
        "match_count": int(k),
    }
    return repo_search_raw(params, dims=dims)
