# backend/rms.py
from __future__ import annotations
from typing import Any, Dict, List
from backend.services.supabase_service import supabase

SUPABASE_RPC_1536 = "repo_search_1536"
SUPABASE_RPC_1024 = "repo_search_1024"


def repo_search_raw(params: Dict[str, Any], *, dims: int) -> List[Dict[str, Any]]:
    """Dispatch to the right RPC based on dims. Params must include:
    q (list[float]), repo, branch_in, prefix, match_count
    """
    rpc_name = SUPABASE_RPC_1024 if dims == 1024 else SUPABASE_RPC_1536
    # PostgREST requires named args for RPCs
    payload = {
        "repo_in": params["repo"],
        "branch_in": params["branch_in"],
        "prefix_in": params.get("prefix"),
        "query_embedding": params["q"],
        "match_count": params.get("match_count", 8),
    }
    res = supabase.rpc(rpc_name, payload).execute()
    return res.data or []


def repo_search(
    vec: List[float], *, repo: str, branch: str, k: int, prefix: str | None
) -> List[Dict[str, Any]]:
    dims = len(vec)
    params = {
        "q": vec,
        "repo": repo,
        "branch_in": branch,
        "prefix": prefix,
        "match_count": k,
    }
    return repo_search_raw(params, dims=dims)
