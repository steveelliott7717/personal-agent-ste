# backend/services/rms.py
from __future__ import annotations
import os, json
from typing import List, Dict, Any, Optional
from urllib.request import Request, urlopen

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ["SUPABASE_KEY"]

def _rest(path: str, method="GET", body: Optional[bytes]=None) -> Any:
    url = f"{SUPABASE_URL.rstrip('/')}{path}"
    req = Request(url, data=body, method=method, headers={
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    })
    with urlopen(req) as resp:
        data = resp.read()
    return json.loads(data) if data else None

def repo_search(vec: list[float], repo: str, branch: str, k: int = 12, prefix: Optional[str] = None):
    body = json.dumps({
        "q": vec,
        "repo": repo,
        "branch_in": branch,
        "prefix": prefix,
        "match_count": k
    }).encode("utf-8")
    return _rest("/rest/v1/rpc/repo_search", method="POST", body=body)

def format_citation(idx: int, path: str, start: int, end: int, sha: str) -> str:
    return f"[{idx}] {path}:{start}–{end}@{sha[:7]}"
