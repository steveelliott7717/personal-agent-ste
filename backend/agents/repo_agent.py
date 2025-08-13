# backend/agents/repo_agent.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import os

from openai import OpenAI

# If you already have a thin RMS helper, keep using it where convenient:
#   - repo_search(vec, repo, branch, k, prefix) -> List[dict]
#   - format_citation(idx, path, start_line, end_line, sha) -> str
from backend.services.rms import repo_search, format_citation

# Your existing embedding util (returns List[float])
from backend.semantics.embeddings import embed_text

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("CHAT_MODEL", "gpt-5")

SYSTEM_PLAN = (
    "ROLE\n"
    "You are a senior engineer operating inside {REPO}@{BRANCH}@{COMMIT}.\n"
    "You MUST rely ONLY on the RMS context provided here; do not invent unseen code.\n"
    "Cite every code reference as [n] path:start–end@sha.\n\n"
    "MISSION\n"
    "Complete the TASK below with minimal, safe changes. If context is missing, STOP and list files/lines to recall.\n\n"
    "RESPONSE SHAPE\n"
    "1) TL;DR\n2) PLAN\n3) PATCH(ES) — unified diff with [n] citations\n4) TESTS\n5) NOTES\n"
)

def _embed(text: str) -> List[float]:
    """Thin shim in case you later swap models."""
    return embed_text(text)

def repo_search_raw(params: Dict[str, Any], dims: int) -> List[Dict[str, Any]]:
    """
    Thin REST call to the dimensioned repo_search RPC so we can capture raw 4xx error bodies.
    Picks the RPC name based on vector length: 1024 -> repo_search_1024, else repo_search_1536.
    """
    import json
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError

    base = os.environ["SUPABASE_URL"]
        # prefer explicit override (helps while 1024 column isn’t populated)
    force = os.getenv("RMS_FORCE_DIMS", "").strip()
    if force == "1536":
        fn = "repo_search_1536"
    elif force == "1024":
        fn = "repo_search_1024"
    else:
        fn = "repo_search_1024" if dims == 1024 else "repo_search_1536"

    url = f"{base}/rest/v1/rpc/{fn}"
    body = json.dumps(params).encode("utf-8")
    key = os.environ.get("SUPABASE_SERVICE_ROLE") or os.environ["SUPABASE_KEY"]
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }
    req = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8")) or []
    except HTTPError as he:
        detail = he.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {he.code} {he.reason} at RPC {fn}\n{detail}")

def propose_changes(
    task: str,
    *,
    repo: str,
    branch: str,
    commit: str = "HEAD",
    k: int = 12,
    path_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    vec = _embed(task)
    hits = repo_search(vec, repo=repo, branch=branch, k=k, prefix=path_prefix) or []

    # Build context block with citations + snippet content
    ctx_parts: List[str] = []
    for i, h in enumerate(hits, start=1):
        cite = format_citation(i, h["path"], h["start_line"], h["end_line"], h.get("commit_sha"))
        ctx_parts.append(f"{cite}\n```\n{h['content']}\n```")

    prompt = (
        SYSTEM_PLAN.format(REPO=repo, BRANCH=branch, COMMIT=commit)
        + "\nTASK\n" + task
        + "\n\nCONTEXT\n" + ("\n\n".join(ctx_parts) if ctx_parts else "(no context)")
    )

    client = OpenAI()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,  # gpt-5 (must use default temperature)
        messages=[
            {"role": "system", "content": "Follow RESPONSE SHAPE exactly; produce minimal, safe diffs."},
            {"role": "user", "content": prompt},
        ],
    )
    draft = resp.choices[0].message.content or ""
    return {"hits": hits, "draft": draft, "prompt": prompt}

# ---------- Repo Q&A helper (bypasses router) ----------

# ---------- Repo Q&A helper (bypasses router) ----------

def answer_about_repo(
    question: str,
    *,
    repo: str,
    branch: str,
    k: int = 8,
    path_prefix: Optional[str] = None,
    commit: Optional[str] = None,
) -> Dict[str, Any]:
    """Answer a natural-language question strictly from repo memory (RMS)."""

    # 1) Embed the question
    vec = _embed(question)
    if not isinstance(vec, list) or not vec or not isinstance(vec[0], (int, float)):
        raise ValueError("bad embedding vector: expected list[float]")

    # 2) Decide which RPC to call (env override wins)
    dims_env = os.getenv("RMS_FORCE_DIMS", "").strip()
    dims = int(dims_env) if dims_env.isdigit() else len(vec)
    used_rpc = "repo_search_1024" if dims == 1024 else "repo_search_1536"

    # 3) Build params for the newer RPC signature
    #    public.repo_search_1536(branch_in, match_count, prefix_in, query_embedding, repo_in)
    params = {
        "query_embedding": vec,      # <-- was 'query_embedding' but value was undefined before
        "repo_in": repo,
        "branch_in": branch,
        "prefix_in": path_prefix,
        "match_count": k,
    }

    # 4) Call RPC
    try:
        hits = repo_search_raw(params, dims=dims)
    except Exception as e:
        err = getattr(e, "message", None) or str(e)
        return {
            "agent": "repo",
            "intent": "error",
            "message": f"repo_search RPC failed: {err}",
            "debug": {
                "params": {**params, "query_embedding": f"[{len(vec)} floats]"},
                "dims": dims,
                "used_rpc": used_rpc,
            },
        }

    # 5) Build LLM context from hits
    ctx_blocks: List[str] = []
    for i, h in enumerate(hits[:8], start=1):
        cite = format_citation(i, h["path"], h["start_line"], h["end_line"], h.get("commit_sha"))
        ctx_blocks.append(f"{cite}\n```\n{h['content']}\n```")

    system = "Answer concisely. Cite sources as [n] path:start–end@sha. If the answer isn't in the context, say you don't know and list files to fetch."
    user = (
        f"QUESTION:\n{question}\n\n"
        "CONTEXT (authoritative; do not invent beyond this):\n"
        + ("\n\n".join(ctx_blocks) if ctx_blocks else "(no context)")
    )

    client = OpenAI()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    answer = resp.choices[0].message.content or ""

    return {
        "hits": hits,
        "answer": answer,
        "used_commit": commit,
        "used_index_dims": dims,
        "used_model": EMBED_MODEL,
        "used_rpc": used_rpc,
    }
