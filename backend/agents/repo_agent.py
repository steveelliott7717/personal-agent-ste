# backend/agents/repo_agent.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import os

from openai import OpenAI

# RMS helpers
from backend.services.rms import repo_search, format_citation

# Embeddings
from backend.semantics.embeddings import embed_text

# Conversation memory (SQLite-backed)
from backend.services.conversation import (
    append_message,
    get_messages,     # <- renamed
    get_session_n,
    default_n,
)

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
    return embed_text(text)

def repo_search_raw(params: Dict[str, Any], dims: int) -> List[Dict[str, Any]]:
    """
    Raw call to the dimensioned Supabase RPC, picking 1024 vs 1536 by vector length.
    """
    import json
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError

    base = os.environ["SUPABASE_URL"]
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

# ---------- Repo planning (produces diffs) ----------

def propose_changes(
    task: str,
    *,
    repo: str,
    branch: str,
    commit: str = "HEAD",
    k: int = 12,
    path_prefix: Optional[str] = None,
    session: Optional[str] = None,
    thread_n: Optional[int] = None,
) -> Dict[str, Any]:
    vec = _embed(task)
    hits = repo_search(vec, repo=repo, branch=branch, k=k, prefix=path_prefix) or []

    # RMS context (citable)
    ctx_parts: List[str] = []
    for i, h in enumerate(hits, start=1):
        cite = format_citation(i, h["path"], h["start_line"], h["end_line"], h.get("commit_sha"))
        ctx_parts.append(f"{cite}\n```\n{h['content']}\n```")

    # Chat history (non-citable)
    effective_n = None
    history_text = "(none)"
    if session:
        effective_n = int(thread_n) if (thread_n is not None and str(thread_n).isdigit()) else (get_session_n(session) or default_n())
        msgs = get_messages(session, limit=effective_n) or []
        if msgs:
            history_text = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)

    prompt = (
        SYSTEM_PLAN.format(REPO=repo, BRANCH=branch, COMMIT=commit)
        + "\nTASK\n" + task
        + "\n\nCHAT HISTORY (recent; do not cite):\n" + history_text
        + "\n\nCONTEXT\n" + ("\n\n".join(ctx_parts) if ctx_parts else "(no context)")
    )

    client = OpenAI()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Follow RESPONSE SHAPE exactly; produce minimal, safe diffs."},
            {"role": "user", "content": prompt},
        ],
    )
    draft = resp.choices[0].message.content or ""

    if session:
        try:
            append_message(session, "user", task)
            append_message(session, "assistant", draft)
        except Exception:
            pass

    return {
        "hits": hits,
        "draft": draft,
        "prompt": prompt,
        "session": session,
        "thread_n": effective_n,
    }

# ---------- Repo Q&A helper (bypasses router) ----------

def answer_about_repo(
    question: str,
    *,
    repo: str,
    branch: str,
    k: int = 8,
    path_prefix: Optional[str] = None,
    commit: Optional[str] = None,
    session: Optional[str] = None,
    thread_n: Optional[int] = None,
) -> Dict[str, Any]:

    # Embed & dims
    vec = _embed(question)
    if not isinstance(vec, list) or not vec or not isinstance(vec[0], (int, float)):
        raise ValueError("bad embedding vector: expected list[float]")
    dims_env = os.getenv("RMS_FORCE_DIMS", "").strip()
    dims = int(dims_env) if dims_env.isdigit() else len(vec)
    used_rpc = "repo_search_1024" if dims == 1024 else "repo_search_1536"

    # --- Call search via the stable wrapper (handles RPC details) ---
    try:
        hits = repo_search(vec, repo=repo, branch=branch, k=k, prefix=path_prefix) or []
        used_rpc = "repo_search"  # for debugging/return
    except Exception as e:
        err = getattr(e, "message", None) or str(e)
        return {
            "agent": "repo",
            "intent": "error",
            "message": f"repo_search failed: {err}",
            "debug": {
                "dims": dims,
                "used_rpc": "repo_search",
            },
        }

    # RMS context (citable)
    ctx_blocks: List[str] = []
    for i, h in enumerate(hits[:8], start=1):
        cite = format_citation(i, h["path"], h["start_line"], h["end_line"], h.get("commit_sha"))
        ctx_blocks.append(f"{cite}\n```\n{h['content']}\n```")

    # Chat history (non-citable)
    effective_n = None
    history_text = "(none)"
    if session:
        effective_n = int(thread_n) if (thread_n is not None and str(thread_n).isdigit()) else (get_session_n(session) or default_n())
        msgs = get_messages(session, limit=effective_n) or []
        if msgs:
            history_text = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)

    system = (
        "Answer concisely. Cite sources as [n] path:start–end@sha. "
        "Only cite repo sources; do not cite chat history. "
        "If the answer isn't in the repo context, say you don't know and list files to fetch."
    )
    user = (
        f"QUESTION:\n{question}\n\n"
        "CHAT HISTORY (recent; do not cite):\n" + history_text + "\n\n"
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

    if session:
        try:
            append_message(session, "user", question)
            append_message(session, "assistant", answer)
        except Exception:
            pass

    return {
        "hits": hits,
        "answer": answer,
        "used_commit": commit,
        "used_index_dims": dims,
        "used_model": EMBED_MODEL,
        "used_rpc": used_rpc,
        "session": session,
        "thread_n": effective_n,
    }
