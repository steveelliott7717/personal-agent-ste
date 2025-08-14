# backend/agents/repo_agent.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os
import re
import time

from openai import OpenAI

from backend.rms import repo_search, repo_search_raw
from backend.services.supabase_service import supabase
from backend.services.conversation import append_message, get_session_n
from backend.services.supabase_service import supabase


# ---------- Config ----------

REPO_SLUG = os.getenv("RMS_REPO", "personal-agent-ste")
REPO_BRANCH = os.getenv("RMS_BRANCH", "main")
EMBED_MODEL = os.getenv("RMS_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("RMS_CHAT_MODEL", "gpt-5")  # keep placeholder; you can change later

# Baseline system prompt (can be disabled or extended via env; see _build_system_prompt)
DEFAULT_RMS_SYSTEM_PROMPT = (
    "You are RMS GPT, a repo modification and Q&A assistant.\n"
    "- When planning changes, produce unified diffs (git-apply ready) and minimal, reversible edits.\n"
    "- Do NOT invent databases, schemas, or large frameworks; keep scope to the task and acceptance criteria.\n"
    "- Respect path_prefix and avoid unrelated files.\n"
    "- Prefer small, safe patches with clear verification steps.\n"
    "- If uncertain, state assumptions and keep changes minimal."
)

_openai = OpenAI()


def _build_system_prompt() -> Optional[str]:
    """
    Returns the effective baseline system prompt or None if disabled.
    Env:
      - RMS_BASE_PROMPT_DISABLE=true  -> disable entirely
      - RMS_BASE_PROMPT               -> appended after baseline (team-specific guardrails)
    """
    if str(os.getenv("RMS_BASE_PROMPT_DISABLE", "")).lower() == "true":
        return None
    base = DEFAULT_RMS_SYSTEM_PROMPT.strip()
    extra = (os.getenv("RMS_BASE_PROMPT") or "").strip()
    if extra:
        return f"{base}\n\n{extra}"
    return base


def _last_n_messages(session: str, n: int = 10) -> list[dict]:
    """
    Fallback: read the most recent n conversation rows for a session.
    Returns list of dicts sorted ascending by id: [{role, content, created_at}, ...]
    """
    try:
        res = (
            supabase.table("conversations")
            .select("role,content,created_at")
            .eq("session", session)
            .order("id", desc=True)
            .limit(n)
            .execute()
        )
        rows = res.data or []
        rows.reverse()  # oldest → newest
        return rows
    except Exception:
        return []

# ---------- Utilities ----------

def _embed(text: str) -> List[float]:
    resp = _openai.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding  # list[float]


def _format_citation(idx: int, path: str, start: int, end: int, commit_sha: Optional[str]) -> str:
    sha = (commit_sha or "HEAD")[:7]
    return f"[{idx}] {path}:{start}–{end}@{sha}"


# ---------- Minimal “session memory” helpers ----------

_TOKEN_STORE_RE = re.compile(r"(?:^|\b)TOKEN::\s*([A-Za-z0-9._\-]+)", re.IGNORECASE)
_TOKEN_NATURAL_STORE_RE = re.compile(
    r"(?:remember\s+this\s+token\s*:?\s*)([A-Za-z0-9._\-]+)", re.IGNORECASE
)
_TOKEN_RECALL_RE = re.compile(r"what\s+token\s+do\s+you\s+remember", re.IGNORECASE)


def _store_session_token(session: Optional[str], text: str) -> Optional[str]:
    """
    Extract a token from user text and persist it into conversations.
    Returns the token if one was stored.
    """
    m = _TOKEN_STORE_RE.search(text) or _TOKEN_NATURAL_STORE_RE.search(text)
    if not m:
        return None
    token = m.group(1).strip()

    # Conversations constraint usually allows roles: 'user','assistant','system'
    # Use 'assistant' to record server-side memory writes.
    if session:
        try:
            append_message(session=session, role="assistant", content=f"MEMORY token={token}")
        except Exception:
            # best-effort; don't crash the request flow
            pass
    return token


def _recall_session_token(session: Optional[str]) -> Optional[str]:
    if not session:
        return None
    try:
        rows = _last_n_messages(session=session, n=50)  # newest last or newest first depending on service
    except Exception:
        rows = []

    # Normalize to iterate newest → oldest
    rows = list(rows) if isinstance(rows, list) else []
    if rows and isinstance(rows[0], dict) and "created_at" in rows[0]:
        # sort ascending on created_at, then reverse so we scan newest-first
        rows = sorted(rows, key=lambda r: r.get("created_at") or "")[::-1]
    else:
        rows = rows[::-1]

    for r in rows:
        content = (r.get("content") or "") if isinstance(r, dict) else str(r)
        mm = re.search(r"MEMORY\s+token=([A-Za-z0-9._\-]+)", content)
        if mm:
            return mm.group(1)
    return None


# ---------- “Propose changes” (kept minimal for completeness) ----------

def propose_changes(
    task: str,
    *,
    repo: str,
    branch: str,
    commit: str = "HEAD",
    k: int = 12,
    path_prefix: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Generates a draft diff (very light-weight stub).
    Keeps the function signature so backend.main import works.
    """
    vec = _embed(task)
    hits = repo_search(vec, repo=repo, branch=branch, k=k, prefix=path_prefix) or []

    ctx_parts: List[str] = []
    for i, h in enumerate(hits, start=1):
        cite = _format_citation(i, h["path"], h["start_line"], h["end_line"], h.get("commit_sha"))
        ctx_parts.append(f"{cite}\n```\n{h['content']}\n```")

    baseline = _build_system_prompt()
    header = f"SYSTEM\n{baseline}\n\n" if baseline else ""

    # You can later swap to a full “planning” prompt. For now, include baseline explicitly.
    draft = "No automatic changes proposed in this stub."
    return {
        "hits": hits,
        "draft": draft,
        "prompt": header + "TASK\n" + task + "\n\nCONTEXT\n" + ("\n\n".join(ctx_parts) if ctx_parts else "(no context)"),
        "meta": {"system_prompt": baseline} if baseline else {"system_prompt": None},
    }


# ---------- Repo Q&A / Endpoint handler ----------

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
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Primary entry the HTTP route uses. It also handles ping + session memory.
    Returns a dict with (at least): agent, intent, answer, session, thread_n.
    """

    q = (question or "").strip()

    # --- Health check: no RMS calls
    if q.lower() == "ping":
        return {
            "agent": "repo",
            "intent": "pong",
            "answer": "pong",
            "session": session,
            "thread_n": thread_n,
            "meta": {"system_prompt": _build_system_prompt()},
        }

    # --- Session memory: explicit store
    token = _store_session_token(session, q)
    if token:
        return {
            "agent": "repo",
            "intent": "memory.store",
            "answer": "Acknowledged. I’ll remember this for this session.",
            "session": session,
            "thread_n": thread_n,
            "meta": {"system_prompt": _build_system_prompt()},
        }

    # --- Session memory: recall
    if _TOKEN_RECALL_RE.search(q):
        recalled = _recall_session_token(session)
        if recalled:
            return {
                "agent": "repo",
                "intent": "memory.recall",
                "answer": f"The token is: {recalled}",
                "session": session,
                "thread_n": thread_n,
                "meta": {"system_prompt": _build_system_prompt()},
            }
        else:
            return {
                "agent": "repo",
                "intent": "memory.miss",
                "answer": "I don’t have a token stored for this session yet.",
                "session": session,
                "thread_n": thread_n,
                "meta": {"system_prompt": _build_system_prompt()},
            }

    # --- Normal RMS-backed Q&A ---
    try:
        vec = _embed(q)
    except Exception as e:
        return {
            "agent": "repo",
            "intent": "error",
            "message": f"embedding failed: {getattr(e, 'message', None) or str(e)}",
            "session": session,
            "thread_n": thread_n,
            "meta": {"system_prompt": _build_system_prompt()},
        }

    # choose dims + rpc via helper; collect top-k hits
    try:
        hits = repo_search(vec, repo=repo, branch=branch, k=k, prefix=path_prefix) or []
    except Exception as e:
        err = getattr(e, "message", None) or str(e)
        return {
            "agent": "repo",
            "intent": "error",
            "message": f"repo_search RPC failed: {err}",
            "debug": {
                "params": {"repo_in": repo, "branch_in": branch, "prefix_in": path_prefix, "query_embedding": vec, "match_count": int(k)},
            },
            "session": session,
            "thread_n": thread_n,
            "meta": {"system_prompt": _build_system_prompt()},
        }

    # Build a concise, cite-aware answer by concatenating contexts (simple baseline)
    if hits:
        cites = []
        for i, h in enumerate(hits[:3], start=1):
            cites.append(_format_citation(i, h["path"], h["start_line"], h["end_line"], h.get("commit_sha")))
        answer = "Here’s what I found in the repo:\n" + "\n".join(cites)
    else:
        answer = "No relevant matches in the repo memory."

    return {
        "agent": "repo",
        "intent": "answer",
        "answer": answer,
        "hits": hits,
        "session": session,
        "thread_n": thread_n,
        "meta": {"system_prompt": _build_system_prompt()},
    }


# ---------- Convenience wrapper the route may call ----------

def handle(task: str, *, session: Optional[str] = None, thread_n: Optional[int] = None,
           repo: Optional[str] = None, branch: Optional[str] = None,
           k: int = 8, path_prefix: Optional[str] = None) -> Dict[str, Any]:
    """
    If your router/route uses a generic `handle`, keep this thin wrapper.
    """
    return answer_about_repo(
        task,
        repo=(repo or REPO_SLUG),
        branch=(branch or REPO_BRANCH),
        k=k,
        path_prefix=path_prefix,
        session=session,
        thread_n=thread_n,
    )
