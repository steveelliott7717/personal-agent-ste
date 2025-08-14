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
DEFAULT_RMS_SYSTEM_PROMPT = ("""
You are RMS GPT, a repo modification + Q&A assistant.

HARD OUTPUT CONTRACT (NON-NEGOTIABLE)

Return ONLY a unified diff patch (git-apply ready) starting with 'diff --git'.
No prose, no explanations, no markdown fences (```), no JSON, no logs, no pre/post text.
Do not include C-style block comments (/* ... */) anywhere in the output.
Only use the comment style appropriate for the file's language (e.g., '# ...' for Python).

Always produce a non-empty patch. If no code change is strictly needed, add a harmless no-op
(e.g., a timestamped comment) inside the specified path_prefix using the correct comment style.

Line endings must be LF.
Patch must apply with: git apply --whitespace=fix.

SCOPE & CONSTRAINTS

- Keep edits minimal, reversible, and limited to the task + acceptance criteria.
- Do NOT introduce new services, databases, or heavy dependencies.
- Respect path_prefix. Do not touch files outside it.
- Prefer small, safe changes with in-patch comments describing verification steps.

REQUIRED DIFF FORMAT

- Start each file change with:
  diff --git a/<path> b/<path>
  --- a/<path>
  +++ b/<path>
  @@ <hunk header>
- Include only files under the given path_prefix.
- For new files, use '/dev/null' as the source and the correct 'new file mode'.

FALLBACK / NO-OP RULE

If you determine no functional changes are required, output a valid diff that
modifies (or adds) only a comment inside a file under path_prefix, e.g.:

# RMS GPT no-op touch: <UTC timestamp>  (Python)

INPUTS

You will receive:
Role/Goal/Requirements/Acceptance Criteria/Constraints/Verification Steps
Repo/Branch/Path prefix
Selected file excerpts for context.

DECISION LOGIC (INTERNAL — DO NOT PRINT)

- Plan minimal changes to satisfy Requirements & Acceptance Criteria.
- Edit or create files under path_prefix only.
- Build a single unified diff covering all changes.
- Self-check for forbidden patterns before output:
  - Forbidden: '```', '/*', '*/', standalone '...'
  - If found, remove or replace with valid language comment syntax before final output.

VIOLATION GUARD

If you’re about to emit anything that is not a valid diff or contains forbidden patterns,
stop and emit a no-op diff instead.
""")

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


# ---------- Patch sanitation/validation helpers (NEW) ----------

# detect fenced blocks (we already try to extract body, but keep as safety)
_CODEBLOCK_RE = re.compile(r"```(?:diff|patch)?\s*(?P<body>.*?)```", re.IGNORECASE | re.DOTALL)
# forbid C-style comment blocks anywhere in the patch
_CBLOCK_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
# forbid standalone ellipsis lines often used as placeholders
_ELLIPSIS_LINE_RE = re.compile(r"^\s*\.\.\.\s*$", re.MULTILINE)

def _sanitize_patch_text(text: str) -> str:
    """
    Remove known bad artifacts and normalize newlines.
    """
    if not text:
        return ""
    # If fenced, drop the fences and keep inner
    m = _CODEBLOCK_RE.search(text)
    candidate = (m.group("body") if m else text)

    # Normalize CRLF/CR to LF
    candidate = candidate.replace("\r\n", "\n").replace("\r", "\n")

    # Remove C-style comment blocks and standalone ellipsis lines
    candidate = _CBLOCK_RE.sub("", candidate)
    candidate = _ELLIPSIS_LINE_RE.sub("", candidate)

    # Trim leading/trailing whitespace
    candidate = candidate.strip()
    return candidate

def _looks_like_unified_diff(text: str) -> bool:
    """
    Quick structural checks to guard bad outputs.
    """
    if not text:
        return False
    if text.startswith("diff --git"):
        return True
    # Some engines omit first diff line on single-file edits; allow classic header pair
    if text.startswith("--- ") and ("\n+++ " in text):
        return True
    return False


# ---------- Patch-generation helpers ----------

def _extract_patch_from_text(text: str) -> Optional[str]:
    """
    Try to extract a unified diff from model output. Accepts fenced or raw text.
    Now sanitized and strictly validated.
    """
    cleaned = _sanitize_patch_text(text)
    if not _looks_like_unified_diff(cleaned):
        return None
    return cleaned


def _coalesce_response_text(resp: Any) -> str:
    """
    Best-effort extraction across OpenAI client variants.
    """
    if not resp:
        return ""
    try:
        if hasattr(resp, "output_text"):
            return resp.output_text or ""
    except Exception:
        pass
    try:
        if getattr(resp, "choices", None):
            return resp.choices[0].message.content or ""
    except Exception:
        pass
    try:
        parts = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", "") == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", "") == "output_text":
                        parts.append(getattr(c, "text", ""))
        return "\n".join(parts).strip()
    except Exception:
        return ""


def generate_patch_from_prompt(
    prompt: str,
    model_env: str = "RMS_PATCH_MODEL",
    default_model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> Optional[str]:
    """
    Ask the LLM to emit a unified diff ONLY (no fences, no prose).
    Returns None if not configured or output doesn't look like a diff.
    """
    if not prompt:
        return None
    model = os.getenv(model_env, default_model)

    system_msg = (
        "You output software patches as unified diffs ONLY. "
        "No prose, no explanations, no code fences. "
        "Do not include C-style comments (/* ... */). Use LF newlines."
    )
    user_msg = (
        "Produce a unified diff patch for the following repo task. "
        "Output ONLY the diff. No backticks, no commentary, no placeholders.\n\n"
        f"{prompt}"
    )

    try:
        resp = _openai.responses.create(
            model=model,
            temperature=temperature,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        text = _coalesce_response_text(resp)
    except Exception:
        try:
            resp = _openai.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
            text = (resp.choices[0].message.content if resp and resp.choices else "") or ""
        except Exception:
            text = ""

    return _extract_patch_from_text(text)


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

    draft = "No automatic changes proposed in this stub."
    prompt = header + "TASK\n" + task + "\n\nCONTEXT\n" + ("\n\n".join(ctx_parts) if ctx_parts else "(no context)")

    # Back-compat JSON (additive hints for callers)
    return {
        "hits": hits,
        "draft": draft,
        "prompt": prompt,
        "meta": {"system_prompt": baseline} if baseline else {"system_prompt": None},
        "summary": "Repo plan generated (JSON mode). Use patch mode to fetch a diff.",
        "patch_present": False,
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
