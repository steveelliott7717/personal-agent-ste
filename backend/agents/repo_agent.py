# backend/agents/repo_agent.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os
import re
import time
import logging

from openai import OpenAI

from backend.rms import repo_search, repo_search_raw
from backend.services.supabase_service import supabase
from backend.services.conversation import append_message, get_session_n

# Logger for this module
logger = logging.getLogger(__name__)

# --- begin: RMS session memory helpers ---
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict as _Dict, Optional as _Optional

@dataclass
class _MemEntry:
    kind: str         # "task" | "prompt" | "patch" | "error"
    text: str         # small snippet (truncated)
    ts: float         # epoch seconds

# session -> ring buffer of recent items (ephemeral, in-process)
_RMS_MEM: _Dict[str, Deque[_MemEntry]] = {}
_RMS_MEM_N: int = int(os.getenv("RMS_MEM_N", "6"))  # keep it small

def _remember(session: _Optional[str], kind: str, text: str, limit: int = _RMS_MEM_N) -> None:
    if not session:
        return
    q = _RMS_MEM.get(session)
    if q is None:
        q = deque(maxlen=max(3, limit))
        _RMS_MEM[session] = q
    # trim long blobs so we don't balloon memory
    snippet = (text or "").strip()
    if len(snippet) > 1200:
        snippet = snippet[:600] + "\n...\n" + snippet[-600:]
    q.append(_MemEntry(kind=kind, text=snippet, ts=time.time()))

def _context_digest(session: _Optional[str]) -> str:
    if not session:
        return ""
    q = _RMS_MEM.get(session)
    if not q:
        return ""
    # Compose a compact, deterministic digest (oldest -> newest)
    lines = ["# Previous context (most recent last)"]
    for it in list(q):
        hdr = f"- [{it.kind} @ {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(it.ts))}Z]"
        lines.append(hdr)
        lines.append(it.text)
    return "\n".join(lines)
# --- end: RMS session memory helpers ---

# ---------- Config ----------
REPO_SLUG = os.getenv("RMS_REPO", "personal-agent-ste")
REPO_BRANCH = os.getenv("RMS_BRANCH", "main")
EMBED_MODEL = os.getenv("RMS_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("RMS_PATCH_MODEL", os.getenv("RMS_CHAT_MODEL", "gpt-4o-mini"))  # single source of truth

# Baseline system prompt
DEFAULT_RMS_SYSTEM_PROMPT = ("""
You are RMS GPT. Output ONLY a single unified diff (git-apply ready).
The FIRST non-empty line MUST be: diff --git a/... b/...

Hard rules:
- No prose, JSON, markdown fences (```), C-style comments (/* ... */), or ellipses (...).
- Edit ONLY under path_prefix.
- For NEW files: use '--- /dev/null' then '+++ b/<path>'; all hunk body lines must start with '+' (except optional '\ No newline').
- For MODIFIED files: hunk body lines must start with ' ', '+', '-', or '\' only.
- One section per file; correct header order; no duplicate '+++'.
- LF newlines; UTF-8 no BOM; end with exactly one trailing newline.
- Must pass: git apply --check --whitespace=nowarn.

Self-check before answering:
1) Output begins with 'diff --git'.
2) All file sections and hunk headers are structurally valid.
3) All paths are within path_prefix.
4) No prose/fences/markers/comments.
If any check fails, regenerate a valid diff; as last resort, emit a minimal no-op diff under path_prefix that still passes 'git apply --check'.
""").strip()

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
    base = DEFAULT_RMS_SYSTEM_PROMPT
    extra = (os.getenv("RMS_BASE_PROMPT") or "").strip()
    if extra:
        return f"{base}\n\n{extra}"
    return base

def _last_n_messages(session: str, n: int = 10) -> list[dict]:
    """Fallback: read the most recent n conversation rows for a session from Supabase."""
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

# ---------- Patch sanitation/validation helpers ----------
_CODEBLOCK_RE = re.compile(r"```(?:diff|patch)?\s*(?P<body>.*?)```", re.IGNORECASE | re.DOTALL)
_CBLOCK_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_ELLIPSIS_LINE_RE = re.compile(r"^\s*\.\.\.\s*$", re.MULTILINE)

def _sanitize_patch_text(text: str) -> str:
    if not text:
        return ""
    m = _CODEBLOCK_RE.search(text)
    candidate = (m.group("body") if m else text)
    candidate = candidate.replace("\r\n", "\n").replace("\r", "\n")
    candidate = _CBLOCK_RE.sub("", candidate)
    candidate = _ELLIPSIS_LINE_RE.sub("", candidate)
    return candidate.strip()

def _looks_like_unified_diff(text: str) -> bool:
    if not text:
        return False
    if text.startswith("diff --git"):
        return True
    if text.startswith("--- ") and ("\n+++ " in text):
        return True
    return False

def _extract_patch_from_text(text: str) -> Optional[str]:
    cleaned = _sanitize_patch_text(text)
    if not _looks_like_unified_diff(cleaned):
        return None
    return cleaned

# ---------- Patch-generation helpers ----------
def generate_patch_from_prompt(
    prompt: str,
    *,
    session: Optional[str] = None,
    model_env: str = "RMS_PATCH_MODEL",
    default_model: str = CHAT_MODEL,
    temperature: float = 0.0,
) -> Optional[str]:
    """
    Ask the LLM to emit a unified diff ONLY (no fences, no prose).
    Uses Chat Completions exclusively (no Responses fallback).
    """
    if not prompt:
        return None

    model = os.getenv(model_env, default_model) or "gpt-4o-mini"

    system_msg = _build_system_prompt() or (
        "You output software patches as unified diffs ONLY. "
        "No prose, no explanations, no code fences. "
        "Do not include C-style comments (/* ... */). Use LF newlines."
    )
    user_msg = (
        "Produce a unified diff patch for the following repo task. "
        "Output ONLY the diff. No backticks, no commentary, no placeholders.\n\n"
        f"{prompt}"
    )

    logger.info("repo-agent: calling chat.completions model=%s", model)
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
    except Exception as e:
        # Remember the error (for follow-up context) and bubble it up
        try:
            _remember(session, "error", f"{type(e).__name__}: {e}")
        except Exception:
            pass
        logger.exception("repo-agent: chat.completions failed")
        raise

    patch = _extract_patch_from_text(text)

    # Remember the generated patch (raw or cleaned) for follow-ups
    try:
        _remember(session, "patch", patch or text or "")
    except Exception:
        pass

    return patch

# ---------- “Propose changes” ----------
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
    Generates a draft diff (very light-weight stub) + a prompt for RMS GPT.
    Now includes small, session-scoped memory to make follow-ups incremental.
    """
    session: Optional[str] = kwargs.get("session")
    thread_n: Optional[int] = kwargs.get("thread_n")

    # 1) Build search context
    vec = _embed(task)
    hits = repo_search(vec, repo=repo, branch=branch, k=k, prefix=path_prefix) or []

    ctx_parts: List[str] = []
    for i, h in enumerate(hits, start=1):
        cite = _format_citation(i, h["path"], h["start_line"], h["end_line"], h.get("commit_sha"))
        ctx_parts.append(f"{cite}\n```\n{h['content']}\n```")

    # 2) System prompt
    baseline = _build_system_prompt()
    header = f"SYSTEM\n{baseline}\n\n" if baseline else ""

    # 3) Session digest → follow-up bias
    prev = _context_digest(session)
    if prev:
        task_for_model = (
            "Apply a minimal FOLLOW-UP fix based on prior context below. "
            "Output ONLY a single unified diff (git-apply ready). "
            "Do not restate previous changes; adjust just what is necessary.\n\n"
            f"{prev}\n\n"
            "----- NEW REQUEST -----\n"
            f"{task}"
        )
    else:
        task_for_model = task

    # 4) Compose model prompt
    draft = "No automatic changes proposed in this stub."
    prompt = header + "TASK\n" + task_for_model + "\n\nCONTEXT\n" + ("\n\n".join(ctx_parts) if ctx_parts else "(no context)")

    # 5) Remember current task/prompt
    _remember(session, "task", task)
    _remember(session, "prompt", prompt)

    return {
        "hits": hits,
        "draft": draft,
        "prompt": prompt,
        "meta": {
            "system_prompt": baseline,
            "session": session,
            "thread_n": thread_n,
        },
        "summary": "Repo plan generated (JSON mode). Use patch mode to fetch a diff.",
        "patch_present": False,
    }

# ---------- Repo Q&A / Endpoint wrapper ----------
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

    # Health check: no RMS calls
    if q.lower() == "ping":
        return {
            "agent": "repo",
            "intent": "pong",
            "answer": "pong",
            "session": session,
            "thread_n": thread_n,
            "meta": {"system_prompt": _build_system_prompt()},
        }

    # Session memory: explicit store
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

    # Session memory: recall
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

    # Normal RMS-backed Q&A
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

def handle(task: str, *, session: Optional[str] = None, thread_n: Optional[int] = None,
           repo: Optional[str] = None, branch: Optional[str] = None,
           k: int = 8, path_prefix: Optional[str] = None) -> Dict[str, Any]:
    """Thin wrapper used by the route."""
    return answer_about_repo(
        task,
        repo=(repo or REPO_SLUG),
        branch=(branch or REPO_BRANCH),
        k=k,
        path_prefix=path_prefix,
        session=session,
        thread_n=thread_n,
    )
