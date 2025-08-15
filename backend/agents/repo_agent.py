from __future__ import annotations

import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple

# --- OpenAI client (Chat Completions only) ------------------------------------
try:
    from openai import OpenAI  # official SDK
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("RMS_PATCH_MODEL", os.getenv("CHAT_MODEL", "gpt-4o-mini"))

_openai = OpenAI(api_key=_OPENAI_API_KEY) if OpenAI else None  # type: ignore

# --- Conversation memory (session-level) --------------------------------------
try:
    # Your existing lightweight conversation service
    from backend.services import conversation as conv
except Exception:  # pragma: no cover
    class _ConvStub:
        _msgs: Dict[str, List[Dict[str, str]]] = {}
        def remember(self, session: str, role: str, content: str) -> None:
            self._msgs.setdefault(session or "_", []).append({"role": role, "content": content})
        def export_messages(self, session: str, limit: int = 50) -> List[Dict[str, str]]:
            return list(self._msgs.get(session or "_", []))[-int(limit):]
        def get_session_n(self, session: str) -> Optional[int]: return None
        def set_session_n(self, session: str, n: int) -> None: ...
        def clear_session(self, session: str) -> int:
            cnt = len(self._msgs.get(session or "_", []))
            self._msgs[session or "_"] = []
            return cnt
        def default_n(self) -> int: return 16
    conv = _ConvStub()  # type: ignore

# --- Repo search / embeddings (best-effort imports; graceful fallback) --------
def _embed(text: str) -> List[float]:
    """Return a vector embedding for repo search. Gracefully degrade if unavailable."""
    try:
        from backend.services.embeddings import embed_text  # your internal helper
        return embed_text(text)
    except Exception:
        # Minimal fallback to keep function signature; caller can continue with empty hits.
        return []

def repo_search(vec: List[float], *, repo: str, branch: str, k: int, prefix: Optional[str]) -> List[Dict[str, Any]]:
    """
    Search the repo for relevant files/snippets. Graceful fallback to empty results.
    Expected hit shape (best effort):
      { "path": "backend/main.py", "start_line": 1, "end_line": 50, "content": "..." , "commit_sha": "..." }
    """
    try:
        from backend.services.repo_index import search_repo  # your internal indexer
        return search_repo(vec, repo=repo, branch=branch, k=k, prefix=prefix) or []
    except Exception:
        return []

# --- RMS system prompt (baseline) ---------------------------------------------
def _build_system_prompt() -> str:
    """Get the RMS system prompt baseline from your config if present."""
    try:
        from backend.rms import DEFAULT_RMS_SYSTEM_PROMPT  # centrally defined hard rules
        return (DEFAULT_RMS_SYSTEM_PROMPT or "").strip()
    except Exception:
        # Minimal, safe fallback
        return (
            "You are RMS GPT. Output ONLY a single unified diff (git-apply ready). "
            "No prose/JSON/fences/comments. Start with 'diff --git'. Use LF newlines."
        )

# --- Pinned, non-negotiable constraints (prepended every time) ----------------
PINNED_BLOCK = (
    "PINNED MUST-HAVES (NON-NEGOTIABLE):\n"
    "1) Output ONLY a single unified diff starting with 'diff --git'. No prose/JSON/fences/comments.\n"
    "2) Edit ONLY under path_prefix. One section per file. No duplicate '+++'.\n"
    "3) If a file already exists (e.g., backend/main.py), emit MODIFIED-FILE headers:\n"
    "   --- a/<path>\n"
    "   +++ b/<path>\n"
    "   Use '--- /dev/null' ONLY for truly new files. New-file hunks must be @@ -0,0 +1,<N> @@ and every body line '+'\n"
    "4) For request logging: set request.state.correlation_id; use header 'X-Correlation-ID'; ASCII logs only; "
    "   DO NOT redefine 'app = FastAPI(...)'.\n"
    "5) MOD-file hunks: body lines begin with ' ', '+', '-', or '\\' only (no bare lines). "
    "   Must pass: git apply --check --whitespace=nowarn (LF, UTF-8 no BOM).\n"
)

# --- Small helpers -------------------------------------------------------------
def _remember(session: Optional[str], role: str, content: str) -> None:
    if not session:
        return
    try:
        conv.remember(session, role, content)  # type: ignore[attr-defined]
    except Exception:
        pass

def _context_digest(session: Optional[str], limit: int = 40) -> str:
    if not session:
        return ""
    try:
        msgs = conv.export_messages(session, limit)  # type: ignore[attr-defined]
    except Exception:
        return ""
    if not msgs:
        return ""
    # Compact digest: keep only recent, short system/user/patch/error snippets
    out: List[str] = []
    for m in msgs[-limit:]:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        # Heuristic: keep short lines & label
        head = content.splitlines()
        snippet = "\n".join(head[:40])
        out.append(f"[{role}] {snippet}")
    return "\n".join(out[-limit:])

# Extract unified diff; tolerate optional BEGIN/END markers if upstream added them.
_BEGIN = re.compile(r"^\s*BEGIN_PATCH\s*$", re.M)
_END   = re.compile(r"^\s*END_PATCH\s*$", re.M)

def _extract_patch_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    mb = _BEGIN.search(s)
    me = _END.search(s)
    if mb and me and mb.start() < me.start():
        body = s[mb.end():me.start()].strip("\n")
        return body if body.startswith("diff --git") else None
    return s if s.lstrip().startswith("diff --git") else None

def _format_citation(idx: int, path: str, start: int, end: int, sha: Optional[str]) -> str:
    sha_part = f" @ {sha[:7]}" if sha else ""
    return f"[{idx}] {path}:{start}-{end}{sha_part}"

# ==============================================================================
# Public: generate_patch_from_prompt
# ==============================================================================
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
    Returns None if not configured or output doesn't look like a diff.
    """
    if not prompt:
        return None
    if not _openai or not _OPENAI_API_KEY:
        _remember(session, "error", "OpenAI client not configured")
        return None

    model = os.getenv(model_env, default_model)
    system_msg = _build_system_prompt()

    # Prepend pinned constraints every time to reduce drift.
    user_msg = (
        "Produce a unified diff patch for the following repo task. "
        "Output ONLY the diff. No backticks, no commentary, no placeholders.\n\n"
        f"{PINNED_BLOCK}\n\n"
        f"{prompt}"
    )

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
    except Exception as e:  # Bubble up, but remember error for follow-ups
        _remember(session, "error", f"{type(e).__name__}: {e}")
        raise

    patch = _extract_patch_from_text(text)

    # Remember raw result for follow-ups (patch or the text the model returned)
    try:
        _remember(session, "patch", patch or text or "")
    except Exception:
        pass

    return patch

# ==============================================================================
# Public: propose_changes (JSON planning endpoint)
# ==============================================================================
def propose_changes(
    task: Any,
    *,
    repo: str,
    branch: str,
    commit: str = "HEAD",
    k: int = 12,
    path_prefix: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Generates a JSON plan and context digest; callers can then ask for a patch via patch mode.
    This preserves backward compatibility with existing callers of /app/api/repo/plan (JSON).
    """
    session: Optional[str] = kwargs.get("session")
    thread_n: Optional[int] = kwargs.get("thread_n")

    # Stringify task safely
    if isinstance(task, str):
        task_str = task
    else:
        try:
            task_str = json.dumps(task, ensure_ascii=False)
        except Exception:
            task_str = str(task)

    # Cap fanout to reduce noisy context
    k_cap = min(int(k or 6), 6)

    vec = _embed(task_str)
    hits = repo_search(vec, repo=repo, branch=branch, k=k_cap, prefix=path_prefix) or []

    # Build compact context citations
    ctx_parts: List[str] = []
    for i, h in enumerate(hits, start=1):
        p = h.get("path") or ""
        s = int(h.get("start_line") or 1)
        e = int(h.get("end_line") or s)
        cite = _format_citation(i, p, s, e, h.get("commit_sha"))
        body = h.get("content") or ""
        # Keep the body under control; model already sees PINNED_BLOCK
        snippet_lines = (body.splitlines() if isinstance(body, str) else [])[:200]
        ctx_parts.append(f"{cite}\n```\n" + "\n".join(snippet_lines) + "\n```")

    baseline = _build_system_prompt()
    header = f"SYSTEM\n{baseline}\n\n" if baseline else ""

    prev = _context_digest(session)
    if prev:
        task_for_model = (
            f"{PINNED_BLOCK}\n\n"
            "Apply a minimal FOLLOW-UP fix based on prior context below. "
            "Output ONLY a single unified diff (git-apply ready). "
            "Do not restate previous changes; adjust just what is necessary.\n\n"
            f"{prev}\n\n"
            "----- NEW REQUEST -----\n"
            f"{task_str}"
        )
    else:
        task_for_model = f"{PINNED_BLOCK}\n\n{task_str}"

    draft = "No automatic changes proposed in this stub."
    prompt = header + "TASK\n" + task_for_model + "\n\nCONTEXT\n" + ("\n\n".join(ctx_parts) if ctx_parts else "(no context)")

    # Remember current task/prompt for follow-ups
    _remember(session, "task", task_str)
    _remember(session, "prompt", prompt)

    # Back-compat JSON (hints for callers)
    return {
        "hits": hits,
        "draft": draft,
        "prompt": prompt,
        "meta": {
            "system_prompt": baseline,
            "session": session,
            "thread_n": thread_n,
            "k_cap": k_cap,
        },
        "summary": "Repo plan generated (JSON mode). Use patch mode to fetch a diff.",
        "patch_present": False,
    }

# ==============================================================================
# Optional: simple repo Q&A passthrough (kept for completeness)
# ==============================================================================
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
    """Minimal placeholder that returns search hits & question; adapt to your needs."""
    vec = _embed(question)
    hits = repo_search(vec, repo=repo, branch=branch, k=min(int(k or 8), 8), prefix=path_prefix) or []
    return {
        "agent": "repo",
        "intent": "answer",
        "question": question,
        "hits": hits,
        "meta": {"session": session, "thread_n": thread_n},
    }
