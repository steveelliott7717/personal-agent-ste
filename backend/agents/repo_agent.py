# backend/agents/repo_agent.py
from __future__ import annotations

import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple

try:
    # OpenAI SDK v1
    from openai import OpenAI  # type: ignore
    _openai = OpenAI()
except Exception:
    _openai = None  # Will raise on use if not configured


# =========================
# System prompts
# =========================
# ---------- RMS system prompts (minimal & deterministic) ----------

FILES_MODE_SYSTEM_PROMPT = (
    "You return updated files, not a diff.\n\n"
    "FORMAT (STRICT):\n"
    "- For each file:\n"
    "  BEGIN_FILE <path>\n"
    "  <ASCII+LF, full content, ends with exactly one LF>\n"
    "  END_FILE\n"
    "- Nothing outside blocks. No prose, JSON, or markdown fences.\n\n"
    "SCOPE:\n"
    "- Only under path_prefix.\n\n"
    "FORBIDDEN:\n"
    "- Extra helpers, duplicate imports, duplicate 'app = FastAPI(...)', new routes, CORS changes, unicode, CRLF.\n\n"
    "SELF-CHECK before answering:\n"
    "1) Output starts with 'BEGIN_FILE '\n"
    "2) One or more blocks, paths under path_prefix\n"
    "3) ASCII-only, LF-only, one trailing LF per file\n"
)

PATCH_MODE_SYSTEM_PROMPT = (
    "You output ONLY a single unified diff (git-apply ready). The FIRST non-empty line MUST be: diff --git a/... b/...\n"
    "No prose, JSON, markdown fences, or block comments. ASCII only. LF newlines. Exactly one trailing newline.\n"
    "Edit ONLY under path_prefix. NEW files: '--- /dev/null' then '+++ b/<path>'; all hunk lines '+' for new files.\n"
    "Modified files: hunk body lines must start with ' ', '+', '-', or '\\'.\n"
    "One section per file, correct header order, no duplicate '+++'. Must pass: git apply --check --whitespace=nowarn.\n"
)




# =========================
# Utilities
# =========================

# --- Files-context helpers (add near imports) ---
import os
from pathlib import Path
from typing import Iterable

def _safe_read_text(p: Path, max_bytes: int = 100_000) -> str:
    try:
        if not p.is_file():
            return ""
        data = p.read_bytes()[:max_bytes]
        # Force ASCII-ish by replacing non-ASCII with '?'
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""

def _excerpt_around(text: str, needle: str, pre: int = 10, post: int = 10) -> str:
    if not text or not needle:
        return ""
    lines = text.splitlines()
    # find first occurrence
    idx = next((i for i, line in enumerate(lines) if needle in line), None)
    if idx is None:
        return ""
    start = max(0, idx - pre)
    end = min(len(lines), idx + post + 1)
    numbered = [f"{i+1:04d}: {lines[i]}" for i in range(start, end)]
    return "\n".join(numbered)

def _head(text: str, n: int = 80) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    numbered = [f"{i+1:04d}: {line}" for i, line in enumerate(lines[:n])]
    return "\n".join(numbered)

def _collect_context_files(path_prefix: str) -> Iterable[Path]:
    """
    Minimal, deterministic context set. Expand if needed, but keep tiny.
    """
    root = Path(__file__).resolve().parents[2]  # repo root (…/backend/agents/ -> repo)
    targets = []
    # Always include main.py if under this prefix
    main_py = root / "backend" / "main.py"
    if str(main_py).startswith(str(root / path_prefix.strip("/"))):
        targets.append(main_py)
    # You can add other high-signal files here if desired.
    return [p for p in targets if p.exists()]

def _build_files_context(path_prefix: str | None) -> str:
    """
    Returns compact, high-signal context from existing files:
    - Top-of-file header (first ~60 lines)
    - Excerpt around 'app = FastAPI(' anchor
    - Excerpt around stable import anchor
    Also emits explicit DO/DON'T pins for main.py insertion tasks.
    """
    if not path_prefix:
        return ""

    parts: list[str] = []
    for p in _collect_context_files(path_prefix):
        txt = _safe_read_text(p)
        if not txt:
            continue
        head = _head(txt, 60)
        a1 = _excerpt_around(txt, "app = FastAPI(", 8, 8)
        a2 = _excerpt_around(txt, "from backend.utils.agent_protocol import AgentResponse", 8, 8)

        parts.append(
            f"=== FILE: {p.as_posix()} ===\n"
            f"-- HEAD (first 60 lines) --\n{head}\n\n"
            f"-- ANCHOR (app = FastAPI) --\n{a1 or '(not found)'}\n\n"
            f"-- ANCHOR (AgentResponse import) --\n{a2 or '(not found)'}\n"
            f"-- PINS --\n"
            f"DO NOT redefine 'app'. DO NOT move load_dotenv(...) or CORS. "
            f"If asked to add logging: add exactly one import "
            f"'from backend.logging_utils import setup_logging, RequestLoggingMiddleware' "
            f"and insert exactly two lines immediately after the anchor:\n"
            f"setup_logging()\n"
            f"app.add_middleware(RequestLoggingMiddleware)\n"
        )
    return "\n".join(parts).strip()


_ASCII_RE = re.compile(r"^[\x00-\x7F]*$")

def _build_system_prompt(mode: str = "files") -> str:
    if mode == "patch":
        return PATCH_MODE_SYSTEM_PROMPT
    return FILES_MODE_SYSTEM_PROMPT

def _coalesce_response_text(resp: Any) -> str:
    """
    Extract best-effort text from OpenAI ChatCompletion-like responses.
    """
    if not resp:
        return ""
    try:
        ch = (resp.choices or [None])[0]
        if not ch:
            return ""
        msg = getattr(ch, "message", None)
        if msg and getattr(msg, "content", None):
            return msg.content or ""
    except Exception:
        pass
    return ""

def _embed(text: str) -> List[float]:
    # Minimal stub – replace with your actual embedding call if needed.
    return [0.0]

def repo_search(vec: List[float], repo: str, branch: str, k: int, prefix: Optional[str]) -> List[Dict[str, Any]]:
    # Minimal stub for compatibility with existing imports.
    return []

def _remember(session: Optional[str], kind: str, content: str) -> None:
    # Hook to your conversation memory service if desired.
    # No-op by default to avoid side effects.
    return

def _ascii_lf(s: str) -> str:
    # Normalize CRLF to LF and enforce ASCII by stripping non-ASCII
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    if not _ASCII_RE.match(s):
        # Replace non-ASCII with '?'
        s = "".join(ch if ord(ch) < 128 else "?" for ch in s)
    # Ensure single trailing LF
    if not s.endswith("\n"):
        s += "\n"
    return s

def _extract_fileset(text: str, path_prefix: Optional[str]) -> List[Tuple[str, str]]:
    """
    Parse BEGIN_FILE blocks into (path, content) tuples.
    Enforce path_prefix and ASCII/LF constraints.
    """
    files: List[Tuple[str, str]] = []
    # Robust, line-based parse
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("BEGIN_FILE "):
            relpath = line[len("BEGIN_FILE "):].strip()
            i += 1
            buf: List[str] = []
            while i < len(lines) and lines[i].strip() != "END_FILE":
                buf.append(lines[i])
                i += 1
            if i >= len(lines):
                raise ValueError("Missing END_FILE for path: {}".format(relpath))
            content = "\n".join(buf)
            content = _ascii_lf(content)
            # Prefix enforcement
            if path_prefix and not relpath.startswith(path_prefix):
                raise ValueError("File outside path_prefix: {}".format(relpath))
            files.append((relpath, content))
        i += 1

    if not files:
        raise ValueError("No BEGIN_FILE blocks found.")
    return files


# =========================
# Public API
# =========================

def generate_artifact_from_task(
    task: str,
    *,
    repo: str,
    branch: str,
    path_prefix: Optional[str] = None,
    session: Optional[str] = None,
    mode: str = "files",               # "files" or "patch" (we’ll use "files")
    model_env: str = "RMS_PATCH_MODEL",
    default_model: str = "gpt-5",
    **kwargs: Any,                     # tolerate extras like thread_n, k, etc.
) -> Dict[str, Any]:
    """
    Returns {'ok': True, 'content': <BEGIN_FILE...END_FILE text>} on success.

    IMPORTANT: This function is NOT the JSON planning stub. It emits the raw
    files-mode artifact by calling chat.completions with a strict system prompt.
    """
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")

    model = os.getenv(model_env, default_model)

    # Strict, minimal system prompt for FILES mode
    system_msg = (
        "You return updated files, not a diff.\n\n"
        "FORMAT (STRICT):\n"
        "- For each file:\n"
        "  BEGIN_FILE <path>\n"
        "  <ASCII+LF, full content, ends with exactly one LF>\n"
        "  END_FILE\n"
        "- Nothing outside blocks. No prose, JSON, or markdown fences.\n\n"
        "SCOPE:\n"
        "- Only under path_prefix.\n\n"
        "FORBIDDEN:\n"
        "- Extra helpers, duplicate imports, duplicate 'app = FastAPI(...)', new routes, CORS changes, unicode, CRLF.\n\n"
        "SELF-CHECK before answering:\n"
        "1) Output starts with 'BEGIN_FILE '\n"
        "2) One or more blocks, paths under path_prefix\n"
        "3) ASCII-only, LF-only, one trailing LF per file\n"
    )

    # Build a tiny context string (optional); keep it short to avoid token bloat
    path_prefix = path_prefix or ""
    instructions = (
        f"PATH_PREFIX={path_prefix}\n"
        f"MODE={mode}\n\n"
        f"TASK\n{task}\n"
    )

    # Call Chat Completions API (never use Responses API here)
    try:
        resp = _openai.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": instructions},
            ],
        )
        text = (resp.choices[0].message.content if resp and resp.choices else "") or ""
    except Exception as e:
        # Surface exact upstream error to caller/route
        raise RuntimeError(f"chat.completions failed: {type(e).__name__}: {e}")

    # Basic validation: must contain at least one BEGIN_FILE/END_FILE
    if "BEGIN_FILE " not in text or "END_FILE" not in text:
        # Return a structured error for the route to show a 400
        return {
            "ok": False,
            "warning": "No BEGIN_FILE/END_FILE blocks found",
            "content": text,
        }

    # Optional: quick ASCII/LF check and path_prefix check (lightweight)
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return {"ok": False, "warning": "Non-ASCII characters in output", "content": text}

    if "\r" in text:
        return {"ok": False, "warning": "CRLF detected; must be LF only", "content": text}

    # If a path_prefix is provided, enforce it in the blocks
    if path_prefix:
        bad = False
        for line in text.split("\n"):
            if line.startswith("BEGIN_FILE "):
                p = line[len("BEGIN_FILE "):].strip()
                if not p.startswith(path_prefix):
                    bad = True
                    break
        if bad:
            return {"ok": False, "warning": f"Paths must start with '{path_prefix}'", "content": text}

    return {"ok": True, "content": text}


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
    JSON 'plan' response for callers that want a preview plus the input prompt.
    This preserves compatibility with existing clients in JSON mode.
    """
    session: Optional[str] = kwargs.get("session")
    thread_n: Optional[int] = kwargs.get("thread_n")

    # Minimal context search placeholder (kept for interface stability)
    vec = _embed(task if isinstance(task, str) else json.dumps(task))
    hits = repo_search(vec, repo=repo, branch=branch, k=k, prefix=path_prefix) or []

    baseline = _build_system_prompt("files")
    header = f"SYSTEM\n{baseline}\n\n" if baseline else ""
    prompt = header + "TASK\n" + (task if isinstance(task, str) else json.dumps(task)) + "\n\nCONTEXT\n(no context)"

    # Memory breadcrumbs
    try:
        _remember(session, "task", task if isinstance(task, str) else json.dumps(task))
        _remember(session, "prompt", prompt)
    except Exception:
        pass

    return {
        "hits": hits,
        "draft": "No automatic changes proposed in this stub.",
        "prompt": prompt,
        "meta": {"system_prompt": baseline, "session": session, "thread_n": thread_n},
        "summary": "Repo plan generated (JSON mode). Use `format=files` or `format=patch` to fetch an artifact.",
        "patch_present": False,
    }


# Back-compat Q&A entrypoint (used elsewhere)
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
    # Minimal placeholder so imports do not break
    return {"agent": "repo", "intent": "say", "message": "Q&A is not implemented in this stub."}
