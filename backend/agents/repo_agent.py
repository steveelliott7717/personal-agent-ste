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

DEFAULT_RMS_SYSTEM_PROMPT_PATCH = (
    "You are RMS GPT. Output ONLY a single unified diff (git-apply ready).\n"
    "The FIRST non-empty line MUST be: diff --git a/... b/...\n\n"
    "Hard rules:\n"
    "- No prose, JSON, markdown fences (```), C-style comments (/* ... */), ellipses (...), or non-ASCII glyphs.\n"
    "- Edit ONLY under path_prefix.\n"
    "- For NEW files: use '--- /dev/null' then '+++ b/<path>' and a single new-file hunk where every body line starts with '+'.\n"
    "- For MODIFIED files: hunk body lines must start with ' ', '+', '-', or '\\' only.\n"
    "- One section per file; correct header order; no duplicate '+++'.\n"
    "- Use LF newlines; UTF-8 no BOM; end with exactly one trailing newline.\n"
    "- Must pass: git apply --check --whitespace=nowarn.\n\n"
    "Self-check before answering:\n"
    "1) Output begins with 'diff --git'.\n"
    "2) All file sections and hunk headers are structurally valid.\n"
    "3) All paths are within path_prefix.\n"
    "4) No prose/fences/markers/comments/non-ASCII.\n"
    "If any check fails, regenerate; as last resort, emit a minimal no-op diff under path_prefix that passes 'git apply --check'."
).strip()

DEFAULT_RMS_SYSTEM_PROMPT_FILES = (
    "You are RMS GPT. Return updated files, not a diff.\n\n"
    "OUTPUT FORMAT (STRICT):\n"
    "- For each file to write, emit:\n"
    "  BEGIN_FILE <path>\n"
    "  <full file content>\n"
    "  END_FILE\n"
    "- Use ONLY ASCII characters. Use LF line endings. One trailing LF at file end.\n"
    "- Do NOT emit any text outside these blocks. No prose, JSON, or markdown fences.\n\n"
    "SCOPE:\n"
    "- Edit ONLY files under the provided path_prefix. Do not create, modify, or touch files outside it.\n\n"
    "CONTENT RULES:\n"
    "- Each BEGIN_FILE path must be under path_prefix.\n"
    "- If you modify an existing file, output its complete new content in one block.\n"
    "- Keep changes minimal and reversible. No heavy dependencies. Preserve API shapes unless requested.\n"
    "- Ensure the code is syntactically valid and imports resolve.\n\n"
    "SELF-CHECK BEFORE ANSWERING:\n"
    "1) Output consists solely of one or more BEGIN_FILE/END_FILE blocks.\n"
    "2) All file paths are under path_prefix.\n"
    "3) All file content is ASCII-only and uses LF line endings with exactly one trailing LF.\n"
    "4) No extra commentary or markup."
   """FILES MODE — HARD RULES
- Output only BEGIN_FILE/END_FILE blocks, ASCII-only, LF-only, one trailing LF.
- No placeholders like "...", no C-style comments, no markdown fences.
- Do not re-declare existing singletons (e.g., `app = FastAPI(...)`) unless explicitly told.
- Use task-provided ANCHORS to integrate edits; if an anchor is missing, do NOT guess—emit a comment at top of the file stating which anchor was missing.
SELF-CHECK (must pass):
1) No "..." anywhere.
2) No new `app = FastAPI(` in backend/main.py.
3) All referenced names (e.g., `logger`) are defined/imported.
4) Only allowed files are emitted; each ends with exactly one LF."""

).strip()


# =========================
# Utilities
# =========================

_ASCII_RE = re.compile(r"^[\x00-\x7F]*$")

def _build_system_prompt(mode: str = "files") -> str:
    if mode == "patch":
        return DEFAULT_RMS_SYSTEM_PROMPT_PATCH
    return DEFAULT_RMS_SYSTEM_PROMPT_FILES

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
    mode: str = "files",               # "files" (default) or "patch"
    model_env: str = "RMS_PATCH_MODEL",
    default_model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Call the LLM to produce either:
      - files: BEGIN_FILE blocks (most robust), or
      - patch: unified diff.

    Returns a dict with:
      { "mode": "files", "files": [(path, content), ...] }  OR
      { "mode": "patch", "patch": "<diff text>" }
    """
    if not _openai:
        raise RuntimeError("OpenAI client not configured.")

    system_msg = _build_system_prompt(mode)
    # Keep the user prompt concise and explicit.
    user_msg = (
        "TASK:\n{}\n\n"
        "REPO: {}\nBRANCH: {}\nPATH_PREFIX: {}\n"
        "Return output in {} mode only."
    ).format(task, repo, branch, path_prefix or "", mode.upper())

    # Chat Completions (Responses API removed to avoid schema mismatches)
    resp = _openai.chat.completions.create(
        model=os.getenv(model_env, default_model),
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    text = _coalesce_response_text(resp) or ""
    text = _ascii_lf(text)

    if mode == "files":
        files = _extract_fileset(text, path_prefix)
        _remember(session, "files", json.dumps([p for p, _ in files]))
        return {"mode": "files", "files": files}

    # mode == "patch"
    if not text.lstrip().startswith("diff --git"):
        raise ValueError("Invalid patch output (missing 'diff --git').")
    _remember(session, "patch", "(patch)")
    return {"mode": "patch", "patch": text}


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
