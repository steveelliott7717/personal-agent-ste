# backend/agents/repo_agent.py (strict files/patch modes)
from __future__ import annotations

import os, json, re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

try:
    from openai import OpenAI  # OpenAI SDK v1

    _openai = OpenAI()
except Exception:
    _openai = None

FILES_SYSTEM = (
    "You return updated files, not a diff.\n\n"
    "OUTPUT FORMAT (STRICT)\n"
    "- For each file to write, emit exactly:\n"
    "  BEGIN_FILE <path>\n"
    "  <full ASCII content with LF newlines, ends with exactly one LF>\n"
    "  END_FILE\n"
    "- Nothing outside these blocks. No prose, JSON, or markdown fences.\n\n"
    "SCOPE\n"
    "- Edit ONLY files under PATH_PREFIX.\n"
    "- Keep edits minimal and surgical. Do not reformat unrelated code.\n\n"
    "FORBIDDEN\n"
    "- Duplicate app creation/middleware/imports.\n"
    "- Changing dotenv/CORS unless explicitly requested.\n"
    "- Non-ASCII, CRLF, or code fences.\n\n"
    "SELF-CHECK BEFORE ANSWERING\n"
    "1) First non-empty line starts with 'BEGIN_FILE '.\n"
    "2) Every BEGIN_FILE path starts with PATH_PREFIX.\n"
    "3) ASCII-only, LF-only, exactly one trailing LF per file.\n"
)

PATCH_SYSTEM = (
    "You output ONLY a single unified diff (git-apply ready). No prose, JSON, or markdown fences. ASCII+LF, one trailing LF.\n"
    "Scope: edit only under PATH_PREFIX.\n"
    "Rules:\n"
    "- Modified files: header order diff/index/--- a/ +++ b/, hunks start @@, body lines start with space,'+','-','\\'.\n"
    "- New files: --- /dev/null then +++ b/<path>, all body lines '+'.\n"
    "- One section per file; no duplicate '+++'. All b/<path> start with PATH_PREFIX.\n"
)


def _ascii_lf(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _collect_context_files(path_prefix: Optional[str]) -> List[Path]:
    root = Path(__file__).resolve().parents[2]
    if not path_prefix:
        return []
    pp = (root / path_prefix.strip("/")).resolve()
    out: List[Path] = []
    cand = root / "backend" / "main.py"
    if cand.exists() and str(cand).startswith(str(pp)):
        out.append(cand)
    return out


def _excerpt(txt: str, needle: str, before: int, after: int) -> str:
    i = txt.find(needle)
    if i < 0:
        return ""
    start = max(0, i - before * 80)
    end = min(len(txt), i + len(needle) + after * 80)
    return txt[start:end]


def _build_context(path_prefix: Optional[str]) -> str:
    parts: List[str] = []
    for p in _collect_context_files(path_prefix):
        try:
            t = p.read_text(encoding="utf-8")
        except Exception:
            continue
        head = "\n".join(t.splitlines()[:60])
        ex1 = _excerpt(t, "app = FastAPI(", 8, 8)
        ex2 = _excerpt(
            t,
            "from backend.logging_utils import setup_logging, RequestLoggingMiddleware",
            8,
            8,
        )
        parts.append(f"FILE {p.as_posix()}\n{head}\n---\n{ex1}\n---\n{ex2}\n")
    pins = (
        "Pins for backend/main.py edits:\n"
        "- Keep ONE 'app = FastAPI(title=\"Personal Agent API\")'\n"
        "- Insert ONE setup_logging() immediately after app creation\n"
        "- Insert ONE app.add_middleware(RequestLoggingMiddleware)\n"
        "- Do not alter dotenv or CORS\n"
    )
    return "\n\n".join(parts + [pins]) if parts else pins


def _parse_files_output(
    text: str, *, path_prefix: Optional[str]
) -> List[Tuple[str, str]]:
    lines = _ascii_lf(text).split("\n")
    out: List[Tuple[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "":
            i += 1
            continue
        if not line.startswith("BEGIN_FILE "):
            raise ValueError(f"Expected 'BEGIN_FILE <path>' at line {i+1}")
        rel = line[len("BEGIN_FILE ") :].strip()
        if not rel:
            raise ValueError(f"Empty path at line {i+1}")
        i += 1
        buf: List[str] = []
        while i < len(lines) and lines[i] != "END_FILE":
            buf.append(lines[i])
            i += 1
        if i >= len(lines):
            raise ValueError(f"Missing END_FILE for {rel}")
        body = "\n".join(buf)
        if not _is_ascii(body):
            raise ValueError(f"{rel}: non-ASCII characters")
        if "\r" in body:
            raise ValueError(f"{rel}: CR found; must be LF only")
        if not body.endswith("\n"):
            raise ValueError(f"{rel}: must end with exactly one LF")
        if path_prefix and not rel.startswith(path_prefix):
            raise ValueError(f"{rel}: outside PATH_PREFIX={path_prefix}")
        out.append((rel, body))
        i += 1  # skip END_FILE
    if not out:
        raise ValueError("No BEGIN_FILE/END_FILE blocks found")
    return out


def _chat(model: str, system: str, user: str) -> str:
    if _openai is None:
        raise RuntimeError("OpenAI client is not configured")
    resp = _openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    if not resp or not resp.choices:
        return ""
    return resp.choices[0].message.content or ""


def generate_artifact_from_task(
    task: str,
    *,
    repo: str,
    branch: str,
    path_prefix: Optional[str] = None,
    session: Optional[str] = None,
    mode: str = "files",
    model_env: str = "RMS_PATCH_MODEL",
    default_model: str = "gpt-5",
    **kwargs: Any,
) -> Dict[str, Any]:
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")
    model = os.getenv(model_env, default_model)
    ctx = _build_context(path_prefix)
    header = f"PATH_PREFIX={path_prefix or ''}\nMODE={mode}\n\n"
    system = FILES_SYSTEM if mode == "files" else PATCH_SYSTEM
    text = _chat(model, system, header + "TASK\n" + task)

    if mode == "files":
        try:
            files = _parse_files_output(text, path_prefix=path_prefix)
        except Exception as e:
            return {"ok": False, "warning": str(e), "content": text}
        return {"ok": True, "files": files, "content": text}
    else:
        t = _ascii_lf(text)
        if not t.endswith("\n"):
            t += "\n"
        if not _is_ascii(t):
            return {"ok": False, "warning": "Non-ASCII in patch", "patch": t}
        return {"ok": True, "patch": t}


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
    return {
        "agent": "repo",
        "intent": "plan",
        "task": task,
        "repo": repo,
        "branch": branch,
        "path_prefix": path_prefix,
    }
