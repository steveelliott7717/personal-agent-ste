# backend/agents/repo_agent.py (strict files/patch modes)
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from backend.repo.retry import retry_artifact_generation

try:
    from openai import OpenAI  # OpenAI SDK v1

    _openai = OpenAI()
except Exception:
    _openai = None


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
    _ = _build_context(path_prefix)
    user_prompt = f"PATH_PREFIX={path_prefix or ''}\nMODE={mode}\n\nTASK\n{task}"

    def _llm_caller(system_prompt: str, user_prompt: str) -> str:
        return _chat(model, system_prompt, user_prompt)

    # We don't pass original_content here, so snippet detection will be skipped.
    # This agent is less critical than the main updater pipeline.
    text, final_mode, _resolution_method = retry_artifact_generation(
        _llm_caller, mode, user_prompt
    )

    if final_mode == "files":
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
