from __future__ import annotations

import os
import json
from typing import Any, Dict, Tuple, Optional, List
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import (
    JSONResponse,
    RedirectResponse,
    HTMLResponse,
    FileResponse,
    PlainTextResponse,
)
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

# add near your other imports
from backend.agents.repo_agent import generate_artifact_from_task
import re

# reuse the verb/plan handlers but mount them on THIS app’s routes below
from backend.api import call_agent_verb, call_agent_plan

from dotenv import load_dotenv

load_dotenv()

# Package-qualified imports
from backend.agents.repo_agent import propose_changes, generate_artifact_from_task

# Logging
from backend.logging_utils import setup_logging, RequestLoggingMiddleware

# NEW: exception types for consistent error shaping
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

app = FastAPI(title="Personal Agent API")

from backend.routers import schema as schema_router

app.include_router(schema_router.router)

setup_logging()
app.add_middleware(RequestLoggingMiddleware)

# Mount the capability-registry endpoints (implemented in backend/api.py)
app.add_api_route("/app/api/agents/verb", call_agent_verb, methods=["POST"])
app.add_api_route("/app/api/agents/plan", call_agent_plan, methods=["POST"])

_BEGIN_RE = re.compile(r"^BEGIN_FILE\s+(.+)$")
_END_RE = re.compile(r"^END_FILE\s*$")


def _is_ascii_lf(s: str) -> bool:
    try:
        s.encode("ascii")
    except UnicodeEncodeError:
        return False
    return "\r" not in s


def _parse_files_artifact(text: str) -> list[tuple[str, str]]:
    lines = text.split("\n")
    i, n = 0, len(lines)
    blocks: list[tuple[str, str]] = []

    while i < n and lines[i].strip() == "":
        i += 1

    if i >= n or not lines[i].startswith("BEGIN_FILE "):
        raise ValueError("first non-empty line must be BEGIN_FILE <path>")

    while i < n:
        m = _BEGIN_RE.match(lines[i])
        if not m:
            raise ValueError(f"expected BEGIN_FILE at line {i+1}")
        path = m.group(1).strip()
        i += 1
        buf: list[str] = []
        found_end = False
        while i < n:
            if _END_RE.match(lines[i]):
                found_end = True
                i += 1
                break
            buf.append(lines[i])
            i += 1
        if not found_end:
            raise ValueError(f"missing END_FILE for {path}")
        content = "\n".join(buf)
        if not content.endswith("\n"):
            content = content + "\n"
        if not _is_ascii_lf(content):
            raise ValueError(f"non-ASCII or CRLF detected in {path}")
        blocks.append((path, content))

        while i < n and lines[i].strip() == "":
            i += 1
        if i < n and not lines[i].startswith("BEGIN_FILE "):
            raise ValueError(f"unexpected text after END_FILE at line {i+1}")

    return blocks


def _try_fix_patch_with_ps1(patch_text: str) -> Optional[str]:
    from pathlib import Path

    script_rel = Path(__file__).resolve().parents[1] / "tools" / "fix-patch.ps1"
    if not script_rel.exists():
        return None
    for exe in ("pwsh", "powershell"):
        try:
            import subprocess, tempfile

            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", delete=False
            ) as f_in:
                f_in.write(patch_text)
                in_path = f_in.name
            out_path = in_path + ".out.patch"
            cmd = [
                exe,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_rel),
                "-InPath",
                in_path,
                "-OutPath",
                out_path,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            if proc.returncode == 0 and Path(out_path).exists():
                fixed = Path(out_path).read_text(encoding="utf-8")
                return fixed
        except Exception:
            continue
    return None


# -------------------- Health --------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/app/api/repo/health")
def repo_health():
    return {"ok": True, "model": os.getenv("CHAT_MODEL", "gpt-5")}


@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/app/")


# ---------- Files-mode validation helpers ----------
_ALLOWED_FILES = {"backend/logging_utils.py", "backend/main.py"}
_MAIN_SENTINELS = [
    "from backend.agents.router_agent import route_request",
    "class NaturalLanguageMiddleware(BaseHTTPMiddleware)",
    "app.add_middleware(NaturalLanguageMiddleware)",
]


def _ascii_lf_only(s: str) -> bool:
    try:
        s.encode("ascii")
    except UnicodeEncodeError:
        return False
    return ("\r" not in s) and s.endswith("\n")


def _parse_files_blocks(text: str) -> dict[str, str]:
    files: dict[str, str] = {}
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = t.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("BEGIN_FILE "):
            rel = line[len("BEGIN_FILE ") :].strip()
            i += 1
            buf = []
            while i < len(lines) and lines[i].strip() != "END_FILE":
                buf.append(lines[i])
                i += 1
            if i >= len(lines):
                raise ValueError(f"Missing END_FILE for {rel}")
            body = "\n".join(buf)
            if not body.endswith("\n"):
                body += "\n"
            files[rel] = body
        i += 1
    if not files:
        raise ValueError("No BEGIN_FILE/END_FILE blocks found")
    return files


def _validate_files_mode(files: dict[str, str]) -> tuple[bool, str]:
    bad = [p for p in files.keys() if p not in _ALLOWED_FILES]
    if bad:
        return False, f"Contains paths outside allowed set: {bad}"
    for p, body in files.items():
        if not _ascii_lf_only(body):
            return False, f"{p} not ASCII/LF or missing trailing LF."
    if "backend/main.py" in files:
        new_main = files["backend/main.py"]
        for s in _MAIN_SENTINELS:
            if s not in new_main:
                return False, f"backend/main.py missing sentinel: {s}"
    return True, "ok"


def _enforce_trailing_lf_per_file(artifact: str) -> str:
    t = artifact.replace("\r\n", "\n").replace("\r", "\n")
    lines = t.split("\n")
    out = []
    i = 0
    L = len(lines)
    while i < L:
        if not lines[i].startswith("BEGIN_FILE "):
            i += 1
            continue
        path = lines[i][len("BEGIN_FILE ") :].strip()
        i += 1
        body_lines = []
        while i < L and lines[i].strip() != "END_FILE":
            body_lines.append(lines[i])
            i += 1
        if i >= L:
            raise ValueError(f"Missing END_FILE for {path}")
        i += 1
        body = "\n".join(body_lines).replace("\r\n", "\n").replace("\r", "\n")
        body = body.rstrip("\n") + "\n"
        out.append(f"BEGIN_FILE {path}\n{body}END_FILE\n")
    return ("".join(out)).rstrip("\n") + "\n"


# -------------------- Repo endpoints --------------------
@app.post("/app/api/repo/plan")
def repo_plan(payload: Dict[str, Any], request: Request):
    task = (payload or {}).get("task")
    if not task:
        raise HTTPException(status_code=400, detail="Missing 'task'")
    repo = payload.get("repo", "personal-agent-ste")
    branch = payload.get("branch", "main")
    prefix = payload.get("path_prefix", "backend/")
    k = int(payload.get("k", 12))
    session = payload.get("session")
    thread_n = payload.get("thread_n")
    fmt = request.query_params.get("format") or request.headers.get("X-RMS-Format")

    if fmt == "files":
        art = generate_artifact_from_task(
            task,
            repo=repo,
            branch=branch,
            path_prefix=prefix,
            session=session,
            mode="files",
        )
        if not art.get("ok"):
            return PlainTextResponse(
                str(art.get("content", "")),
                status_code=422,
                media_type="text/plain; charset=utf-8",
            )
        content = art.get("content", "")
        content = _enforce_trailing_lf_per_file(content)
        content = content.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"
        return PlainTextResponse(content, media_type="text/plain; charset=utf-8")

    if fmt == "patch":
        art = generate_artifact_from_task(
            task,
            repo=repo,
            branch=branch,
            path_prefix=prefix,
            session=session,
            mode="patch",
        )
        patch = art.get("patch") or art.get("content") or ""
        from backend.utils.patch_sanitizer import (
            sanitize_patch,
            validate_patch_structure,
        )

        patch, warnings = sanitize_patch(patch)
        ok, msg = validate_patch_structure(patch, path_prefix=prefix)
        if not ok:
            fixed = _try_fix_patch_with_ps1(patch)
            if fixed is not None:
                patch = fixed
                ok, msg = validate_patch_structure(patch, path_prefix=prefix)
        if not ok:
            raise HTTPException(status_code=422, detail=f"Invalid patch: {msg}")
        return PlainTextResponse(patch, media_type="text/x-patch; charset=utf-8")

    out = propose_changes(
        task,
        repo=repo,
        branch=branch,
        commit="HEAD",
        k=k,
        path_prefix=prefix,
        session=session,
        thread_n=thread_n,
    )
    return out


@app.post("/app/api/repo/files")
def repo_files(payload: Dict[str, Any]):
    task = (payload or {}).get("task")
    if not task:
        raise HTTPException(status_code=400, detail="Missing 'task'")
    repo = payload.get("repo", "personal-agent-ste")
    branch = payload.get("branch", "main")
    prefix = payload.get("path_prefix", "backend/")
    session = payload.get("session")

    art = generate_artifact_from_task(
        task,
        repo=repo,
        branch=branch,
        path_prefix=prefix,
        session=session,
        mode="files",
    )
    content = art.get("content", "")
    content = _enforce_trailing_lf_per_file(content)
    content = content.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"
    return PlainTextResponse(content, media_type="text/plain; charset=utf-8")


# =========================
# NEW: Global error shaping
# =========================
def _structured_error_payload(
    message: str,
    *,
    code: str = "RequestError",
    correlation_id: str | None = None,
    idempotency_key: str | None = None,
) -> dict:
    return {
        "ok": False,
        "result": None,
        "error": {
            "version": 1,
            "code": code,
            "message": message,
            "hint": None,
            "details": None,
        },
        "latency_ms": 0,
        "correlation_id": correlation_id or "",
        "idempotency_key": idempotency_key or "",
    }


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse(
        _structured_error_payload(
            "invalid_request: " + str(exc.errors()), code="ValidationError"
        ),
        status_code=200,
    )


@app.exception_handler(StarletteHTTPException)
async def on_http_exception(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        _structured_error_payload(
            f"http_error_{exc.status_code}: {exc.detail}",
            code=f"HTTP_{exc.status_code}",
        ),
        status_code=200,
    )
