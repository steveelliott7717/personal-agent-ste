# backend/main.py
from __future__ import annotations

import os
import json
from typing import Any, Dict, Tuple, Optional, List
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
# add near your other imports
from fastapi.responses import PlainTextResponse
from backend.agents.repo_agent import generate_artifact_from_task
import re


from dotenv import load_dotenv
load_dotenv()

# Package-qualified imports
from backend.agents.repo_agent import propose_changes, generate_artifact_from_task
# Add these imports after the existing dotenv and package imports
from backend.logging_utils import setup_logging, RequestLoggingMiddleware

app = FastAPI(title="Personal Agent API")


app.add_middleware(RequestLoggingMiddleware)

_BEGIN_RE = re.compile(r"^BEGIN_FILE\s+(.+)$")
_END_RE = re.compile(r"^END_FILE\s*$")

def _is_ascii_lf(s: str) -> bool:
    try:
        s.encode("ascii")
    except UnicodeEncodeError:
        return False
    return "\r" not in s

def _parse_files_artifact(text: str) -> list[tuple[str, str]]:
    """
    Parse BEGIN_FILE/END_FILE blocks.
    Returns list of (path, content).
    Raises ValueError with a short message when invalid.
    """
    lines = text.split("\n")
    i, n = 0, len(lines)
    blocks: list[tuple[str, str]] = []

    # Skip leading empties
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
        # Ensure file ends with exactly one LF
        content = "\n".join(buf)
        if not content.endswith("\n"):
            content = content + "\n"
        if not _is_ascii_lf(content):
            raise ValueError(f"non-ASCII or CRLF detected in {path}")
        blocks.append((path, content))

        # Skip intervening empties
        while i < n and lines[i].strip() == "":
            i += 1
        if i < n and not lines[i].startswith("BEGIN_FILE "):
            raise ValueError(f"unexpected text after END_FILE at line {i+1}")

    return blocks


# -------------------- Health --------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/app/")

# ---------- Files-mode validation helpers ----------
from fastapi.responses import PlainTextResponse  # add if not already imported

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
            rel = line[len("BEGIN_FILE "):].strip()
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
    # 1) only allowed paths
    bad = [p for p in files.keys() if p not in _ALLOWED_FILES]
    if bad:
        return False, f"Contains paths outside allowed set: {bad}"

    # 2) ascii + lf + trailing lf
    for p, body in files.items():
        if not _ascii_lf_only(body):
            return False, f"{p} not ASCII/LF or missing trailing LF."

    # 3) keep anchors in main.py
    if "backend/main.py" in files:
        new_main = files["backend/main.py"]
        for s in _MAIN_SENTINELS:
            if s not in new_main:
                return False, f"backend/main.py missing sentinel: {s}"

    return True, "ok"



# -------------------- Repo endpoints --------------------
from fastapi import Request  # ensure this import exists

@app.post("/app/api/repo/plan")
def repo_plan(payload: Dict[str, Any], request: Request):
    task = payload.get("task")
    if not task:
        raise HTTPException(status_code=400, detail="Missing 'task'")

    repo    = payload.get("repo", "personal-agent-ste")
    branch  = payload.get("branch", "main")
    prefix  = payload.get("path_prefix")
    k       = int(payload.get("k", 12))
    session = payload.get("session")
    thread_n = payload.get("thread_n")

    # NEW: honor ?format=files|patch
    fmt = request.query_params.get("format") or request.headers.get("X-RMS-Format")
    if fmt in ("files", "patch"):
        try:
            art = generate_artifact_from_task(
                task if isinstance(task, str) else json.dumps(task),
                repo=repo,
                branch=branch,
                path_prefix=prefix,
                session=session,
                mode=fmt,
            )
            if fmt == "files":
                # Stitch BEGIN_FILE blocks for the client to materialize
                blocks = []
                for path, content in art["files"]:
                    # ASCII/LF already normalized inside repo_agent
                    blocks.append(f"BEGIN_FILE {path}\n{content}END_FILE\n")
                return PlainTextResponse("".join(blocks), media_type="text/plain; charset=utf-8")
            else:
                return PlainTextResponse(art["patch"], media_type="text/x-patch; charset=utf-8")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Fallback: original JSON “plan” mode (unchanged)
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
from fastapi.responses import PlainTextResponse
import logging
logger = logging.getLogger("app.files")

@app.post("/app/api/repo/files")
def repo_files(payload: Dict[str, Any]):
    """
    Returns BEGIN_FILE/END_FILE blocks for 'files' mode.
    Expected JSON:
      {
        "task": "<instructions>",
        "repo": "personal-agent-ste",
        "branch": "main",
        "path_prefix": "backend/",
        "session": "optional"
      }
    """
    try:
        task_text = (payload or {}).get("task")
        if not task_text:
            raise HTTPException(status_code=400, detail="Missing 'task'")
        repo = payload.get("repo", "personal-agent-ste")
        branch = payload.get("branch", "main")
        prefix = payload.get("path_prefix", "backend/")
        session = payload.get("session")

        out = generate_artifact_from_task(
            task_text,
            repo=repo,
            branch=branch,
            path_prefix=prefix,
            session=session,
            mode="files",           # IMPORTANT: force files mode
        )

        if not isinstance(out, dict):
            raise HTTPException(status_code=500, detail="Upstream returned non-dict")

        if not out.get("ok"):
            warn = out.get("warning") or "files-mode validation failed"
            raise HTTPException(status_code=400, detail=f"{warn}")

        content = out.get("content") or ""
        if not content.strip():
            raise HTTPException(status_code=400, detail="Empty files content")

        # Return raw text blocks
        return PlainTextResponse(content, media_type="text/plain; charset=utf-8")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("files-mode generation failed")
        return JSONResponse(
            {
                "agent": "system",
                "intent": "error",
                "message": f"Upstream generation error: {type(e).__name__}: {e}",
            },
            status_code=500,
        )



# -------------------- Static frontend (SPA at /app) --------------------
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    INDEX_FILE = Path(static_dir) / "index.html"

    app.mount("/app", StaticFiles(directory=static_dir, html=True), name="static")

    @app.get("/app", include_in_schema=False, response_class=HTMLResponse)
    @app.get("/app/", include_in_schema=False, response_class=HTMLResponse)
    def serve_index():
        if INDEX_FILE.exists():
            return HTMLResponse(INDEX_FILE.read_text(encoding="utf-8"))
        raise HTTPException(status_code=404, detail="Frontend build not found.")

    @app.get("/app/{path:path}", include_in_schema=False, response_class=HTMLResponse)
    def spa_fallback(path: str):
        candidate = Path(static_dir) / path
        if candidate.is_file():
            return FileResponse(str(candidate))
        if INDEX_FILE.exists():
            return HTMLResponse(INDEX_FILE.read_text(encoding="utf-8"))
        raise HTTPException(status_code=404, detail="Frontend build not found.")
