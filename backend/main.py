# backend/main.py
from __future__ import annotations

import os
import json
from typing import Any, Dict, Tuple, Optional, List
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse, FileResponse, PlainTextResponse
from backend.utils.patch_sanitizer import sanitize_patch, validate_patch_structure
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

setup_logging()


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

from typing import Optional
from pathlib import Path

def _try_fix_patch_with_ps1(patch_text: str) -> Optional[str]:
    """
    Optionally invoke tools/fix-patch.ps1 if available in runtime.
    Returns fixed text on success, or None if not executed or failed.
    """
    script_rel = Path(__file__).resolve().parents[1] / "tools" / "fix-patch.ps1"
    if not script_rel.exists():
        return None
    # Prefer pwsh if present
    for exe in ("pwsh", "powershell"):
        try:
            import subprocess, tempfile
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f_in:
                f_in.write(patch_text)
                in_path = f_in.name
            out_path = in_path + ".out.patch"
            cmd = [
                exe, "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-File", str(script_rel),
                "-InPath", in_path,
                "-OutPath", out_path
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

# add somewhere near your other health routes in backend/main.py

@app.get("/app/api/repo/health")
def repo_health():
    return {"ok": True, "model": os.getenv("CHAT_MODEL", "gpt-5")}



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
    """
    RMS plan endpoint with strict output enforcement.
    - ?format=files -> returns BEGIN_FILE/END_FILE blocks as text/plain
    - ?format=patch -> returns unified diff as text/x-patch after sanitation+validation
    - default -> JSON preview for UI
    """
    task = (payload or {}).get("task")
    if not task:
        raise HTTPException(status_code=400, detail="Missing 'task'")
    repo    = payload.get("repo", "personal-agent-ste")
    branch  = payload.get("branch", "main")
    prefix  = payload.get("path_prefix", "backend/")
    k       = int(payload.get("k", 12))
    session = payload.get("session")
    thread_n = payload.get("thread_n")
    fmt = request.query_params.get("format") or request.headers.get("X-RMS-Format")

    if fmt == "files":
        art = generate_artifact_from_task(
            task, repo=repo, branch=branch, path_prefix=prefix, session=session, mode="files"
        )
        if not art.get("ok"):
            # Return raw content for inspection but mark 422
            return PlainTextResponse(str(art.get("content","")), status_code=422, media_type="text/plain; charset=utf-8")
        content = art.get("content","")
        # Lightweight validation: ASCII/LF + BEGIN/END markers + trailing LF per file
        try:
            _ = _parse_files_artifact(content)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Invalid files artifact: {e}")
        return PlainTextResponse(content, media_type="text/plain; charset=utf-8")

    if fmt == "patch":
        art = generate_artifact_from_task(
            task, repo=repo, branch=branch, path_prefix=prefix, session=session, mode="patch"
        )
        patch = art.get("patch") or art.get("content") or ""
        # Sanitize and validate
        patch, warnings = sanitize_patch(patch)
        ok, msg = validate_patch_structure(patch, path_prefix=prefix)
        if not ok:
            # Optional: attempt external fixer if enabled
            fixed = _try_fix_patch_with_ps1(patch)
            if fixed is not None:
                patch = fixed
                ok, msg = validate_patch_structure(patch, path_prefix=prefix)
        if not ok:
            raise HTTPException(status_code=422, detail=f"Invalid patch: {msg}")
        return PlainTextResponse(patch, media_type="text/x-patch; charset=utf-8")

    # Default JSON preview (no strict guarantees)
    out = propose_changes(
        task, repo=repo, branch=branch, commit="HEAD", k=k, path_prefix=prefix, session=session, thread_n=thread_n
    )
    return out@app.post("/app/api/repo/files")
def repo_files(payload: Dict[str, Any]):
    """
    Strict files-mode endpoint: returns only BEGIN_FILE/END_FILE blocks (ASCII+LF).
    """
    task = (payload or {}).get("task")
    if not task:
        raise HTTPException(status_code=400, detail="Missing 'task'")
    repo    = payload.get("repo", "personal-agent-ste")
    branch  = payload.get("branch", "main")
    prefix  = payload.get("path_prefix", "backend/")
    session = payload.get("session")

    art = generate_artifact_from_task(
        task, repo=repo, branch=branch, path_prefix=prefix, session=session, mode="files"
    )
    content = art.get("content","")
    # Validate files artifact
    try:
        _ = _parse_files_artifact(content)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid files artifact: {e}")

    return PlainTextResponse(content, media_type="text/plain; charset=utf-8")