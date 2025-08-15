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

from dotenv import load_dotenv
load_dotenv()

# Package-qualified imports
from backend.agents.repo_agent import propose_changes, generate_artifact_from_task

app = FastAPI(title="Personal Agent API")


# -------------------- Health --------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/app/")


# -------------------- Repo endpoints --------------------
@app.post("/app/api/repo/plan")
def repo_plan(payload: Dict[str, Any], request: Request):
    """
    Accepts JSON payload with:
      - task (str)
      - repo, branch (optional with defaults)
      - path_prefix (e.g., "backend/")
      - k (int), session, thread_n (optional)

    Supports artifacts via query string:
      - ?format=files  -> returns BEGIN_FILE blocks (text/plain)
      - ?format=patch  -> returns unified diff (text/x-patch)
      - default -> JSON plan preview
    """
    task = (payload or {}).get("task")
    if not isinstance(task, str) or not task.strip():
        raise HTTPException(status_code=400, detail="Missing or invalid 'task'")

    repo = (payload or {}).get("repo") or "personal-agent-ste"
    branch = (payload or {}).get("branch") or "main"
    prefix = (payload or {}).get("path_prefix")
    k = int((payload or {}).get("k") or 12)
    session = (payload or {}).get("session")
    thread_n = (payload or {}).get("thread_n")

    fmt = request.query_params.get("format", "").lower().strip()
    if fmt in ("files", "patch"):
        try:
            art = generate_artifact_from_task(
                task,
                repo=repo,
                branch=branch,
                path_prefix=prefix,
                session=session,
                mode=fmt,
            )
        except Exception as e:
            # Return raw error text for easier debugging
            return PlainTextResponse(
                f"Artifact generation error: {type(e).__name__}: {e}",
                status_code=400,
                media_type="text/plain",
            )

        if art["mode"] == "files":
            # Render as plain blocks the client can write out directly
            blocks: List[str] = []
            for rel, content in art["files"]:
                # Ensure one trailing LF in content
                if not content.endswith("\n"):
                    content = content + "\n"
                blocks.append(f"BEGIN_FILE {rel}\n{content}END_FILE\n")
            body = "".join(blocks)
            return PlainTextResponse(body, media_type="text/plain")

        # mode == "patch"
        return PlainTextResponse(art["patch"], media_type="text/x-patch")

    # Fallback: JSON plan mode
    return propose_changes(
        task,
        repo=repo,
        branch=branch,
        commit="HEAD",
        k=k,
        path_prefix=prefix,
        session=session,
        thread_n=thread_n,
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
