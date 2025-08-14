# backend/main.py
from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, Tuple, Union

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv

# Linter (kept) — optional guardrails in JSON & warning headers
from backend.utils.patch_linter import lint_patch, make_followup_prompt

# ✅ package-qualified imports (works when running: uvicorn backend.main:app)
from backend.agents.router_agent import route_request
from backend.agents.repo_agent import propose_changes, answer_about_repo, generate_patch_from_prompt

from backend.utils.nl_formatter import ensure_natural
from backend.utils.agent_protocol import AgentResponse
from backend.services import conversation as conv

load_dotenv()

app = FastAPI(title="Personal Agent API")

@app.get("/app/api/debug/supabase-key")
async def debug_supabase_key():
    key = os.getenv("SUPABASE_SERVICE_ROLE")
    if key:
        return {"SUPABASE_SERVICE_ROLE_start": key[:8] + "..."}
    return {"SUPABASE_SERVICE_ROLE": None}

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

# -------------------- Minimal sanitizer/gate for patch responses --------------------
# --- begin sanitizer/gate helpers (replace your existing block) ---
# --- begin sanitizer/gate helpers (REPLACE your existing block) ---
import re

_CODEBLOCK_RE = re.compile(r"```(?:diff|patch)?\s*(?P<body>.*?)```", re.IGNORECASE | re.DOTALL)
_CBLOCK_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_ELLIPSIS_LINE_RE = re.compile(r"^\s*\.\.\.\s*$", re.MULTILINE)
_HUNK_HDR_RE = re.compile(r"^@@ -(\d+),(\d+) \+(\d+),(\d+) @@")

def _auto_fix_newfile_hunks(text: str) -> str:
    """
    Make malformed *new-file* hunks git-apply-ready:
      - A section is treated as 'new file' if either:
          a) the section headers include '--- /dev/null', OR
          b) a hunk header has minus-len == 0 (e.g., @@ -0,0 +1,N @@)
      - For such hunks, ensure every content line begins with '+' (unless it's already '+', '-', ' ', or '\\').
      - After fixing, rewrite the hunk header to '@@ -0,0 +<start>,<len> @@' where <len> is the count of '+' lines.
    """
    if not text:
        return text

    lines = text.split("\n")
    out: list[str] = []

    in_section = False
    saw_devnull_header = False  # saw '--- /dev/null' in this section
    in_hunk = False
    hunk_is_newfile = False

    # Track current hunk to fix header after we count + lines
    pending_hunk_idx: int | None = None
    plus_start: int = 1  # keep existing start if present, default to 1
    plus_len: int = 0

    def is_file_header(l: str) -> bool:
        return (
            l.startswith("diff --git ")
            or l.startswith("index ")
            or l.startswith("--- ")
            or l.startswith("+++ ")
            or l.startswith("new file mode ")
            or l.startswith("deleted file mode ")
            or l.startswith("rename from ")
            or l.startswith("rename to ")
        )

    def flush_hunk_if_needed():
        nonlocal pending_hunk_idx, plus_start, plus_len, hunk_is_newfile
        if pending_hunk_idx is None:
            return
        if hunk_is_newfile:
            # Rewrite header to reflect actual +len
            hdr = out[pending_hunk_idx]
            m = _HUNK_HDR_RE.match(hdr)
            if m:
                # old_minus_start, old_minus_len = int(m.group(1)), int(m.group(2))
                old_plus_start = int(m.group(3))
                # keep existing +start if >0, default to 1
                new_plus_start = old_plus_start if old_plus_start > 0 else max(1, plus_start)
                new_plus_len = max(0, plus_len)
                out[pending_hunk_idx] = f"@@ -0,0 +{new_plus_start},{new_plus_len} @@"
        # reset hunk state
        pending_hunk_idx = None
        plus_start = 1
        plus_len = 0
        hunk_is_newfile = False

    for l in lines:
        # Start of a new diff section
        if l.startswith("diff --git "):
            flush_hunk_if_needed()
            in_section = True
            saw_devnull_header = False
            in_hunk = False
            out.append(l)
            continue

        if not in_section:
            out.append(l)
            continue

        # Track devnull header for new-file detection
        if l.startswith("--- "):
            if l.strip() == "--- /dev/null":
                saw_devnull_header = True

        # Hunk header
        if l.startswith("@@ "):
            flush_hunk_if_needed()
            in_hunk = True
            pending_hunk_idx = len(out)
            out.append(l)  # placeholder; may be rewritten
            # Determine if this hunk is effectively "new-file"
            m = _HUNK_HDR_RE.match(l)
            minus_len_is_zero = False
            if m:
                minus_len_is_zero = (int(m.group(2)) == 0)
                # carry over the +start; we may keep it if >0
                try:
                    plus_start = int(m.group(3)) if int(m.group(3)) > 0 else 1
                except Exception:
                    plus_start = 1
            hunk_is_newfile = bool(saw_devnull_header or minus_len_is_zero)
            plus_len = 0
            continue

        # Any file header line ends hunk
        if is_file_header(l):
            flush_hunk_if_needed()
            in_hunk = False
            out.append(l)
            continue

        # Inside a hunk: for new-file hunks, prefix + where missing
        if in_hunk and hunk_is_newfile:
            if not (l.startswith("+") or l.startswith("-") or l.startswith(" ") or l.startswith("\\")):
                l = "+" + l
            if l.startswith("+"):
                plus_len += 1
            out.append(l)
            continue

        # Other lines (outside hunk or not new-file)
        out.append(l)

    # End of file: close any open hunk
    flush_hunk_if_needed()

    return "\n".join(out)

def _sanitize_patch_text(text: str) -> str:
    if not text:
        return ""
    # 1) Strip code fences (keep inner), normalize to LF
    m = _CODEBLOCK_RE.search(text)
    s = (m.group("body") if m else text)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # 2) Remove C-style comments and standalone ellipsis lines
    s = _CBLOCK_RE.sub("", s)
    s = _ELLIPSIS_LINE_RE.sub("", s)
    s = s.strip()
    # 3) Auto-fix new-file hunks (prefix '+' + fix headers)
    s = _auto_fix_newfile_hunks(s)
    return s

def _looks_like_unified_diff(text: str) -> bool:
    if not text:
        return False
    if text.startswith("diff --git"):
        return True
    return text.startswith("--- ") and ("\n+++ " in text)
# --- end sanitizer/gate helpers ---



# -------------------- Repo endpoints (bypass router) --------------------
@app.post("/app/api/repo/query")
def repo_query(payload: Dict[str, Any]):
    q = (payload or {}).get("task") or (payload or {}).get("question") or (payload or {}).get("q")
    if not q:
        raise HTTPException(status_code=400, detail="Missing 'task'/'question'/'q'")
    repo   = payload.get("repo", "personal-agent-ste")
    branch = payload.get("branch", "main")
    prefix = payload.get("path_prefix")
    k      = int(payload.get("k", 8))
    commit = payload.get("commit")
    session = payload.get("session")
    thread_n = payload.get("thread_n")

    try:
        out = answer_about_repo(
            q,
            repo=repo,
            branch=branch,
            k=k,
            path_prefix=prefix,
            commit=commit,
            session=session,
            thread_n=thread_n,
        )
        if isinstance(out, str):
            return PlainTextResponse(out)
        if isinstance(out, dict) and out.get("agent") == "repo":
            text = out.get("answer") or out.get("message") or out.get("data") or json.dumps(out, ensure_ascii=False)
            return PlainTextResponse(str(text))
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/app/api/repo/plan", response_model=None)
async def repo_plan(request: Request) -> Response:
    try:
        payload: Dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    task = payload.get("task")
    if not task:
        raise HTTPException(status_code=400, detail="Missing 'task'")

    repo   = payload.get("repo", "personal-agent-ste")
    branch = payload.get("branch", "main")
    prefix = payload.get("path_prefix")  # e.g., "backend/"
    k      = int(payload.get("k", 12))
    session = payload.get("session")
    thread_n = payload.get("thread_n")

    # Optional caller-provided checks (regex lists)
    checks = payload.get("checks") or {}
    required = checks.get("required") or []
    forbidden = checks.get("forbidden") or []
    max_files = checks.get("max_files")
    max_bytes = checks.get("max_bytes")

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

    fmt = (request.query_params.get("format") or "").lower()
    accept = (request.headers.get("accept") or "").lower()
    wants_patch = (fmt == "patch") or ("text/x-patch" in accept) or ("text/x-diff" in accept)

    # Build/obtain patch text (for both JSON & patch modes)
    raw_patch = (out.get("patch") if isinstance(out, dict) else None) or ""
    if not raw_patch:
        # generate from prompt when available
        raw_patch = generate_patch_from_prompt(out.get("prompt", "")) or ""

    # Sanitize/gate before linting/return
    sanitized = _sanitize_patch_text(raw_patch)
    is_sanitized = (sanitized != (raw_patch or "").strip())
    if wants_patch:
        # For patch responses, enforce structure up front
        if not _looks_like_unified_diff(sanitized):
            return PlainTextResponse(
                "No patch generated or invalid diff. Re-run in JSON mode and fix the plan.",
                status_code=400,
                media_type="text/plain",
            )

    # Lint the (sanitized) patch
    lint = lint_patch(
        patch=sanitized,
        path_prefix=prefix,
        required=required,
        forbidden=forbidden,
        max_files=max_files,
        max_bytes=max_bytes,
    )
    followup = None if lint["ok"] else make_followup_prompt(task=task, issues=lint["issues"], path_prefix=prefix)

    if not wants_patch:
        # JSON mode: include linter results + ready-to-send followup prompt
        enriched = dict(out)
        enriched["lint"] = lint
        if followup:
            enriched["followup_prompt"] = followup
        return JSONResponse(enriched)

    # Patch mode: return plain text with headers
    if not sanitized:
        return PlainTextResponse(
            "No patch generated for this task. Use JSON mode to inspect the plan/prompt.",
            status_code=400,
            media_type="text/plain",
        )

    headers: Dict[str, str] = {}
    if not lint["ok"]:
        summary = str(lint["summary"])[:180]
        headers["X-RMS-Warnings"] = f"{summary} (+{max(0, len(lint['issues'])-1)} more)"
    if is_sanitized:
        headers["X-RMS-Sanitized"] = "true"
    headers["Content-Disposition"] = 'attachment; filename="repo_plan.patch"'

    return PlainTextResponse(content=sanitized, media_type="text/x-patch", headers=headers)

@app.post("/app/api/repo/memory/reset")
def repo_memory_reset(payload: Dict[str, Any]):
    session = (payload or {}).get("session")
    if not session:
        raise HTTPException(status_code=400, detail="Missing 'session'")
    deleted = conv.clear_session(session)
    return {"ok": True, "session": session, "deleted": deleted}

@app.post("/app/api/repo/memory/config")
def repo_memory_config(payload: Dict[str, Any]):
    session = (payload or {}).get("session")
    n = (payload or {}).get("n")
    if not session or n is None:
        raise HTTPException(status_code=400, detail="Missing 'session' or 'n'")
    conv.set_session_n(session, int(n))
    return {"ok": True, "session": session, "n": conv.get_session_n(session) or conv.default_n()}

@app.get("/app/api/repo/memory/export")
def repo_memory_export(session: str, limit: int = 50):
    if not session:
        raise HTTPException(status_code=400, detail="Missing 'session'")
    return {"ok": True, "session": session, "limit": int(limit), "messages": conv.export_messages(session, limit)}

# -------------------- Middleware --------------------
class NaturalLanguageMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)

            # Only transform JSONResponse payloads
            if not isinstance(response, JSONResponse):
                return response

            # Clone response body safely
            body_bytes = b""
            try:
                async for chunk in response.body_iterator:  # type: ignore[attr-defined]
                    body_bytes += chunk
            except Exception:
                return response  # if we can't read it, return as-is

            response.body_iterator = None  # prevent double iteration

            # If not JSON, passthrough
            try:
                payload = json.loads(body_bytes.decode("utf-8"))
            except Exception:
                return JSONResponse(
                    content=body_bytes.decode("utf-8") if body_bytes else None,
                    status_code=response.status_code,
                )

            # Naturalize
            try:
                formatted = ensure_natural(payload)
            except Exception as e:
                formatted = {
                    "agent": (payload.get("agent") if isinstance(payload, dict) else "system"),
                    "intent": "error",
                    "message": f"Formatting error: {e}",
                    "raw": payload,
                }

            return JSONResponse(content=formatted, status_code=response.status_code)

        except Exception as e:
            return JSONResponse(
                content={"agent": "system", "intent": "error", "message": f"Middleware error: {e}"},
                status_code=500,
            )

app.add_middleware(NaturalLanguageMiddleware)

# -------------------- Helpers --------------------
def _extract_query(query: str | None, body: Dict[str, Any] | None) -> Tuple[str | None, Dict[str, Any] | None]:
    """
    Accepts either:
      - form 'query'
      - JSON { query | prompt | q }
    """
    if query:
        return query, body
    if body and isinstance(body, dict):
        q = body.get("query") or body.get("prompt") or body.get("q")
        return q, body
    return None, body

def _normalize(agent: str, raw_result: Any) -> AgentResponse:
    if isinstance(raw_result, str):
        return {"agent": agent, "intent": "say", "message": raw_result}
    if isinstance(raw_result, dict):
        if "agent" not in raw_result:
            raw_result = {"agent": agent, **raw_result}
        raw_result.setdefault("intent", "unknown")
        return raw_result  # type: ignore[return-value]
    if isinstance(raw_result, list):
        return {"agent": agent, "intent": "list", "data": raw_result}
    return {"agent": agent, "intent": "unknown", "message": str(raw_result)}

# -------------------- Universal request endpoint (router path) --------------------
@app.post("/api/request")
@app.post("/app/api/request")
@app.post("/api/route")       # alias to support earlier clients/tests
@app.post("/app/api/route")   # alias to support earlier clients/tests
async def handle_request(request: Request):
    # Try JSON first (don’t declare Body param so form requests don’t get validated as JSON)
    body: Dict[str, Any] | None = None
    try:
        if request.headers.get("content-type", "").lower().startswith("application/json"):
            body = await request.json()
            if not isinstance(body, dict):
                body = None
    except Exception:
        body = None

    # Fall back to form fields
    form_query: str | None = None
    if body is None:
        try:
            form = await request.form()
            if form:
                form_query = form.get("query") or form.get("prompt") or form.get("q")  # type: ignore[assignment]
        except Exception:
            form_query = None

    q, _ = _extract_query(form_query, body)
    if not q:
        return JSONResponse(
            {"agent": "system", "intent": "error", "message": "Missing 'query' in form or JSON body"},
            status_code=400,
        )

    agent, raw_result = route_request(q)
    resp = _normalize(agent, raw_result)

    # If the routed agent is the repo agent, return text/plain so the client gets raw text.
    if isinstance(resp, dict) and resp.get("agent") == "repo":
        text = resp.get("answer") or resp.get("message") or resp.get("data") or json.dumps(resp, ensure_ascii=False)
        return PlainTextResponse(str(text))

    # Pre-format so clients without middleware still get a nice shape
    try:
        natural = ensure_natural(resp)
    except Exception as e:
        natural = {
            "agent": resp.get("agent", "system"),
            "intent": "error",
            "message": f"Formatting error: {e}",
            "raw": resp,
        }

    return JSONResponse(natural)

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
