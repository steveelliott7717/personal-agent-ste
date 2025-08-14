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

# -------------------- Sanitizer / gate for patch responses --------------------
# -------------------- Sanitizer / gate for patch responses --------------------
import re
import json  # ensure json is imported for headers later

# Strip wrappers/noise first
_CODEBLOCK_RE     = re.compile(r"```(?:diff|patch)?\s*(?P<body>.*?)```", re.IGNORECASE | re.DOTALL)
_CBLOCK_RE        = re.compile(r"/\*.*?\*/", re.DOTALL)
_ELLIPSIS_LINE_RE = re.compile(r"^\s*\.\.\.\s*$", re.MULTILINE)

# Hunk headers:
#   4-number: @@ -a,b +c,d @@
#   3-number: @@ -a,b +d @@   (new len only)  OR  @@ -a +c,d @@ (old len omitted)
#   2-number: @@ -0,0 +20 @@  (start omitted on + side)
_HUNK_HDR_RE_ANY = re.compile(r"^@@\s+-([0-9]+)(?:,([0-9]+))?\s+\+([0-9]+)?(?:,([0-9]+))?\s+@@")

# diff --git line
_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$")

def _unwrap_and_clean(text: str) -> str:
    if not text:
        return ""
    m = _CODEBLOCK_RE.search(text)
    s = (m.group("body") if m else text)
    s = s.replace("\r\n", "\n").replace("\r", "\n")  # normalize to LF
    s = _CBLOCK_RE.sub("", s)                        # strip C-style block comments
    s = _ELLIPSIS_LINE_RE.sub("", s)                 # strip standalone ellipsis lines
    return s.strip()

def _ensure_file_headers(text: str) -> str:
    """
    Guarantee every '@@ ... @@' hunk sits under a valid file header within its section.
    If a hunk appears before '---'/'+++':
      - Insert '--- /dev/null' if the section has 'new file mode'
      - Else insert '--- a/<path>' (path from 'diff --git')
      - Always insert '+++ b/<path>'
    """
    lines = text.split("\n")
    out: list[str] = []

    in_section = False
    have_old = False
    have_new = False
    new_file_mode = False
    current_path_b: str | None = None  # from 'diff --git a/<a> b/<b>'

    def start_section():
        nonlocal have_old, have_new, new_file_mode
        have_old = False
        have_new = False
        new_file_mode = False

    for i, l in enumerate(lines):
        if l.startswith("diff --git "):
            # New section
            in_section = True
            start_section()
            m = _DIFF_GIT_RE.match(l)
            current_path_b = m.group(2) if m else None
            out.append(l)
            continue

        if not in_section:
            out.append(l)
            continue

        if l.startswith("new file mode "):
            new_file_mode = True
            out.append(l)
            continue

        if l.startswith("--- "):
            have_old = True
            out.append(l)
            continue

        if l.startswith("+++ "):
            have_new = True
            out.append(l)
            continue

        if l.startswith("@@ "):
            # If hunks appear without headers in this section, synthesize them
            if not have_old or not have_new:
                # We need a path to build headers
                path_b = current_path_b or ""
                if new_file_mode:
                    out.append("--- /dev/null")
                else:
                    out.append(f"--- a/{path_b}")
                out.append(f"+++ b/{path_b}")
                have_old = True
                have_new = True
            out.append(l)
            continue

        # A new file header line also resets have_old/have_new if we see another 'diff --git'
        if l.startswith("index ") or l.startswith("rename from ") or l.startswith("rename to "):
            # index/rename lines appear between diff and headers; leave flags as-is
            out.append(l)
            continue

        # Any other line is part of hunks or context; just pass through
        out.append(l)

    return "\n".join(out)

def _auto_fix_newfile_hunks(text: str) -> tuple[str, dict]:
    """
    Brute-force, hunk-aware normalizer:
      - Marks a hunk as 'new file' if the section has '--- /dev/null' OR minus-len == 0 (or omitted while minus-start == 0).
      - Inside such hunks, prefixes '+' on any non-control line (including empty lines).
      - Rewrites header to '@@ -0,0 +<start>,<len> @@' where <len> is the actual '+' count.
      - Returns (fixed_text, debug_stats).
    """
    if not text:
        return "", {"sections": 0, "newfile_hunks": 0, "prefixed_lines": 0, "rewritten_headers": 0}

    lines = text.split("\n")
    out: list[str] = []

    sections = 0
    newfile_hunks = 0
    prefixed = 0
    rewritten = 0

    in_section = False
    section_saw_devnull = False
    in_hunk = False
    hunk_is_newfile = False
    hdr_idx: int | None = None
    plus_start = 1
    plus_count = 0

    def flush_hunk():
        nonlocal hdr_idx, plus_start, plus_count, hunk_is_newfile, rewritten, newfile_hunks
        if hdr_idx is not None and hunk_is_newfile:
            ps = plus_start if (plus_start and plus_start > 0) else 1
            out[hdr_idx] = f"@@ -0,0 +{ps},{max(0, plus_count)} @@"
            rewritten += 1
            newfile_hunks += 1
        hdr_idx = None
        plus_start = 1
        plus_count = 0
        hunk_is_newfile = False

    def is_file_header(l: str) -> bool:
        return (
            l.startswith("diff --git ") or l.startswith("index ") or
            l.startswith("--- ") or l.startswith("+++ ") or
            l.startswith("new file mode ") or l.startswith("deleted file mode ") or
            l.startswith("rename from ") or l.startswith("rename to ")
        )

    for l in lines:
        if l.startswith("diff --git "):
            flush_hunk()
            in_section = True
            section_saw_devnull = False
            in_hunk = False
            sections += 1
            out.append(l)
            continue

        if not in_section:
            out.append(l)
            continue

        if l.startswith("--- "):
            if l.strip() == "--- /dev/null":
                section_saw_devnull = True

        if l.startswith("@@ "):
            flush_hunk()
            in_hunk = True
            hdr_idx = len(out)
            out.append(l)  # placeholder; may be rewritten
            # Decide if this hunk is a 'new file' hunk
            ms = ml = ps = pl = None
            m = _HUNK_HDR_RE_ANY.match(l)
            if m:
                ms = int(m.group(1))
                ml = int(m.group(2)) if m.group(2) is not None else None
                ps = int(m.group(3)) if m.group(3) is not None else None
                pl = int(m.group(4)) if m.group(4) is not None else None
            minus_len_zero = (ml == 0) or (ml is None and ms == 0)
            hunk_is_newfile = bool(section_saw_devnull or minus_len_zero)
            plus_start = ps if (ps and ps > 0) else 1
            plus_count = 0
            continue

        if is_file_header(l):
            flush_hunk()
            in_hunk = False
            out.append(l)
            continue

        if in_hunk and hunk_is_newfile:
            # Prefix '+' if missing and not a control line
            if not (l.startswith("+") or l.startswith("-") or l.startswith(" ") or l.startswith("\\")):
                l = "+" + l
                prefixed += 1
            if l.startswith("+"):
                plus_count += 1
            out.append(l)
            continue

        out.append(l)

    flush_hunk()

    return "\n".join(out), {
        "sections": sections,
        "newfile_hunks": newfile_hunks,
        "prefixed_lines": prefixed,
        "rewritten_headers": rewritten,
    }

def _sanitize_patch_text(text: str) -> tuple[str, dict]:
    """
    1) Unwrap + normalize, strip forbidden patterns
    2) Ensure each hunk has proper file headers (---/+++), synthesizing as needed
    3) Auto-fix all new-file hunks (+ prefixes and header lengths)
    """
    s = _unwrap_and_clean(text)
    s = _ensure_file_headers(s)
    fixed, stats = _auto_fix_newfile_hunks(s)
    return fixed, stats

def _looks_like_unified_diff(text: str) -> bool:
    if not text:
        return False
    if text.startswith("diff --git"):
        return True
    # fallback: single-file unified diff with ---/+++ present
    return text.startswith("--- ") and ("\n+++ " in text)


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
    sanitized, stats = _sanitize_patch_text(raw_patch)
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
    headers["X-RMS-Sanitized"] = "true" if is_sanitized else "false"
    headers["X-RMS-Sanitizer-Stats"] = json.dumps(stats, separators=(",", ":"))
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
