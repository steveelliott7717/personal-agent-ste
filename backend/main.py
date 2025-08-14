# backend/main.py
from __future__ import annotations

import os
import json
from typing import Any, Dict, Tuple, Union

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from starlette.middleware.base import BaseHTTPMiddleware

from dotenv import load_dotenv
load_dotenv()

# ✅ package-qualified imports (works when running: uvicorn backend.main:app)
from backend.agents.router_agent import route_request
from backend.agents.repo_agent import propose_changes, answer_about_repo, generate_patch_from_prompt
from backend.utils.nl_formatter import ensure_natural
from backend.utils.agent_protocol import AgentResponse
from backend.services import conversation as conv


app = FastAPI(title="Personal Agent API")

# backend/main.py
from fastapi import FastAPI
import os

app = FastAPI()

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
    # Simple probe the Repo endpoints are up (and which model is configured)
    return {"ok": True, "model": os.getenv("CHAT_MODEL", "gpt-5")}

@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/app/")

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
        # If the repo agent produced a string, or we can infer it's a repo-agent dict, return text/plain.
        if isinstance(out, str):
            return PlainTextResponse(out)
        if isinstance(out, dict) and out.get("agent") == "repo":
            text = out.get("answer") or out.get("message") or out.get("data") or json.dumps(out, ensure_ascii=False)
            return PlainTextResponse(str(text))
        return out
    except Exception as e:
        # bubble up precise error so we can see the RPC’s complaint
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/app/api/repo/plan", response_model=None)
async def repo_plan(request: Request) -> Response:
    """
    Default: return JSON plan (backward compatible).
    Patch mode: return text/x-patch unified diff when:
      - query param ?format=patch OR
      - Accept header contains text/x-patch or text/x-diff
    """
    try:
        payload: Dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    task = payload.get("task")
    if not task:
        raise HTTPException(status_code=400, detail="Missing 'task'")
    repo   = payload.get("repo", "personal-agent-ste")
    branch = payload.get("branch", "main")
    prefix = payload.get("path_prefix")
    k      = int(payload.get("k", 12))
    session = payload.get("session")
    thread_n = payload.get("thread_n")

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

    if not wants_patch:
        return JSONResponse(out)

    patch_text = out.get("patch") if isinstance(out, dict) else None
    if not patch_text:
        patch_text = generate_patch_from_prompt(out.get("prompt", ""))

    if not patch_text:
        return PlainTextResponse(
            "No patch generated for this task. Use default JSON mode to inspect the plan/prompt.",
            status_code=400,
            media_type="text/plain",
        )

    return PlainTextResponse(content=patch_text, media_type="text/x-patch")


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
        # Pull a plausible textual field; if unavailable, fall back to the whole normalized/pretty object.
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

    # Serve the SPA at /app (and optionally /)
    app.mount("/app", StaticFiles(directory=static_dir, html=True), name="static")

    @app.get("/app", include_in_schema=False, response_class=HTMLResponse)
    @app.get("/app/", include_in_schema=False, response_class=HTMLResponse)
    def serve_index():
        if INDEX_FILE.exists():
            return HTMLResponse(INDEX_FILE.read_text(encoding="utf-8"))
        raise HTTPException(status_code=404, detail="Frontend build not found.")

    # SPA fallback: return index.html for any /app/* path that isn't a real file
    @app.get("/app/{path:path}", include_in_schema=False, response_class=HTMLResponse)
    def spa_fallback(path: str):
        candidate = Path(static_dir) / path
        if candidate.is_file():
            return FileResponse(str(candidate))
        if INDEX_FILE.exists():
            return HTMLResponse(INDEX_FILE.read_text(encoding="utf-8"))
        raise HTTPException(status_code=404, detail="Frontend build not found.")
