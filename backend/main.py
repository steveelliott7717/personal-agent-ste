# backend/main.py
from __future__ import annotations

import os
import json
from typing import Any, Dict, Tuple

from fastapi import FastAPI, Form, Body, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from agents.router_agent import route_request
from utils.nl_formatter import ensure_natural
from utils.agent_protocol import AgentResponse

app = FastAPI(title="Personal Agent API")

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

# -------------------- Health --------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------- Universal request endpoint --------------------
@app.post("/api/request")
async def handle_request(
    query: str | None = Form(default=None),
    body: Dict[str, Any] | None = Body(default=None),
):
    q, _ = _extract_query(query, body)
    if not q:
        return JSONResponse(
            {"agent": "system", "intent": "error", "message": "Missing 'query' in form or JSON body"},
            status_code=400,
        )

    agent, raw_result = route_request(q)
    resp = _normalize(agent, raw_result)

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

# -------------------- Static frontend (optional) --------------------
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/app", StaticFiles(directory=static_dir, html=True), name="static")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static_root")
