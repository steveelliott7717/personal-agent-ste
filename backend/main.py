# backend/main.py
from fastapi import FastAPI, Form, Body
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from agents.router_agent import route_request

from utils.nl_formatter import ensure_natural
from utils.agent_protocol import AgentResponse
import os

app = FastAPI()
from fastapi.responses import JSONResponse
import json

class NaturalLanguageMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)

        try:
            # Only format JSON responses
            if isinstance(response, JSONResponse):
                body = b"".join([chunk async for chunk in response.body_iterator])
                response.body_iterator = None

                try:
                    payload = json.loads(body.decode("utf-8"))
                except Exception:
                    return response  # not valid JSON, skip formatting

                # Apply natural formatting
                try:
                    payload = ensure_natural(payload)
                except Exception as e:
                    payload = {
                        "agent": payload.get("agent", "system"),
                        "intent": "error",
                        "message": f"Formatting error: {str(e)}"
                    }

                return JSONResponse(content=payload, status_code=response.status_code)

            return response
        # backend/main.py, inside NaturalLanguageMiddleware.dispatch
        except Exception as e:
            return JSONResponse(content={
                "agent": payload.get("agent", "system") if isinstance(payload, dict) else "system",
                "intent": "error",
                "message": f"Formatting error: {str(e)}",
                "raw": payload  # <= add this so we see what the agent returned
            }, status_code=500)


# Add middleware to the app
app.add_middleware(NaturalLanguageMiddleware)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/request")
async def handle_request(
    query: str | None = Form(None),
    body: dict | None = Body(None),
):
    """
    Accepts either:
      - application/x-www-form-urlencoded with field 'query'
      - application/json with key 'query' (or 'prompt' / 'q')
    """
    # Prefer form, but allow JSON too
    if query is None and body and isinstance(body, dict):
        query = body.get("query") or body.get("prompt") or body.get("q")

    if not query:
        return JSONResponse(
            {"agent": "system", "intent": "error", "message": "Missing 'query' in form or JSON body"},
            status_code=400,
        )

    print(f"[api] incoming query: {query!r}")

    agent, raw_result = route_request(query)

    # Normalize agent result a bit
    if isinstance(raw_result, str):
        resp: AgentResponse = {"agent": agent, "intent": "say", "message": raw_result}
    elif isinstance(raw_result, dict):
        resp = {"agent": agent, **raw_result} if "agent" not in raw_result else raw_result  # type: ignore
        resp.setdefault("intent", "unknown")
        resp.setdefault("agent", agent)
    elif isinstance(raw_result, list):
        resp = {"agent": agent, "intent": "list", "data": raw_result}
    else:
        resp = {"agent": agent, "intent": "unknown", "message": str(raw_result)}

    try:
        natural = ensure_natural(resp)
    except Exception as e:
        # Don’t hide errors—surface the raw payload for debugging
        natural = {
            "agent": resp.get("agent", "system"),
            "intent": "error",
            "message": f"Formatting error: {e}",
            "raw": resp,
        }

    return JSONResponse(natural)

# near your existing mount
static_dir = os.path.join(os.path.dirname(__file__), "static")

# ALSO serve the same bundle at "/app"
app.mount("/app", StaticFiles(directory=static_dir, html=True), name="static")

# Serve at "/" (already there)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static_root")



