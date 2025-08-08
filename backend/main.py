# backend/main.py
from fastapi import FastAPI, Form
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
        except Exception as e:
            return JSONResponse(content={
                "agent": "system",
                "intent": "error",
                "message": f"Middleware failure: {str(e)}"
            }, status_code=500)

# Add middleware to the app
app.add_middleware(NaturalLanguageMiddleware)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/request")
async def handle_request(query: str = Form(...)):
    agent, raw_result = route_request(query)

    # Legacy agents might return plain strings or lists; normalize minimally
    if isinstance(raw_result, str):
        resp: AgentResponse = {"agent": agent, "intent": "say", "message": raw_result}
    elif isinstance(raw_result, dict):
        # assume it's an AgentResponse-like dict
        resp = {"agent": agent, **raw_result} if "agent" not in raw_result else raw_result  # type: ignore
        resp.setdefault("intent", "unknown")
        resp.setdefault("agent", agent)
    elif isinstance(raw_result, list):
        resp = {"agent": agent, "intent": "list", "data": raw_result}
    else:
        resp = {"agent": agent, "intent": "unknown", "message": str(raw_result)}

    natural = ensure_natural(resp)
    return JSONResponse(natural)

# near your existing mount
static_dir = os.path.join(os.path.dirname(__file__), "static")

# ALSO serve the same bundle at "/app"
app.mount("/app", StaticFiles(directory=static_dir, html=True), name="static")

# Serve at "/" (already there)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static_root")



