# backend/api.py
from __future__ import annotations
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env for SUPABASE etc.
load_dotenv()

# your existing router entry
from backend.agents.router_agent import route_request

BASE = "/app"  # matches your frontend base
app = FastAPI(title="Personal Agents API")

# ---------- SPA static serving at /app/ ----------
# Serve built assets from frontend/dist (Vite build) and index fallback for SPA routes.
DIST_DIR = (Path(__file__).resolve().parent.parent / "frontend" / "dist").resolve()
INDEX_FILE = DIST_DIR / "index.html"

# Serve built asset files (Vite emits to dist/assets/*). With base='/app/', URLs are /app/assets/...
if (DIST_DIR / "assets").exists():
    app.mount(f"{BASE}/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")

def _index_response() -> HTMLResponse:
    if INDEX_FILE.exists():
        return HTMLResponse(INDEX_FILE.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="Frontend build not found. Run `npm run build` in /frontend.")

@app.get(BASE, include_in_schema=False, response_class=HTMLResponse)
def serve_index_root():
    # GET /app -> index.html
    return _index_response()

@app.get(f"{BASE}/", include_in_schema=False, response_class=HTMLResponse)
def serve_index_slash():
    # GET /app/ -> index.html
    return _index_response()

@app.get(f"{BASE}" + "/{path:path}", include_in_schema=False)
def spa_fallback(path: str):
    # Let API continue to be handled by your /app/api/* routes
    if path.startswith("api"):
        raise HTTPException(status_code=404)
    # If a real file was requested (e.g., /app/assets/...), serve it directly
    candidate = DIST_DIR / path
    if candidate.is_file():
        return FileResponse(str(candidate))
    # Otherwise, return index.html for client-side routes
    return _index_response()

# CORS for local dev (frontend vite port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RouteIn(BaseModel):
    text: str
    client_meta: dict | None = None
    user_id: str | None = "web-guest"

@app.post(f"{BASE}/api/route")
def route(incoming: RouteIn):
    try:
        agent, result = route_request(query=incoming.text, user_id=incoming.user_id or "web-guest")
        # result might already be a dict/str from your make_response; normalize
        if isinstance(result, dict):
            payload = result
        else:
            payload = {"agent": agent, "intent": "unknown", "message": str(result)}
        # attach meta if present
        if incoming.client_meta:
            payload.setdefault("meta", {})["client_meta"] = incoming.client_meta

        # Special-case: if the routed agent is "repo", return text/plain for humans.
        if (payload.get("agent") == "repo"):
            # Prefer explicit 'answer', then 'message', then stringify the raw result.
            text = (
                (payload.get("answer") if isinstance(payload, dict) else None)
                or (payload.get("message") if isinstance(payload, dict) else None)
                or (str(result) if result is not None else "")
            )
            return PlainTextResponse(text)

        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
