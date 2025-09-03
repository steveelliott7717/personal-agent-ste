# backend/main.py
from __future__ import annotations

"""
Delegator entrypoint for Uvicorn (Option B).

Keeps Dockerfile as: uvicorn backend.main:app --host 0.0.0.0 --port 8080
Loads the real FastAPI app from backend/api.py (primary) or api.py (secondary).
If both fail, starts a tiny fallback that surfaces the import error.
"""

try:
    # Primary: your app lives at backend/api.py
    from backend.api import app  # type: ignore
except Exception:
    try:
        # Secondary: if you ever move app to top-level api.py
        from api import app  # type: ignore
    except Exception:
        from fastapi import FastAPI
        from fastapi.responses import PlainTextResponse

        app = FastAPI(title="Fallback App (import failed)")

        @app.get("/health", response_class=PlainTextResponse)
        async def health() -> str:
            return "ok"

        @app.get("/", response_class=PlainTextResponse)
        async def root() -> str:
            return (
                "fallback: failed to import FastAPI app\n"
                f"backend.api error: {e_backend.__class__.__name__}: {e_backend}\n"
                f"api error: {e_top.__class__.__name__}: {e_top}\n"
            )
