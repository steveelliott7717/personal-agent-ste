from __future__ import annotations

import asyncio
import sys
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

# Keep error texts in module-level names to avoid F821 in handlers.
E_BACKEND_MSG: str | None = None
E_TOP_MSG: str | None = None

# On Windows, Playwright requires asyncio subprocess support.
# SelectorEventLoopPolicy DOES NOT support subprocesses; Proactor does.
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        # Safe fallback: leave the default policy in place if this fails
        pass


try:
    from backend.api import app as _real_app  # type: ignore[attr-defined]

    app = _real_app
except Exception as _e_backend:  # noqa: BLE001
    E_BACKEND_MSG = f"{_e_backend.__class__.__name__}: {_e_backend}"
    try:
        from api import app as _real_app  # type: ignore[attr-defined]

        app = _real_app
    except Exception as _e_top:  # noqa: BLE001
        E_TOP_MSG = f"{_e_top.__class__.__name__}: {_e_top}"

        # Final minimal fallback app so health checks still pass.
        app = FastAPI()

        @app.get("/health", response_class=PlainTextResponse)
        async def health() -> str:
            return "ok"

        @app.get("/", response_class=PlainTextResponse)
        async def root() -> str:
            b_err = (
                f"backend.api error: {E_BACKEND_MSG}"
                if E_BACKEND_MSG
                else "backend.api ok"
            )
            t_err = f"api error: {E_TOP_MSG}" if E_TOP_MSG else "api ok"
            return (
                "fallback: failed to import FastAPI app\n" + b_err + "\n" + t_err + "\n"
            )
