# backend/logging_utils.py
from __future__ import annotations
import logging
import os
import time
import uuid
from contextvars import ContextVar
from typing import Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Per-request correlation ID
_cid_var: ContextVar[str] = ContextVar("correlation_id", default="-")
_CONFIGURED = False


class CorrelationIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.correlation_id = _cid_var.get()
        except Exception:
            record.correlation_id = "-"
        return True


def _truthy(s: Optional[str], default: bool = True) -> bool:
    if s is None:
        return default
    return s.strip().lower() in {"true", "1", "t", "yes", "y", "on"}


def setup_logging(level: Optional[int] = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    filt = CorrelationIdFilter()
    root = logging.getLogger()

    # Always attach filter to root
    root.addFilter(filt)

    if not root.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)s [%(correlation_id)s] %(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(handler)
        root.setLevel(level or logging.INFO)
    else:
        for h in root.handlers:
            if "%(correlation_id)" not in getattr(h.formatter, "_fmt", ""):
                h.setFormatter(logging.Formatter(
                    "%(asctime)s %(levelname)s [%(correlation_id)s] %(name)s: %(message)s"
                ))
            h.addFilter(filt)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logging.getLogger(name).addFilter(filt)

    _CONFIGURED = True


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    - Generates a UUID per request and stores it in ContextVar
    - Logs start/end/errors (gated by LOG_REQUESTS; default: enabled)
    - Adds X-Correlation-ID to responses
    """
    async def dispatch(self, request: Request, call_next):
        cid = request.headers.get("X-Correlation-ID") or uuid.uuid4().hex
        token = _cid_var.set(cid)
        request.state.correlation_id = cid

        log_requests = _truthy(os.getenv("LOG_REQUESTS"), default=True)
        logger = logging.getLogger("request")
        start = time.perf_counter()

        if log_requests:
            logger.info(">> %s %s", request.method, request.url.path)

        try:
            response: Response = await call_next(request)
            duration_ms = int((time.perf_counter() - start) * 1000)
            try:
                response.headers["X-Correlation-ID"] = cid
            except Exception:
                pass
            if log_requests:
                logger.info("<< %s %s %d %dms", request.method, request.url.path, response.status_code, duration_ms)
            return response
        except Exception as e:
            duration_ms = int((time.perf_counter() - start) * 1000)
            logger.exception("!! %s %s error after %dms: %s", request.method, request.url.path, duration_ms, e)
            raise
        finally:
            try:
                _cid_var.reset(token)
            except Exception:
                pass
