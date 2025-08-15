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
correlation_id_ctx: ContextVar[str] = ContextVar("correlation_id", default="-")


class CorrelationIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # provide %(correlation_id)s to all formatters
        try:
            record.correlation_id = correlation_id_ctx.get()
        except Exception:
            record.correlation_id = "-"
        return True


_CONFIGURED = False


# Add near the top (helpers)
def _env_truthy(name: str, default: str = "true") -> bool:
    val = os.getenv(name, default)
    return str(val).strip().lower() in {"true", "1", "t", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _truthy_env(name: str, default: str = "true") -> bool:
    val = os.getenv(name, default)
    return str(val or default).strip().lower() in {"true", "1", "t", "yes", "y", "on"}


def setup_logging(level: Optional[int] = None) -> None:
    """
    Idempotent logging setup that ensures %(correlation_id)s is available in all log lines.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    filt = CorrelationIdFilter()
    root = logging.getLogger()
    root.propagate = False

    # Always attach filter to root so any existing/new handlers see correlation_id
    try:
        root.addFilter(filt)
    except Exception:
        pass

    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(correlation_id)s] %(name)s: %(message)s"
            )
        )
        handler.addFilter(filt)
        root.addHandler(handler)
        root.setLevel(level or logging.INFO)
    else:
        for h in list(root.handlers):
            # Ensure formatter includes correlation_id
            fmt = getattr(h.formatter, "_fmt", "") if h.formatter else ""
            if "%(correlation_id)" not in fmt:
                try:
                    h.setFormatter(
                        logging.Formatter(
                            "%(asctime)s %(levelname)s [%(correlation_id)s] %(name)s: %(message)s"
                        )
                    )
                except Exception:
                    pass
            try:
                h.addFilter(filt)
            except Exception:
                pass
        # Optional: switch to JSON logs
    if _env_truthy("LOG_JSON", "false"):

        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                import json, time

                payload = {
                    "ts": time.strftime(
                        "%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)
                    ),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                    "correlation_id": getattr(record, "correlation_id", "-"),
                }
                return json.dumps(payload, ensure_ascii=True)

        for h in logging.getLogger().handlers:
            h.setFormatter(_JsonFormatter())

    # Common FastAPI/Uvicorn loggers
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        try:
            logging.getLogger(name).addFilter(filt)
        except Exception:
            pass

    _CONFIGURED = True


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    - Generates UUID correlation ID per request (also in request.state.correlation_id)
    - Logs start/end/errors (gated by LOG_REQUESTS, default true)
    - Adds X-Correlation-ID response header
    """

    def __init__(self, app):
        super().__init__(app)
        self._logger = logging.getLogger("request")
        self._log_requests = _truthy_env("LOG_REQUESTS", "true")

    async def dispatch(self, request: Request, call_next):
        cid = uuid.uuid4().hex
        token = correlation_id_ctx.set(cid)
        try:
            request.state.correlation_id = cid
        except Exception:
            pass

        method = request.method
        path = request.url.path
        client = request.client.host if request.client else "-"
        start = time.perf_counter()

        if self._log_requests:
            self._logger.info(">> %s %s client=%s", method, path, client)

        try:
            response: Response = await call_next(request)
            dur_ms = int((time.perf_counter() - start) * 1000)
            try:
                response.headers["X-Correlation-ID"] = cid
            except Exception:
                pass
            if self._log_requests:
                self._logger.info(
                    "<< %s %s %d %dms", method, path, response.status_code, dur_ms
                )
            return response
        except Exception as e:
            dur_ms = int((time.perf_counter() - start) * 1000)
            # Always log exceptions
            self._logger.exception(
                "!! %s %s error after %dms: %s", method, path, dur_ms, e
            )
            raise
        finally:
            try:
                correlation_id_ctx.reset(token)
            except Exception:
                pass


# At the very bottom of logging_utils.py, after all class and function definitions

# Auto-configure logging on import
try:
    setup_logging()
except Exception as e:
    import logging

    logging.getLogger("logging_utils").warning(
        "setup_logging() failed at import: %s", e
    )
