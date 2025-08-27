from __future__ import annotations

from typing import Any, Callable, Dict, Optional, TypedDict
from datetime import datetime, timezone
import traceback
import uuid
import time


# Import only adapter entrypoints here
from backend.registry.adapters.http_fetch import http_fetch_adapter
from backend.registry.adapters.db_read import db_read_adapter
from backend.registry.adapters.db_write import db_write_adapter
from backend.registry.adapters.notify_push import notify_push_adapter
from backend.registry.adapters.browser_adapter import browser_run_adapter

# -----------------------------
# Structured error helpers
# -----------------------------
import traceback

ERR_VERSION = 1


def _mk_error(
    code: str, message: str, hint: str | None = None, details: str | dict | None = None
) -> dict:
    return {
        "version": ERR_VERSION,
        "code": str(code),
        "message": str(message),
        "hint": hint,
        "details": details,
    }


def _normalize_exception(exc: Exception) -> dict:
    """
    Map Python/PostgREST/browser/http exceptions to a stable {code,message,hint,details}.
    """
    # PostgREST/Supabase-style dict in args?  e.g. {'message': '...', 'code': 'PGRST100', ...}
    if exc.args and isinstance(exc.args[0], dict):
        d = exc.args[0]
        if "message" in d or "code" in d:
            return _mk_error(
                code=d.get("code") or exc.__class__.__name__,
                message=d.get("message") or str(exc),
                hint=d.get("hint"),
                details=d.get("details"),
            )

    # Some SDKs attach .data (dict with code/message)
    data = getattr(exc, "data", None)
    if isinstance(data, dict) and ("code" in data or "message" in data):
        return _mk_error(
            code=data.get("code") or exc.__class__.__name__,
            message=data.get("message") or str(exc),
            hint=data.get("hint"),
            details=data.get("details"),
        )

    # Common Python exceptions
    if isinstance(exc, TimeoutError):
        return _mk_error(
            "Timeout", str(exc), "Increase timeout_ms or simplify the request."
        )
    if isinstance(exc, (ValueError, TypeError)):
        return _mk_error("ValidationError", str(exc))
    if isinstance(exc, KeyError):
        return _mk_error("ValidationError", f"missing key: {exc}")

    # Fallback
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return _mk_error("InternalError", str(exc), details=tb)


def _extract_error_from_result(res: Any) -> Optional[Dict[str, Any]]:
    """
    If an adapter returned a dict that *looks* like an error (status:0 or 'error'/'message' keys),
    convert to a structured error so envelopes are consistent.
    """
    if not isinstance(res, dict):
        return None

    # Adapter already returned structured error?
    if isinstance(res.get("error"), dict) and "message" in res["error"]:
        return res["error"]

    # status==0 or string error
    if res.get("status") == 0 or "error" in res:
        err = res.get("error")
        code = (
            res.get("code") or (err.get("code") if isinstance(err, dict) else None)
        ) or "AdapterError"
        msg = (
            res.get("message")
            or (err if isinstance(err, str) else None)
            or "Adapter error"
        )
        return _mk_error(code, msg, res.get("hint"), res.get("details"))

    return None


class Envelope(TypedDict, total=False):
    ok: bool
    result: Dict[str, Any] | None
    error: Dict[str, Any] | None
    latency_ms: int
    correlation_id: str
    idempotency_key: str


MetaDict = Dict[str, Any]


class CapabilityRegistry:
    def __init__(self) -> None:
        self._adapters: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Any]] = {}
        self._register()

    def register(
        self, verb: str, fn: Callable[[Dict[str, Any], Dict[str, Any]], Any]
    ) -> None:
        self._adapters[verb] = fn

    def has(self, verb: str) -> bool:
        return verb in self._adapters

    # -----------------------------
    # Central dispatch with envelope
    # -----------------------------
    def dispatch(
        self, verb: str, args: Dict[str, Any], meta: Dict[str, Any]
    ) -> Envelope:
        t0 = time.perf_counter()
        correlation_id = (meta or {}).get("correlation_id", "")
        idempotency_key = (meta or {}).get("idempotency_key", "")

        # Resolve handler
        handler = self._adapters.get(verb)
        if handler is None:
            return {
                "ok": False,
                "result": None,
                "error": _mk_error(
                    "UnknownVerb",
                    f"Unknown verb: {verb}",
                    "See /app/api/agents/verbs for supported verbs.",
                ),
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "correlation_id": correlation_id,
                "idempotency_key": idempotency_key,
            }

        # Execute handler
        try:
            raw = handler(args or {}, meta or {})
            latency_ms = int((time.perf_counter() - t0) * 1000)

            # Coerce in-band adapter error to structured error
            err = _extract_error_from_result(raw)
            if err:
                return {
                    "ok": False,
                    "result": None,
                    "error": err,
                    "latency_ms": latency_ms,
                    "correlation_id": correlation_id,
                    "idempotency_key": idempotency_key,
                }

            # Success; pass result through as-is
            result = raw if isinstance(raw, dict) else {"data": raw}
            return {
                "ok": True,
                "result": result,
                "error": None,
                "latency_ms": latency_ms,
                "correlation_id": correlation_id,
                "idempotency_key": idempotency_key,
            }

        except Exception as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return {
                "ok": False,
                "result": None,
                "error": _normalize_exception(exc),
                "latency_ms": latency_ms,
                "correlation_id": correlation_id,
                "idempotency_key": idempotency_key,
            }

    # -----------------------------
    # Registry
    # -----------------------------
    def _register(self) -> None:
        self.register("health.echo", lambda a, m: {"echo": {"args": a, "meta": m}})
        self.register(
            "time.now", lambda _a, _m: {"iso": datetime.now(timezone.utc).isoformat()}
        )
        self.register("db.read", db_read_adapter)
        self.register("db.write", db_write_adapter)
        self.register("notify.push", notify_push_adapter)
        self.register("http.fetch", http_fetch_adapter)
        self.register("browser.run", browser_run_adapter)
