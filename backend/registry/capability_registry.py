from __future__ import annotations

from typing import Any, Callable, Dict, Optional, TypedDict
from datetime import datetime, timezone
import traceback
import uuid
import time
import os
import json

# --- add: async + db client for audit logs ---
import asyncio

try:
    import psycopg  # psycopg v3 (sync + async)
except Exception:  # pragma: no cover
    psycopg = None


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


# ---- Error taxonomy & helpers ----

REDACT_KEYS = {
    k.strip().lower()
    for k in (
        os.getenv("REGISTRY_REDACT_KEYS", "authorization,api_key,bearer,token").split(
            ","
        )
    )
}


def _redact_headers(hdrs: dict | None) -> dict | None:
    if not isinstance(hdrs, dict):
        return None
    out = {}
    for k, v in hdrs.items():
        if k.lower() in REDACT_KEYS:
            out[k] = "REDACTED"
        else:
            out[k] = v
    return out


def _redact_obj(obj):
    """
    Recursively redact dict/list values whose keys match REDACT_KEYS.
    Leaves non-dict/list as-is.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = "REDACTED" if k.lower() in REDACT_KEYS else _redact_obj(v)
        return out
    if isinstance(obj, list):
        return [_redact_obj(x) for x in obj]
    return obj


# --- add: audit logging config ---
_AUDIT_ENABLED = os.getenv("REGISTRY_AUDIT_LOGS", "1").lower() not in {
    "0",
    "false",
    "no",
}
_AGENT_LOGS_TABLE = os.getenv("AGENT_LOGS_TABLE", "public.agent_logs")
_DB_DSN = (
    os.getenv("SUPABASE_DB_POOLER_URL")
    or os.getenv("DATABASE_URL")
    or os.getenv("SUPABASE_DB_URL")
)


def _prepare_safe_json(obj):
    if obj is None:
        return None
    try:
        return json.dumps(_redact_obj(obj))
    except Exception:
        # last-resort: best-effort stringify so logging never breaks
        try:
            return json.dumps({"_unsafe": str(obj)[:2000]})
        except Exception:
            return None


async def _emit_agent_log_async(
    *,
    verb: str,
    ok: bool,
    code: str | None,
    latency_ms: int,
    args_json: dict | list | None,
    result_json: dict | list | None,
    correlation_id: str | None = None,
    cost_cents: int | None = None,
) -> None:
    """
    Async, best-effort audit insert. Swallows all exceptions.
    Uses psycopg async if available; else runs a sync insert in a thread.
    """
    if not _AUDIT_ENABLED or psycopg is None or not _DB_DSN:
        return

    trace_id = str(uuid.uuid4())
    args_safe = _prepare_safe_json(args_json)
    res_safe = _prepare_safe_json(result_json)

    async def _do_async():
        try:
            async with await psycopg.AsyncConnection.connect(_DB_DSN) as conn:
                async with conn.cursor() as cur:
                    try:
                        await cur.execute(
                            f"""insert into {_AGENT_LOGS_TABLE}
                                (trace_id, verb, ok, code, latency_ms, cost_cents, args_json, result_json, correlation_id)
                              values (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                            (
                                trace_id,
                                verb,
                                ok,
                                code,
                                latency_ms,
                                cost_cents,
                                args_safe,
                                res_safe,
                                correlation_id,
                            ),
                        )
                    except Exception:
                        await cur.execute(
                            f"""insert into {_AGENT_LOGS_TABLE}
                                (trace_id, verb, ok, code, latency_ms, cost_cents, args_json, result_json)
                              values (%s,%s,%s,%s,%s,%s,%s,%s)""",
                            (
                                trace_id,
                                verb,
                                ok,
                                code,
                                latency_ms,
                                cost_cents,
                                args_safe,
                                res_safe,
                            ),
                        )
                await conn.commit()
        except Exception:
            return  # swallow

    def _do_sync():
        try:
            with psycopg.connect(_DB_DSN, autocommit=True) as conn:
                with conn.cursor() as cur:
                    try:
                        cur.execute(
                            f"""insert into {_AGENT_LOGS_TABLE}
                                (trace_id, verb, ok, code, latency_ms, cost_cents, args_json, result_json, correlation_id)
                              values (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                            (
                                trace_id,
                                verb,
                                ok,
                                code,
                                latency_ms,
                                cost_cents,
                                args_safe,
                                res_safe,
                                correlation_id,
                            ),
                        )
                    except Exception:
                        cur.execute(
                            f"""insert into {_AGENT_LOGS_TABLE}
                                (trace_id, verb, ok, code, latency_ms, cost_cents, args_json, result_json)
                              values (%s,%s,%s,%s,%s,%s,%s,%s)""",
                            (
                                trace_id,
                                verb,
                                ok,
                                code,
                                latency_ms,
                                cost_cents,
                                args_safe,
                                res_safe,
                            ),
                        )
        except Exception:
            return  # swallow

    # Prefer async path if available; otherwise offload sync insert to thread
    if hasattr(psycopg, "AsyncConnection"):
        await _do_async()
    else:
        await asyncio.to_thread(_do_sync)


def _fire_and_forget(coro) -> None:
    """
    Schedule a coroutine if there's a running loop; otherwise run it in a short-lived loop.
    Never raises.
    """
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        # no running loop
        try:
            asyncio.run(coro)
        except RuntimeError:
            # already running loop but in non-async context; fallback to background thread
            asyncio.get_event_loop().create_task(coro)


def _exception_chain(exc: BaseException, max_depth: int = 3) -> list[dict]:
    chain = []
    cur = exc
    depth = 0
    seen = set()
    while cur and depth < max_depth and id(cur) not in seen:
        seen.add(id(cur))
        chain.append({"type": cur.__class__.__name__, "message": str(cur)[:500]})
        cur = cur.__cause__ or cur.__context__
        depth += 1
    return chain


def _extract_retry_after(headers: dict | None) -> int | None:
    try:
        ra = (headers or {}).get("Retry-After")
        if not ra:
            return None
        # seconds or HTTP date
        return int(ra) if ra.isdigit() else None
    except Exception:
        return None


def _map_error_code(exc: Exception) -> tuple[str, str | None, bool]:
    """
    Returns: (code, subcode, retryable)
    """
    name = exc.__class__.__name__
    msg = str(exc)

    # PostgREST / DB
    if hasattr(exc, "code"):  # your APIError dicts already carry .code
        code = getattr(exc, "code") or "DBError"
        retryable = code in {
            "40001",
            "40P01",
            "PGRST116",
            "PGRST114",
            "PGRST204",
        }  # deadlock/timeout-ish
        return code, None, retryable

    # HTTP adapters
    if name in ("TimeoutError", "socket.timeout"):
        return "Timeout", None, True
    if "connection refused" in msg.lower():
        return "ConnRefused", None, True
    if "dns" in msg.lower() or "name or service not known" in msg.lower():
        return "DNS", None, True
    if "ssl" in msg.lower():
        return "TLS", None, False

    # Browser/Playwright
    if "BrowserType.launch" in msg or "playwright" in msg.lower():
        return "BrowserError", None, False
    if "Timeout " in msg and "exceeded" in msg:
        return "Timeout", "BrowserWait", True

    # Validation / policy
    if name in ("ValueError", "KeyError", "ValidationError", "TypeError"):
        return "ValidationError", None, False

    # Fallback
    return name, None, False


def build_error_envelope(
    exc: Exception,
    *,
    source: str | None = None,
    request_ctx: dict | None = None,
    response_ctx: dict | None = None,
    invalid_params: list[dict] | None = None,
    partial_result: dict | None = None,
) -> dict:
    code, subcode, retryable = _map_error_code(exc)

    # Pull retry_after from response headers if present
    retry_after_ms = _extract_retry_after((response_ctx or {}).get("headers"))

    # Sanitize request headers if present
    safe_request = None
    if request_ctx:
        safe_request = dict(request_ctx)
        if "headers" in safe_request:
            safe_request["headers"] = _redact_headers(safe_request["headers"])

    err = {
        "version": 1,
        "code": code,
        "subcode": subcode,
        "message": str(exc)[:1000],
        "retryable": retryable,
        "retry_after_ms": retry_after_ms,
        "source": source,
        "causes": _exception_chain(exc),
        "hint": getattr(exc, "hint", None) if hasattr(exc, "hint") else None,
        "details": getattr(exc, "details", None) if hasattr(exc, "details") else None,
        "request": safe_request,
        "response": response_ctx,
        "invalid_params": invalid_params or None,
        "partial_result": partial_result or None,
    }
    # Drop empty keys
    return {k: v for k, v in err.items() if v is not None}


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
                # --- add: fire-and-forget audit (adapter error) ---
                try:
                    _fire_and_forget(
                        _emit_agent_log_async(
                            verb=verb,
                            ok=False,
                            code=err.get("code") or "AdapterError",
                            latency_ms=latency_ms,
                            args_json={"args": args, "meta": meta},
                            result_json={"error": err},
                            correlation_id=correlation_id or None,
                            cost_cents=None,
                        )
                    )
                except Exception:
                    pass

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

            # --- add: fire-and-forget audit (success) ---
            try:
                _fire_and_forget(
                    _emit_agent_log_async(
                        verb=verb,
                        ok=True,
                        code="OK",
                        latency_ms=latency_ms,
                        args_json={"args": args, "meta": meta},
                        result_json=result,
                        correlation_id=correlation_id or None,
                        cost_cents=(
                            result.get("cost_cents")
                            if isinstance(result, dict)
                            else None
                        ),
                    )
                )
            except Exception:
                pass

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
            norm = _normalize_exception(exc)

            # --- add: fire-and-forget audit (exception path) ---
            try:
                _fire_and_forget(
                    _emit_agent_log_async(
                        verb=verb,
                        ok=False,
                        code=norm.get("code") or "InternalError",
                        latency_ms=latency_ms,
                        args_json={"args": args, "meta": meta},
                        result_json={"error": norm},
                        correlation_id=correlation_id or None,
                        cost_cents=None,
                    )
                )
            except Exception:
                pass

            return {
                "ok": False,
                "result": None,
                "error": norm,
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
