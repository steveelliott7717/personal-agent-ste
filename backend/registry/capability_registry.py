from __future__ import annotations

from typing import Any, Callable, Dict, Optional, TypedDict, Tuple
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
from backend.registry.adapters.browser_adapter import browser_warmup_adapter

from backend.registry.adapters.browser_adapter import browser_run_adapter


# Heuristic: decide if we should escalate from http.fetch -> browser.run
def _needs_browser(
    status: int, body: bytes | str | None, headers: Dict[str, Any]
) -> Tuple[bool, str]:
    if body is None:
        return True, "empty-body"
    # Normalize to text for checks (limit to avoid big decode)
    sample = (
        body[:200000] if isinstance(body, (bytes, bytearray)) else (body or "")[:200000]
    )
    if isinstance(sample, (bytes, bytearray)):
        try:
            sample = sample.decode("utf-8", "ignore")
        except Exception:
            sample = ""

    ct = (headers.get("content-type") or headers.get("Content-Type") or "").lower()

    # Hard signals: HTTP throttle/forbid
    if status in (401, 403, 407, 429):
        return True, f"http-{status}"

    # Suspicious content types or no content-type for HTML pages
    if "text/html" in ct or ct == "":
        # Common bot-wall / JS-wall phrases
        wall_markers = (
            "are you a robot",
            "enable javascript",
            "turn on javascript",
            "access denied",
            "attention required",
            "sorry, you have been blocked",
            "captcha",
            "the page could not be loaded",
            "verification required",
            "please verify you are human",
        )
        low_signal = len(sample.strip()) < 512  # page likely rendered by JS or blocked
        if low_signal:
            return True, "low-signal-html"
        if any(m in sample.lower() for m in wall_markers):
            return True, "bot-wall-markers"

    # Non-HTML, but tiny payloads from an HTML URL path may also be suspicious
    if len(sample.strip()) < 64 and any(k in ct for k in ("html", "xml", "xhtml")):
        return True, "tiny-html"

    return False, "ok"


def _to_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8", "ignore")
        except Exception:
            return ""
    return str(x)


def _web_smart_get(
    registry, args: Dict[str, Any], meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Try http.fetch first (fast). If blocked/empty/JS-only, escalate to browser.run.
    Args:
      {
        "url": "...",
        # optional shared knobs passed to both layers:
        "timeout_ms": 15000,
        "timeout_total_ms": 30000,         # honored by browser.run
        "user_agent": "...",
        "user_agent_pool": [...],
        "accept_language_pool": [...],     # http.fetch only
        "rate_limit": {"per_host": {...}}, # both
        "jitter": {...},                   # browser.run only
        "locale_pool": [...],              # browser.run only
        "viewport": {...},                 # browser.run only
        "ua_rotate_per": "run"|"host",     # browser.run only
        # browser extras:
        "wait_until": "domcontentloaded",
        "save_dir": "downloads",
        "return_html": True
      }
    Returns:
      Same envelope style as adapters: {"ok": bool, "result": {...}, "error": {...}}
    """
    url = (args.get("url") or "").strip()
    if not url:
        return {
            "ok": False,
            "error": {"code": "BadRequest", "message": "url is required"},
        }

    # ---------- First pass: http.fetch ----------
    http_args = {
        "url": url,
        "method": "GET",
        "timeout_ms": int(args.get("timeout_ms", 15000)),
        "headers": args.get("headers") or {},
        "user_agent_pool": args.get("user_agent_pool") or [],
        "accept_language_pool": args.get("accept_language_pool") or [],
        "rate_limit": args.get("rate_limit") or {},
        "retry": args.get("retry")
        or {
            "max": 2,
            "on": [429, 500, 502, 503, 504],
            "backoff_ms": 300,
            "jitter": True,
        },
        "redirect": args.get("redirect") or {"policy": "follow", "max": 5},
    }

    http_out = registry.dispatch("http.fetch", http_args, meta)
    if not http_out.get("ok"):
        # Hard fail on fetch → go straight to browser
        reason = (
            f"http.fetch.error:{(http_out.get('error') or {}).get('code','unknown')}"
        )
        return _smart_escalate_browser(registry, args, meta, reason, http_out)

    status = int(http_out.get("result", {}).get("status", 0))
    body = http_out.get("result", {}).get("body")
    headers = http_out.get("result", {}).get("headers") or {}

    need, why = _needs_browser(status, body, headers)
    if not need:
        # Return the http result but normalize a text body field for convenience
        text_body = _to_text(body)
        res = dict(http_out)
        res["result"] = dict(http_out.get("result", {}))
        res["result"]["text"] = text_body
        res["result"]["source"] = "http.fetch"
        return res

    # ---------- Escalate: browser.run ----------
    return _smart_escalate_browser(registry, args, meta, why, http_out)


def _smart_escalate_browser(
    registry,
    args: Dict[str, Any],
    meta: Dict[str, Any],
    reason: str,
    http_out: Dict[str, Any],
) -> Dict[str, Any]:
    # Minimal scripted flow: goto → (optional) wait → capture HTML + optional screenshot
    steps = [
        {"goto": args["url"], "wait_until": args.get("wait_until", "domcontentloaded")},
        {"wait_for": {"state": "load"}},
    ]
    if args.get("screenshot"):
        steps.append({"screenshot": {"path": args["screenshot"], "full_page": True}})

    br_args = {
        "url": args["url"],
        "steps": steps,
        "timeout_ms": int(args.get("timeout_ms", 30000)),
        "timeout_total_ms": int(args.get("timeout_total_ms", 45000)),
        "user_agent": args.get("user_agent"),
        "user_agent_pool": args.get("user_agent_pool") or [],
        "ua_rotate_per": args.get("ua_rotate_per", "run"),
        "locale_pool": args.get("locale_pool") or ["en-US", "en-GB", "en"],
        "viewport": args.get("viewport"),
        "rate_limit": args.get("rate_limit") or {},
        "jitter": args.get("jitter")
        or {
            "ms_min": 120,
            "ms_max": 900,
            "prob": 0.85,
            "between_steps": True,
            "between_actions": True,
            "post_action_prob": 0.30,
        },
        "return_html": True,
        "save_dir": args.get("save_dir", "downloads"),
        "wait_until": args.get("wait_until", "domcontentloaded"),
    }

    br_out = registry.dispatch("browser.run", br_args, meta)
    # Annotate result with provenance & reason
    if br_out.get("ok"):
        r = dict(br_out)
        r["result"] = dict(br_out.get("result", {}))
        r["result"]["source"] = "browser.run"
        r["result"]["escalated_from"] = {
            "verb": "http.fetch",
            "reason": reason,
            "http_status": (http_out.get("result") or {}).get("status"),
        }
        return r

    # Both paths failed → return browser error, include http.fetch summary for debugging
    return {
        "ok": False,
        "error": br_out.get("error")
        or {"code": "BrowserError", "message": "browser.run failed"},
        "result": {
            "escalated_from": "http.fetch",
            "escalation_reason": reason,
            "http_status": (http_out.get("result") or {}).get("status"),
        },
    }


# -----------------------------
# Structured error helpers
# -----------------------------
import traceback

ERR_VERSION = 1


def _apply_http_fetch_env_defaults(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    If caller didn't specify, inherit safe HTTP defaults from environment.
    """
    out = dict(args or {})

    # allow/deny host lists (comma-separated env)
    if "allow_hosts" not in out and os.getenv("HTTP_FETCH_ALLOW_HOSTS"):
        out["allow_hosts"] = os.getenv("HTTP_FETCH_ALLOW_HOSTS")
    if "deny_hosts" not in out and os.getenv("HTTP_FETCH_DENY_HOSTS"):
        out["deny_hosts"] = os.getenv("HTTP_FETCH_DENY_HOSTS")

    # rate limit defaults (only if caller omitted the per_host block)
    ph = (out.get("rate_limit") or {}).get("per_host") or {}
    if not ph:
        cap = os.getenv("HTTP_FETCH_RL_CAPACITY")
        ref = os.getenv("HTTP_FETCH_RL_REFILL")
        mw = os.getenv("HTTP_FETCH_RL_MAXWAIT")
        if any(v is not None for v in (cap, ref, mw)):
            out["rate_limit"] = {
                "per_host": {
                    "capacity": int(cap) if cap is not None else 5,
                    "refill_per_sec": float(ref) if ref is not None else 2.0,
                    "max_wait_ms": int(mw) if mw is not None else 2000,
                }
            }

    # max redirects default (adapter will clamp again)
    if "max_redirects" not in out and os.getenv("HTTP_FETCH_MAX_REDIRECTS_DEFAULT"):
        out["max_redirects"] = int(os.getenv("HTTP_FETCH_MAX_REDIRECTS_DEFAULT", "5"))

    return out


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
            if verb == "http.fetch":
                args = _apply_http_fetch_env_defaults(args or {})
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
        self.register("browser.warmup", browser_warmup_adapter)
        self.register(
            "web.smart_get", lambda args, meta: _web_smart_get(self, args, meta)
        )
