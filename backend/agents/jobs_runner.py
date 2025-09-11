# backend/agents/jobs_runner.py
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, List

from supabase import create_client, Client

from importlib import import_module


# ===========================================================================


from backend.llm.llm_runner import run_llm_agent  # your existing LLM wrapper

from fastapi.encoders import jsonable_encoder

# Prefer canonical registry path first; then fall back.
capmod = None
for _name in (
    "backend.registry.capability_registry",
    "backend.capability_registry",
    "capability_registry",
):
    try:
        capmod = import_module(_name)
        break
    except Exception:
        continue

if capmod is None:
    raise ImportError("CapabilityRegistry module not found in expected paths")

CapabilityRegistry = getattr(capmod, "CapabilityRegistry")
cap = CapabilityRegistry()

# --- self-heal: ensure 'notify.push' exists on this instance ---
try:
    has_method = getattr(cap, "has", None)
    register_method = getattr(cap, "register", None)
    need_register = (has_method is None) or (not has_method("notify.push"))
    if need_register and callable(register_method):
        # Try both adapter locations
        _np = None
        try:
            _np = import_module(
                "backend.registry.adapters.notify_push"
            ).notify_push_adapter
        except Exception:
            try:
                _np = import_module("notify_push").notify_push_adapter
            except Exception:
                _np = None
        if _np:
            register_method("notify.push", _np)
except Exception:
    # Never crash runner on registry patch-up failures
    pass


def _json_safe(obj):
    return jsonable_encoder(obj, custom_encoder={set: list})


def _log_direct(row: dict):
    """Insert a row into agent_logs via the global Supabase client; prints errors."""
    try:
        sb.table("agent_logs").insert([_json_sanitize(row)]).execute()
    except Exception as e:
        print("[jobs_notifier] direct agent_logs insert failed:", e)


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def _resolve_capability(verb: str):
    """
    Always return a callable that goes through CapabilityRegistry.dispatch.
    This avoids relying on internal adapter maps (adapters/_adapters) or
    optional helper methods (get/resolve/adapter_for) that may not exist.
    The returned callable matches the jobs_runner expectation: fn(payload) -> dict.
    """
    if hasattr(cap, "dispatch") and callable(getattr(cap, "dispatch")):
        # Wrap dispatch so callers can keep doing: fn(payload)
        def _wrapped(payload: Dict[str, Any]):
            meta = {"source": "jobs_runner"}  # lightweight context
            return cap.dispatch(verb, payload or {}, meta)  # returns Envelope dict

        return _wrapped

    # Fallbacks for exotic registries (kept for compatibility)
    if hasattr(cap, "get") and callable(getattr(cap, "get")):
        return cap.get(verb)  # type: ignore[attr-defined]
    if hasattr(cap, "resolve") and callable(getattr(cap, "resolve")):
        return cap.resolve(verb)  # type: ignore[attr-defined]
    if hasattr(cap, "adapter_for") and callable(getattr(cap, "adapter_for")):
        return cap.adapter_for(verb)  # type: ignore[attr-defined]
    if hasattr(cap, "adapters"):
        adapters = getattr(cap, "adapters")
        if isinstance(adapters, dict) and verb in adapters and callable(adapters[verb]):
            return adapters[verb]
    if hasattr(cap, "_adapters"):
        adapters = getattr(cap, "_adapters")
        if isinstance(adapters, dict) and verb in adapters and callable(adapters[verb]):
            return adapters[verb]
    if hasattr(cap, "registry"):
        registry = getattr(cap, "registry")
        if isinstance(registry, dict) and verb in registry and callable(registry[verb]):
            return registry[verb]

    raise KeyError(f"verb '{verb}' not found in CapabilityRegistry")


# ---- small helpers ---------------------------------------------------------


def _safe_patch_jobs_run(run_id: str, patch: Dict[str, Any]) -> None:
    if not patch:
        return
    try:
        sb.table("jobs_runs").update(patch).eq("run_id", run_id).execute()
    except Exception as e:
        msg = str(e)
        if "PGRST204" in msg:
            print(f"[jobs_runner] jobs_runs patch (ignored): {e}")
        else:
            print(f"[jobs_runner] jobs_runs patch (warn): {e}")


def _new_run() -> str:
    payload = {"status": "running"}
    res = sb.table("jobs_runs").insert(payload).execute()
    run_id = res.data[0]["run_id"]
    return run_id


def _finish_run(run_id: str, ok: bool, summary: Optional[str] = None) -> None:
    patch = {"status": "succeeded" if ok else "failed"}
    if summary:
        patch["summary"] = summary  # ignored if column missing
    _safe_patch_jobs_run(run_id, patch)


def _execute_capability(verb: str, args: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    try:
        fn = _resolve_capability(verb)
    except KeyError:
        return {
            "ok": False,
            "result": None,
            "error": {
                "version": 1,
                "code": "UnknownVerb",
                "message": f"verb '{verb}' not found",
            },
            "latency_ms": 0,
        }

    call_args = dict(args or {})
    call_args.setdefault("run_id", run_id)

    # Ensure run_id on queue inserts/upserts
    if call_args.get("table") == "fetch_queue":
        rows = call_args.get("rows") or call_args.get("insert")
        if isinstance(rows, list):
            for r in rows:
                if isinstance(r, dict) and "run_id" not in r:
                    r["run_id"] = run_id

    t0 = time.time()
    out = fn(call_args)
    dt = int((time.time() - t0) * 1000)
    if isinstance(out, dict) and "latency_ms" not in out:
        out["latency_ms"] = dt
    return (
        out if isinstance(out, dict) else {"ok": True, "result": out, "latency_ms": dt}
    )


# ---- settings helpers ------------------------------------------------------


def _get_setting(agent_slug: str, key: str, default: Any = None) -> Any:
    try:
        res = (
            sb.table("agent_settings")
            .select("value")
            .eq("agent_slug", agent_slug)
            .eq("key", key)
            .limit(1)
            .execute()
        )
        if res.data:
            val = res.data[0]["value"]
            return val if val is not None else default
    except Exception as e:
        print(f"[jobs_runner] read setting {agent_slug}.{key} failed: {e}")
    return default


# ---- fetcher safety-net ----------------------------------------------------


def _fetcher_autopilot_from_claim(
    tool_result: Dict[str, Any], run_id: str
) -> Dict[str, Any]:
    """
    From a successful fetch_queue_claim tool_result, do:
      - http.fetch (or later: headless fallback)
      - write to fetched_pages
      - mark queue row done on success, or error on failure
    """
    rows = (tool_result or {}).get("rows") or (tool_result or {}).get("result") or []
    if not isinstance(rows, list) or not rows:
        return {"ok": True, "fetched": 0}

    fetched_count = 0
    for r in rows:
        url = r.get("url")
        source = r.get("source")
        row_id = r.get("id")
        if not url or not source or not row_id:
            _execute_capability(
                "db.write",
                {
                    "rpc": "fetch_queue_done",
                    "args": {
                        "id": row_id,
                        "status": "error",
                        "error": "Malformed claim row",
                    },
                },
                run_id,
            )
            continue

        # 1) Fetch
        fetch_out = _execute_capability(
            "http.fetch",
            {
                "url": url,
                "method": "GET",
                "timeout_ms": 20000,
                "allow_redirects": True,
                "max_bytes": 2_000_000,
                "headers": {"User-Agent": "personal-agent/1.0"},
            },
            run_id,
        )
        if not fetch_out.get("ok"):
            _execute_capability(
                "db.write",
                {
                    "rpc": "fetch_queue_done",
                    "args": {
                        "id": row_id,
                        "status": "error",
                        "error": str(fetch_out.get("error"))[:800],
                    },
                },
                run_id,
            )
            continue

        # Extract a text sample
        text_excerpt = None
        if isinstance(fetch_out.get("result"), dict):
            text_excerpt = fetch_out["result"].get("text_excerpt") or fetch_out[
                "result"
            ].get("text")
        if not text_excerpt:
            text_excerpt = fetch_out.get("text_excerpt") or ""

        # 2) Write to fetched_pages — try upsert first, then insert as fallback
        upsert_payload = {
            "table": "fetched_pages",
            "mode": "upsert",
            "on_conflict": "source,url",
            "rows": [
                {
                    "source": source,
                    "url": url,
                    "sample_text": text_excerpt,
                    "extracted": False,
                }
            ],
            "returning": "minimal",
        }
        write_out = _execute_capability("db.write", upsert_payload, run_id)
        if not write_out.get("ok"):
            insert_payload = {
                "table": "fetched_pages",
                "insert": [
                    {
                        "source": source,
                        "url": url,
                        "sample_text": text_excerpt,
                        "extracted": False,
                    }
                ],
                "on_conflict": "source,url",
                "returning": "minimal",
            }
            write_out = _execute_capability("db.write", insert_payload, run_id)

        # 3) Mark queue row and count
        if write_out.get("ok"):
            _execute_capability(
                "db.write",
                {"rpc": "fetch_queue_done", "args": {"id": row_id, "status": "done"}},
                run_id,
            )
            fetched_count += 1
        else:
            err_txt = str(write_out.get("error") or write_out)[:800]
            _execute_capability(
                "db.write",
                {
                    "rpc": "fetch_queue_done",
                    "args": {"id": row_id, "status": "error", "error": err_txt},
                },
                run_id,
            )

    return {"ok": True, "fetched": fetched_count}


# --- capability: assistant_text ----------------------------------------------
def cap_assistant_text(
    agent_slug: str, args: Dict[str, Any], run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Write a human summary string into agent_logs.
    Creates a row with role='assistant', verb='assistant_text',
    and content JSON: {"text": "..."}.
    """
    text = ((args or {}).get("text") or "").strip()

    _log_direct(
        {
            "agent_slug": agent_slug,
            "verb": "assistant_text",
            "ok": True,
            "role": "assistant",
            "step": 0,
            "args_json": None,
            "result_json": None,
            "content": {"text": text},
            "run_id": run_id,
        }
    )

    return {"ok": True}


# -----------------------------------------------------------------------------


def _dispatch_capability(
    *,
    agent_slug: Optional[str],
    verb: str,
    args: Dict[str, Any],
    run_id: str,
    verbs_allowlist: Optional[List[str]] = None,
) -> Dict[str, Any]:
    # 0) allowlist gate
    if verbs_allowlist and verb not in set(verbs_allowlist):
        return {
            "ok": False,
            "result": None,
            "error": {
                "version": 1,
                "code": "VerbNotAllowed",
                "message": f"{agent_slug or 'agent'} is not allowed to call '{verb}'",
            },
            "latency_ms": 0,
        }

    # 1) fetch adapter fn
    try:
        fn = _resolve_capability(verb)
    except KeyError:
        return {
            "ok": False,
            "result": None,
            "error": {
                "version": 1,
                "code": "UnknownVerb",
                "message": f"verb '{verb}' not found",
            },
            "latency_ms": 0,
        }

    # 2) clone & inject run_id
    payload = dict(args or {})
    payload.setdefault("run_id", run_id)

    # 3) light payload normalization by verb
    if verb == "db.write":
        table = payload.get("table")
        rows = payload.get("rows") or payload.get("insert")
        if isinstance(rows, list):
            for r in rows:
                if isinstance(r, dict) and "run_id" not in r:
                    r["run_id"] = run_id

        if "mode" not in payload and "insert" not in payload:
            if table in {
                "fetch_queue",
                "fetched_pages",
                "jobs_normalized",
                "jobs_scores",
            }:
                payload["mode"] = "upsert"
                if "rows" not in payload and rows:
                    payload["rows"] = rows
                payload.setdefault("on_conflict", "source,url")
                payload.setdefault("returning", "minimal")

    elif verb == "http.fetch":
        payload.setdefault("method", "GET")
        payload.setdefault("timeout_ms", 20000)
        payload.setdefault("max_bytes", 2_000_000)
        headers = dict(payload.get("headers") or {})
        headers.setdefault("User-Agent", "personal-agent/1.0")
        payload["headers"] = headers

    # 4) execute with timing
    t0 = time.time()
    out = fn(payload)
    latency_ms = int((time.time() - t0) * 1000)

    def _normalize_ret(o):
        if isinstance(o, dict):
            o.setdefault("latency_ms", latency_ms)
            return o
        return {"ok": True, "result": o, "latency_ms": latency_ms}

    out = _normalize_ret(out)

    # 5) gentle db.write fallback between upsert/insert shapes
    if verb == "db.write" and not out.get("ok"):
        tbl = payload.get("table")
        rows = payload.get("rows") or payload.get("insert")
        err_txt = str(out.get("error") or out)
        if isinstance(rows, list) and tbl:
            if payload.get("mode") == "upsert":
                alt_payload = {
                    "table": tbl,
                    "insert": rows,
                    "on_conflict": payload.get("on_conflict"),
                    "returning": payload.get("returning", "minimal"),
                    "run_id": run_id,
                }
                alt = fn(alt_payload)
                alt = _normalize_ret(alt)
                if alt.get("ok"):
                    return alt
            if "insert" in payload and "rows" not in payload:
                alt_payload = {
                    "table": tbl,
                    "mode": "upsert",
                    "rows": rows,
                    "on_conflict": payload.get("on_conflict", "source,url"),
                    "returning": payload.get("returning", "minimal"),
                    "run_id": run_id,
                }
                alt = fn(alt_payload)
                alt = _normalize_ret(alt)
                if alt.get("ok"):
                    return alt
        out.setdefault("error_text", err_txt[:800])
        return out

    return out


def _json_sanitize(obj):
    """
    Convert non-JSON types (e.g., set, bytes, UUID, datetime) into JSON-safe forms.
    - set/tuple -> list
    - bytes -> str (utf-8 with replacement)
    - UUID -> str
    - datetime/date -> isoformat
    - fallback -> str(obj)
    """
    import uuid
    import datetime

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [_json_sanitize(x) for x in list(obj)]
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return repr(obj)
    if isinstance(obj, (uuid.UUID,)):
        return str(obj)
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    # last resort
    return str(obj)


# ---- agent loop ------------------------------------------------------------


def _agent_loop(
    agent_slug: str, initial_user_text: str, run_id: str, max_steps: int = 20
) -> Dict[str, Any]:
    tool_result: Optional[Dict[str, Any]] = None
    last_assistant: Optional[Dict[str, Any]] = None
    insist_left = (
        2  # retries when assistant returns plain text (no tool call, no finish)
    )

    # Pull an allowlist (if present) to enforce at dispatch time
    verbs_allowlist = _get_setting(agent_slug, "verbs_allowlist", None)

    for step in range(1, max_steps + 1):
        if tool_result is None:
            user_text = initial_user_text
        else:
            # SANITIZE before embedding to JSON string
            safe_tool_result = _json_sanitize(tool_result)
            user_text = (
                "Continue with your task. Here is tool_result from the previous step:\n"
                + json.dumps(safe_tool_result, ensure_ascii=False)
            )

        assistant_json = run_llm_agent(agent_slug=agent_slug, user_text=user_text)
        last_assistant = (
            assistant_json
            if isinstance(assistant_json, dict)
            else {"assistant_text": str(assistant_json)}
        )

        # Finish?
        if isinstance(assistant_json, dict) and "finish" in assistant_json:
            return assistant_json

        # Tool call?
        verb = assistant_json.get("verb") if isinstance(assistant_json, dict) else None
        args = assistant_json.get("args") if isinstance(assistant_json, dict) else None

        # If no tool call AND no finish, insist once or twice.
        if not verb:
            if insist_left > 0:
                insist_left -= 1
                # Force a single tool call next turn — keep lists (NOT sets)
                tool_result = {
                    "ok": False,
                    "error": "No tool call returned. You must respond with STRICT JSON using one of your tools.",
                    "hint": {
                        "allowed_verbs": [
                            "db.read",
                            "db.write",
                            "http.fetch",
                            "browser.adapter",
                            "notify.push",
                        ],
                        "format": {"verb": "<one_of_allowed>", "args": {"...": "..."}},
                    },
                }
                continue
            # Give up and return whatever we have
            return last_assistant or {}

        # Execute the tool
        exec_out = _dispatch_capability(
            agent_slug=agent_slug,
            verb=verb,
            args=args or {},
            run_id=run_id,
            verbs_allowlist=verbs_allowlist,
        )

        print(
            f"[jobs_runner] {agent_slug} -> {json.dumps({'verb': verb, 'args': args}, ensure_ascii=False)}"
        )

        # Fetcher autopilot
        if (
            agent_slug == "jobs_fetcher"
            and verb == "db.read"
            and ((args or {}).get("rpc") == "fetch_queue_claim")
            and exec_out.get("ok") is True
            and (exec_out.get("rows") or exec_out.get("result"))
        ):
            auto = _fetcher_autopilot_from_claim(exec_out, run_id)
            tool_result = {"ok": True, "auto_fetched": auto.get("fetched", 0)}
            try:
                f = int(auto.get("fetched", 0))
                if f:
                    _safe_patch_jobs_run(run_id, {"fetched": f})
            except Exception:
                pass
            continue

        tool_result = exec_out

    return last_assistant or {}


# ---- notifier / overseer helpers ------------------------------------------


def _format_job_line(row: Dict[str, Any]) -> str:
    title = row.get("title") or ""
    company = row.get("company") or ""
    location = row.get("location") or ""
    score = row.get("score")
    url = row.get("url") or ""
    left = f"{title} — {company}".strip(" —")
    loc = f" ({location})" if location else ""
    sc = f" [{score}]" if score is not None else ""
    bullet = f"• {left}{loc}{sc}\n{url}".strip()
    return bullet


def _notifier_fallback_push(shortlist: List[str], run_id: str) -> Dict[str, Any]:
    """
    Deterministic notifier path used when the LLM didn't emit a notify.push call.
    1) If shortlist is empty, backfill from jobs_scores using min_score/max_items settings.
    2) Compose a compact message from jobs_normalized metadata (fallback to plain URLs).
    3) Log intent, dispatch notify.push via capability registry, log result, and patch jobs_runs.
    """

    # --- helpers --------------------------------------------------------------
    def _coerce_int(v, default):
        try:
            return int(v)
        except Exception:
            try:
                return int(float(v))
            except Exception:
                return default

    # --- (1) backfill from jobs_scores when shortlist is empty ----------------
    try:
        _min_score_raw = _get_setting("jobs_notifier", "min_score", 80)
        _max_items_raw = _get_setting("jobs_notifier", "max_items", 5)
        min_score = _coerce_int(_min_score_raw, 80)
        max_items = _coerce_int(_max_items_raw, 5)

        if not shortlist:
            res = (
                sb.table("jobs_scores")
                .select("source,url,score,created_at")
                .gte("score", min_score)
                .order("created_at", desc=True)
                .limit(max_items)
                .execute()
            )
            rows = (
                (getattr(res, "data", None) or res.get("data") or [])
                if isinstance(res, dict) is False
                else (res.get("data") or [])
            )
            shortlist = [r.get("url") for r in rows if r.get("url")]

            _log_direct(
                {
                    "agent_slug": "jobs_notifier",
                    "verb": "notifier.backfill",
                    "ok": True,
                    "args_json": {"min_score": min_score, "max_items": max_items},
                    "result_json": {"rows": rows, "shortlist_len": len(shortlist)},
                    "run_id": run_id,
                }
            )
    except Exception as e:
        _log_direct(
            {
                "agent_slug": "jobs_notifier",
                "verb": "notifier.backfill.error",
                "ok": False,
                "args_json": {"hint": "jobs_scores read failed or RLS"},
                "result_json": {"error": str(e)},
                "run_id": run_id,
            }
        )
        # Do not abort; continue with whatever shortlist we have.

    # --- read configs (channel, min_score/max_items for compose/push limits) ---
    channel = _get_setting("jobs_notifier", "channel", "jobs")
    _min_score_raw = _get_setting("jobs_notifier", "min_score", 80)
    _max_items_raw = _get_setting("jobs_notifier", "max_items", 5)
    min_score = _coerce_int(_min_score_raw, 80)
    max_items = _coerce_int(_max_items_raw, 5)

    # --- (2) compose text from shortlist (enrich from jobs_normalized) -------
    lines: List[str] = []
    details_by_url: Dict[str, Dict[str, Any]] = {}

    if shortlist:
        try:
            res = (
                sb.table("jobs_normalized")
                .select("source,url,title,company,location,work_mode")
                .in_("url", shortlist)
                .execute()
            )
            if hasattr(res, "data") and isinstance(res.data, list):
                details_by_url = {r.get("url"): r for r in res.data if r.get("url")}
            else:
                details_by_url = {}
        except Exception as e:
            details_by_url = {}
            _log_direct(
                {
                    "agent_slug": "jobs_notifier",
                    "verb": "notifier.compose.warn",
                    "ok": False,
                    "args_json": {"hint": "jobs_normalized lookup failed"},
                    "result_json": {"error": str(e)},
                    "run_id": run_id,
                }
            )

        for u in shortlist[:max_items]:
            r = details_by_url.get(u) or {}
            title = (r.get("title") or "").strip()
            company = (r.get("company") or "").strip()
            location = (r.get("location") or "").strip()
            wm = (r.get("work_mode") or "").strip()

            if title or company or location or wm:
                parts = []
                if title:
                    parts.append(title)
                if company:
                    parts.append(company)
                locwm = ", ".join([p for p in [location, wm] if p])
                if locwm:
                    parts.append(f"({locwm})")
                header = " — ".join(parts) if parts else "Role"
                line = f"• {header}\n{u}"
            else:
                line = f"• {u}"

            lines.append(line.strip())

    text = "\n".join(lines).strip()

    _log_direct(
        {
            "agent_slug": "jobs_notifier",
            "verb": "notifier.compose",
            "ok": True,
            "args_json": {"shortlist_len": len(shortlist)},
            "result_json": {"lines": len(lines), "text_len": len(text)},
            "run_id": run_id,
        }
    )

    # If nothing to send, record a dry-run and stop.
    if not lines or not text:
        _log_direct(
            {
                "agent_slug": "jobs_notifier",
                "verb": "notify.push.dryrun",
                "ok": True,
                "role": "assistant",
                "step": 0,
                "args_json": {
                    "reason": "no composed lines",
                    "min_score": min_score,
                    "shortlist": shortlist,
                },
                "result_json": {"sent": 0},
                "run_id": run_id,
            }
        )
        return {"ok": True, "finish": {"sent": 0, "reason": "no composed lines"}}

    # --- (3) intent → push via capability registry → post-log ---------------
    push_args = {"channel": channel, "title": "Top roles", "text": text}

    _log_direct(
        {
            "agent_slug": "jobs_notifier",
            "verb": "notify.push.intent",
            "ok": True,
            "role": "assistant",
            "step": 0,
            "args_json": push_args,
            "run_id": run_id,
        }
    )

    push: Dict[str, Any] = {}
    try:
        push = (
            _dispatch_capability(
                agent_slug="jobs_notifier",
                verb="notify.push",
                args=push_args,
                run_id=run_id,
                verbs_allowlist=_get_setting(
                    "jobs_notifier", "verbs_allowlist", ["db.read", "notify.push"]
                ),
            )
            or {}
        )
    except Exception as e:
        push = {"ok": False, "error": f"notify.push exception: {e}"}

    # If registry couldn’t find the verb, call the adapter directly as a last resort.
    if not push.get("ok"):
        err_txt = (push.get("error") or push.get("code") or "").lower()
        if "unknownverb" in err_txt or "verb 'notify.push' not found" in err_txt:
            try:
                try:
                    _np = import_module(
                        "backend.registry.adapters.notify_push"
                    ).notify_push_adapter
                except Exception:
                    _np = import_module("notify_push").notify_push_adapter
                # adapter returns {"ok": bool, ...}
                push = _np(push_args)
            except Exception as e:
                push = {"ok": False, "error": f"notify.push direct-call exception: {e}"}

    _log_direct(
        {
            "agent_slug": "jobs_notifier",
            "verb": "notify.push",
            "ok": bool(push.get("ok")),
            "role": "assistant",
            "step": 0,
            "args_json": push_args,
            "result_json": _json_sanitize(push),
            "run_id": run_id,
        }
    )

    sent = len(lines) if push.get("ok") else 0
    if sent:
        _safe_patch_jobs_run(run_id, {"notified_count": sent})
    else:
        err_txt = str(push.get("error") or push.get("error_text") or push)[:400]
        _safe_patch_jobs_run(run_id, {"summary": f"notify.push failed: {err_txt}"})

    return {"ok": True, "finish": {"sent": sent}}


def _run_notifier(shortlist: List[str], run_id: str) -> Dict[str, Any]:
    """
    Kick the jobs_notifier agent with the shortlist context.
    If the LLM doesn't actually push, we do a deterministic fallback push.
    """
    if not shortlist:
        payload_text = "Proceed with your task using the rules above. Shortlist is empty for this run."
    else:
        payload_text = (
            "Proceed with your task using the rules above. "
            "Here is the shortlist of URLs to notify about:\n" + json.dumps(shortlist)
        )

    _log_direct(
        {
            "agent_slug": "jobs_notifier",
            "verb": "notifier.entry",
            "ok": True,
            "args_json": {"note": "pre LLM"},
            "run_id": run_id,
        }
    )

    out = _agent_loop("jobs_notifier", payload_text, run_id)

    # Count LLM-driven sends if it returned a finish
    sent = 0
    if isinstance(out, dict):
        fin = out.get("finish") or {}
        sent = int(fin.get("sent", fin.get("notified", 0)) or 0)

    # Fallback (deterministic) if nothing was sent
    if sent <= 0:
        out = _notifier_fallback_push(shortlist, run_id)

    # Patch notified_count if we have something
    try:
        fin = out.get("finish", {}) if isinstance(out, dict) else {}
        notified = int(fin.get("sent", fin.get("notified", 0)) or 0)
        if notified:
            _safe_patch_jobs_run(run_id, {"notified_count": notified})
    except Exception:
        pass
    return out


def _run_overseer(
    run_id: str,
    disc_out: Dict[str, Any],
    fetch_out: Dict[str, Any],
    extr_out: Dict[str, Any],
    score_out: Dict[str, Any],
    notifier_out: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Kick the jobs_overseer agent with the overall state of the run so it can summarize,
    and persist a human-readable report row into public.jobs_overseer_reports.
    """

    def _safe_int(x, *keys, default=0):
        try:
            cur = x or {}
            for k in keys:
                if cur is None:
                    return default
                cur = cur.get(k)
            return int(cur) if cur is not None else default
        except Exception:
            return default

    # Pre-digest best-effort metrics so the overseer has a quick view.
    discoveries = _safe_int(disc_out, "finish", "discoveries", default=0)
    fetched = _safe_int(fetch_out, "finish", "fetched", default=0)
    extracted = _safe_int(extr_out, "finish", "extracted", default=0)
    upserted = _safe_int(extr_out, "finish", "inserted", default=0) or _safe_int(
        extr_out, "finish", "upserted", default=0
    )
    shortlist_count = len(
        (score_out or {}).get("finish", {}).get("shortlist", []) or []
    )
    # Greedy fallback; the overseer should verify via notifications_log (see instructions below)
    notified_fallback = _safe_int(notifier_out, "finish", "sent", default=0)

    context = {
        "run_id": str(run_id),
        "metrics": {
            "discoveries": discoveries,
            "fetched": fetched,
            "extracted": extracted,
            "upserted": upserted,
            "shortlist": shortlist_count,
            "notified_fallback": notified_fallback,
        },
        "discoverer_finish": (disc_out or {}).get("finish"),
        "fetcher_finish": (fetch_out or {}).get("finish"),
        "extractor_finish": (extr_out or {}).get("finish"),
        "scorer_finish": (score_out or {}).get("finish"),
        "notifier_finish": (notifier_out or {}).get("finish"),
    }

    # Clear, strict directives for the overseer agent:
    # - Read latest jobs_runs + recent tables for audit
    # - Derive "notified" by reading notifications_log and counting bullet lines (• or "- ")
    # - Then write ONE row into jobs_overseer_reports with agent_slug set
    # - Use strict JSON tool calls (one per message)
    user_text = (
        "You are the OVERSEER. Follow your system_prompt. "
        "Important additions for this run:\n"
        "1) Compute a truthful 'notified' count by reading public.notifications_log (latest row), "
        "   counting bullet lines in its text that start with '•' or '- '. If notifications_log is empty, "
        f"   fall back to this hint: notified_fallback={notified_fallback}.\n"
        "2) Persist a summary via db.write(table='jobs_overseer_reports', mode='insert') including: "
        "   {run_id, agent_slug:'jobs_overseer', title:'Pipeline summary', text:'<human-readable lines>'}. "
        "   Use returning='minimal'.\n"
        "3) Use STRICT JSON tool calls only (one call per assistant message). "
        "   No prose unless inside the final 'text' string in the db.write row.\n"
        "4) Suggested text format:\n"
        '   "Run <RUN_ID> — discoveries: <N>; fetched: <N>; extracted: <N>; upserted: <N>; '
        'shortlist: <N>; notified: <N>."\n'
        "5) If you notify.push, keep it compact; otherwise it's optional.\n"
        "Here is the summary JSON from this run so far:\n"
        + json.dumps(context, ensure_ascii=False)
    )

    out = _agent_loop("jobs_overseer", user_text, run_id)

    # Best-effort: if the agent returns a human summary string, attempt to store on jobs_runs
    # via existing helper; this will safely no-op if the column doesn't exist.
    try:
        fin = out.get("finish", {}) if isinstance(out, dict) else {}
        summary = fin.get("summary") or fin.get("notes") or None
        if summary:
            _finish_run(run_id, ok=True, summary=str(summary)[:2000])
    except Exception:
        pass

    return out


def _overseer_fallback_persist(
    run_id: str,
    disc_out: Dict[str, Any],
    fetch_out: Dict[str, Any],
    extr_out: Dict[str, Any],
    score_out: Dict[str, Any],
    notifier_out: Dict[str, Any],
    overseer_out: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    If the overseer didn't persist a report, insert a compact human-readable summary
    into public.jobs_overseer_reports. Tolerates missing table/schemas.
    """
    # 1) If a report already exists for this run, no-op.
    try:
        res = (
            sb.table("jobs_overseer_reports")
            .select("id")
            .eq("run_id", run_id)
            .limit(1)
            .execute()
        )
        if getattr(res, "data", None):
            return {"ok": True, "skipped": "exists"}
    except Exception as e:
        # If table doesn't exist (or schema cache issues), just bail quietly.
        return {"ok": False, "skipped": "check-failed", "error": str(e)}

    def _fin(d):
        return d.get("finish", {}) if isinstance(d, dict) else {}

    def _int(d, key):
        try:
            return int((_fin(d).get(key, 0) or 0))
        except Exception:
            return 0

    # 2) Build a compact line summary
    discoveries = _int(disc_out, "discoveries")
    fetched = _int(fetch_out, "fetched")
    extracted = _int(extr_out, "extracted")
    upserted = _int(extr_out, "inserted") or _int(extr_out, "upserted")

    fin_sc = _fin(score_out)
    shortlist = fin_sc.get("shortlist", []) if isinstance(fin_sc, dict) else []
    shortlist_count = len(shortlist) if isinstance(shortlist, list) else 0

    fin_not = _fin(notifier_out)
    try:
        notified = int(fin_not.get("sent", fin_not.get("notified", 0)) or 0)
    except Exception:
        notified = 0

    title = "Pipeline summary"
    text = (
        f"Run {run_id} — "
        f"discoveries: {discoveries}; fetched: {fetched}; extracted: {extracted}; "
        f"upserted: {upserted}; shortlist: {shortlist_count}; notified: {notified}."
    )

    # 3) Insert the fallback report
    try:
        sb.table("jobs_overseer_reports").insert(
            [{"run_id": run_id, "title": title, "text": text}]
        ).execute()
        return {"ok": True, "inserted": 1}
    except Exception as e:
        # Don't fail the whole run if we can't persist this; just surface the error.
        return {"ok": False, "error": str(e)}


# ---- public entry -----------------------------------------------------------


def run_once(_: Dict[str, Any]) -> Dict[str, Any]:
    run_id = _new_run()
    os.environ["CURRENT_RUN_ID"] = run_id

    try:
        if hasattr(cap, "warmup"):
            try:
                _ = cap.warmup(
                    [
                        "db.read",
                        "db.write",
                        "http.fetch",
                        "browser.adapter",
                        "notify.push",
                    ]
                )
            except NotImplementedError:
                pass
            except Exception as e:
                print(f"[jobs_runner] warmup (ignored): {e}")

        # DISCOVER
        disc_out = _agent_loop(
            "jobs_discoverer", "Proceed with your task using the rules above.", run_id
        )
        try:
            discoveries = int(disc_out.get("finish", {}).get("discoveries", 0))
        except Exception:
            discoveries = 0
        _safe_patch_jobs_run(run_id, {"discoveries": discoveries})

        # FETCH
        fetch_out = _agent_loop(
            "jobs_fetcher", "Proceed with your task using the rules above.", run_id
        )
        try:
            fetched = int(fetch_out.get("finish", {}).get("fetched", 0))
        except Exception:
            fetched = 0
        if fetched:
            _safe_patch_jobs_run(run_id, {"fetched": fetched})

        # EXTRACT
        extr_out = _agent_loop(
            "jobs_extractor", "Proceed with your task using the rules above.", run_id
        )
        fin_ex = extr_out.get("finish", {}) if isinstance(extr_out, dict) else {}
        extracted = int(fin_ex.get("extracted", 0) or 0)
        inserted = int(fin_ex.get("inserted", fin_ex.get("upserted", 0)) or 0)
        _safe_patch_jobs_run(run_id, {"extracted": extracted, "upserted": inserted})

        # SCORE
        score_out = _agent_loop(
            "jobs_scorer", "Proceed with your task using the rules above.", run_id
        )
        fin_sc = score_out.get("finish", {}) if isinstance(score_out, dict) else {}
        shortlist = fin_sc.get("shortlist", []) if isinstance(fin_sc, dict) else []
        _safe_patch_jobs_run(run_id, {"shortlist_count": len(shortlist)})

        # NOTIFY (LLM-first; deterministic fallback ensures pushes happen)
        notifier_out = _run_notifier(
            shortlist if isinstance(shortlist, list) else [], run_id
        )

        # OVERSEER
        overseer_out = _run_overseer(
            run_id, disc_out, fetch_out, extr_out, score_out, notifier_out
        )

        # Ensure a persisted summary always exists even if the LLM didn't write one
        _ = _overseer_fallback_persist(
            run_id, disc_out, fetch_out, extr_out, score_out, notifier_out, overseer_out
        )

        _finish_run(run_id, ok=True)
        return {
            "run_id": run_id,
            "discoverer": disc_out,
            "fetcher": fetch_out,
            "extractor": extr_out,
            "scorer": score_out,
            "notifier": notifier_out,
            "overseer": overseer_out,
        }
    except Exception as e:
        _finish_run(run_id, ok=False, summary=str(e))
        raise
