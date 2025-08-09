from __future__ import annotations
from typing import Any, Dict, List, Optional
from importlib import import_module
from services.supabase_service import supabase

# --- tiny event emitter (idempotent by key) ---
def emit(topic: str, payload: dict, source_agent: str, idem_key: Optional[str] = None):
    row = {"topic": topic, "payload": payload, "source_agent": source_agent, "idempotency_key": idem_key}
    try:
        supabase.table("events").insert(row).execute()
    except Exception as e:
        if "duplicate key" in str(e).lower():
            return  # idempotent
        raise

# --- helpers ---
def _get_settings(slug: str) -> dict:
    try:
        res = supabase.table("agent_settings").select("key,value").eq("agent_slug", slug).execute()
        pairs = res.data or []
        return {p["key"]: p["value"] for p in pairs if p.get("key") is not None}
    except Exception:
        return {}

def _get(d: dict, path: str, default=None):
    """Dot-path fetch: 'set.status' -> d['set']['status'] if present."""
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def _render(template: Any, row: dict, op: dict, ctx: dict):
    """
    Render a payload template:
    - strings: "${row.id}" or "${op.table}" or "${ctx.user_text}"
    - dict/list: recurse
    - primitives: pass-through
    """
    if isinstance(template, str):
        out = template
        for key, val in (("row.", row), ("op.", op), ("ctx.", ctx)):
            if isinstance(val, dict):
                for k, v in val.items():
                    out = out.replace("${" + key + k + "}", str(v))
        return out
    if isinstance(template, dict):
        return {k: _render(v, row, op, ctx) for k, v in template.items()}
    if isinstance(template, list):
        return [_render(v, row, op, ctx) for v in template]
    return template

# --- main generic post hook ---
def post_publish(agent_slug: str, user_text: str, data: dict) -> dict:
    """
    Reads rules from agent_settings['bus.publish_rules'] and emits events.
    Rule shape (list):
    {
      "when": { "table": "meal_plan", "op": "update", "equals": {"set.status":"done"} },
      "emit": {
        "topic": "meal.done",
        "idempotency_key": "meal.done:${row.id}",
        "payload": {"meal_plan_id": "${row.id}", "source": "${agent}", "note": "${ctx.user_text}"}
      }
    }

    The rule matches an operation at index i and applies to each dict row
    returned in results[i].
    """
    settings = _get_settings(agent_slug)
    rules: List[dict] = settings.get("bus.publish_rules") or []
    if not rules:
        return data

    ops: List[dict] = data.get("operations") or []
    results: List[Any] = data.get("results") or []
    ctx = {"user_text": user_text, "agent": agent_slug}

    for i, op in enumerate(ops):
        op = op or {}
        res = results[i] if i < len(results) else None
        # normalize rows list
        rows = res if isinstance(res, list) else ([res] if isinstance(res, dict) else [])
        for rule in rules:
            cond = rule.get("when") or {}
            if cond.get("op") and str(cond["op"]).lower() != str(op.get("op","")).lower():
                continue
            if cond.get("table") and cond["table"] != op.get("table"):
                continue
            ok = True
            for k, v in (cond.get("equals") or {}).items():
                if _get(op, k) != v:
                    ok = False; break
            if not ok:
                continue

            emit_spec = rule.get("emit") or {}
            topic = emit_spec.get("topic")
            if not topic: 
                continue
            for row in (rows or [{}]):
                payload_tpl = emit_spec.get("payload") or {}
                idem_tpl = emit_spec.get("idempotency_key")
                payload = _render(payload_tpl, row or {}, op, ctx)
                idem = _render(idem_tpl, row or {}, op, ctx) if idem_tpl else None
                emit(topic, payload, source_agent=agent_slug, idem_key=idem)

    return data
