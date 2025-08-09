# backend/agents/finance_agent.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from decimal import Decimal
import json, logging, re
from datetime import datetime, timezone

from services.supabase_service import supabase
from utils.agent_protocol import make_response, AgentResponse
from reasoner.policy import reason_with_memory

logger = logging.getLogger("finance")

# ------------------------------------------------------------------------------
# Agent metadata (used for self-registration + router discovery)
# ------------------------------------------------------------------------------
AGENT_META = {
    "slug": "finance",
    "title": "Finance",
    "description": "Personal finance: expenses, budgets, hypothetical changes, ledger queries.",
    "handler_key": "finance.handle",          # maps to router HANDLERS registry
    "namespaces": ["expenses"],
    "capabilities": [
        "List and search transactions",
        "Summarize spending by time window",
        "Propose hypothetical budgets",
        "Insert/update/delete ledger rows",
    ],
    "keywords": ["expense","expenses","spend","spent","budget","purchase","money","cost"],
    "status": "enabled",
}

# ------------------------------------------------------------------------------
# Utilities for safe self-registration
# ------------------------------------------------------------------------------
def _table_columns(table: str) -> List[str]:
    try:
        sample = supabase.table(table).select("*").limit(1).execute().data or []
        return list(sample[0].keys()) if sample else []
    except Exception:
        return []

def _payload_with_existing_cols(table: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    cols = set(_table_columns(table))
    if not cols:
        # best-effort defaults for common columns
        cols = {"slug","title","description","handler_key","status"}
    return {k: v for k, v in payload.items() if k in cols}

def _register_agent_if_needed() -> None:
    """Idempotent agent registration that only inserts columns that exist."""
    try:
        slug = AGENT_META["slug"]
        exists = supabase.table("agents").select("id").eq("slug", slug).limit(1).execute().data
        if exists:
            return
        payload = _payload_with_existing_cols("agents", {
            "slug": slug,
            "title": AGENT_META["title"],
            "description": AGENT_META["description"],
            "handler_key": AGENT_META["handler_key"],
            "namespaces": AGENT_META.get("namespaces"),
            "capabilities": AGENT_META.get("capabilities"),
            "keywords": AGENT_META.get("keywords"),
            "status": AGENT_META.get("status", "enabled"),
        })
        supabase.table("agents").insert(payload).execute()
        logger.info("[finance] self-registered agent in Supabase")
    except Exception:
        logger.exception("[finance] agent self-registration failed (safe to ignore in dev)")

# Call once on import
_register_agent_if_needed()

# ------------------------------------------------------------------------------
# Instruction loading (core + optional tags) from Supabase
# ------------------------------------------------------------------------------
FALLBACK_SYSTEM = """You are a finance operator with autonomy to plan and act on the database.
Return ONLY minified JSON with keys: "thoughts", "operations", and optional "response_template".
Supported ops: select/update/insert/delete/upsert with fields: op, table, where (dict or simple AND string), order, limit, values, set.
Use amount < 0 to mean expenses. Do NOT refer to a 'category' column unless it exists in the schema hint.
When inferring recurrence: weekly/biweekly/monthly/bimonthly/quarterly/semiannual/annual by gaps across 6–12 months; only update when confident.
"""

def _get_instruction_row(agent_name: str, tag: str) -> Optional[str]:
    try:
        res = (
            supabase.table("agent_instructions")
            .select("instructions")
            .eq("agent_name", agent_name)
            .eq("tag", tag)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if res.data:
            return res.data[0]["instructions"]
    except Exception:
        logger.exception("[finance] load instructions failed")
    return None

def _get_core_instructions() -> str:
    return _get_instruction_row("finance", "core") or FALLBACK_SYSTEM

def _choose_extra_tags(user_text: str) -> List[str]:
    """Lightweight LLM assist to choose up to 3 extra instruction tags."""
    try:
        prompt = f"""You are selecting instruction tags for the finance agent.
User query: {user_text}

Given tags: ["recurring-expenses","categorization","budgets","grooming-link","query-parsing","reporting"]
Return ONLY a comma-separated list (0-3 items)."""
        raw = reason_with_memory(agent_name="router", query=prompt, namespace="routing", k=2)
        if not isinstance(raw, str):
            return []
        tags = [t.strip() for t in raw.split(",") if t.strip()]
        allowed = {"recurring-expenses","categorization","budgets","grooming-link","query-parsing","reporting"}
        return [t for t in tags if t in allowed][:3]
    except Exception:
        logger.exception("[finance] tag selection failed")
        return []

def _get_relevant_instructions(user_text: str) -> str:
    core = _get_core_instructions()
    extras: List[str] = []
    for tag in _choose_extra_tags(user_text):
        row = _get_instruction_row("finance", tag)
        if row:
            extras.append(row)
    return core + ("\n\n" + "\n\n".join(extras) if extras else "")

# ------------------------------------------------------------------------------
# Schema hint (ONLY real columns you actually have)
# ------------------------------------------------------------------------------
def _introspect_schema_hint() -> str:
    def cols(table: str):
        try:
            sample = supabase.table(table).select("*").limit(1).execute().data or []
            return list(sample[0].keys()) if sample else []
        except Exception:
            return []

    hint = {
        "tables": {
            "finance_ledger": {"columns": cols("finance_ledger")},
            "budgets": {"columns": cols("budgets")},
        }
    }
    return json.dumps(hint)



def _hint_columns(table: str) -> List[str]:
    try:
        hint = json.loads(_introspect_schema_hint())
        return (hint.get("tables", {}).get(table, {}) or {}).get("columns", []) or []
    except Exception:
        return []

def _real_columns(table: str) -> List[str]:
    cols = _table_columns(table)
    return cols if cols else _hint_columns(table)

# ------------------------------------------------------------------------------
# WHERE helpers (accept dict / list / simple string)
# ------------------------------------------------------------------------------
def _prune_where(where: Any, columns: List[str]) -> Any:
    if not where or not columns:
        return where
    cols = set(columns)

    if isinstance(where, str):
        parts = re.split(r"\s+and\s+", where, flags=re.I)
        kept = []
        for cond in parts:
            m = re.match(r"\s*([a-zA-Z0-9_]+)\s*=\s*(.+)\s*$", cond)
            if m and m.group(1) in cols:
                kept.append(cond)
        return " AND ".join(kept) if kept else None

    if isinstance(where, list):
        pruned = [w for w in (_prune_where(w, columns) for w in where) if w]
        return pruned or None

    if isinstance(where, dict):
        out = {}
        for k, v in where.items():
            if k in cols:
                out[k] = v
        return out or None

    return None

def _apply_where(q, where: Any):
    if not where:
        return q

    if isinstance(where, str):
        parts = re.split(r"\s+and\s+", where, flags=re.I)
        for cond in parts:
            m = re.match(r"\s*([a-zA-Z0-9_]+)\s*=\s*(.+)\s*$", cond)
            if not m:
                continue
            key, val = m.group(1), m.group(2).strip()
            if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
                val = val[1:-1]
            q = q.eq(key, val)
        return q

    if isinstance(where, list):
        for w in where:
            q = _apply_where(q, w)
        return q

    if isinstance(where, dict):
        for k, v in where.items():
            if isinstance(v, dict):
                for oper, val in v.items():
                    o = str(oper).strip()
                    if o == ">=": q = q.gte(k, val)
                    elif o == "<=": q = q.lte(k, val)
                    elif o == ">":  q = q.gt(k, val)
                    elif o == "<":  q = q.lt(k, val)
                    elif o == "!=": q = q.neq(k, val)
                    elif o == "like": q = q.like(k, val)
                    else: q = q.eq(k, val)
            else:
                q = q.eq(k, v)
        return q

    return q

# ------------------------------------------------------------------------------
# Transform helper (e.g., {"set":{"amount__mul":0.9}})
# ------------------------------------------------------------------------------
def _apply_transforms(table: str, step: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    setv = dict(step.get("set") or {})
    for k in list(setv.keys()):
        if k.endswith("__mul"):
            field = k[:-5]
            factor = Decimal(str(setv.pop(k)))
            rows = supabase.table(table).select("*").execute().data or []
            updates = []
            for r in rows:
                if field in r and r[field] is not None:
                    new_val = float(Decimal(str(r[field])) * factor)
                    updates.append({"op":"update","table":table,"where":{"id":r["id"]},"set":{field:new_val}})
            return updates
    return None

# ------------------------------------------------------------------------------
# DB op executor (select / update / insert / delete / upsert)
# ------------------------------------------------------------------------------
def _execute_ops(ops: List[Dict[str, Any]]) -> List[Any]:
    out: List[Any] = []
    for step in ops:
        op = (step.get("op") or "").lower()
        table = step.get("table")
        if not op or not table:
            out.append({"error": "missing op/table", "step": step})
            continue

        # Expand abstract transforms
        if op == "update" and "set" in step:
            expanded = _apply_transforms(table, step)
            if expanded is not None:
                out.extend(_execute_ops(expanded))
                continue

        cols = _real_columns(table)
        where = _prune_where(step.get("where"), cols)

        if op == "select":
            q = supabase.table(table).select("*")
            q = _apply_where(q, where or {})
            for pair in (step.get("order") or []):
                if isinstance(pair, list) and len(pair) == 2:
                    q = q.order(pair[0], desc=(str(pair[1]).lower() == "desc"))
            if step.get("limit"):
                q = q.limit(int(step["limit"]))
            res = q.execute()
            out.append(res.data or [])
            continue

        if op == "update":
            q = supabase.table(table).update(step.get("set") or {})
            q = _apply_where(q, where or {})
            res = q.execute()
            out.append(res.data or [])
            continue

        if op == "insert":
            values = step.get("values")
            if values and isinstance(values, dict):
                values = [values]
            res = supabase.table(table).insert(values or []).execute()
            out.append(res.data or [])
            continue

        if op == "delete":
            q = supabase.table(table).delete()
            q = _apply_where(q, where or {})
            res = q.execute()
            out.append(res.data or [])
            continue

        if op == "upsert":
            values = step.get("values")
            if values and isinstance(values, dict):
                values = [values]
            res = supabase.table(table).upsert(values or []).execute()
            out.append(res.data or [])
            continue

        out.append({"error": f"unsupported op {op}", "step": step})
    return out

# ------------------------------------------------------------------------------
# Build prompt (dynamic instructions + schema hint + user request)
# ------------------------------------------------------------------------------
def _build_action_prompt(user_text: str) -> str:
    system = _get_relevant_instructions(user_text)
    schema_hint = _introspect_schema_hint()
    return (
        f"{system}\n\n"
        "SCHEMA_HINT:\n"
        f"{schema_hint}\n\n"
        "USER_REQUEST:\n"
        f"{user_text}\n\n"
        "Return ONLY the JSON Action Plan."
    )

# ------------------------------------------------------------------------------
# Main agent entry
# ------------------------------------------------------------------------------
def run_finance_agent(user_text: str) -> AgentResponse:
    prompt = _build_action_prompt(user_text)

    try:
        plan_raw = reason_with_memory(agent_name="finance", query=prompt, namespace="expenses", k=8)
    except Exception as e:
        logger.exception("reason_with_memory failed")
        return make_response(agent="finance", intent="error", data={"message": f"reasoner error: {e}"})

    # Robust JSON extraction
    if not isinstance(plan_raw, str) or not plan_raw.strip():
        return make_response(agent="finance", intent="error", data={"message": "LLM returned empty plan", "raw": str(plan_raw)})

    s = plan_raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        lines = s.splitlines()
        if lines and lines[0].lower().startswith("json"):
            s = "\n".join(lines[1:])
    start = s.find("{"); end = s.rfind("}")
    candidate = s[start:end+1] if (start != -1 and end != -1 and end > start) else s

    try:
        plan = json.loads(candidate)
    except Exception as e:
        logger.exception("LLM plan parse failed")
        return make_response(agent="finance", intent="error", data={"message": f"Could not parse LLM plan: {e}", "raw": plan_raw})

    ops: List[Dict[str, Any]] = plan.get("operations", [])
    try:
        results = _execute_ops(ops)
    except Exception as e:
        logger.exception("execution failed")
        return make_response(agent="finance", intent="error", data={"message": f"execution error: {e}", "operations": ops})

    return make_response(
        agent="finance",
        intent="auto",
        data={
            "thoughts": plan.get("thoughts"),
            "operations": ops,
            "results": results,
            "response_template": plan.get("response_template")
        }
    )

# ------------------------------------------------------------------------------
# Public wrapper for router
# ------------------------------------------------------------------------------
def handle_finance(query: str) -> AgentResponse:
    return run_finance_agent(query)

# ------------------------------------------------------------------------------
# Optional UI formatter registration (returns AgentResponse, not a string)
# ------------------------------------------------------------------------------
try:
    from utils.nl_formatter import register_formatter

    def _fmt_auto(resp: AgentResponse) -> AgentResponse:
        # Ensure dict shape
        if not isinstance(resp, dict):
            return make_response(agent="finance", intent="auto", message=str(resp), data={"raw": resp})

        data = resp.get("data") or {}
        if not isinstance(data, dict):
            return make_response(agent="finance", intent="auto", message=str(data), data=data)

        ops = data.get("operations") or []
        results = data.get("results") or []

        lines = ["Plan:"]
        for i, step in enumerate(ops, 1):
            try:
                lines.append(f"{i}. {json.dumps(step, separators=(',',':'))}")
            except Exception:
                lines.append(f"{i}. {repr(step)}")

        lines.append("Results (first item only per step):")
        for i, r in enumerate(results, 1):
            summary = r[0] if isinstance(r, list) and r else r
            try:
                lines.append(f"{i}. {json.dumps(summary, separators=(',',':'))}")
            except Exception:
                lines.append(f"{i}. {repr(summary)}")

        resp["message"] = "\n".join(lines)
        return resp

    def _fmt_error(resp: AgentResponse) -> AgentResponse:
        msg = None
        if isinstance(resp, dict):
            msg = resp.get("message") or resp.get("error") or resp.get("detail")
        if not msg:
            msg = "An error occurred."
        resp = resp if isinstance(resp, dict) else make_response(agent="finance", intent="error", data={"raw": str(resp)})
        resp["message"] = f"Error: {msg}"
        return resp

    register_formatter("finance", "auto", _fmt_auto)
    register_formatter("finance", "error", _fmt_error)
except Exception:
    # Formatter is best-effort; never fail agent import
    pass
