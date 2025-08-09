# backend/utils/nl_formatter.py
from __future__ import annotations
from typing import Any, Callable, Dict, List, Tuple, Union
from datetime import datetime
import json

from .agent_protocol import AgentResponse

Formatter = Callable[[AgentResponse], AgentResponse]

# (agent,intent) -> formatter
_registry: Dict[Tuple[str, str], Formatter] = {}


def register_formatter(agent: str, intent: str, fn: Formatter) -> None:
    _registry[(agent, intent)] = fn


# ---------------- helpers ----------------

def _fmt_money(v: Any) -> str:
    try:
        return f"${float(v):,.2f}"
    except Exception:
        return str(v)


def _fmt_dt(v: Any) -> str:
    if isinstance(v, str):
        try:
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return v
    return str(v)


def _coerce_agent_response(resp_like: Any) -> AgentResponse:
    """
    Accept literally anything and coerce to a valid AgentResponse shape.
    This prevents UI formatting crashes when a handler returns a raw string.
    """
    # Already close enough
    if isinstance(resp_like, dict):
        agent = str(resp_like.get("agent", "") or "unknown")
        intent = str(resp_like.get("intent", "") or "unknown")
        out: AgentResponse = {
            "agent": agent,
            "intent": intent,
            # Keep original data/message if present; otherwise carry raw through
            "data": resp_like.get("data", resp_like),
            "message": resp_like.get("message", ""),
        }
        # Preserve any other keys (debug, meta, etc.)
        for k, v in resp_like.items():
            if k not in out:
                out[k] = v
        return out

    # String / number / list / whatever
    text = json.dumps(resp_like, default=str) if not isinstance(resp_like, str) else resp_like
    return {
        "agent": "unknown",
        "intent": "unknown",
        "data": resp_like,
        "message": text,  # shows *something* readable in the UI
    }


# ---------------- core formatters ----------------

def _default_formatter(resp: AgentResponse) -> AgentResponse:
    """
    Generic summary if a specific formatter isn't registered.
    - If data is a list of dicts: summarize count + preview a few
    - If message exists already: keep it
    - Otherwise craft a simple sentence
    """
    if resp.get("message"):
        return resp

    data = resp.get("data")

    # List[dict] -> compact preview
    if isinstance(data, list) and data and isinstance(data[0], dict):
        count = len(data)
        preview = data[-5:]

        def one_line(row: dict) -> str:
            desc = row.get("description") or row.get("name") or row.get("title") or "Item"
            amount = row.get("amount") or row.get("value")
            created = row.get("created_at") or row.get("created") or row.get("date")
            bits = [str(desc)]
            if amount is not None:
                bits.append(f"({_fmt_money(amount)})")
            if created is not None:
                bits.append(f"on {_fmt_dt(created)}")
            return " ".join(bits)

        preview_txt = "; ".join(one_line(r) for r in preview)
        more = f" …and {count-5} more." if count > 5 else ""
        resp["message"] = (
            f"{resp['agent'].capitalize()} • {resp['intent'].replace('_',' ')}: "
            f"{count} item(s). Recent: {preview_txt}{more}"
        )
        return resp

    # Dict payload -> generic success line
    if isinstance(data, dict):
        resp["message"] = f"{resp['agent'].capitalize()} • {resp['intent'].replace('_',' ')} done."
        return resp

    # Fallback for any other type
    resp["message"] = resp.get("message") or f"{resp['agent'].capitalize()} • {resp['intent'].replace('_',' ')}."
    return resp


def ensure_natural(resp_like: Any) -> AgentResponse:
    """
    Apply a specific formatter if registered; otherwise use a sensible default.
    Accepts ANY input (string, dict, etc.) and always returns a valid AgentResponse.
    Never raises — if a formatter explodes, we fall back with a helpful message.
    """
    resp = _coerce_agent_response(resp_like)
    key = (resp.get("agent", ""), resp.get("intent", ""))

    try:
        fn = _registry.get(key)
        if fn:
            out = fn(resp)
            # ⬇⬇⬇ IMPORTANT: coerce formatter *output* too
            out = _coerce_agent_response(out)
        else:
            out = _default_formatter(resp)
    except Exception as e:
        safe_raw = json.dumps(resp, default=str) if not isinstance(resp, str) else resp
        out = {
            "agent": resp.get("agent", "unknown"),
            "intent": resp.get("intent", "unknown"),
            "data": resp.get("data"),
            "message": f"Formatting error: {e}\nRaw: {safe_raw}",
        }

    # Guarantee a non-empty message
    if not out.get("message"):
        out = _default_formatter(out)
    return out

