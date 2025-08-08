# backend/utils/nl_formatter.py
from __future__ import annotations
from typing import Any, Callable, Dict, List, Tuple, Union
from datetime import datetime

from .agent_protocol import AgentResponse, make_response

Formatter = Callable[[AgentResponse], AgentResponse]

_registry: Dict[Tuple[str, str], Formatter] = {}  # (agent,intent) -> formatter


def register_formatter(agent: str, intent: str, fn: Formatter) -> None:
    _registry[(agent, intent)] = fn


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


def _default_formatter(resp: AgentResponse) -> AgentResponse:
    """
    Generic summary if a specific formatter isn't registered.
    - If data is a list of dicts: summarize count + show up to 5 recent
    - If message exists: keep it
    """
    if "message" in resp and resp["message"]:
        return resp

    data = resp.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        count = len(data)
        preview = data[-5:]
        # Try to pick nice fields if they exist
        # desc/name/title, amount/value, created/created_at/date
        def one_line(row: dict) -> str:
            desc = row.get("description") or row.get("name") or row.get("title") or "Item"
            amount = row.get("amount") or row.get("value")
            created = row.get("created_at") or row.get("created") or row.get("date")
            bits = [desc]
            if amount is not None:
                bits.append(f"({_fmt_money(amount)})")
            if created is not None:
                bits.append(f"on {_fmt_dt(created)}")
            return " ".join(bits)

        preview_txt = "; ".join(one_line(r) for r in preview)
        more = f" …and {count-5} more." if count > 5 else ""
        message = f"{resp['agent'].capitalize()} • {resp['intent'].replace('_',' ')}: {count} item(s). Recent: {preview_txt}{more}"
        resp["message"] = message
        return resp

    # Fallbacks
    if isinstance(data, dict):
        resp["message"] = f"{resp['agent'].capitalize()} • {resp['intent'].replace('_',' ')} done."
        return resp

    if not resp.get("message"):
        resp["message"] = f"{resp['agent'].capitalize()} • {resp['intent'].replace('_',' ')}."
    return resp


def ensure_natural(resp: AgentResponse) -> AgentResponse:
    """
    Apply a specific formatter if registered; otherwise use a sensible default.
    Always returns an AgentResponse with a 'message'.
    """
    key = (resp.get("agent", ""), resp.get("intent", ""))
    fn = _registry.get(key)
    if fn:
        out = fn(resp)
    else:
        out = _default_formatter(resp)

    # Guarantee message exists
    if "message" not in out or not out["message"]:
        out = _default_formatter(out)
    return out
