# backend/agents/finance_agent.py
from __future__ import annotations
from typing import Dict, List, Tuple
from datetime import datetime

from services.supabase_service import supabase
from utils.agent_protocol import make_response
from utils.nl_formatter import register_formatter


def _fetch_all_expenses() -> List[Dict]:
    res = (
        supabase.table("finance_ledger")
        .select("id, description, amount, created_at")
        .order("created_at", desc=False)
        .execute()
    )
    return res.data or []


def _insert_expense(description: str, amount: float) -> None:
    supabase.table("finance_ledger").insert(
        {"description": description, "amount": amount}
    ).execute()


def _parse_log_intent(q: str) -> Tuple[bool, str, float]:
    lowered = q.lower().strip()
    if lowered.startswith("log expense") or lowered.startswith("add expense"):
        parts = q.split()
        try:
            amt = float(parts[-1].replace("$", ""))
            idx = parts.index("expense")
            description = " ".join(parts[idx + 1 : -1]).strip() or "Unlabeled"
            return True, description.strip('"\' '), amt
        except Exception:
            return False, "", 0.0
    return False, "", 0.0


def handle_finance(query: str):
    q = query.strip().lower()

    # Create
    is_log, desc, amt = _parse_log_intent(query)
    if is_log:
        _insert_expense(desc, amt)
        return make_response(
            agent="finance",
            intent="log_expense",
            message=f"Logged expense: {desc} (${amt:,.2f}).",
            data={"description": desc, "amount": amt},
        )

    # Read
    if any(p in q for p in ["list expenses", "show expenses", "expenses", "list expense", "show expense"]):
        rows = _fetch_all_expenses()
        return make_response(agent="finance", intent="list_expenses", data=rows)

    # Fallback
    return make_response(
        agent="finance",
        intent="help",
        message="Finance agent. Try: ‘log expense coffee 4.50’ or ‘list expenses’.",
    )


# (Optional) Example of a custom formatter for finance/list_expenses
def _finance_list_formatter(resp):
    rows = resp.get("data") or []
    if not rows:
        resp["message"] = "No expenses yet. Try: ‘log expense coffee 4.50’."
        return resp
    total = sum(float(r.get("amount") or 0.0) for r in rows)
    # show last 5
    recent = rows[-5:]
    def line(r):
        dt = r.get("created_at")
        try:
            dtp = datetime.fromisoformat(str(dt).replace("Z","+00:00"))
            ds = dtp.strftime("%Y-%m-%d %H:%M")
        except Exception:
            ds = str(dt)
        return f"{r.get('description','Item')} (${float(r.get('amount') or 0):,.2f} on {ds})"
    preview = "; ".join(line(x) for x in recent)
    more = f" …and {len(rows)-5} more." if len(rows) > 5 else ""
    resp["message"] = f"You have {len(rows)} expense(s) totaling ${total:,.2f}. Recent: {preview}{more}"
    return resp

register_formatter("finance", "list_expenses", _finance_list_formatter)
