# backend/agents/finance_agent.py
from __future__ import annotations
from typing import Dict, List, Tuple
from datetime import datetime
import traceback
import logging

from services.supabase_service import supabase
from utils.agent_protocol import make_response
from utils.nl_formatter import register_formatter
from reasoner.policy import reason_with_memory  # Cohere-based reasoning

logger = logging.getLogger("finance")
# --------------------------
# DB helpers
# --------------------------
def _fetch_all_expenses() -> List[Dict]:
    """Return all rows from finance_ledger (id, description, amount, created_at)."""
    res = (
        supabase.table("finance_ledger")
        .select("id, description, amount, created_at")
        .order("created_at", desc=False)
        .execute()
    )
    return res.data or []

def _insert_expense(description: str, amount: float) -> None:
    """Insert a single expense row."""
    supabase.table("finance_ledger").insert(
        {"description": description, "amount": amount}
    ).execute()

def _most_expensive_row() -> Dict | None:
    """Safely get the single most expensive purchase (or None)."""
    try:
        res = (
            supabase.table("finance_ledger")
            .select("description, amount, created_at")
            .order("amount", desc=True)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        return rows[0] if rows else None
    except Exception as e:
        return {"error": f"DB error: {e}"}


# --------------------------
# Intent parsing
# --------------------------
def _parse_log_intent(q: str) -> Tuple[bool, str, float]:
    """
    Very simple parser:
      "log expense coffee 4.50"  -> (True, "coffee", 4.50)
      "add expense dinner 12.00" -> (True, "dinner", 12.00)
    Anything else -> (False, "", 0.0)
    """
    lowered = (q or "").lower().strip()
    if lowered.startswith("log expense") or lowered.startswith("add expense"):
        parts = q.split()
        try:
            amt = float(parts[-1].replace("$", ""))
            # use the first occurrence of the token "expense"
            idx = parts.index("expense")
            description = " ".join(parts[idx + 1 : -1]).strip() or "Unlabeled"
            return True, description.strip('"\' '), amt
        except Exception:
            return False, "", 0.0
    return False, "", 0.0


# --------------------------
# Main handler
# --------------------------
def handle_finance(query: str):
    logging.info(f"[finance] handling query: {query}")
    try:
        """
        Handles:
        - "log expense <desc> <amount>"
        - "list expenses" / "show expenses"
        - freeform questions via Cohere reasoning over 'expenses' memory
        - never crashes the API (safeguards in place)
        """
        q = (query or "").strip()
        q_lower = q.lower()

        # 1) Explicit: logging an expense
        is_log, desc, amt = _parse_log_intent(q)
        if is_log:
            try:
                _insert_expense(desc, amt)
                return make_response(
                    agent="finance",
                    intent="log_expense",
                    message=f"Logged expense: {desc} (${amt:,.2f}).",
                    data={"description": desc, "amount": amt},
                )
            except Exception as e:
                return make_response(
                    agent="finance",
                    intent="error",
                    message=f"Failed to log expense: {e}",
                )

        # 2) Explicit: list expenses
        if any(p in q_lower for p in ["list expenses", "show expenses", "expenses", "list expense", "show expense"]):
            try:
                rows = _fetch_all_expenses()
                return make_response(agent="finance", intent="list_expenses", data=rows)
            except Exception as e:
                return make_response(agent="finance", intent="error", message=f"List error: {e}")

        # 3) Common finance Q: most expensive purchase (DB direct, fast path)
        if "most expensive" in q_lower or ("highest" in q_lower and "purchase" in q_lower):
            row = _most_expensive_row()
            if not row:
                return make_response(agent="finance", intent="answer", message="No expenses found yet.")
            if isinstance(row, dict) and "error" in row:
                return make_response(agent="finance", intent="error", message=row["error"])
            return make_response(
                agent="finance",
                intent="answer",
                message=f"Most expensive: {row.get('description','(no description)')} — ${float(row.get('amount') or 0):,.2f} on {row.get('created_at')}",
                data=row,
            )

        # 4) Freeform: let the reasoner think with memory — SAFEGUARDED
        try:
            semantic_answer = reason_with_memory(
                agent_name="finance",
                query=query,
                namespace="expenses",
                k=10
            )
            if semantic_answer:
                # keep this shape so the NL middleware can render it naturally
                return {"agent": "finance", "final_answer": semantic_answer}
        except Exception as e:
            # Do not 500; report a friendly, debuggable error
            return make_response(
                agent="finance",
                intent="error",
                message=f"Reasoner error: {e}"
            )

        # 5) Fallback help
        return make_response(
            agent="finance",
            intent="help",
            message="Finance agent. Try: ‘log expense coffee 4.50’, ‘list expenses’, or ask a question like ‘What was my most expensive purchase this month?’",
        )
    except Exception as e:
        logging.error(f"[finance] ERROR: {e}\n{traceback.format_exc()}")
        return {"intent": "error", "message": f"Finance agent failed: {e}"}

# --------------------------
# Formatters
# --------------------------
def _finance_list_formatter(resp):
    rows = resp.get("data") or []
    if not rows:
        resp["message"] = "No expenses yet. Try: ‘log expense coffee 4.50’."
        return resp

    total = sum(float(r.get("amount") or 0.0) for r in rows)
    recent = rows[-5:]

    def line(r):
        dt = r.get("created_at")
        try:
            # coerce to readable local-ish string
            dtp = datetime.fromisoformat(str(dt).replace("Z", "+00:00"))
            ds = dtp.strftime("%Y-%m-%d %H:%M")
        except Exception:
            ds = str(dt)
        return f"{r.get('description','Item')} (${float(r.get('amount') or 0):,.2f} on {ds})"

    preview = "; ".join(line(x) for x in recent)
    more = f" …and {len(rows) - 5} more." if len(rows) > 5 else ""
    resp["message"] = f"You have {len(rows)} expense(s) totaling ${total:,.2f}. Recent: {preview}{more}"
    return resp

register_formatter("finance", "list_expenses", _finance_list_formatter)
