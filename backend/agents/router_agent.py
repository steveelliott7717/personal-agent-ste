# backend/agents/router_agent.py
from __future__ import annotations
from typing import Tuple
import re
import traceback

# import your agents
from .finance_agent import handle_finance
from .meals_agent import handle_meals
from .workouts_agent import handle_workouts
from .grooming_agent import handle_grooming
# add others as you have them

ROUTER_VERSION = "2025-08-08-logging"
print(f"[router] loaded version={ROUTER_VERSION}")

# --- helpers ---------------------------------------------------------

FINANCE_KEYWORDS = {
    "market", "stock", "stocks", "ticker", "price", "quote", "portfolio", "index",
    "sp500", "s&p", "snp", "s&p500", "s&p 500", "spx", "spy", "nasdaq", "dow",
    "earnings", "buy", "sell", "trade", "option", "call", "put",
    "expense", "expenses", "purchase", "purchased", "spend", "spent", "budget",
}

MEALS_KEYWORDS = {"meal", "meals", "breakfast", "lunch", "dinner", "calories", "food", "diet"}
WORKOUTS_KEYWORDS = {"workout", "workouts", "gym", "run", "lift", "strength", "cardio", "exercise"}
GROOMING_KEYWORDS = {"groom", "grooming", "shower", "skincare", "haircut", "clip", "trim"}

import re

def _normalize(q: str) -> str:
    q = (q or "").strip().lower()
    q = q.replace("^&", "&")  # windows-escaped
    q = q.replace("&", " & ") # ensure tokenization
    q = re.sub(r"[^a-z0-9 &]+", " ", q)  # drop odd punctuation
    q = re.sub(r"\s+", " ", q)
    return q


def _contains_any(q: str, words: set[str]) -> bool:
    qn = _normalize(q)
    return any(w in qn for w in words)

# --- main router -----------------------------------------------------

def route_request(query: str) -> Tuple[str, dict | str]:
    q = _normalize(query)
    print(f"[router] incoming={q!r}")

    try:
        if _contains_any(q, FINANCE_KEYWORDS):
            print("[router] -> finance (keyword match)")
            return "finance", handle_finance(query)

        if _contains_any(q, MEALS_KEYWORDS):
            print("[router] -> meals")
            return "meals", handle_meals(query)

        if _contains_any(q, WORKOUTS_KEYWORDS):
            print("[router] -> workouts")
            return "workouts", handle_workouts(query)

        if _contains_any(q, GROOMING_KEYWORDS):
            print("[router] -> grooming")
            return "grooming", handle_grooming(query)

        if any(phrase in q for phrase in ["most expensive", "purchase", "spent", "spend", "lowest expense"]):
            print("[router] -> finance (heuristic match)")
            return "finance", handle_finance(query)

    except Exception as e:
        print("[router] ERROR in routing:")
        traceback.print_exc()  # <––– will dump full stack trace to logs
        raise

    print("[router] -> default meals")
    return "meals", handle_meals(query)