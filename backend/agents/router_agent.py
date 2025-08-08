# backend/agents/router_agent.py
from typing import Callable, Dict, Tuple, Any
from agents.meals_agent import handle_meals
from agents.finance_agent import handle_finance
try:
    from agents.workouts_agent import handle_workouts
except Exception:
    def handle_workouts(q: str):  # optional
        return {"agent":"workouts","intent":"help","message":"Workouts agent not configured yet."}
try:
    from agents.grooming_agent import handle_grooming
except Exception:
    def handle_grooming(q: str):
        return {"agent":"grooming","intent":"help","message":"Grooming agent not configured yet."}

ROUTES: Dict[str, Callable[[str], Any]] = {
    "meals": handle_meals,
    "finance": handle_finance,
    "expenses": handle_finance,  # alias
    "workouts": handle_workouts,
    "grooming": handle_grooming,
}

KEYWORDS = {
    "meals": ["meal", "breakfast", "lunch", "dinner", "calories", "ate", "food"],
    "expenses": ["expense", "expenses", "spend", "spent", "purchase", "bill"],
    "workouts": ["workout", "run", "lift", "gym", "exercise"],
    "grooming": ["groom", "shampoo", "rinse", "deodorant", "sunscreen", "moisturizer", "lotion", "beard", "trim", "nails", "hair style"],
}

def route_request(query: str):
    q = query.lower()

    # Finance direct patterns
    if any(pat in q for pat in ["list expenses", "show expenses", "log expense"]):
        return "finance", handle_finance(query)

    # Keyword routing
    for agent, kws in KEYWORDS.items():
        if any(kw in q for kw in kws):
            handler = ROUTES.get(agent) or ROUTES["meals"]
            return agent, handler(query)

    # Default to meals (or a generic agent)
    return "meals", ROUTES["meals"](query)
