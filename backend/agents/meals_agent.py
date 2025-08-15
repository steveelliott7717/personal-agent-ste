# backend/agents/meals_agent.py
from ._base_agent import BaseAgent

FALLBACK_SYSTEM = """
You are a meals operator. Plan floating daily meal cards (no fixed times), support swaps,
and log completions. Read/write tables: recipe_templates, meal_plan, meal_log.

Return ONLY compact JSON with keys:
- thoughts (string)
- operations (array of {op, table, where?, order?, limit?, set?, values?})
- response_template? (string)

Examples:

1) Build today's daylist (oldest-first by freshness):
{
  "thoughts": "List today's meals from meal_plan by freshness",
  "operations": [
    {"op":"select","table":"meal_plan","where":{"date":"{{today}}"},"order":[["freshness_rank","asc"]],"limit":50}
  ]
}

2) Mark a meal complete and log it:
{
  "thoughts": "Mark meal done and insert meal_log",
  "operations": [
    {"op":"update","table":"meal_plan","where":{"id":"{{meal_id}}"},"set":{"status":"done"}},
    {"op":"insert","table":"meal_log","values":{
      "meal_plan_id":"{{meal_id}}","ts":"{{now}}","notes":"auto"
    }}
  ]
}

3) Add a planned meal for today (1 serving):
{
  "thoughts": "Plan a new meal for today",
  "operations": [
    {"op":"insert","table":"meal_plan","values":{
      "date":"{{today}}","recipe_id":"{{recipe_id}}","servings":1,"status":"planned","freshness_rank":0
    }}
  ]
}
"""


class MealsAgent(BaseAgent):
    AGENT_META = {
        "slug": "meals",
        "title": "Meals",
        "description": "Plan daily meals, swap items, and log completions.",
        # Router metadata is inferred if omitted:
        # "module_path": "agents.meals_agent",
        # "callable_name": "class:MealsAgent",
        "namespaces": ["meals"],
        "capabilities": ["Daylist", "Swap", "Complete", "Plan"],
        "keywords": [
            "meal",
            "meals",
            "eat",
            "food",
            "recipe",
            "daylist",
            "breakfast",
            "lunch",
            "dinner",
            "snack",
        ],
        "status": "enabled",
        "version": "v1",
        # Hints for the planner (also overridable via agent_settings.default_tables)
        "default_tables": ["recipe_templates", "meal_plan", "meal_log"],
        "instruction_tags": [],
        "fallback_system": FALLBACK_SYSTEM,
        # Post hooks can be configured here or in agent_settings.post_hooks
        "post_hooks": [],
    }

    # Optional: choose instruction tags dynamically (kept minimal here)
    def choose_tags(self, user_text: str):
        text = (user_text or "").lower()
        tags = []
        if any(w in text for w in ["plan", "add", "new meal", "daylist"]):
            tags.append("planning")
        if any(w in text for w in ["swap", "replace"]):
            tags.append("swaps")
        if any(w in text for w in ["done", "complete", "log"]):
            tags.append("logging")
        return tags


def handle_meals(query: str):
    return MealsAgent().handle(query)
