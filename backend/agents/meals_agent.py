# backend/agents/meals_agent.py
from ._base_agent import BaseAgent

FALLBACK_SYSTEM = """
You are a meals operator. You plan floating daily meal cards (no timestamps) and log completions.
You may read/write tables: recipe_templates, meal_plan, meal_log.
Return ONLY compact JSON with keys: thoughts, operations, response_template?.

Examples:

1) Build today's daylist (oldest-first):
{
  "thoughts": "List meals for today from meal_plan",
  "operations": [
    {"op":"select","table":"meal_plan","where":{"date":"{{today}}"},"order":[["freshness_rank","asc"]],"limit":50}
  ]
}

2) Mark a meal complete and log it:
{
  "thoughts": "Mark meal done and insert into meal_log",
  "operations": [
    {"op":"update","table":"meal_plan","where":{"id":"{{meal_id}}"},"set":{"status":"done"}},
    {"op":"insert","table":"meal_log","values":{
      "meal_plan_id":"{{meal_id}}","ts":"{{now}}","notes":"auto"
    }}
  ]
}
"""

class MealsAgent(BaseAgent):
    AGENT_META = {
        "slug": "meals",
        "title": "Meals",
        "description": "Plan daily meals, swaps, and log completions.",
        "handler_key": "meals.handle",
        "namespaces": ["meals"],
        "capabilities": ["Build daylist","Swap meals","Log meal completion"],
        "keywords": ["meal","meals","eat","food","recipe","daylist","lunch","breakfast","dinner","snack"],
        "status": "enabled",
        "default_tables": ["recipe_templates","meal_plan","meal_log"],
        "instruction_tags": [],       # optionally use ["planning","swaps","logging"]
        "fallback_system": FALLBACK_SYSTEM,
        "post_hooks": [],             # e.g., ["plugins.meals.attach_daylist_summary:post_summarize"]
    }

def handle_meals(query: str):
    return MealsAgent().handle(query)
