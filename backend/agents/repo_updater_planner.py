# backend/agents/repo_updater_planner.py
from __future__ import annotations

import json
from typing import Any, Dict

from backend.llm.llm_runner import run_llm_agent


def handle_planner_task(query: str) -> Dict[str, Any]:
    """
    Delegates the planning task to the LLM runner with the 'repo_updater_planner' slug.
    """
    result = run_llm_agent(agent_slug="repo_updater_planner", user_text=query)
    return {"ok": result.ok, "response": result.response_text, "error": result.error}
