# backend/agents/repo_updater_scout.py
from __future__ import annotations


from typing import Any, Dict

from backend.llm.llm_runner import run_llm_agent


def handle_scout_task(query: str) -> Dict[str, Any]:
    """
    Delegates the scouting task to the LLM runner with the 'repo_updater_scout' slug.
    """
    result = run_llm_agent(agent_slug="repo_updater_scout", user_text=query)
    return {"ok": result.ok, "response": result.response_text, "error": result.error}
