# backend/agents/repo_updater_reviewer.py
from __future__ import annotations


from typing import Any, Dict

from backend.llm.llm_runner import run_llm_agent


def handle_reviewer_task(query: str) -> Dict[str, Any]:
    """
    Delegates the reviewing task to the LLM runner with the 'repo_updater_reviewer' slug.
    """
    result = run_llm_agent(agent_slug="repo_updater_reviewer", user_text=query)
    return {"ok": result.ok, "response": result.response_text, "error": result.error}
