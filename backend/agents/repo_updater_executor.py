# backend/agents/repo_updater_executor.py
from __future__ import annotations


from typing import Any, Dict

from backend.llm.llm_runner import run_llm_agent


def handle_executor_task(query: str) -> Dict[str, Any]:
    """
    Delegates the execution task to the LLM runner with the 'repo_updater_executor' slug.
    """
    result = run_llm_agent(agent_slug="repo_updater_executor", user_text=query)
    return {"ok": result.ok, "response": result.response_text, "error": result.error}
