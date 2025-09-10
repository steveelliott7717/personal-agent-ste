# backend/agents/repo_updater_generator.py
from __future__ import annotations

import json
from typing import Any, Dict

from backend.llm.llm_runner import run_llm_agent


def handle_generator_task(query: str) -> Dict[str, Any]:
    """
    Delegates the generation task to the LLM runner with the 'repo_updater_generator' slug.
    """
    result = run_llm_agent(agent_slug="repo_updater_generator", user_text=query)
    return {"ok": result.ok, "response": result.response_text, "error": result.error}
