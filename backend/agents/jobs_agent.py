# backend/agents/jobs_agent.py
from __future__ import annotations

import logging
from typing import Any, Dict

from backend.agents.jobs_runner import run_once
from backend.agents.mixins import SemanticAgentMixin

logger = logging.getLogger(__name__)


class JobsAgent(SemanticAgentMixin):
    """
    A multi-stage agent responsible for the entire job discovery pipeline.
    It acts as a simple entry point to trigger the jobs_runner.
    """

    def __init__(self):
        self.namespace = "jobs"

    def handle(self, query: str) -> Dict[str, Any]:
        """
        The main entry point for the JobsAgent. It ignores the query and
        triggers the full job processing pipeline.
        """
        logger.info(f"JobsAgent triggered with query: {query}")
        # The `run_once` function from jobs_runner handles the entire pipeline.
        result = run_once({})
        return {
            "ok": True,
            "status": "pipeline_triggered",
            "run_id": result.get("run_id"),
        }
