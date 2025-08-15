# backend/utils/agent_protocol.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict, Union


class AgentResponse(TypedDict, total=False):
    agent: str  # e.g. "finance"
    intent: str  # e.g. "list_expenses", "log_expense"
    message: str  # natural language response
    data: Union[List[Dict[str, Any]], Dict[str, Any], None]
    meta: Dict[str, Any]


def make_response(
    *,
    agent: str,
    intent: str,
    message: Optional[str] = None,
    data: Union[List[Dict[str, Any]], Dict[str, Any], None] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> AgentResponse:
    resp: AgentResponse = {"agent": agent, "intent": intent}
    if message is not None:
        resp["message"] = message
    if data is not None:
        resp["data"] = data
    if meta:
        resp["meta"] = meta
    return resp
