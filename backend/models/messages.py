# backend/models/messages.py
from __future__ import annotations

from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field


class Actor(BaseModel):
    user_id: Optional[str] = None
    session: Optional[str] = None


class PolicyCtx(BaseModel):
    scopes: Optional[List[str]] = Field(default_factory=list)
    flags: Dict[str, Any] = Field(default_factory=dict)


class AgentCallRequest(BaseModel):
    verb: str
    args: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)

    # Correlation / idempotency can come via headers or inline here
    trace_id: Optional[str] = None
    idempotency_key: Optional[str] = None

    # Optional typed helpers (also mirrored inside meta if you prefer)
    actor: Optional[Actor] = None
    policy_ctx: Optional[PolicyCtx] = None


class AgentCallResponse(BaseModel):
    ok: bool
    result: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: int
    correlation_id: str
    idempotency_key: str


# ---- Planning contracts ----


class PlanStep(BaseModel):
    verb: str
    args: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)


class PlanRequest(BaseModel):
    steps: List[PlanStep] = Field(default_factory=list)
    continue_on_error: bool = False
    # Optional correlation/idempotency at the plan level
    trace_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    # Optional structured helpers
    actor: Optional[Actor] = None
    policy_ctx: Optional[PolicyCtx] = None
    # NEW: allow referencing a named micro-plan
    plan: Optional[str] = None


class PlanStepResult(BaseModel):
    verb: str
    ok: bool
    result: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: int


class PlanResponse(BaseModel):
    ok: bool
    steps: List[PlanStepResult] = Field(default_factory=list)
    total_latency_ms: int
    correlation_id: str
    idempotency_key: str
    # NEW: these must be present or FastAPI will drop them
    start: Optional[str] = None  # ISO 8601 UTC
    end: Optional[str] = None  # ISO 8601 UTC
    failed_step_index: Optional[int] = None
