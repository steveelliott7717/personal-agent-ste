# backend/api.py
from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

# models
from backend.models.messages import (
    AgentCallRequest,
    AgentCallResponse,
    PlanRequest,
    PlanResponse,
    PlanStepResult,
    PlanStep,
)

# orchestrator (meta-agent)
from backend.agents.orchestrator import Orchestrator

# ---------------------------------------------------------------------
# Environment & App bootstrap
# ---------------------------------------------------------------------

# Load .env (Supabase keys, allowlists, etc.)
load_dotenv()

BASE = "/app"

app = FastAPI(title="Personal Agent API", version="0.1.0")

# permissive CORS for local/dev; tighten in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Detect potential frontend bundle locations; actual mount happens at the bottom
_frontend_root_candidates: List[Path] = [
    Path("frontend/dist"),
    Path("app/dist"),
    Path("app"),
    Path("frontend"),
]
_frontend_root: Optional[Path] = next(
    (p for p in _frontend_root_candidates if p.exists()), None
)

# ---------------------------------------------------------------------
# Static health & version endpoints
# ---------------------------------------------------------------------


@app.get(f"{BASE}/api/health", response_class=PlainTextResponse)
async def health() -> str:
    return "ok"


@app.get(f"{BASE}/api/version", response_class=PlainTextResponse)
async def version() -> str:
    return "0.1.0"


# ---------------------------------------------------------------------
# Optional: legacy router passthrough
# ---------------------------------------------------------------------
try:
    from backend.agents.router_agent import route_request  # type: ignore
except Exception:
    route_request = None


@app.post(f"{BASE}/api/route")
async def legacy_route(body: dict) -> dict:
    """
    Back-compat endpoint that forwards to router_agent if available.
    """
    if route_request is None:
        # not available – return a helpful message
        return {
            "ok": False,
            "error": "router_agent not loaded",
            "hint": "use /app/api/agents/verb instead",
        }
    text = body.get("text") if isinstance(body, dict) else None
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' in request body")
    return route_request(text)


# ---------------------------------------------------------------------
# Orchestrator + Capability Registry Entrypoints
# ---------------------------------------------------------------------

orchestrator = Orchestrator()

# Named micro-plans you asked for
MICRO_PLANS = {
    "quick.health_check": [
        {"verb": "time.now"},
        {"verb": "notify.push", "args": {"text": "health_check: alive"}},
    ],
    "db.ping": [
        {"verb": "db.read", "args": {"table": "agents", "limit": 1}},
    ],
}


@app.post(f"{BASE}/api/agents/verb", response_model=AgentCallResponse)
async def call_agent_verb(
    req: AgentCallRequest,
    # bind explicitly to the canonical header names
    x_correlation_id: Optional[str] = Header(default=None, alias="X-Correlation-Id"),
    x_idempotency_key: Optional[str] = Header(default=None, alias="X-Idempotency-Key"),
):
    """
    Canonical, auditable entrypoint for a single capability verb.
    Delegates to the Orchestrator (which applies policies, auditing, idempotency).
    """
    correlation_id = x_correlation_id or req.trace_id or uuid.uuid4().hex
    idempotency_key = x_idempotency_key or req.idempotency_key or uuid.uuid4().hex

    out = orchestrator.call_verb(
        req.verb,
        req.args or {},
        req.meta or {},
        correlation_id=correlation_id,
        idempotency_key=idempotency_key,
        actor=(req.actor.dict() if req.actor else None),
        policy_ctx=(req.policy_ctx.dict() if req.policy_ctx else None),
    )

    return AgentCallResponse(
        ok=bool(out.get("ok")),
        result=out.get("result", {}),
        latency_ms=int(out.get("latency_ms", 0)),
        correlation_id=out["correlation_id"],
        idempotency_key=out["idempotency_key"],
    )


@app.post(f"{BASE}/api/agents/plan", response_model=PlanResponse)
async def call_agent_plan(
    req: PlanRequest,
    x_correlation_id: Optional[str] = Header(default=None, alias="X-Correlation-Id"),
    x_idempotency_key: Optional[str] = Header(default=None, alias="X-Idempotency-Key"),
):
    """
    Execute an ordered list of verbs via the Orchestrator.
    - Applies policies & auditing per step.
    - Propagates correlation id across the whole plan.
    - Creates per-step idempotency keys: "<plan_key>#<idx>:<verb>".
    - Aborts on first failure unless continue_on_error=True.
    - Supports named micro-plans via req.plan (e.g., "quick.health_check").
    Adds plan start/end timestamps and failed_step_index.
    """
    plan_start = datetime.now(timezone.utc).isoformat()
    plan_corr = x_correlation_id or req.trace_id or uuid.uuid4().hex
    plan_idem = x_idempotency_key or req.idempotency_key or uuid.uuid4().hex

    # Expand named micro-plan (append to any explicit steps)
    steps_in: List[PlanStep] = list(req.steps or [])
    if getattr(req, "plan", None):
        micro = MICRO_PLANS.get(req.plan)
        if not micro:
            raise HTTPException(status_code=400, detail=f"Unknown plan '{req.plan}'")
        steps_in.extend(PlanStep(**s) for s in micro)

    # --- plan start audit ---
    orchestrator._log_event(
        "plan.start",
        {
            "plan": getattr(req, "plan", None),
            "steps_count": len(steps_in),
            "continue_on_error": bool(req.continue_on_error),
        },
        correlation_id=plan_corr,
        idempotency_key=plan_idem,
    )

    steps_out: List[PlanStepResult] = []
    total_latency = 0
    ok = True
    failed_idx: Optional[int] = None

    for idx, step in enumerate(steps_in):
        # derive per-step idempotency key FIRST so it exists for all logs
        step_idem = f"{plan_idem}#{idx}:{step.verb}"

        # --- step start audit ---
        # Tip: we log only lightweight summaries to keep your events table small.
        # If you need full payloads, we can add a LOG_VERBOSE=true toggle later.
        orchestrator._log_event(
            "plan.step.start",
            {
                "index": idx,
                "verb": step.verb,
                "step_idempotency_key": step_idem,
                # keep args light in logs; include keys only
                "arg_keys": sorted(list((step.args or {}).keys())),
            },
            correlation_id=plan_corr,
            idempotency_key=plan_idem,
        )

        out = orchestrator.call_verb(
            step.verb,
            step.args or {},
            step.meta or {},
            correlation_id=plan_corr,
            idempotency_key=step_idem,
            actor=(req.actor.dict() if req.actor else None),
            policy_ctx=(req.policy_ctx.dict() if req.policy_ctx else None),
        )

        latency = int(out.get("latency_ms", 0))
        total_latency += latency
        step_ok = bool(out.get("ok"))

        # --- step end audit ---
        orchestrator._log_event(
            "plan.step.end",
            {
                "index": idx,
                "verb": step.verb,
                "ok": step_ok,
                "latency_ms": latency,
                "step_idempotency_key": step_idem,
                # keep result light; include common summary fields if present
                "summary": {
                    "keys": sorted(list((out.get("result") or {}).keys())),
                    "status": (out.get("result") or {}).get("status"),
                    "rows": (
                        len((out.get("result") or {}).get("rows", []))
                        if isinstance((out.get("result") or {}).get("rows"), list)
                        else None
                    ),
                },
            },
            correlation_id=plan_corr,
            idempotency_key=plan_idem,
        )

        steps_out.append(
            PlanStepResult(
                verb=step.verb,
                ok=step_ok,
                result=out.get("result", {}),
                latency_ms=latency,
            )
        )

        if not step_ok:
            failed_idx = idx
            if not req.continue_on_error:
                ok = False
                break

    # If any step failed overall plan is false; preserve first failing index if continuing
    if ok and any(not s.ok for s in steps_out):
        ok = False
        if failed_idx is None:
            failed_idx = next(i for i, s in enumerate(steps_out) if not s.ok)

    plan_end = datetime.now(timezone.utc).isoformat()

    # --- plan end audit ---
    orchestrator._log_event(
        "plan.end",
        {
            "plan": getattr(req, "plan", None),
            "ok": bool(ok),
            "total_latency_ms": int(total_latency),
            "failed_step_index": failed_idx,
            "steps_run": len(steps_out),
            "start": plan_start,
            "end": plan_end,
        },
        correlation_id=plan_corr,
        idempotency_key=plan_idem,
    )

    return PlanResponse(
        ok=ok,
        steps=steps_out,
        total_latency_ms=total_latency,
        correlation_id=plan_corr,
        idempotency_key=plan_idem,
        start=plan_start,
        end=plan_end,
        failed_step_index=failed_idx,
    )


# ---------------------------------------------------------------------
# Local dev convenience endpoints (optional)
# ---------------------------------------------------------------------


@app.get(BASE + "/api/files/{path:path}")
async def serve_file(path: str):
    """
    Best-effort static file server for debugging.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path)


@app.get("/")
async def root_note():
    return {"ok": True, "hint": f"try {BASE}/api/health or {BASE}/api/agents/verb"}


# ---------------------------------------------------------------------
# Serve PWA (MOUNT LAST so API routes are matched first)
# ---------------------------------------------------------------------

if _frontend_root:
    app.mount(
        f"{BASE}", StaticFiles(directory=str(_frontend_root), html=True), name="app"
    )

    @app.get(f"{BASE}/", response_class=HTMLResponse)
    async def app_index() -> HTMLResponse:
        index_path = _frontend_root / "index.html"
        if index_path.exists():
            return HTMLResponse(index_path.read_text(encoding="utf-8"))
        # Fallback to directory listing note
        return HTMLResponse("<h1>App</h1><p>Frontend assets mounted.</p>")

else:

    @app.get(f"{BASE}/", response_class=HTMLResponse)
    async def app_index_missing() -> HTMLResponse:
        return HTMLResponse(
            "<h1>App</h1><p>No frontend bundle found. "
            "Place your built assets under ./app or ./frontend/dist.</p>"
        )
