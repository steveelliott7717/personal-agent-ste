from __future__ import annotations

import os
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

# Load .env (Supabase keys, allowlists, etc.)
load_dotenv()

BASE = "/app"
app = FastAPI(title="Personal Agent API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_frontend_root_candidates: List[Path] = [
    Path("frontend/dist"),
    Path("app/dist"),
    Path("app"),
    Path("frontend"),
]
_frontend_root: Optional[Path] = next(
    (p for p in _frontend_root_candidates if p.exists()), None
)

# Optional: legacy router passthrough
try:
    from backend.agents.router_agent import route_request  # type: ignore
except Exception:
    route_request = None


@app.get(f"{BASE}/api/health", response_class=PlainTextResponse)
async def health() -> str:
    return "ok"


@app.get(f"{BASE}/api/version", response_class=PlainTextResponse)
async def version() -> str:
    return "0.1.0"


orchestrator = Orchestrator()

MICRO_PLANS = {
    "quick.health_check": [
        {"verb": "time.now"},
        {"verb": "notify.push", "args": {"text": "health_check: alive"}},
    ],
    "db.ping": [{"verb": "db.read", "args": {"table": "agents", "limit": 1}}],
}


def _structured_error(
    message: str, *, correlation_id: str, idempotency_key: str
) -> dict:
    """Return a 200 with our canonical structured error envelope."""
    return {
        "ok": False,
        "result": None,
        "error": {
            "version": 1,
            "code": "RequestError",
            "message": message,
            "hint": None,
            "details": None,
        },
        "latency_ms": 0,
        "correlation_id": correlation_id,
        "idempotency_key": idempotency_key,
    }


@app.post(f"{BASE}/api/agents/verb")
async def call_agent_verb(
    req: AgentCallRequest,
    x_correlation_id: Optional[str] = Header(default=None, alias="X-Correlation-Id"),
    x_idempotency_key: Optional[str] = Header(default=None, alias="X-Idempotency-Key"),
):
    correlation_id = x_correlation_id or req.trace_id or uuid.uuid4().hex
    idempotency_key = x_idempotency_key or req.idempotency_key or uuid.uuid4().hex

    try:
        out = orchestrator.call_verb(
            req.verb,
            req.args or {},
            req.meta or {},
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
            actor=(req.actor.dict() if req.actor else None),
            policy_ctx=(req.policy_ctx.dict() if req.policy_ctx else None),
        )
        # IMPORTANT: do not reshape/filter; return as-is so `error` survives
        # Also ensure correlation/idempotency are present
        out.setdefault("correlation_id", correlation_id)
        out.setdefault("idempotency_key", idempotency_key)
        return out
    except Exception as e:
        # Always shape errors as 200 with structured body
        return _structured_error(
            str(e), correlation_id=correlation_id, idempotency_key=idempotency_key
        )


@app.post(f"{BASE}/api/agents/plan", response_model=PlanResponse)
async def call_agent_plan(
    req: PlanRequest,
    x_correlation_id: Optional[str] = Header(default=None, alias="X-Correlation-Id"),
    x_idempotency_key: Optional[str] = Header(default=None, alias="X-Idempotency-Key"),
):
    plan_start = datetime.now(timezone.utc).isoformat()
    plan_corr = x_correlation_id or req.trace_id or uuid.uuid4().hex
    plan_idem = x_idempotency_key or req.idempotency_key or uuid.uuid4().hex

    # Expand named micro-plan
    steps_in: List[PlanStep] = list(req.steps or [])
    if getattr(req, "plan", None):
        micro = MICRO_PLANS.get(req.plan)
        if not micro:
            # Don’t raise HTTP 400; return structured failure
            return PlanResponse(
                ok=False,
                steps=[],
                total_latency_ms=0,
                correlation_id=plan_corr,
                idempotency_key=plan_idem,
                start=plan_start,
                end=datetime.now(timezone.utc).isoformat(),
                failed_step_index=0,
            )
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

    try:
        for idx, step in enumerate(steps_in):
            step_idem = f"{plan_idem}#{idx}:{step.verb}"

            orchestrator._log_event(
                "plan.step.start",
                {
                    "index": idx,
                    "verb": step.verb,
                    "step_idempotency_key": step_idem,
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

            orchestrator._log_event(
                "plan.step.end",
                {
                    "index": idx,
                    "verb": step.verb,
                    "ok": step_ok,
                    "latency_ms": latency,
                    "step_idempotency_key": step_idem,
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
                    result=(out.get("result") or {}),
                    latency_ms=latency,
                )
            )

            if not step_ok:
                failed_idx = idx
                if not req.continue_on_error:
                    ok = False
                    break

        if ok and any(not s.ok for s in steps_out):
            ok = False
            if failed_idx is None:
                failed_idx = next(i for i, s in enumerate(steps_out) if not s.ok)

        plan_end = datetime.now(timezone.utc).isoformat()

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

    except Exception as e:
        # Shape catastrophic plan failures (e.g., orchestrator crash in step)
        plan_end = datetime.now(timezone.utc).isoformat()
        orchestrator._log_event(
            "plan.end",
            {
                "plan": getattr(req, "plan", None),
                "ok": False,
                "total_latency_ms": int(total_latency),
                "failed_step_index": failed_idx,
                "steps_run": len(steps_out),
                "start": plan_start,
                "end": plan_end,
                "error": str(e),
            },
            correlation_id=plan_corr,
            idempotency_key=plan_idem,
        )
        return PlanResponse(
            ok=False,
            steps=steps_out,
            total_latency_ms=total_latency,
            correlation_id=plan_corr,
            idempotency_key=plan_idem,
            start=plan_start,
            end=plan_end,
            failed_step_index=failed_idx,
        )


# ------------ Local dev helpers ------------
@app.get(BASE + "/api/files/{path:path}")
async def serve_file(path: str):
    p = Path(path)
    if not p.exists() or not p.is_file():
        # Return structured 200 from the global handlers in main.py via mount,
        # but here we keep the standard exception for local dev:
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path)


@app.get("/")
async def root_note():
    return {"ok": True, "hint": f"try {BASE}/api/health or {BASE}/api/agents/verb"}


# Serve PWA last
if _frontend_root:
    app.mount(
        f"{BASE}", StaticFiles(directory=str(_frontend_root), html=True), name="app"
    )

    @app.get(f"{BASE}/", response_class=HTMLResponse)
    async def app_index() -> HTMLResponse:
        index_path = _frontend_root / "index.html"
        if index_path.exists():
            return HTMLResponse(index_path.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>App</h1><p>Frontend assets mounted.</p>")

else:

    @app.get(f"{BASE}/", response_class=HTMLResponse)
    async def app_index_missing() -> HTMLResponse:
        return HTMLResponse(
            "<h1>App</h1><p>No frontend bundle found. "
            "Place your built assets under ./app or ./frontend/dist.</p>"
        )
