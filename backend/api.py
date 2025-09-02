from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
import asyncio
import time
from backend.registry.capability_registry import CapabilityRegistry


from dotenv import load_dotenv
from functools import lru_cache
import logging, json, sys

from fastapi import FastAPI, HTTPException, Header, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse,
    FileResponse,
    PlainTextResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

from pydantic_settings import BaseSettings, SettingsConfigDict

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

# ⬇️ moved in from main.py so nothing is lost
from backend.routers import schema as schema_router
from backend.agents.repo_agent import generate_artifact_from_task, propose_changes

# Load .env (Supabase keys, allowlists, etc.)
load_dotenv()


# ---------------------------
# Settings (env-driven config)
# ---------------------------
class Settings(BaseSettings):
    VERSION: str = "0.1.1"
    BASE: str = "/app"
    CORS_ALLOW_ORIGINS: str = "*"  # CSV

    # Policy envs (documented here even if adapters read them directly)
    DBWRITE_DISABLE_TABLE_GUARD: str = "0"
    DBWRITE_TABLE_ALLOWLIST: str = ""
    DBWRITE_COL_ALLOWLIST_events: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
VERSION = settings.VERSION
BASE = settings.BASE


def _csv_env(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [s.strip() for s in raw.split(",") if s.strip()]


# --- Structured logging (JSON to stdout) --
class JsonFormatter(logging.Formatter):
    def format(self, record):
        msg = {"level": record.levelname, "message": record.getMessage()}
        # If adapter passed an 'event' payload via logger.info(..., extra={"event": {...}})
        evt = getattr(record, "event", None)
        if isinstance(evt, dict):
            msg.update(evt)
        return json.dumps(msg, ensure_ascii=False)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
root = logging.getLogger()
# avoid duplicate handlers on reload
if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
    root.setLevel(logging.INFO)
    root.addHandler(handler)


# ---------------------------
# App & middleware
# ---------------------------
app = FastAPI(title="Personal Agent API", version=VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_csv_env("CORS_ALLOW_ORIGINS", settings.CORS_ALLOW_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TraceMiddleware(BaseHTTPMiddleware):
    """Inject correlation & idempotency into request.state and echo back in response headers."""

    async def dispatch(self, request: StarletteRequest, call_next):
        cid = request.headers.get("X-Correlation-Id") or uuid.uuid4().hex
        ikey = request.headers.get("X-Idempotency-Key") or uuid.uuid4().hex
        request.state.correlation_id = cid
        request.state.idempotency_key = ikey
        response = await call_next(request)
        response.headers.setdefault("X-Correlation-Id", cid)
        response.headers.setdefault("X-Idempotency-Key", ikey)
        return response


app.add_middleware(TraceMiddleware)


@app.on_event("startup")
async def _warmup_playwright() -> None:
    """
    Fire-and-forget warmup to reduce Playwright cold starts after deploys.
    Disable via env: BROWSER_WARMUP_DISABLED=1
    """
    if os.getenv("BROWSER_WARMUP_DISABLED", "0") == "1":
        logging.info("[warmup] Browser warmup disabled by env")
        return

    # ensure we only run once per process (guards duplicate startup events)
    if getattr(app.state, "did_browser_warmup", False):
        return
    app.state.did_browser_warmup = True

    async def _job():
        try:
            registry = getattr(app.state, "registry", None) or CapabilityRegistry()

            # Local-only warmup: no network. Small per-action timeout.
            args = {
                "timeout_ms": 2000,
                "headless": True,
            }
            meta = {
                "correlation_id": f"warmup.browser.{int(time.time())}",
                "idempotency_key": f"warmup-{int(time.time())}",
            }

            async def _dispatch():
                # call the new lightweight verb
                return await asyncio.to_thread(
                    registry.dispatch, "browser.warmup", args, meta
                )

            try:
                # OUTER wall-clock guard so startup never stalls (even if Playwright is grumpy)
                result = await asyncio.wait_for(_dispatch(), timeout=4.0)
                ok = isinstance(result, dict) and result.get("ok", True)
                logging.info(
                    "[warmup] browser.warmup %s", "ok" if ok else f"non-ok: {result}"
                )
            except asyncio.TimeoutError:
                logging.info("[warmup] browser.warmup outer timeout; continuing")
        except Exception as e:
            logging.info("[warmup] skipped due to error: %s", e)

    asyncio.create_task(_job())


# ---------------------------
# Static frontend mount (PWA)
# ---------------------------
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


# ---------------------------
# Health / version (keep both plain /health and BASE health)
# ---------------------------
@app.get("/health", response_class=PlainTextResponse)
async def health_plain() -> str:
    return "ok"


@app.get(f"{BASE}/api/health", response_class=PlainTextResponse)
async def health() -> str:
    return "ok"


@app.get(f"{BASE}/api/version", response_class=PlainTextResponse)
async def version() -> str:
    return VERSION


# Root redirect (main.py behavior)
@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url=f"{BASE}/")


# ---------------------------
# Orchestrator & micro-plans
# ---------------------------
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


# ---------------------------
# Agents API
# ---------------------------
agents_router = APIRouter(prefix=f"{BASE}/api/agents", tags=["agents"])


@agents_router.get("/verbs")
def list_verbs() -> dict:
    """Optional verb discovery; returns [] if orchestrator can't enumerate."""
    verbs = []
    try:
        if hasattr(orchestrator, "list_verbs"):
            verbs = sorted(orchestrator.list_verbs())  # type: ignore[attr-defined]
    except Exception:
        verbs = []
    return {"ok": True, "verbs": verbs}


@agents_router.post("/verb")
async def call_agent_verb(
    req: AgentCallRequest,
    request: Request,
    x_correlation_id: Optional[str] = Header(default=None, alias="X-Correlation-Id"),
    x_idempotency_key: Optional[str] = Header(default=None, alias="X-Idempotency-Key"),
):
    correlation_id = (
        x_correlation_id
        or getattr(request.state, "correlation_id", None)
        or req.trace_id
        or uuid.uuid4().hex
    )
    idempotency_key = (
        x_idempotency_key
        or getattr(request.state, "idempotency_key", None)
        or req.idempotency_key
        or uuid.uuid4().hex
    )
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
        out.setdefault("correlation_id", correlation_id)
        out.setdefault("idempotency_key", idempotency_key)
        return out
    except Exception as e:
        return _structured_error(
            str(e), correlation_id=correlation_id, idempotency_key=idempotency_key
        )


@agents_router.post("/plan", response_model=PlanResponse)
async def call_agent_plan(
    req: PlanRequest,
    request: Request,
    x_correlation_id: Optional[str] = Header(default=None, alias="X-Correlation-Id"),
    x_idempotency_key: Optional[str] = Header(default=None, alias="X-Idempotency-Key"),
):
    plan_start = datetime.now(timezone.utc).isoformat()
    plan_corr = (
        x_correlation_id
        or getattr(request.state, "correlation_id", None)
        or req.trace_id
        or uuid.uuid4().hex
    )
    plan_idem = (
        x_idempotency_key
        or getattr(request.state, "idempotency_key", None)
        or req.idempotency_key
        or uuid.uuid4().hex
    )

    # Expand named micro-plan
    steps_in: List[PlanStep] = list(req.steps or [])
    if getattr(req, "plan", None):
        micro = MICRO_PLANS.get(req.plan)
        if not micro:
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


app.include_router(agents_router)


# ---------------------------
# Debug router (policy/introspection) — cache endpoints already existed
# ---------------------------
def _norm_table(s: str) -> str:
    s = (s or "").strip().lower()
    return s[7:] if s.startswith("public.") else s


def _csv_set(name: str) -> set[str]:
    raw = os.getenv(name, "") or ""
    return {x.strip() for x in raw.split(",") if x.strip()}


debug_router = APIRouter(prefix=f"{BASE}/api/debug", tags=["debug"])
debug_router_nobase = APIRouter(prefix="/api/debug", tags=["debug"])


@debug_router.get("/cache")
@debug_router_nobase.get("/cache")
def debug_cache():
    return {"ok": True, "cache": orchestrator.get_cache_stats()}


@debug_router.post("/cache/clear")
@debug_router_nobase.post("/cache/clear")
def debug_cache_clear():
    return {"ok": True, "cache": orchestrator.clear_cache()}


# Optional: route lister for quick diagnostics
@debug_router.get("/routes")
@debug_router_nobase.get("/routes")
def debug_routes():
    return {
        "ok": True,
        "paths": sorted([getattr(r, "path", str(r)) for r in app.routes]),
        "ts": datetime.now(timezone.utc).isoformat(),
    }


app.include_router(debug_router)
app.include_router(debug_router_nobase)


# ---------------------------
# ⬇️ REPO API (moved here from main.py)
# ---------------------------
def _enforce_trailing_lf_per_file(artifact: str) -> str:
    t = artifact.replace("\r\n", "\n").replace("\r", "\n")
    lines = t.split("\n")
    out: list[str] = []
    i = 0
    L = len(lines)
    while i < L:
        if not lines[i].startswith("BEGIN_FILE "):
            i += 1
            continue
        path = lines[i][len("BEGIN_FILE ") :].strip()
        i += 1
        body_lines: list[str] = []
        while i < L and lines[i].strip() != "END_FILE":
            body_lines.append(lines[i])
            i += 1
        if i >= L:
            raise ValueError(f"Missing END_FILE for {path}")
        i += 1
        body = "\n".join(body_lines).replace("\r\n", "\n").replace("\r", "\n")
        body = body.rstrip("\n") + "\n"
        out.append(f"BEGIN_FILE {path}\n{body}END_FILE\n")
    return ("".join(out)).rstrip("\n") + "\n"


repo_router = APIRouter(prefix=f"{BASE}/api/repo", tags=["repo"])


@repo_router.get("/health")
def repo_health():
    return {"ok": True, "model": os.getenv("CHAT_MODEL", "gpt-5")}


@repo_router.post("/plan")
def repo_plan(payload: Dict[str, Any], request: Request):
    task = (payload or {}).get("task")
    if not task:
        raise HTTPException(status_code=400, detail="Missing 'task'")

    repo = payload.get("repo", "personal-agent-ste")
    branch = payload.get("branch", "main")
    prefix = payload.get("path_prefix", "backend/")
    k = int(payload.get("k", 12))
    session = payload.get("session")
    thread_n = payload.get("thread_n")

    fmt = request.query_params.get("format") or request.headers.get("X-RMS-Format")

    if fmt == "files":
        art = generate_artifact_from_task(
            task,
            repo=repo,
            branch=branch,
            path_prefix=prefix,
            session=session,
            mode="files",
        )
        if not art.get("ok"):
            return PlainTextResponse(
                str(art.get("content", "")),
                status_code=422,
                media_type="text/plain; charset=utf-8",
            )
        content = art.get("content", "")
        content = _enforce_trailing_lf_per_file(content)
        content = content.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"
        return PlainTextResponse(content, media_type="text/plain; charset=utf-8")

    if fmt == "patch":
        art = generate_artifact_from_task(
            task,
            repo=repo,
            branch=branch,
            path_prefix=prefix,
            session=session,
            mode="patch",
        )
        patch = art.get("patch") or art.get("content") or ""
        from backend.utils.patch_sanitizer import (
            sanitize_patch,
            validate_patch_structure,
        )

        patch, _warnings = sanitize_patch(patch)
        ok, msg = validate_patch_structure(patch, path_prefix=prefix)
        if not ok:
            # best-effort PowerShell fixer if available
            from pathlib import Path as _Path
            import subprocess, tempfile

            try:
                with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", delete=False
                ) as f_in:
                    f_in.write(patch)
                    in_path = f_in.name
                out_path = in_path + ".out.patch"
                script = (
                    _Path(__file__).resolve().parents[1] / "tools" / "fix-patch.ps1"
                )
                if script.exists():
                    proc = subprocess.run(
                        [
                            "pwsh",
                            "-NoProfile",
                            "-ExecutionPolicy",
                            "Bypass",
                            "-File",
                            str(script),
                            "-InPath",
                            in_path,
                            "-OutPath",
                            out_path,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=20,
                    )
                    if proc.returncode == 0 and _Path(out_path).exists():
                        patch = _Path(out_path).read_text(encoding="utf-8")
                        ok, msg = validate_patch_structure(patch, path_prefix=prefix)
            except Exception:
                pass
        if not ok:
            raise HTTPException(status_code=422, detail=f"Invalid patch: {msg}")
        return PlainTextResponse(patch, media_type="text/x-patch; charset=utf-8")

    out = propose_changes(
        task,
        repo=repo,
        branch=branch,
        commit="HEAD",
        k=k,
        path_prefix=prefix,
        session=session,
        thread_n=thread_n,
    )
    return out


@repo_router.post("/files")
def repo_files(payload: Dict[str, Any]):
    task = (payload or {}).get("task")
    if not task:
        raise HTTPException(status_code=400, detail="Missing 'task'")

    repo = payload.get("repo", "personal-agent-ste")
    branch = payload.get("branch", "main")
    prefix = payload.get("path_prefix", "backend/")
    session = payload.get("session")

    art = generate_artifact_from_task(
        task,
        repo=repo,
        branch=branch,
        path_prefix=prefix,
        session=session,
        mode="files",
    )
    content = art.get("content", "")
    content = _enforce_trailing_lf_per_file(content)
    content = content.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"
    return PlainTextResponse(content, media_type="text/plain; charset=utf-8")


app.include_router(repo_router)

# ---------------------------
# Schema router (moved here from main.py)
# ---------------------------
app.include_router(schema_router.router)


# ---------------------------
# Global error handler
# ---------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    cid = getattr(request.state, "correlation_id", uuid.uuid4().hex)
    ikey = getattr(request.state, "idempotency_key", uuid.uuid4().hex)
    return _structured_error(str(exc), correlation_id=cid, idempotency_key=ikey)


# ---------------------------
# Lifespan hooks (light)
# ---------------------------
@app.on_event("startup")
async def on_startup():
    try:
        _ = orchestrator
    except Exception as e:
        print("Startup warning:", e)


@app.on_event("shutdown")
async def on_shutdown():
    pass


# ---------------------------
# Local dev helpers
# ---------------------------
@app.get(BASE + "/api/files/{path:path}")
async def serve_file(path: str):
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path)


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
