from __future__ import annotations
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
import asyncio
import time
import sys
from backend.registry.capability_registry import (
    CapabilityRegistry,
    register_builtin_capabilities,
)
from backend.connectors.mcp_supabase import router as supabase_router

from dotenv import load_dotenv
from functools import lru_cache
import logging


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
from starlette.background import BackgroundTasks

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

# models
from backend.models.messages import (
    AgentCallRequest,
    PlanRequest,
    PlanResponse,
    PlanStepResult,
    PlanStep,
)

# orchestrator (meta-agent)
from backend.agents.orchestrator import Orchestrator
from backend.agents.article_summarizer_agent import handle_article_summary
from backend.agents.jobs_runner import _agent_loop, _run_notifier, _run_overseer
from backend.logging_utils import RequestLoggingMiddleware

import json
import difflib
import re


# ⬇️ moved in from main.py so nothing is lost
from backend.routers import schema as schema_router
from backend.agents.repo_agent import generate_artifact_from_task


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

# Shared capability registry (used by /agents/verbs and /agents/verb)
registry = CapabilityRegistry()
registry_builtin = register_builtin_capabilities(registry)
app.state.registry_builtin = registry_builtin

orchestrator = Orchestrator()


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
app.add_middleware(RequestLoggingMiddleware)


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
# Health / version
# ---------------------------
@app.get("/health", response_class=PlainTextResponse)
async def health_plain() -> str:
    return "ok"


@app.get("/api/health", response_class=PlainTextResponse)
async def health() -> str:
    return "ok"


@app.get("/api/version", response_class=PlainTextResponse)
async def version() -> str:
    return VERSION


# Root redirect (main.py behavior)
@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url=f"{BASE}/")


# supabase mcp router
app.include_router(supabase_router)

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
# Agents API (prefix-free; mounted at /api/agents)
# ---------------------------
agents_router = APIRouter(tags=["agents"])


@agents_router.get("/verbs")
def list_verbs() -> dict:
    """Verb discovery from the shared CapabilityRegistry."""
    reg = getattr(app.state, "registry", registry)
    try:
        if hasattr(reg, "list_verbs"):
            names = sorted(reg.list_verbs())  # type: ignore[attr-defined]
        else:
            names = sorted(list(getattr(reg, "_adapters", {}).keys()))
    except Exception:
        names = []
    return {"ok": True, "verbs": names}


if route_request:

    class RouteRequestBody(BaseModel):
        query: str | dict
        user_id: str = "local-user"

    @agents_router.post("/route", tags=["router"])
    async def handle_route_request(body: RouteRequestBody):
        """
        Main entry point for the intelligent router agent.
        It receives a query and uses an LLM to dispatch it to the
        best-suited agent.
        """
        try:
            # The handler expects a JSON string for complex queries, so we dump it.
            query_input = (
                json.dumps(body.query) if isinstance(body.query, dict) else body.query
            )
            agent, result = route_request(query=query_input, user_id=body.user_id)
            return {"ok": True, "agent": agent, "response": result}
        except Exception as e:
            logging.exception("Router agent failed")
            raise HTTPException(status_code=500, detail=str(e))


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
        reg = getattr(app.state, "registry", registry)
        meta = {
            **(req.meta or {}),
            "correlation_id": correlation_id,
            "idempotency_key": idempotency_key,
            "actor": (req.actor.dict() if req.actor else None),
            "policy_ctx": (req.policy_ctx.dict() if req.policy_ctx else None),
        }
        out = reg.dispatch(req.verb, req.args or {}, meta)

        if isinstance(out, dict):
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


# Mount the agents router canonically at /api/agents
app.include_router(agents_router, prefix="/api/agents")


# ---------------------------
# Root /route (router entrypoint with fallback)
# ---------------------------
class RouteRequestBody(BaseModel):
    query: str | dict
    user_id: str = "api"


@app.post("/route", tags=["router"])
def handle_route_root(body: RouteRequestBody):
    """
    Canonical router entry: if the intelligent router is unavailable,
    fall back to article summarizer so this endpoint never 404s.
    """
    # Prefer router if import succeeded above
    if route_request is not None:
        try:
            q = json.dumps(body.query) if isinstance(body.query, dict) else body.query
            agent, result = route_request(query=q, user_id=body.user_id)
            return {"ok": True, "agent": agent, "response": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Fallback to direct summarizer
    if not handle_article_summary:
        raise HTTPException(
            status_code=503, detail="Router unavailable and summarizer not importable"
        )
    return {
        "ok": True,
        "agent": "article_summarizer",
        "response": handle_article_summary(
            json.dumps(body.query) if isinstance(body.query, dict) else body.query
        ),
    }


# Optional back-compat: also mount under {BASE}/api/agents if BASE is set
if BASE and BASE.strip("/") and BASE.strip() != "/":
    app.include_router(agents_router, prefix=f"{BASE}/api/agents")


class ArticleSummarizeBody(BaseModel):
    url: str | None = None
    query: str | None = None


class CurateIn(BaseModel):
    run_id: str | None = None
    limit: int | None = 250


@app.post(f"{BASE}/api/article/summarize", tags=["article"])
def api_article_summarize_base(body: ArticleSummarizeBody):
    return api_article_summarize(body)


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


def _parse_files_artifact(s: str) -> dict[str, str]:
    """
    Parse a FILES artifact:

      BEGIN_FILE path/to/file
      <entire new file content>
      END_FILE

    Returns: { "path/to/file": "<content>", ... }
    """
    files: dict[str, str] = {}
    cur: str | None = None
    buf: list[str] = []
    for line in s.splitlines(keepends=True):
        if line.startswith("BEGIN_FILE "):
            if cur is not None:
                files[cur] = "".join(buf)
                buf.clear()
            cur = line[len("BEGIN_FILE ") :].strip()
            continue
        if line.startswith("END_FILE"):
            if cur is not None:
                files[cur] = "".join(buf)
                cur = None
                buf.clear()
            continue
        if cur is not None:
            buf.append(line)
    if cur is not None:
        files[cur] = "".join(buf)
    return files


def _synthesize_unified_patch(files: dict[str, str]) -> str:
    """
    Build a proper unified diff with valid @@ hunk ranges from on-disk content -> new file bodies.
    Uses difflib.unified_diff so `git apply --check` will pass.
    """
    chunks: list[str] = []
    for rel, new_text in files.items():
        p = Path(rel)
        try:
            old_text = p.read_text(encoding="utf-8")
        except FileNotFoundError:
            old_text = ""
        diff = difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            n=3,
        )
        chunk = "".join(diff)
        if chunk:
            chunks.append(chunk)
    return "".join(chunks)


# --- BEGIN: anti-destructive synthesis helpers ---
def _deletion_ratio(unified: str) -> float:
    removed = kept = 0
    for ln in unified.splitlines():
        if ln.startswith(("--- ", "+++ ", "@@ ")):
            continue
        if ln.startswith("-") and not ln.startswith("---"):
            removed += 1
        elif ln.startswith((" ", "+")):
            kept += 1
    denom = (removed + kept) or 1
    return removed / denom


def _merge_preserving(original: str, model_new: str) -> str:
    """
    Preserve ALL original lines and insert ONLY the model's added lines.
    Never delete original lines.
    """
    merged: list[str] = []
    a = original.splitlines(keepends=True)
    b = model_new.splitlines(keepends=True)
    for tag in difflib.ndiff(a, b):
        if tag.startswith("  "):  # unchanged (from original)
            merged.append(tag[2:])
        elif tag.startswith("- "):  # present only in original -> KEEP
            merged.append(tag[2:])
        elif tag.startswith("+ "):  # present only in model -> INSERT
            merged.append(tag[2:])
    if merged and not merged[-1].endswith("\n"):
        merged[-1] = merged[-1] + "\n"
    return "".join(merged)


def _apply_request_logging_anchors(original: str, model_new: str) -> str | None:
    """
    If the model_new contains the logging import/middleware, insert them
    at stable anchors in the original file and return the updated text.
    Otherwise return None.
    """
    import_line = "from backend.logging_utils import RequestLoggingMiddleware"
    mw_line = "app.add_middleware(RequestLoggingMiddleware)"

    if (import_line not in model_new) and (mw_line not in model_new):
        return None

    lines = original.splitlines(keepends=True)
    out = []
    inserted_import = import_line in original
    inserted_mw = mw_line in original

    # 1) insert import after 'from fastapi import FastAPI'
    for ln in lines:
        out.append(ln)
        # Match combined import lines like: from fastapi import FastAPI, HTTPException, ...
        if (not inserted_import) and re.match(
            r"^from fastapi import .*FastAPI.*$", ln.strip()
        ):
            out.append(import_line + "\n")
            inserted_import = True

    updated = "".join(out)
    if not updated.endswith("\n"):
        updated += "\n"

    # 2) insert middleware after first 'app = FastAPI()'
    out2 = []
    seen_app = False
    for ln in updated.splitlines(keepends=True):
        out2.append(ln)
        # Match app initializers with args, e.g. app = FastAPI(title="...", version=VERSION)
        if (
            (not inserted_mw)
            and (not seen_app)
            and ln.strip().startswith("app = FastAPI(")
        ):
            out2.append(mw_line + "\n")
            inserted_mw = True
            seen_app = True

    result = "".join(out2)
    if import_line in result and mw_line in result:
        return result
    return None


def _synthesize_diff_from_disk(rel: str, new_text: str) -> str:
    p = Path(rel)
    try:
        old_text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        old_text = ""
    return "".join(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            n=3,
        )
    )


# --- END: anti-destructive synthesis helpers ---


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
    """
    Plan a repo change and return either:
      - format=files  -> FILES artifact (BEGIN_FILE/END_FILE)
      - format=patch  -> a unified diff synthesized on the server from FILES
    This avoids model-made diffs (which often have malformed @@ hunks).
    """
    # ---- inputs / defaults ----
    task = (payload or {}).get("task")
    if not task:
        raise HTTPException(status_code=400, detail="Missing 'task'")

    repo = payload.get("repo", "personal-agent-ste")
    branch = payload.get("branch", "main")

    fmt = (request.query_params.get("format") or "files").strip().lower()
    if fmt not in {"files", "patch"}:
        raise HTTPException(status_code=400, detail="format must be 'files' or 'patch'")

    # ---- common: always ask the generator for FILES first ----
    def _get_files_artifact_text() -> str:
        """
        Call your artifact generator in 'files' mode and return the raw FILES text.
        """
        prefix = payload.get("path_prefix", "backend/")
        session = payload.get("session")

        files_art = generate_artifact_from_task(
            task,
            repo=repo,
            branch=branch,
            path_prefix=prefix,
            session=session,
            mode="files",
        )
        if not files_art.get("ok"):
            return str(files_art.get("content", ""))  # error text for client
        txt = files_art.get("content", "") or ""
        txt = _enforce_trailing_lf_per_file(txt)
        # normalize EOL
        txt = txt.replace("\r\n", "\n").replace("\r", "\n")
        return txt

    # ---------------------------
    # 1) Return raw FILES artifact (unchanged)
    # ---------------------------
    if fmt == "files":
        files_text = _get_files_artifact_text()
        # If the model returned a diff/fenced output by mistake, surface a clear 422
        if (
            re.search(r"^(diff --git|--- a/|\+\+\+ b/|@@ )", files_text, flags=re.M)
            or "```" in files_text
            or "~~~" in files_text
        ):
            raise HTTPException(
                status_code=422,
                detail="Generator returned a diff/fenced output. Expected FILES artifact (BEGIN_FILE/END_FILE).",
            )
        return PlainTextResponse(
            files_text.rstrip("\n") + "\n", media_type="text/plain; charset=utf-8"
        )

    # ---------------------------
    # 2) Synthesize a patch from FILES (do NOT trust model-made diffs)
    # ---------------------------
    # fmt == "patch"
    files_text = _get_files_artifact_text()
    if (
        re.search(r"^(diff --git|--- a/|\+\+\+ b/|@@ )", files_text, flags=re.M)
        or "```" in files_text
        or "~~~" in files_text
    ):
        raise HTTPException(
            status_code=422,
            detail="Generator returned a diff/fenced output. Expected FILES artifact (BEGIN_FILE/END_FILE).",
        )

    # Parse FILES artifact; accept a minimal JSON fallback too
    files = _parse_files_artifact(files_text)
    if not files:
        try:
            obj = json.loads(files_text)
            if isinstance(obj, dict) and "file_path" in obj and "file_content" in obj:
                files = {obj["file_path"]: obj["file_content"]}
        except Exception:
            pass

    if not files:
        raise HTTPException(status_code=422, detail="No FILES artifacts found.")

    # Build a proper unified diff from disk -> new file bodies
    chunks: list[str] = []
    # deterministic ordering (useful for tests)
    for rel in sorted(files.keys()):
        new_text = files[rel]
        p = Path(rel)
        try:
            old_text = p.read_text(encoding="utf-8")
        except FileNotFoundError:
            old_text = ""
        # Use difflib for correct @@ hunk ranges
        diff = difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            n=3,
        )
        chunk = "".join(diff)
        if chunk:
            chunks.append(chunk)

    patch = "".join(chunks)
    if not patch.strip():
        # no-op vs disk; emit a harmless diff header for the first file
        first = next(iter(sorted(files.keys())))
        empty = "".join(
            difflib.unified_diff(
                [], [], fromfile=f"a/{first}", tofile=f"b/{first}", n=3
            )
        )
        patch = empty or f"--- a/{first}\n+++ b/{first}\n"

    # ---- auto-repair if destructive (>20% deletions overall) ----
    if _deletion_ratio(patch) > 0.20:
        repaired_chunks: list[str] = []
        for rel in sorted(files.keys()):
            new_text = files[rel]
            p = Path(rel)
            try:
                original = p.read_text(encoding="utf-8")
            except FileNotFoundError:
                original = ""

            # 1) task-specific anchors first (for backend/api.py)
            merged = None
            if rel == "backend/api.py":
                merged = _apply_request_logging_anchors(original, new_text)

            # 2) fallback to preserve-all-original + add-only merge
            if merged is None:
                merged = _merge_preserving(original, new_text)

            repaired = _synthesize_diff_from_disk(rel, merged)
            if repaired.strip():
                repaired_chunks.append(repaired)

            repaired_patch = "".join(repaired_chunks)

            # If the repo is already up-to-date (repair produced no textual change),
            # return a harmless no-op diff header instead of 422.
            if not repaired_patch.strip():
                first = next(iter(sorted(files.keys())))
                empty = "".join(
                    difflib.unified_diff(
                        [], [], fromfile=f"a/{first}", tofile=f"b/{first}", n=3
                    )
                )
                return PlainTextResponse(
                    (empty or f"--- a/{first}\n+++ b/{first}\n").rstrip("\n") + "\n",
                    media_type="text/x-patch; charset=utf-8",
                )

            # If the repaired patch still looks destructive, block it.
            if _deletion_ratio(repaired_patch) > 0.20:
                raise HTTPException(
                    status_code=422,
                    detail="Destructive patch detected; executor must preserve content and only insert anchors.",
                )

            # Otherwise, return the repaired, non-destructive patch.
            return PlainTextResponse(
                repaired_patch.rstrip("\n") + "\n",
                media_type="text/x-patch; charset=utf-8",
            )

        # Non-destructive path: return the patch as-is
    return PlainTextResponse(
        patch.rstrip("\n") + "\n", media_type="text/x-patch; charset=utf-8"
    )


@repo_router.get("/runs/{run_id}/timeline")
def get_run_timeline(run_id: uuid.UUID):
    """
    Retrieves the structured timeline of events for a specific agent run.
    This provides a detailed audit trail for debugging and analysis.
    """
    try:
        from backend.services.supabase_service import supabase

        res = (
            supabase.table("agent_runs")
            .select("timeline")
            .eq("id", str(run_id))
            .single()
            .execute()
        )

        if not res.data:
            raise HTTPException(
                status_code=404, detail=f"Run with ID '{run_id}' not found."
            )

        timeline = res.data.get("timeline", [])
        return {"ok": True, "run_id": run_id, "timeline": timeline}
    except HTTPException as e:
        raise e  # Re-raise HTTPException to preserve status code
    except Exception as e:
        logging.exception(f"Failed to retrieve timeline for run_id={run_id}")
        raise HTTPException(status_code=500, detail=str(e))


@repo_router.post("/branch/cleanup")
def repo_branch_cleanup(payload: Dict[str, Any]):
    """
    Endpoint to be triggered by a webhook (e.g., from a GitHub Action)
    after a pull request has been merged. It cleans up the feature branch.
    """
    branch_name = payload.get("branch_name")
    if not branch_name:
        raise HTTPException(status_code=400, detail="Missing 'branch_name' in payload.")

    try:
        # Use the orchestrator to call the new verb
        result = orchestrator.call_verb(
            "repo.git.delete_branch",
            {"branch_name": branch_name},
            meta={},
            correlation_id=f"cleanup-{branch_name}",
            idempotency_key=f"cleanup-{branch_name}",
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
# ⬇️ JOBS API (new runner)
# ---------------------------
jobs_router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class JobsRunRequest(BaseModel):
    run_id: Optional[str] = None
    max_fetch_loops: int = 5
    shortlist_limit: int = 25


def run_jobs_pipeline_sync(req: JobsRunRequest, correlation_id: str):
    """The synchronous implementation of the jobs pipeline runner."""
    run_id = req.run_id or uuid.uuid4().hex
    orchestrator = Orchestrator()

    def _call_agent(agent_slug: str, user_text: str, run_id: str) -> Dict[str, Any]:
        """Helper to run an agent via the LLM loop."""
        return _agent_loop(agent_slug, user_text, run_id)

    def _call_verb(verb: str, args: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Helper to call a deterministic verb via the orchestrator."""
        meta = {"source": "jobs-runner"}
        idem_key = f"{run_id}-{verb}-{json.dumps(args, sort_keys=True)}"
        return orchestrator.call_verb(
            verb, args, meta, correlation_id=run_id, idempotency_key=idem_key
        )

    def _update_run(patch: Dict[str, Any]):
        try:
            _call_verb(
                "db.write",
                {"table": "jobs_runs", "update": patch, "where": {"run_id": run_id}},
                run_id,
            )
        except Exception:
            pass  # best-effort

    try:
        # 1. Initialize run
        _update_run({"status": "running", "run_id": run_id})

        # 2. Discoverer
        disc_out = _call_agent("jobs_discoverer", "Proceed with your task.", run_id)
        discoveries = (disc_out.get("finish", {}) or {}).get("discoveries", 0)
        _update_run({"discoveries": discoveries})

        # 3. Fetcher (loop)
        total_fetched = 0
        for i in range(req.max_fetch_loops):
            fetch_out = _call_agent("jobs_fetcher", "Proceed with your task.", run_id)
            fetched_this_loop = (fetch_out.get("finish", {}) or {}).get("fetched", 0)
            if fetched_this_loop == 0:
                break  # Queue is likely empty
            total_fetched += fetched_this_loop
        _update_run({"fetched": total_fetched})

        # 4. Extractor
        extr_out = _call_agent("jobs_extractor", "Proceed with your task.", run_id)
        extracted = (extr_out.get("finish", {}) or {}).get("extracted", 0)
        _update_run({"extracted": extracted})

        # 5. Scorer
        score_out = _call_agent("jobs_scorer", "Proceed with your task.", run_id)
        shortlist = (score_out.get("finish", {}) or {}).get("shortlist", [])
        _update_run({"shortlist_count": len(shortlist)})

        # 6. Curator (new)
        curator_out = _call_verb(
            "jobs.curator", {"run_id": run_id, "limit": 300}, run_id
        )
        upserted = ((curator_out.get("result") or {}).get("finish") or {}).get(
            "upserted", 0
        )
        _update_run({"upserted": upserted})

        # 7. Notifier
        notifier_out = _run_notifier(shortlist[: req.shortlist_limit], run_id)
        notified = ((notifier_out.get("finish", {}) or {}) or {}).get("sent", 0)
        _update_run({"notified_count": notified})

        # 8. Overseer
        _run_overseer(
            run_id,
            disc_out,
            {"finish": {"fetched": total_fetched}},
            extr_out,
            score_out,
            notifier_out,
        )

        # 9. Finalize run
        final_patch = {
            "status": "succeeded",
            "finished_at": "now()",
        }
        _update_run(final_patch)

        return {
            "run_id": run_id,
            "status": "succeeded",
            "discoveries": discoveries,
            "fetched": total_fetched,
            "extracted": extracted,
            "upserted": upserted,
            "notified": notified,
        }

    except Exception as e:
        error_message = f"Pipeline failed: {e}"
        _update_run(
            {"status": "failed", "error": error_message, "finished_at": "now()"}
        )
        return {
            "run_id": run_id,
            "status": "failed",
            "error": error_message,
        }


@jobs_router.post("/run")
async def run_jobs_pipeline(
    req: JobsRunRequest,
    background_tasks: BackgroundTasks,
    request: Request,
):
    """
    Triggers the end-to-end jobs pipeline asynchronously.
    """
    correlation_id = getattr(request.state, "correlation_id", uuid.uuid4().hex)

    # Run the pipeline in the background to avoid long-hanging HTTP requests.
    background_tasks.add_task(run_jobs_pipeline_sync, req, correlation_id)

    return {
        "ok": True,
        "message": "Jobs pipeline triggered successfully in the background.",
        "run_id": req.run_id or "newly-generated",
    }


app.include_router(jobs_router)

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


# Non-BASE alias for convenience
@app.get("/api/files/{path:path}")
async def serve_file_nobase(path: str):
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
