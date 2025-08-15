# backend/main.py
from __future__ import annotations

import os
import json
import re
import tempfile
import subprocess
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, Tuple
import re, subprocess, tempfile, os
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import (
    JSONResponse,
    RedirectResponse,
    HTMLResponse,
    FileResponse,
    PlainTextResponse,
)
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv

# Optional linter (leave as-is if present in your repo)
from backend.utils.patch_linter import lint_patch, make_followup_prompt  # type: ignore

# Agents / services
from backend.agents.router_agent import route_request
from backend.agents.repo_agent import (
    propose_changes,
    answer_about_repo,
    generate_patch_from_prompt,
)
from backend.utils.nl_formatter import ensure_natural  # type: ignore
from backend.utils.agent_protocol import AgentResponse  # type: ignore
from backend.services import conversation as conv  # type: ignore

load_dotenv()

app = FastAPI(title="Personal Agent API")
logger = logging.getLogger("rms.main")

# -------------------- Debug / Health --------------------

@app.get("/app/api/debug/supabase-key")
async def debug_supabase_key():
    key = os.getenv("SUPABASE_SERVICE_ROLE")
    if key:
        return {"SUPABASE_SERVICE_ROLE_start": key[:8] + "..."}
    return {"SUPABASE_SERVICE_ROLE": None}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/app/api/repo/health")
def repo_health():
    return {"ok": True, "model": os.getenv("CHAT_MODEL", "gpt-5")}

@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/app/")

# -------------------- Patch Sanitization (pure Python) --------------------

# Wrappers / noise
_CODEBLOCK_RE     = re.compile(r"```(?:diff|patch)?\s*(?P<body>.*?)```", re.IGNORECASE | re.DOTALL)
_CBLOCK_RE        = re.compile(r"/\*.*?\*/", re.DOTALL)
_ELLIPSIS_LINE_RE = re.compile(r"^\s*\.\.\.\s*$", re.MULTILINE)
_ENVELOPE_RE      = re.compile(r"(?ms)^\s*BEGIN_PATCH\s*\n(?P<body>.*?)(?:\n)?\s*END_PATCH\s*$")

# Headers / hunk regexes
_DIFF_GIT_RE       = re.compile(r"^diff --git a/(.+?) b/(.+?)$")
_HUNK_HDR_RE_ANY   = re.compile(r"^@@\s+-([0-9]+)(?:,([0-9]+))?\s+\+([0-9]+)?(?:,([0-9]+))?\s+@@$")
_HUNK_HDR_RE_FULL  = re.compile(r"^@@\s+-([0-9]+)(?:,([0-9]+))?\s+\+([0-9]+)(?:,([0-9]+))?\s+@@$")

_HDR_NEWFILE   = re.compile(r"^\s*---\s*/dev/null\s*$")
_HDR_OLD_A     = re.compile(r"^\s*---\s*a/(.+)\s*$")
_HDR_NEW_B     = re.compile(r"^\s*\+\+\+\s*b/(.+)\s*$")

def _unwrap_and_clean(text: str) -> str:
    """Remove BEGIN/END envelopes, code fences, C-style comments, and standalone ellipsis; normalize to LF."""
    if not text:
        return ""
    s = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip optional BEGIN/END envelope
    m_env = _ENVELOPE_RE.search(s)
    if m_env:
        s = m_env.group("body")

    # Remove all fenced code blocks, keeping their bodies
    fence = re.compile(r"```(?:diff|patch)?\s*([\s\S]*?)```", re.IGNORECASE)
    while True:
        before = s
        s = fence.sub(lambda m: (m.group(1) or ""), s)
        if s == before:
            break

    # Strip stray ``` lines
    s = re.sub(r"(?m)^\s*```\s*$", "", s)
    # Strip C-style comment blocks and ellipsis-only lines
    s = _CBLOCK_RE.sub("", s)
    s = _ELLIPSIS_LINE_RE.sub("", s)

    # IMPORTANT: allow leading whitespace before diff header
    return s.lstrip()

def _ensure_file_headers(text: str) -> str:
    """
    Ensure each '@@' hunk is preceded by proper file headers within a section.
    If headers are missing, synthesize:
      - '--- /dev/null' for new files,
      - otherwise '--- a/<path>' and always '+++ b/<path>'.
    """
    lines = text.split("\n")
    out: list[str] = []

    in_section = False
    have_old = False
    have_new = False
    new_file_mode = False
    path_b: str | None = None

    def reset_section_flags():
        nonlocal have_old, have_new, new_file_mode
        have_old = False
        have_new = False
        new_file_mode = False

    for l in lines:
        if l.startswith("diff --git "):
            in_section = True
            reset_section_flags()
            m = _DIFF_GIT_RE.match(l)
            path_b = (m.group(2) if m else None)
            out.append(l)
            continue

        if not in_section:
            out.append(l)
            continue

        if l.startswith("new file mode "):
            new_file_mode = True
            out.append(l)
            continue

        if l.startswith("--- "):
            have_old = True
            out.append(l)
            continue

        if l.startswith("+++ "):
            have_new = True
            out.append(l)
            continue

        if l.startswith("@@ "):
            if not (have_old and have_new):
                if new_file_mode:
                    out.append("--- /dev/null")
                else:
                    out.append(f"--- a/{path_b or ''}")
                out.append(f"+++ b/{path_b or ''}")
                have_old = have_new = True
            out.append(l)
            continue

        out.append(l)

    return "\n".join(out)

def _normalize_and_synthesize_newfile_hunks(text: str) -> tuple[str, dict]:
    """
    For new-file sections, ensure content is inside hunks with '+'-prefixed lines and corrected header counts.
    Also synthesize a hunk if content appears outside any hunk in a new-file section.
    """
    if not text:
        return "", {"sections": 0, "newfile_hunks": 0, "prefixed_lines": 0, "rewritten_headers": 0, "synth_hunks": 0}

    lines = text.split("\n")
    out: list[str] = []

    sections = 0
    newfile_hunks = 0
    prefixed = 0
    rewritten = 0
    synthesized = 0

    in_section = False
    section_is_newfile = False
    in_hunk = False
    hunk_is_newfile = False
    hdr_idx: int | None = None
    plus_start = 1
    plus_count = 0

    def is_header(l: str) -> bool:
        return (
            l.startswith("diff --git ") or l.startswith("index ")
            or l.startswith("new file mode ") or l.startswith("deleted file mode ")
            or l.startswith("rename from ") or l.startswith("rename to ")
            or l.startswith("--- ") or l.startswith("+++ ")
        )

    def flush_hunk():
        nonlocal hdr_idx, plus_start, plus_count, hunk_is_newfile, rewritten, newfile_hunks
        if hdr_idx is not None and hunk_is_newfile:
            ps = plus_start if (plus_start and plus_start > 0) else 1
            out[hdr_idx] = f"@@ -0,0 +{ps},{max(0, plus_count)} @@"
            rewritten += 1
            newfile_hunks += 1
        hdr_idx = None
        plus_start = 1
        plus_count = 0
        hunk_is_newfile = False

    i = 0
    while i < len(lines):
        l = lines[i]

        if l.startswith("diff --git "):
            flush_hunk()
            in_section = True
            section_is_newfile = False
            in_hunk = False
            sections += 1
            out.append(l)
            i += 1
            continue

        if not in_section:
            out.append(l)
            i += 1
            continue

        if l.startswith("--- "):
            if l.strip() == "--- /dev/null":
                section_is_newfile = True
            out.append(l)
            i += 1
            continue

        if l.startswith("@@ "):
            flush_hunk()
            in_hunk = True
            hdr_idx = len(out)
            out.append(l)  # placeholder
            m = _HUNK_HDR_RE_ANY.match(l)
            ms = ml = ps = None
            if m:
                ms = int(m.group(1))
                ml = int(m.group(2)) if m.group(2) is not None else None
                ps = int(m.group(3)) if m.group(3) is not None else None
            minus_len_zero = (ml == 0) or (ml is None and ms == 0)
            hunk_is_newfile = bool(section_is_newfile or minus_len_zero)
            plus_start = ps if (ps and ps > 0) else 1
            plus_count = 0
            i += 1
            continue

        if is_header(l):
            flush_hunk()
            in_hunk = False
            out.append(l)
            i += 1
            continue

        # Content
        if section_is_newfile and not in_hunk:
            # Synthesize a hunk
            in_hunk = True
            hunk_is_newfile = True
            hdr_idx = len(out)
            out.append("@@ -0,0 +1,0 @@")  # will be rewritten on flush
            plus_start = 1
            plus_count = 0
            synthesized += 1

        if in_hunk and hunk_is_newfile:
            if not (l.startswith("+") or l.startswith("-") or l.startswith(" ") or l.startswith("\\")):
                l = "+" + l
                prefixed += 1
            if l.startswith("+"):
                plus_count += 1
            out.append(l)
            i += 1
            continue

        out.append(l)
        i += 1

    flush_hunk()

    return "\n".join(out), {
        "sections": sections,
        "newfile_hunks": newfile_hunks,
        "prefixed_lines": prefixed,
        "rewritten_headers": rewritten,
        "synth_hunks": synthesized,
    }

def _normalize_modified_hunks(text: str) -> str:
    """
    For modified-file hunks:
      - Prefix any bare in-hunk line with a single space (context).
      - Recompute old/new lengths from body and rewrite the header.
    """
    if not text:
        return ""
    lines = text.split("\n")
    out: list[str] = []

    in_section = False
    in_hunk = False
    hdr_idx: int | None = None
    old_count = 0
    new_count = 0

    def flush():
        nonlocal hdr_idx, old_count, new_count, in_hunk
        if hdr_idx is not None:
            hdr = out[hdr_idx]
            m = _HUNK_HDR_RE_FULL.match(hdr)
            if m:
                ms = int(m.group(1))
                ps = int(m.group(3))
                out[hdr_idx] = f"@@ -{ms},{max(0, old_count)} +{ps},{max(0, new_count)} @@"
        hdr_idx = None
        old_count = 0
        new_count = 0
        in_hunk = False

    def is_header(l: str) -> bool:
        return (
            l.startswith("diff --git ")
            or l.startswith("index ")
            or l.startswith("--- ")
            or l.startswith("+++ ")
            or l.startswith("new file mode ")
            or l.startswith("deleted file mode ")
            or l.startswith("rename from ")
            or l.startswith("rename to ")
        )

    for l in lines:
        if l.startswith("diff --git "):
            flush()
            in_section = True
            out.append(l)
            continue

        if not in_section:
            out.append(l)
            continue

        if l.startswith("@@ "):
            flush()
            in_hunk = True
            hdr_idx = len(out)
            out.append(l)  # placeholder
            old_count = 0
            new_count = 0
            continue

        if is_header(l):
            flush()
            out.append(l)
            continue

        if in_hunk:
            if not (l.startswith(" ") or l.startswith("+") or l.startswith("-") or l.startswith("\\")):
                l = " " + l
            if l.startswith(" ") or l.startswith("-"):
                old_count += 1
            if l.startswith(" ") or l.startswith("+"):
                new_count += 1
            out.append(l)
            continue

        out.append(l)

    flush()
    return "\n".join(out)

def _sanitize_patch_text(text: str) -> tuple[str, dict]:
    s = _unwrap_and_clean(text)
    s = _ensure_file_headers(s)
    s, stats = _normalize_and_synthesize_newfile_hunks(s)
    s = _normalize_modified_hunks(s)
    if not s.endswith("\n"):
        s += "\n"
    return s, stats

def _fix_patch_structure(patch_text: str) -> tuple[str, dict]:
    """Secondary structural pass (hook for future tweaks)."""
    s = (patch_text or "").replace("\r\n", "\n").replace("\r", "\n")
    return s, {}

def _maybe_run_powershell_fix(patch_text: str) -> str:
    """
    If on Windows and tools\\fix-patch.ps1 exists, run it to post-fix the patch.
    Opt-out by setting PATCH_FIXER=off.
    """
    try:
        if os.environ.get("PATCH_FIXER", "").lower() == "off":
            return patch_text
        if os.name != "nt":
            return patch_text
        here = Path(__file__).resolve()
        repo_root = here.parents[1] if len(here.parents) >= 2 else here.parent
        script = repo_root / "tools" / "fix-patch.ps1"
        if not script.exists():
            return patch_text
        ps = shutil.which("powershell")
        if not ps:
            return patch_text
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "in.patch"
            outp = Path(td) / "out.patch"
            inp.write_text(patch_text.replace("\r\n", "\n").replace("\r", "\n"), encoding="utf-8")
            cmd = [
                ps, "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-File", str(script),
                "-InPath", str(inp),
                "-OutPath", str(outp),
                "-RepoRoot", str(repo_root),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if r.returncode == 0 and outp.exists():
                fixed = outp.read_text(encoding="utf-8")
                return fixed if fixed else patch_text
            else:
                logger.warning("fix-patch.ps1 returned %s: %s", r.returncode, r.stderr.strip())
    except Exception as e:
        logger.warning("fix-patch.ps1 error: %s", e)
    return patch_text

def _looks_like_unified_diff(text: str) -> bool:
    if not text:
        return False
    if text.lstrip().startswith("diff --git"):
        return True
    return text.startswith("--- ") and ("\n+++ " in text)

_NON_ASCII = re.compile(r'[^\x09\x0A\x0D\x20-\x7E]')
_DIFF_HEADER = re.compile(r'^diff --git a/(?P<a>.+?) b/(?P<b>.+?)$', re.M)
_FILE_SPLIT = re.compile(r'(?=^diff --git a/.+ b/.+$)', re.M)

def _validate_unified_diff(diff_text: str) -> None:
    """
    Raise HTTPException(400) if the diff is malformed or violates policy.
    """
    from fastapi import HTTPException

    if not diff_text.strip().startswith("diff --git"):
        raise HTTPException(status_code=400, detail="Not a unified diff (missing 'diff --git').")

    sections = [s for s in _FILE_SPLIT.split(diff_text) if s.strip()]
    seen = set()
    for sec in sections:
        m = _DIFF_HEADER.search(sec)
        if not m:
            raise HTTPException(status_code=400, detail="Patch fragment without header.")
        path = m.group("b")
        if path in seen:
            raise HTTPException(status_code=400, detail=f"Duplicate file section for {path}.")
        seen.add(path)

        # Reject marking existing main.py as new
        if path == "backend/main.py" and "--- /dev/null" in sec:
            raise HTTPException(status_code=400, detail="backend/main.py must be a modified file, not new.")

        # Reject bad content in added lines
        for line in sec.splitlines():
            if not line.startswith('+'):
                continue
            if _NON_ASCII.search(line):
                raise HTTPException(status_code=400, detail="Non-ASCII characters in added lines.")
            if "X-Request-ID" in line:
                raise HTTPException(status_code=400, detail="Use X-Correlation-ID, not X-Request-ID.")
            if re.search(r'\bapp\s*=\s*FastAPI\s*\(', line):
                raise HTTPException(status_code=400, detail="Do not re-create the FastAPI app object.")

    # Final gate: git apply --check against the current repo
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", newline="\n")
        tmp.write(diff_text if diff_text.endswith("\n") else diff_text + "\n")
        tmp.flush()
        tmp.close()
        proc = subprocess.run(
            ["git", "apply", "--check", "--whitespace=nowarn", tmp.name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(Path(".").resolve())
        )
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout).strip().splitlines()[0] if (proc.stderr or proc.stdout) else "git apply --check failed"
            raise HTTPException(status_code=400, detail=msg)
    finally:
        if tmp:
            try: os.unlink(tmp.name)
            except Exception: pass


# -------------------- Repo endpoints --------------------

@app.post("/app/api/repo/query")
def repo_query(payload: Dict[str, Any]):
    q = (payload or {}).get("task") or (payload or {}).get("question") or (payload or {}).get("q")
    if not q:
        raise HTTPException(status_code=400, detail="Missing 'task'/'question'/'q'")
    repo   = payload.get("repo", "personal-agent-ste")
    branch = payload.get("branch", "main")
    prefix = payload.get("path_prefix")
    k      = int(payload.get("k", 8))
    commit = payload.get("commit")
    session = payload.get("session")
    thread_n = payload.get("thread_n")

    try:
        out = answer_about_repo(
            q, repo=repo, branch=branch, k=k, path_prefix=prefix, commit=commit, session=session, thread_n=thread_n
        )
        if isinstance(out, str):
            return PlainTextResponse(out)
        if isinstance(out, dict) and out.get("agent") == "repo":
            text = out.get("answer") or out.get("message") or out.get("data") or json.dumps(out, ensure_ascii=False)
            return PlainTextResponse(str(text))
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/app/api/repo/plan", response_model=None)
async def repo_plan(request: Request) -> Response:
    try:
        payload: Dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    task = payload.get("task")
    if not task:
        raise HTTPException(status_code=400, detail="Missing 'task'")

    repo   = payload.get("repo", "personal-agent-ste")
    branch = payload.get("branch", "main")
    prefix = payload.get("path_prefix")  # e.g., "backend/"
    k      = int(payload.get("k", 12))
    session = payload.get("session")
    thread_n = payload.get("thread_n")

    # Optional checks/lint inputs from caller (pass-through to future linter if needed)
    checks = payload.get("checks") or {}
    required = checks.get("required") or []
    forbidden = checks.get("forbidden") or []
    max_files = checks.get("max_files")
    max_bytes = checks.get("max_bytes")

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

    fmt = (request.query_params.get("format") or "").lower()
    accept = (request.headers.get("accept") or "").lower()
    wants_patch = (fmt == "patch") or ("text/x-patch" in accept) or ("text/x-diff" in accept)

    # Extract raw model text / patch
    # ...
    raw_patch = ""
    if isinstance(out, dict):
        raw_patch = out.get("patch") or ""
        if not raw_patch:
            maybe_prompt = out.get("prompt") or ""
            if maybe_prompt:
                # 👇 pass session so the generator can remember this patch
                raw_patch = generate_patch_from_prompt(maybe_prompt, session=session) or ""
    # ...

    # if out is not a dict, skip fallback to avoid attribute errors

    # Sanitize / normalize
    sanitized, stats = _sanitize_patch_text(raw_patch)
    is_sanitized = (sanitized != (raw_patch or "").replace("\r\n", "\n").replace("\r", "\n").lstrip())

    # Secondary structural pass (currently a no-op hook; keep for future)
    sanitized, struct_stats = _fix_patch_structure(sanitized)
    if struct_stats:
        is_sanitized = True
        stats = (stats or {})
        stats.update(struct_stats)

    # Optional local PowerShell fixer (Windows dev only)
    sanitized = _maybe_run_powershell_fix(sanitized)

    if wants_patch:
        _validate_unified_diff(sanitized)
        return PlainTextResponse(sanitized, media_type="text/x-patch")


    # Otherwise, return a JSON plan (legacy behavior)
    return JSONResponse(out)

# -------------------- Memory endpoints --------------------

@app.post("/app/api/repo/memory/reset")
def repo_memory_reset(payload: Dict[str, Any]):
    session = (payload or {}).get("session")
    if not session:
        raise HTTPException(status_code=400, detail="Missing 'session'")
    deleted = conv.clear_session(session)
    return {"ok": True, "session": session, "deleted": deleted}

@app.post("/app/api/repo/memory/config")
def repo_memory_config(payload: Dict[str, Any]):
    session = (payload or {}).get("session")
    n = (payload or {}).get("n")
    if not session or n is None:
        raise HTTPException(status_code=400, detail="Missing 'session' or 'n'")
    conv.set_session_n(session, int(n))
    return {"ok": True, "session": session, "n": conv.get_session_n(session) or conv.default_n()}

@app.get("/app/api/repo/memory/export")
def repo_memory_export(session: str, limit: int = 50):
    if not session:
        raise HTTPException(status_code=400, detail="Missing 'session'")
    return {"ok": True, "session": session, "limit": int(limit), "messages": conv.export_messages(session, limit)}

# -------------------- Natural-language response middleware --------------------

class NaturalLanguageMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)

            if not isinstance(response, JSONResponse):
                return response

            # Clone response body safely
            body_bytes = b""
            try:
                async for chunk in response.body_iterator:  # type: ignore[attr-defined]
                    body_bytes += chunk
            except Exception:
                return response  # return as-is if we can't read it

            response.body_iterator = None  # prevent double iteration

            # If not JSON, passthrough
            try:
                payload = json.loads(body_bytes.decode("utf-8"))
            except Exception:
                return JSONResponse(
                    content=body_bytes.decode("utf-8") if body_bytes else None,
                    status_code=response.status_code,
                )

            # Naturalize
            try:
                formatted = ensure_natural(payload)
            except Exception as e:
                formatted = {
                    "agent": (payload.get("agent") if isinstance(payload, dict) else "system"),
                    "intent": "error",
                    "message": f"Formatting error: {e}",
                    "raw": payload,
                }

            return JSONResponse(content=formatted, status_code=response.status_code)

        except Exception as e:
            return JSONResponse(
                content={"agent": "system", "intent": "error", "message": f"Middleware error: {e}"},
                status_code=500,
            )

app.add_middleware(NaturalLanguageMiddleware)

# -------------------- Helpers for router endpoint --------------------

def _extract_query(query: str | None, body: Dict[str, Any] | None) -> Tuple[str | None, Dict[str, Any] | None]:
    if query:
        return query, body
    if body and isinstance(body, dict):
        q = body.get("query") or body.get("prompt") or body.get("q")
        return q, body
    return None, body

def _normalize(agent: str, raw_result: Any) -> AgentResponse:
    if isinstance(raw_result, str):
        return {"agent": agent, "intent": "say", "message": raw_result}
    if isinstance(raw_result, dict):
        if "agent" not in raw_result:
            raw_result = {"agent": agent, **raw_result}
        raw_result.setdefault("intent", "unknown")
        return raw_result  # type: ignore[return-value]
    if isinstance(raw_result, list):
        return {"agent": agent, "intent": "list", "data": raw_result}
    return {"agent": agent, "intent": "unknown", "message": str(raw_result)}

# -------------------- Universal request endpoint --------------------

@app.post("/api/request")
@app.post("/app/api/request")
@app.post("/api/route")
@app.post("/app/api/route")
async def handle_request(request: Request):
    body: Dict[str, Any] | None = None
    try:
        if request.headers.get("content-type", "").lower().startswith("application/json"):
            body = await request.json()
            if not isinstance(body, dict):
                body = None
    except Exception:
        body = None

    form_query: str | None = None
    if body is None:
        try:
            form = await request.form()
            if form:
                form_query = form.get("query") or form.get("prompt") or form.get("q")  # type: ignore[assignment]
        except Exception:
            form_query = None

    q, _ = _extract_query(form_query, body)
    if not q:
        return JSONResponse(
            {"agent": "system", "intent": "error", "message": "Missing 'query' in form or JSON body"},
            status_code=400,
        )

    agent, raw_result = route_request(q)
    resp = _normalize(agent, raw_result)

    # If the routed agent is the repo agent, return text/plain so the client gets raw text.
    if isinstance(resp, dict) and resp.get("agent") == "repo":
        text = resp.get("answer") or resp.get("message") or resp.get("data") or json.dumps(resp, ensure_ascii=False)
        return PlainTextResponse(str(text))

    # Pre-format so clients without middleware still get a nice shape
    try:
        natural = ensure_natural(resp)
    except Exception as e:
        natural = {
            "agent": resp.get("agent", "system"),
            "intent": "error",
            "message": f"Formatting error: {e}",
            "raw": resp,
        }

    return JSONResponse(natural)

# -------------------- Static frontend (SPA at /app) --------------------

static_dir = os.path.join(os.path.dirname(__file__), "static")

if os.path.isdir(static_dir):
    INDEX_FILE = Path(static_dir) / "index.html"

    app.mount("/app", StaticFiles(directory=static_dir, html=True), name="static")

    @app.get("/app", include_in_schema=False, response_class=HTMLResponse)
    @app.get("/app/", include_in_schema=False, response_class=HTMLResponse)
    def serve_index():
        if INDEX_FILE.exists():
            return HTMLResponse(INDEX_FILE.read_text(encoding="utf-8"))
        raise HTTPException(status_code=404, detail="Frontend build not found.")

    @app.get("/app/{path:path}", include_in_schema=False, response_class=HTMLResponse)
    def spa_fallback(path: str):
        candidate = Path(static_dir) / path
        if candidate.is_file():
            return FileResponse(str(candidate))
        if INDEX_FILE.exists():
            return HTMLResponse(INDEX_FILE.read_text(encoding="utf-8"))
        raise HTTPException(status_code=404, detail="Frontend build not found.")
