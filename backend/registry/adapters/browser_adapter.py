# backend/adapters/browser_adapter.py
from __future__ import annotations

import asyncio
import sys
import os
import threading
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from playwright.async_api import async_playwright
from urllib.parse import urlparse
from backend.registry.adapters.http_fetch import (
    host_port_allowed,
    is_disallowed_ip_host,
    _tb_take,
)
from backend.registry.adapters.http_fetch import (
    _upload_to_github,
    _upload_to_supabase_storage,
    _insert_into_supabase_table,
)

# Max HTML bytes when return_html=true (default 2MB; override via env)
BROWSER_MAX_HTML_BYTES = int(os.getenv("BROWSER_MAX_HTML_BYTES", "2000000"))

# Allowed top-level step keys (light schema validation)
_ALLOWED_STEP_KEYS = {
    "goto",
    "wait_until",
    "click",
    "fill",
    "type",
    "wait_for",
    "screenshot",
    "pdf",
    "evaluate",
    "extract",
    "set_files",
}
# NEW: defensive cap on number of steps
MAX_STEPS = int(os.getenv("BROWSER_MAX_STEPS", "50"))

# -----------------------------------------------------------------------------
# Safety / defaults
# -----------------------------------------------------------------------------
SAFE_SCHEMES = {"http", "https"}
DISALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1"}
DEFAULT_UA = "personal-agent/1.0 (+registry-browser.run)"


def _host_ok(url: str) -> Tuple[bool, str]:
    """Quick allowlist for schemes + disallow localhost targets."""
    try:
        u = urlparse(url)
    except Exception:
        return False, "bad URL"
    if u.scheme not in SAFE_SCHEMES:
        return False, f"scheme not allowed: {u.scheme}"
    host = (u.hostname or "").lower()
    if host in DISALLOWED_HOSTS:
        return False, f"host disallowed: {host}"
    return True, ""


# -----------------------------------------------------------------------------
# Test seams / small helpers
# -----------------------------------------------------------------------------
def _build_context_kwargs(
    args: Dict[str, Any],
    default_ua: str,
    default_viewport: Dict[str, int],
    default_locale: str,
) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {
        "user_agent": args.get("user_agent") or default_ua,
        "locale": args.get("locale") or default_locale,
        "viewport": args.get("viewport") or default_viewport,
    }
    proxy = args.get("proxy")
    if proxy:
        ctx["proxy"] = proxy

    geolocation = args.get("geolocation")
    if geolocation:
        ctx["geolocation"] = geolocation
        ctx["permissions"] = ["geolocation"]

    tz = args.get("timezone_id")
    if tz:
        ctx["timezone_id"] = tz

    # If the caller provided a storage state file to load, pass it here
    load_state_path = args.get("load_storage_state_path")
    if load_state_path and Path(load_state_path).exists():
        try:
            ctx["storage_state"] = load_state_path
        except Exception:
            # ignore bad / unreadable state
            pass

    return ctx


def _artifact_path(save_dir: str, requested: str | None, default_name: str) -> str:
    """
    If user passes "downloads\\foo.png", don't prefix save_dir again.
    If it's relative, join to save_dir. If absolute, return as-is.
    """
    name = requested or default_name
    if os.path.isabs(name):
        return name
    norm_save = os.path.normpath(save_dir)
    norm_name = os.path.normpath(name)
    # If the requested path already starts with save_dir, keep it
    if norm_name.split(os.sep, 1)[0].lower() == os.path.basename(norm_save).lower():
        return norm_name
    return os.path.join(save_dir, name)


def _build_stealth_script(stealth: Optional[dict]) -> str:
    """Return a tiny stealth script – off by default, opt-in via args.stealth."""
    if not stealth:
        return ""
    lines: List[str] = []
    # Hide webdriver
    if stealth.get("navigator_webdriver", True):
        lines.append(
            "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
        )
    # Tweak touch points if requested
    if "max_touch_points" in stealth:
        try:
            m = int(stealth["max_touch_points"])
        except Exception:
            m = 0
        lines.append(
            f"Object.defineProperty(navigator, 'maxTouchPoints', {{ get: () => {m} }});"
        )
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Core async runner
# -----------------------------------------------------------------------------


def _validate_steps(steps: List[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(steps, list):
        return "steps must be a list"
    if len(steps) > MAX_STEPS:
        return f"too many steps: {len(steps)} > {MAX_STEPS}"
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            return f"steps[{i}] must be an object"
        extra = set(s.keys()) - _ALLOWED_STEP_KEYS
        if extra:
            return f"steps[{i}] has unsupported keys: {sorted(list(extra))}"
        # spot checks
        if "goto" in s and not isinstance(s["goto"], str):
            return f"steps[{i}].goto must be a string"
        if "click" in s and not isinstance(s["click"], str):
            return f"steps[{i}].click must be a string CSS selector"
        if "fill" in s:
            f = s["fill"]
            if not (isinstance(f, dict) and isinstance(f.get("selector"), str)):
                return f"steps[{i}].fill must have selector:string"
        if "type" in s:
            t = s["type"]
            if not (isinstance(t, dict) and isinstance(t.get("selector"), str)):
                return f"steps[{i}].type must have selector:string"
        if "wait_for" in s and not isinstance(s["wait_for"], dict):
            return f"steps[{i}].wait_for must be an object"
        if "screenshot" in s and not isinstance(s["screenshot"], dict):
            return f"steps[{i}].screenshot must be an object"
        if "pdf" in s and not isinstance(s["pdf"], dict):
            return f"steps[{i}].pdf must be an object"
        if "evaluate" in s:
            ev = s["evaluate"]
            if not (isinstance(ev, dict) and isinstance(ev.get("js"), str)):
                return f"steps[{i}].evaluate must include js:string"
            if "args" in ev and not isinstance(ev["args"], list):
                return f"steps[{i}].evaluate.args must be an array"
        if "set_files" in s:
            sf = s["set_files"]
            if not (
                isinstance(sf, dict)
                and isinstance(sf.get("selector"), str)
                and isinstance(sf.get("files"), list)
            ):
                return (
                    f"steps[{i}].set_files must include selector:string and files:list"
                )
            for j, fp in enumerate(sf.get("files") or []):
                if not isinstance(fp, str):
                    return f"steps[{i}].set_files.files[{j}] must be a string path"
    return None


async def _with_browser(args: Dict[str, Any]) -> Dict[str, Any]:
    start = time.time()

    # total timeout (applied by wrapper); compute here for clarity
    total_timeout_sec = None
    if args.get("timeout_total_ms") is not None:
        try:
            total_timeout_sec = max(0.05, int(args["timeout_total_ms"]) / 1000.0)
        except Exception:
            return {
                "ok": False,
                "error": {
                    "code": "BadRequest",
                    "message": "timeout_total_ms must be an integer",
                },
            }

    # General args / defaults
    try:
        nav_timeout = int(args.get("timeout_ms", 30000))
    except Exception:
        return {
            "ok": False,
            "error": {"code": "BadRequest", "message": "timeout_ms must be an integer"},
        }
    nav_timeout = max(100, nav_timeout)

    ua = args.get("user_agent") or DEFAULT_UA
    viewport = args.get("viewport") or {"width": 1366, "height": 768}
    locale = args.get("locale") or "en-US"
    headless = bool(args.get("headless", True))
    stealth = args.get("stealth") or {}

    # Downloads (screenshots / pdfs / file saves)
    save_dir = args.get("save_dir") or "downloads"
    os.makedirs(save_dir, exist_ok=True)

    # Scripted steps + start URL
    flows: List[Dict[str, Any]] = args.get("steps") or []
    err = _validate_steps(flows)
    if err:
        return {"ok": False, "error": {"code": "BadRequest", "message": err}}

        # Pre-validate set_files existence so missing files are caught even if the browser can't start
    for i, step in enumerate(flows):
        sf = step.get("set_files")
        if isinstance(sf, dict):
            files = sf.get("files") or []
            missing = [p for p in files if not os.path.exists(p)]
            if missing:
                return {
                    "ok": False,
                    "error": {
                        "code": "BadRequest",
                        "message": f"missing local files: {', '.join(missing)}",
                    },
                }

    artifact_paths: List[Tuple[str, str]] = []  # (kind, path)
    start_url = args.get("url") or (flows[0].get("goto") if flows else None)
    if not start_url:
        return {
            "ok": False,
            "error": {"code": "BadRequest", "message": "missing url or steps[0].goto"},
        }

    ok_host, why = _host_ok(start_url)
    if not ok_host:
        return {"ok": False, "error": {"code": "DeniedURL", "message": why}}

    # Validate URL against allowlist / denylist before launching
    ok_port, why_port = host_port_allowed(start_url)
    if not ok_port:
        return {"ok": False, "error": {"code": "PortNotAllowed", "message": why_port}}
    _host = (urlparse(start_url).hostname or "").lower()
    if is_disallowed_ip_host(_host):
        return {
            "ok": False,
            "error": {
                "code": "DeniedIP",
                "message": f"Disallowed host for: {start_url}",
            },
        }

    result: Dict[str, Any] = {"events": []}
    download_paths: List[str] = []

    ua_pool = args.get("user_agent_pool") or []
    ua = args.get("user_agent") or DEFAULT_UA
    if ua_pool:
        if args.get("ua_rotate_per") == "host":
            # choose based on start host for stability across steps
            key = (urlparse(start_url).hostname or "").lower()
            random.seed(hash(key))  # deterministic per host if you want
        ua = random.choice(ua_pool)

    accept_locales = args.get("locale_pool") or ["en-US", "en-GB", "en"]
    locale = args.get("locale") or random.choice(accept_locales)

    if "viewport" not in args:
        viewport = {
            "width": random.choice([1280, 1366, 1440, 1536]),
            "height": random.choice([720, 768, 800, 900]),
        }
    else:
        viewport = args["viewport"]

    # --- JITTER CONFIG (human-like delays) ---
    # args["jitter"] = {
    #   "ms_min": 120, "ms_max": 900,           # range for sleep
    #   "prob": 0.85,                           # chance to apply each time
    #   "between_steps": True,                  # jitter before each step
    #   "between_actions": True,                # jitter before each action within a step
    #   "post_action_prob": 0.30,               # chance to add a short dwell AFTER click/type
    #   "debug_events": False,                  # include jitter events in result["events"]
    #   "max_debug_events": 1000                # cap debug jitter logs
    # }
    jit_cfg = args.get("jitter") or {}
    _jit_ms_min = int(jit_cfg.get("ms_min", 120))
    _jit_ms_max = int(jit_cfg.get("ms_max", 900))
    _jit_prob = float(jit_cfg.get("prob", 0.85))
    _jit_between_steps = bool(jit_cfg.get("between_steps", True))
    _jit_between_actions = bool(jit_cfg.get("between_actions", True))
    _jit_post_prob = float(jit_cfg.get("post_action_prob", 0.30))
    _jit_debug = bool(jit_cfg.get("debug_events", False))
    _jit_max_debug = int(jit_cfg.get("max_debug_events", 1000))

    def _remaining_total_sec() -> float:
        if total_timeout_sec is None:
            return 86400.0  # effectively unlimited
        return max(0.0, total_timeout_sec - (time.time() - start))

    async def _maybe_jitter(tag: str):
        """Random human-like pause. Honors total timeout if set."""
        if random.random() > _jit_prob:
            return
        # choose a duration, but don't exceed remaining total timeout
        dur = random.uniform(_jit_ms_min / 1000.0, _jit_ms_max / 1000.0)
        rem = _remaining_total_sec()
        if dur > rem:
            dur = max(0.0, rem * 0.5)  # be conservative if we're close to the cap
        if dur > 0:
            await asyncio.sleep(dur)

        if _jit_debug:
            evs = result.setdefault("events", [])
            if len(evs) < _jit_max_debug:  # cap to prevent unbounded growth
                evs.append({"jitter": {"tag": tag, "sleep_s": round(dur, 3)}})

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=headless)

            # Build context args (proxy, geo, tz, locale, UA, viewport, storage_state)
            context_kwargs = _build_context_kwargs(args, ua, viewport, locale)
            context = await browser.new_context(**context_kwargs)

            # Optional stealth tweaks
            stealth_js = _build_stealth_script(stealth)
            if stealth_js:
                await context.add_init_script(stealth_js)

            # Page + timeouts
            page = await context.new_page()
            page.set_default_timeout(nav_timeout)

            # Optional human-like pause before first navigation
            if _jit_between_steps:
                await _maybe_jitter("pre_first_nav")

            # Capture downloads
            async def _on_download(d):
                path = os.path.join(save_dir, await d.suggested_filename())
                await d.save_as(path)
                download_paths.append(path)
                artifact_paths.append(("download", path))

            page.on("download", _on_download)

            # Optionally set cookies
            if args.get("cookies"):
                try:
                    await context.add_cookies(args["cookies"])
                except Exception:
                    pass

            # Per-host token-bucket; same shape as http.fetch
            rl_cfg = (args.get("rate_limit") or {}).get("per_host") or {}
            try:
                rl_capacity = int(rl_cfg.get("capacity", 50))
                rl_refill = float(rl_cfg.get("refill_per_sec", 20.0))
                rl_maxwait = int(rl_cfg.get("max_wait_ms", 5000))
            except Exception:
                return {
                    "ok": False,
                    "error": {
                        "code": "BadRequest",
                        "message": "rate_limit.per_host fields must be numeric",
                    },
                }

            # First navigation with rate limiting (retry once). If it still fails, record and continue.
            goto_wait = args.get("wait_until", "load")
            nav_error = None
            for attempt in range(2):
                try:
                    _tb_take(_host, rl_capacity, rl_refill, rl_maxwait)
                    await page.goto(start_url, wait_until=goto_wait)
                    # Ensure DOM is ready
                    await page.wait_for_load_state("domcontentloaded")
                    await page.wait_for_load_state("load")
                    break
                except Exception as e:
                    nav_error = str(e)
                    if attempt == 0:
                        await asyncio.sleep(0.2)  # small backoff then one retry
                    else:
                        # Do NOT fail the run; log and continue with a blank page
                        result["nav_error"] = nav_error
                        try:
                            await page.goto("about:blank")
                            await page.set_content("<html><body></body></html>")
                        except Exception:
                            pass
                        break

            top_error: Optional[Dict[str, str]] = None

            # Execute scripted steps
            for i, step in enumerate(flows):
                # Per-step override: allow step["jitter"]=False to disable all jitter for this step
                _step_jitter = step.get("jitter", True)

                # optional jitter before the step begins
                if _jit_between_steps and _step_jitter:
                    await _maybe_jitter(f"pre_step_{i}")

                evt = {"i": i}
                try:
                    # goto
                    if "goto" in step:
                        if _jit_between_actions and _step_jitter:
                            await _maybe_jitter(f"pre_goto_{i}")
                        url = step["goto"]
                        okp, whyp = host_port_allowed(url)
                        if not okp:
                            raise ValueError(whyp)
                        _h2 = (urlparse(url).hostname or "").lower()
                        if is_disallowed_ip_host(_h2):
                            raise ValueError(f"Disallowed host for: {url}")
                        try:
                            _tb_take(_h2, rl_capacity, rl_refill, rl_maxwait)
                            await page.goto(
                                url,
                                wait_until=step.get("wait_until", "domcontentloaded"),
                            )
                        except TimeoutError as e:
                            raise ValueError(f"RateLimited: {e}")
                        evt["goto"] = url

                    # click
                    if sel := step.get("click"):
                        if _jit_between_actions and _step_jitter:
                            await _maybe_jitter(f"pre_click_{i}")
                        await page.click(sel)
                        # optional short dwell AFTER the click
                        if (
                            _jit_between_actions
                            and _step_jitter
                            and random.random() < _jit_post_prob
                        ):
                            await _maybe_jitter(f"post_click_{i}")
                        evt["click"] = sel

                    # fill
                    if fill := step.get("fill"):
                        if _jit_between_actions and _step_jitter:
                            await _maybe_jitter(f"pre_fill_{i}")
                        await page.fill(fill["selector"], fill.get("value", ""))
                        evt["fill"] = fill

                    # type
                    if type_ := step.get("type"):
                        if _jit_between_actions and _step_jitter:
                            await _maybe_jitter(f"pre_type_{i}")
                        await page.type(
                            type_["selector"],
                            type_.get("text", ""),
                            delay=type_.get("delay", 0),
                        )
                        # optional short dwell AFTER typing
                        if (
                            _jit_between_actions
                            and _step_jitter
                            and random.random() < _jit_post_prob
                        ):
                            await _maybe_jitter(f"post_type_{i}")
                        evt["type"] = type_

                    # wait_for
                    if wait := step.get("wait_for"):
                        # avoid double-sleep if this step already has an explicit time_ms wait
                        if (
                            _jit_between_actions
                            and _step_jitter
                            and ("time_ms" not in wait)
                        ):
                            await _maybe_jitter(f"pre_wait_for_{i}")

                        if "selector" in wait:
                            try:
                                wf_timeout = int(
                                    wait.get("timeout_ms", max(nav_timeout, 45000))
                                )
                            except Exception:
                                wf_timeout = max(nav_timeout, 45000)
                            await page.wait_for_selector(
                                wait["selector"],
                                state=wait.get("state", "attached"),
                                timeout=wf_timeout,
                            )
                        elif "state" in wait:
                            await page.wait_for_load_state(wait["state"])
                        elif "time_ms" in wait:
                            await asyncio.sleep(wait["time_ms"] / 1000.0)
                        evt["wait_for"] = wait

                    # screenshot
                    if shot := step.get("screenshot"):
                        if _jit_between_actions and _step_jitter:
                            await _maybe_jitter(f"pre_screenshot_{i}")
                        sc_req = step["screenshot"]
                        path = _artifact_path(
                            save_dir, sc_req.get("path"), f"shot_{i}.png"
                        )
                        Path(path).parent.mkdir(parents=True, exist_ok=True)
                        await page.screenshot(
                            path=path, full_page=sc_req.get("full_page", True)
                        )
                        evt["screenshot"] = path
                        artifact_paths.append(("screenshot", path))

                    # pdf
                    if pdf := step.get("pdf"):
                        if _jit_between_actions and _step_jitter:
                            await _maybe_jitter(f"pre_pdf_{i}")
                        pdf_req = step["pdf"]
                        path = _artifact_path(
                            save_dir, pdf_req.get("path"), f"page_{i}.pdf"
                        )
                        await page.pdf(path=path, print_background=True)
                        evt["pdf"] = path
                        artifact_paths.append(("pdf", path))

                    # evaluate
                    if ev := step.get("evaluate"):
                        if _jit_between_actions and _step_jitter:
                            await _maybe_jitter(f"pre_evaluate_{i}")
                        js = ev.get("js")
                        js_args = ev.get("args", [])
                        if not isinstance(js, str) or not js:
                            raise ValueError(
                                "steps[%d].evaluate must include js:string" % i
                            )
                        val = await page.evaluate(js, *js_args)
                        evt["evaluate"] = {"result": val}

                    # extract
                    if ex := step.get("extract"):
                        if _jit_between_actions and _step_jitter:
                            await _maybe_jitter(f"pre_extract_{i}")
                        sel2 = ex["selector"]
                        inner_text = bool(ex.get("inner_text", True))
                        loc = page.locator(sel2)
                        val = await (
                            loc.inner_text() if inner_text else loc.inner_html()
                        )
                        evt["extract"] = {"selector": sel2, "value": val}

                    # set_files
                    if step.get("set_files"):
                        if _jit_between_actions and _step_jitter:
                            await _maybe_jitter(f"pre_set_files_{i}")
                        sf = step["set_files"]
                        files = sf.get("files") or []
                        selector = sf["selector"]
                        to_ms = int(sf.get("timeout_ms", 15000))
                        missing = [p for p in files if not os.path.exists(p)]
                        if missing:
                            raise ValueError(
                                f"missing local files: {', '.join(missing)}"
                            )
                        evt["set_files"] = {"selector": selector, "files": files}
                        loc = page.locator(selector)
                        await loc.set_input_files(files, timeout=to_ms)

                    result["events"].append(evt)

                except Exception as e:
                    msg = str(e) or "browser step failed"
                    evt["error"] = msg
                    result["events"].append(evt)
                    # Mark top-level error so we fail the call
                    top_error = {"code": "BadRequest", "message": msg}
                    break  # stop on first error

            # Final page basics
            try:
                content = await page.content()
                title = await page.title()
            except Exception:
                content, title = "", ""

            # Save storage state if requested
            if args.get("save_storage_state_path"):
                try:
                    await context.storage_state(path=args["save_storage_state_path"])
                except Exception:
                    pass

            # tidy up browser
            await context.close()
            await browser.close()

    except Exception as outer_e:
        import traceback

        traceback.print_exc()
        return {
            "ok": False,
            "error": {"code": "BrowserError", "message": repr(outer_e)},
        }

    elapsed_ms = int((time.time() - start) * 1000)

    saved_info = None
    destination = args.get("destination")
    if destination and artifact_paths:
        backend = (destination.get("type") or "").lower()
        items = []
        for kind, path in artifact_paths:
            try:
                with open(path, "rb") as f:
                    content_bytes = f.read()
            except Exception as e:
                items.append(
                    {"kind": kind, "local_path": path, "error": f"read_failed: {e}"}
                )
                continue
            try:
                if backend == "github":
                    up = _upload_to_github(destination, content_bytes)
                elif backend == "supabase_storage":
                    up = _upload_to_supabase_storage(destination, content_bytes)
                elif backend == "supabase_table":
                    up = _insert_into_supabase_table(destination, content_bytes)
                else:
                    up = {"error": f"unknown destination: {backend}"}
                items.append({"kind": kind, "local_path": path, **up})
            except Exception as e:
                items.append(
                    {"kind": kind, "local_path": path, "error": f"upload_failed: {e}"}
                )
        if items:
            saved_info = {"backend": backend, "items": items}

    html_out = None
    if args.get("return_html"):
        raw_html = content or ""
        if len(raw_html.encode("utf-8")) > BROWSER_MAX_HTML_BYTES:
            enc = raw_html.encode("utf-8")[:BROWSER_MAX_HTML_BYTES]
            try:
                html_out = enc.decode("utf-8", "ignore")
            except Exception:
                html_out = ""
        else:
            html_out = raw_html

    # If any step errored, fail the call (tests expect ok: False)
    # (title is still useful to include on error paths for debugging)
    base_result = {
        "status": 200,
        "url": start_url,
        "title": title,
        "html_len": len(content or ""),
        "downloads": download_paths,
        "elapsed_ms": elapsed_ms,
        "html": html_out,
        "saved": saved_info,
        "events": result.get("events", []),
    }

    # If there was a top-level step error, return ok:false + message (contains "missing local files" when applicable)
    if "top_error" in locals() and top_error:
        return {"ok": False, "error": top_error, "result": base_result}

    return {"ok": True, "result": base_result}


# -----------------------------------------------------------------------------
# Sync wrapper (works inside & outside existing event loops)
# -----------------------------------------------------------------------------
def _run_coro_in_new_thread(coro):
    holder = {"result": None, "error": None}

    def runner():
        # On Windows, Playwright needs subprocess support (Proactor policy).
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        try:
            holder["result"] = asyncio.run(coro)
        except Exception as e:
            holder["error"] = e

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()

    if holder["error"] is not None:
        raise holder["error"]
    return holder["result"]


async def _runner_with_timeout(args: Dict[str, Any]):
    # Apply coroutine-level timeout if requested
    t_ms = args.get("timeout_total_ms")
    if t_ms is None:
        return await _with_browser(args)
    try:
        t_sec = max(0.05, float(int(t_ms)) / 1000.0)
    except Exception:
        return {
            "ok": False,
            "error": {
                "code": "BadRequest",
                "message": "timeout_total_ms must be an integer",
            },
        }
    try:
        return await asyncio.wait_for(_with_browser(args), timeout=t_sec)
    except asyncio.TimeoutError:
        return {
            "ok": False,
            "error": {
                "code": "Timeout",
                "message": "browser.run timed out (timeout_total_ms)",
            },
        }


# --- Lightweight Playwright warmup (no network) ------------------------------
def browser_warmup_adapter(args: dict, meta: dict) -> dict:
    """
    Launch Chromium headless, open a blank page with inline HTML, then close.
    No external navigation (no DNS/HTTPS). Safe to call from sync or async contexts.
    If running inside an asyncio loop, the Sync API work is offloaded to a worker thread.
    Args:
      - timeout_ms (int, default 2000): per-action timeout
      - headless   (bool, default True)
    """
    import time
    import asyncio
    import concurrent.futures

    timeout_ms = int(args.get("timeout_ms", 2000))
    headless = bool(args.get("headless", True))

    t0 = time.time()

    def _do_sync_warmup():
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context()
            page = context.new_page()
            # keep everything local; small timeouts
            page.set_default_timeout(timeout_ms)
            page.set_default_navigation_timeout(timeout_ms)
            page.set_content("<!doctype html><title>warmup</title>", wait_until="load")
            context.close()
            browser.close()

    try:
        # If we're inside an asyncio loop (e.g., called from an async route),
        # run Playwright Sync API in a dedicated worker thread.
        in_loop = False
        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False

        if in_loop:
            # wall clock cap ~ timeout_ms/1000 + a small buffer
            thread_timeout = max(3.0, (timeout_ms / 1000.0) + 2.0)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_do_sync_warmup)
                fut.result(timeout=thread_timeout)
        else:
            _do_sync_warmup()

        return {
            "ok": True,
            "status": 200,
            "result": {"warmed": True},
            "latency_ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        return {
            "ok": False,
            "status": 0,
            "error": "WarmupFailed",
            "message": str(e),
            "latency_ms": int((time.time() - t0) * 1000),
        }


def browser_run_adapter(args: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point used by the registry: runs Playwright flows with guardrails + total-timeout
    without ever calling asyncio.run() inside an already-running event loop.
    """
    try:
        # Always offload to a fresh thread/loop to avoid nested-loop errors.
        return _run_coro_in_new_thread(_runner_with_timeout(args))
    except asyncio.TimeoutError:
        return {
            "ok": False,
            "error": {
                "code": "Timeout",
                "message": "browser.run timed out (timeout_total_ms)",
            },
        }
    except Exception as e:
        # Tests accept AdapterError or BadRequest
        return {"ok": False, "error": {"code": "AdapterError", "message": str(e)}}
