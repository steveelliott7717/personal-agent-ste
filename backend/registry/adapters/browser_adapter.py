# backend/adapters/browser_adapter.py
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

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


import os


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
async def _with_browser(args: Dict[str, Any]) -> Dict[str, Any]:
    start = time.time()

    # General args / defaults
    nav_timeout = int(args.get("timeout_ms", 15000))
    ua = args.get("user_agent") or DEFAULT_UA
    viewport = args.get("viewport") or {"width": 1366, "height": 768}
    locale = args.get("locale") or "en-US"
    stealth = (
        args.get("stealth") or {}
    )  # e.g., {"navigator_webdriver": True, "max_touch_points": 0}

    # Downloads (screenshots / pdfs / file saves)
    save_dir = args.get("save_dir") or "downloads"
    os.makedirs(save_dir, exist_ok=True)

    # Scripted steps + start URL
    flows: List[Dict[str, Any]] = args.get("steps") or []
    artifact_paths: List[Tuple[str, str]] = []  # (kind, path)
    start_url = args.get("url") or (flows[0].get("goto") if flows else None)
    if not start_url:
        return {
            "status": 0,
            "error": "BadRequest",
            "message": "missing url or steps[0].goto",
        }

    ok, why = _host_ok(start_url)
    if not ok:
        return {"status": 0, "error": "DeniedURL", "message": why}

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)

        # Build context args (proxy, geo, tz, locale, UA, viewport, storage_state)
        context_kwargs = _build_context_kwargs(args, DEFAULT_UA, viewport, locale)
        context = await browser.new_context(**context_kwargs)

        # Optional stealth tweaks
        stealth_js = _build_stealth_script(stealth)
        if stealth_js:
            await context.add_init_script(stealth_js)

        # Page + timeouts
        page = await context.new_page()
        page.set_default_timeout(nav_timeout)

        result: Dict[str, Any] = {"events": []}
        download_paths: List[str] = []

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

        # Navigate to starting URL
        # AFTER (host allowlist + per-host rate limit around first navigation)

        # Validate URL against allowlist / denylist
        ok, why = host_port_allowed(start_url)
        if not ok:
            return {"status": 0, "error": "PortNotAllowed", "message": why}
        host = (urlparse(start_url).hostname or "").lower()
        if is_disallowed_ip_host(host):
            return {
                "status": 0,
                "error": "DeniedIP",
                "message": f"Disallowed host for: {start_url}",
            }

        # Per-host token-bucket; same shape as http.fetch
        rl_cfg = (args.get("rate_limit") or {}).get("per_host") or {}
        rl_capacity = int(rl_cfg.get("capacity", 5))
        rl_refill = float(rl_cfg.get("refill_per_sec", 2.0))
        rl_maxwait = int(rl_cfg.get("max_wait_ms", 2000))

        try:
            _tb_take(host, rl_capacity, rl_refill, rl_maxwait)
            await page.goto(
                start_url, wait_until=args.get("wait_until", "domcontentloaded")
            )
        except TimeoutError as e:
            return {"status": 0, "error": "RateLimited", "message": str(e)}

        # Execute scripted steps
        for i, step in enumerate(flows):
            evt = {"i": i}
            try:
                # AFTER (guard + rate limit around each step.goto)
                if "goto" in step:
                    url = step["goto"]
                    ok, why = host_port_allowed(url)
                    if not ok:
                        raise ValueError(why)
                    _host = (urlparse(url).hostname or "").lower()
                    if is_disallowed_ip_host(_host):
                        raise ValueError(f"Disallowed host for: {url}")
                    try:
                        _tb_take(_host, rl_capacity, rl_refill, rl_maxwait)
                        await page.goto(
                            url, wait_until=step.get("wait_until", "domcontentloaded")
                        )
                    except TimeoutError as e:
                        raise ValueError(
                            f"RateLimited: {e}"
                        )  # will be captured into evt["error"]
                    evt["goto"] = url

                # click
                if sel := step.get("click"):
                    await page.click(sel)
                    evt["click"] = sel

                # fill
                if fill := step.get("fill"):
                    await page.fill(fill["selector"], fill.get("value", ""))
                    evt["fill"] = fill

                # type
                if type_ := step.get("type"):
                    await page.type(
                        type_["selector"],
                        type_.get("text", ""),
                        delay=type_.get("delay", 0),
                    )
                    evt["type"] = type_

                # wait_for
                if wait := step.get("wait_for"):
                    if "selector" in wait:
                        await page.wait_for_selector(
                            wait["selector"], state=wait.get("state", "visible")
                        )
                    elif "state" in wait:
                        await page.wait_for_load_state(wait["state"])
                    elif "time_ms" in wait:
                        await asyncio.sleep(wait["time_ms"] / 1000.0)
                    evt["wait_for"] = wait

                # screenshot
                if shot := step.get("screenshot"):
                    sc_req = step["screenshot"]
                    path = _artifact_path(save_dir, sc_req.get("path"), f"shot_{i}.png")
                    await page.screenshot(
                        path=path, full_page=sc_req.get("full_page", True)
                    )
                    evt["screenshot"] = path
                    artifact_paths.append(("screenshot", path))

                # pdf
                if pdf := step.get("pdf"):
                    pdf_req = step["pdf"]
                    path = _artifact_path(
                        save_dir, pdf_req.get("path"), f"page_{i}.pdf"
                    )
                    await page.pdf(path=path, print_background=True)
                    evt["pdf"] = path
                    artifact_paths.append(("pdf", path))

                if ev := step.get("evaluate"):
                    js = ev.get("js")
                    js_args = ev.get("args", [])
                    val = await page.evaluate(js, *js_args)
                    evt["evaluate"] = {"result": val}

                # extract (inner_text by default; inner_html if inner_text=False)
                if ex := step.get("extract"):
                    sel = ex["selector"]
                    inner_text = bool(ex.get("inner_text", True))
                    loc = page.locator(sel)
                    val = await (loc.inner_text() if inner_text else loc.inner_html())
                    evt["extract"] = {"selector": sel, "value": val}

                # set_files (upload to <input type=file>)
                if step.get("set_files"):
                    sf = step["set_files"]
                    files = sf.get("files") or []
                    selector = sf["selector"]
                    to_ms = int(sf.get("timeout_ms", 15000))
                    evt["set_files"] = {"selector": selector, "files": files}
                    loc = page.locator(selector)
                    await loc.set_input_files(files, timeout=to_ms)

                result["events"].append(evt)
            except Exception as e:
                evt["error"] = str(e)
                result["events"].append(evt)
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

        await context.close()
        await browser.close()

        saved_info = None
        destination = args.get("destination")
        if destination and artifact_paths:
            backend = (destination.get("type") or "").lower()
            items = []
            for kind, path in artifact_paths:
                try:
                    with open(path, "rb") as f:
                        content = f.read()
                except Exception as e:
                    items.append(
                        {"kind": kind, "local_path": path, "error": f"read_failed: {e}"}
                    )
                    continue
                try:
                    if backend == "github":
                        up = _upload_to_github(destination, content)
                    elif backend == "supabase_storage":
                        up = _upload_to_supabase_storage(destination, content)
                    elif backend == "supabase_table":
                        up = _insert_into_supabase_table(destination, content)
                    else:
                        up = {"error": f"unknown destination: {backend}"}
                    items.append({"kind": kind, "local_path": path, **up})
                except Exception as e:
                    items.append(
                        {
                            "kind": kind,
                            "local_path": path,
                            "error": f"upload_failed: {e}",
                        }
                    )
            if items:
                saved_info = {"backend": backend, "items": items}

        elapsed_ms = int((time.time() - start) * 1000)
        result.update(
            {
                "status": 200,
                "url": start_url,
                "title": title,
                "html_len": len(content or ""),
                "downloads": download_paths,
                "elapsed_ms": elapsed_ms,
                "html": content if args.get("return_html") else None,
                "saved": saved_info,
            }
        )

        return result


# -----------------------------------------------------------------------------
# Sync wrapper (works inside & outside existing event loops)
# -----------------------------------------------------------------------------
def _run_coro_in_new_thread(coro):
    holder = {"result": None, "error": None}

    def runner():
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


def browser_run_adapter(args: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point used by the registry: runs Playwright flows with guardrails."""
    try:
        try:
            # If already in an event loop (e.g., FastAPI), hop to a fresh one in a thread.
            asyncio.get_running_loop()
            return _run_coro_in_new_thread(_with_browser(args))
        except RuntimeError:
            # Not in an event loop → safe to run directly.
            return asyncio.run(_with_browser(args))
    except Exception as e:
        return {"status": 0, "error": "BrowserError", "message": str(e)}
