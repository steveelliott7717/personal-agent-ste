# backend/workers/export_browser.py
import os, time, json, re, html, urllib.parse as up
from pathlib import Path
from playwright.sync_api import sync_playwright, Error as PwError

PW_STATE_PATH = Path(os.environ.get("PW_STATE_PATH", "/data/pw_state.json"))
SEED_TOKEN = os.environ.get("CHATGPT_SESSION_TOKEN", "")  # optional seed
DOWNLOAD_DIR = Path(os.environ.get("PW_DOWNLOAD_DIR", "/data/pw_downloads"))
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)


def _ensure_state_seeded():
    """
    If /data/pw_state.json doesn't exist, create a minimal storage state
    with the session cookie so Playwright starts authenticated.
    """
    if PW_STATE_PATH.exists():
        return
    if not SEED_TOKEN:
        raise RuntimeError("PW state missing and CHATGPT_SESSION_TOKEN not set")
    state = {
        "cookies": [
            {
                "name": "__Secure-next-auth.session-token",
                "value": SEED_TOKEN,
                "domain": ".chatgpt.com",
                "path": "/",
                "httpOnly": True,
                "secure": True,
                "sameSite": "None",
                # best-effort expiry far in the future (Playwright will update this if site sets new cookies)
                "expires": int(time.time()) + 60 * 60 * 24 * 365,
            }
        ],
        "origins": [],
    }
    PW_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PW_STATE_PATH.write_text(json.dumps(state), encoding="utf-8")


def _looks_like_zip_ct(headers: dict) -> bool:
    ct = (headers.get("content-type") or headers.get("Content-Type") or "").lower()
    return ("application/zip" in ct) or ("application/octet-stream" in ct)


def download_via_browser(url: str, out_path: str, timeout_ms: int = 180_000) -> None:
    """
    Open the export URL in a real browser session and save the ZIP to out_path.
    - seeds storage state from CHATGPT_SESSION_TOKEN the first time
    - follows whatever the page needs (HTML landing, click-to-download, JS redirects)
    """
    _ensure_state_seeded()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(
            storage_state=str(PW_STATE_PATH), user_agent=UA, accept_downloads=True
        )
        page = context.new_page()

        # Keep session file up-to-date after activity
        def persist_state():
            try:
                state = context.storage_state()
                PW_STATE_PATH.write_text(json.dumps(state), encoding="utf-8")
            except Exception:
                pass

        # 1) Go to URL
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

        # 2) If the URL itself triggers a download, catch it
        # Try fast-path: the page may start a download immediately.
        try:
            with page.expect_download(timeout=3_000) as dl_info:
                # No click; just wait a moment to see if the server pushes a download
                page.wait_for_timeout(1000)
            dl = dl_info.value
            dl.save_as(out_path)
            persist_state()
            context.close()
            browser.close()
            return
        except Exception:
            pass

        # 3) If the page is an HTML landing, try to click a .zip link
        zip_selector = "a[href$='.zip'], a[href*='.zip?'], a[href*='estuary/content']"
        links = page.locator(zip_selector)
        if links.count() == 0:
            # Sometimes it’s a meta-refresh or JS redirect; give it a moment
            page.wait_for_timeout(1500)
            links = page.locator(zip_selector)

        if links.count() > 0:
            with page.expect_download(timeout=timeout_ms) as dl_info:
                links.nth(0).click()
            dl = dl_info.value
            dl.save_as(out_path)
            persist_state()
            context.close()
            browser.close()
            return

        # 4) As a fallback, try swapping host chatgpt.com <-> chat.openai.com and retry once
        parsed = up.urlparse(url)
        alt_host = (
            "chat.openai.com" if parsed.netloc == "chatgpt.com" else "chatgpt.com"
        )
        alt_url = up.urlunparse(parsed._replace(netloc=alt_host))
        page.goto(alt_url, wait_until="domcontentloaded", timeout=timeout_ms)
        links = page.locator(zip_selector)
        if links.count() > 0:
            with page.expect_download(timeout=timeout_ms) as dl_info:
                links.nth(0).click()
            dl = dl_info.value
            dl.save_as(out_path)
            persist_state()
            context.close()
            browser.close()
            return

        persist_state()
        context.close()
        browser.close()
        raise RuntimeError(
            "Browser could not find or trigger a ZIP download on the export page."
        )
