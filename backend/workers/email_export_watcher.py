# backend/workers/email_export_watcher.py
# Robust Gmail/IMAP watcher for ChatGPT export emails:
# - Selects Gmail folders safely (quotes names with spaces/brackets)
# - Finds latest unread export email from OpenAI
# - Extracts the export URL from HTML (handles &amp;, relative hrefs, meta refresh, JS redirects)
# - Tries HTTP first; if still HTML, falls back to Playwright to download with a real session
# - Stores ZIP in Supabase Storage and upserts conversations into a table
# - Marks the email as read (and optionally deletes) on success

import os
import re
import time
import ssl
import email
import imaplib
import datetime as dt
import requests
import io
import zipfile
import json
import hashlib

from email.header import decode_header
from typing import Optional, List
from supabase import create_client
import html
import string
import urllib.parse as up

# Playwright fallback helper
from backend.workers.export_browser import download_via_browser

# ---------- Config from env ----------
IMAP_HOST = os.environ["IMAP_HOST"]
IMAP_PORT = int(os.environ.get("IMAP_PORT", "993"))
IMAP_SSL = os.environ.get("IMAP_SSL", "true").lower() == "true"
IMAP_USER = os.environ["IMAP_USER"]
IMAP_PASS = os.environ["IMAP_PASS"]
IMAP_FOLDER = os.environ.get("IMAP_FOLDER", "[Gmail]/All Mail")
POLL_SECS = int(os.environ.get("POLL_SECS", "120"))

# Sender/subject: be permissive; default to substring 'openai.com'
SENDER = os.environ.get("EXPORT_SENDER", "openai.com").lower()
SUBJECT_PHRASE = os.environ.get("EXPORT_SUBJECT_PHRASE", "export").lower()
URL_REGEX = re.compile(r'https?://[^\s">]+', re.IGNORECASE)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ[
    "SUPABASE_SERVICE_ROLE"
]  # SERVICE ROLE key required for writes
BUCKET = os.environ.get("SUPABASE_BUCKET", "chatgpt_exports")

DELETE_EMAIL_AFTER_SUCCESS = (
    os.environ.get("DELETE_EMAIL_AFTER_SUCCESS", "false").lower() == "true"
)
CLEANUP_DAYS = int(os.environ.get("CLEANUP_DAYS", "0"))
CLEANUP_EVERY_N_LOOPS = int(os.environ.get("CLEANUP_EVERY_N_LOOPS", "3600"))

# HTTP defaults
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)
BASE_HEADERS = {
    "User-Agent": UA,
    "Accept": "application/zip,application/octet-stream,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://chatgpt.com/",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

supa = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------- Helpers ----------
def decode_subj(raw: Optional[str]) -> str:
    if not raw:
        return ""
    parts = decode_header(raw)
    out = ""
    for txt, enc in parts:
        out += (
            txt.decode(enc or "utf-8", "ignore")
            if isinstance(txt, bytes)
            else (txt or "")
        )
    return out


def _select_quoted(M, mailbox: str) -> bool:
    # Quote mailbox names with spaces/brackets for Gmail (e.g. "[Gmail]/All Mail")
    mbx = (
        mailbox
        if (mailbox.startswith('"') and mailbox.endswith('"'))
        else f'"{mailbox}"'
    )
    try:
        typ, _ = M.select(mbx)
        if typ == "OK":
            print(f"[imap] selected mailbox: {mbx}")
            return True
    except Exception as e:
        print(f"[imap] select {mbx} failed: {e}")
    return False


def connect_imap():
    if IMAP_SSL:
        M = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    else:
        M = imaplib.IMAP4(IMAP_HOST, IMAP_PORT)
        M.starttls(ssl_context=ssl.create_default_context())
    M.login(IMAP_USER, IMAP_PASS)

    candidates: List[str] = [IMAP_FOLDER, "[Gmail]/All Mail", "INBOX"]
    for box in [c for c in candidates if c]:
        if _select_quoted(M, box):
            return M
    raise RuntimeError(f"Could not select IMAP folder; tried: {candidates}")


def extract_urls(payload, charset: Optional[str]) -> List[str]:
    if isinstance(payload, (bytes, bytearray)):
        try:
            text = payload.decode(charset or "utf-8", "ignore")
        except Exception:
            text = ""
    else:
        text = payload or ""
    candidates = [u for u in URL_REGEX.findall(text) if u.startswith("https://")]
    cleaned: List[str] = []
    for u in candidates:
        u = html.unescape(u)
        u = u.rstrip(")\"' ;" + string.whitespace)
        cleaned.append(u)
    cleaned.sort(
        key=lambda u: (".zip" in u, "estuary/content" in u, "openai" in u, len(u)),
        reverse=True,
    )
    return cleaned


def _normalize_relative(base_url: str, candidate: str) -> str:
    base = up.urlparse(base_url)
    cand = up.urlparse(candidate)
    # Keep candidate's query string; use base host/scheme
    return up.urlunparse((base.scheme, base.netloc, cand.path, "", cand.query, ""))


def _extract_zip_from_html(page_html: str, base_url: str) -> Optional[str]:
    body = html.unescape(page_html or "")

    # 1) <a href="...zip...">
    m = re.search(r'href=["\']([^"\']+?\.zip[^"\']*)["\']', body, re.I)
    if m:
        href = m.group(1).strip()
        return _normalize_relative(base_url, href) if href.startswith("/") else href

    # 2) <meta http-equiv="refresh" content="0;url=...zip...">
    m = re.search(
        r'http-equiv=["\']refresh["\'][^>]*content=["\'][^"\']*url=([^"\']+?\.zip[^"\']*)["\']',
        body,
        re.I,
    )
    if m:
        href = m.group(1).strip()
        return _normalize_relative(base_url, href) if href.startswith("/") else href

    # 3) JS redirects: location.href / location.assign("...zip...")
    m = re.search(
        r'location(?:\.href|\.assign)\((["\'])([^"\']+?\.zip[^"\']*)\1\)', body, re.I
    )
    if m:
        href = m.group(2).strip()
        return _normalize_relative(base_url, href) if href.startswith("/") else href

    # 4) any estuary content link (absolute or relative) as fallback
    m = re.search(
        r'["\'](https?://[^"\']*?/backend-api/estuary/content[^"\']*?)["\']', body, re.I
    )
    if m:
        return m.group(1).strip()
    m = re.search(r'["\'](/backend-api/estuary/content[^"\']*?)["\']', body, re.I)
    if m:
        return _normalize_relative(base_url, m.group(1).strip())

    return None


def find_download_url_from_msg(msg) -> Optional[str]:
    def pick(urls: List[str]) -> Optional[str]:
        for u in urls:
            if ".zip" in u:
                return u
        return urls[0] if urls else None

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() in ("text/html", "text/plain"):
                urls = extract_urls(
                    part.get_payload(decode=True), part.get_content_charset()
                )
                u = pick(urls)
                if u:
                    return u
    else:
        urls = extract_urls(msg.get_payload(decode=True), msg.get_content_charset())
        u = pick(urls)
        if u:
            return u
    return None


def download_zip(url: str) -> bytes:
    session = requests.Session()
    session.headers.update({**BASE_HEADERS})

    def try_get(u: str, attempt: int, label: str = "") -> requests.Response:
        r = session.get(u, timeout=180, allow_redirects=True)
        ct = (r.headers.get("Content-Type") or "").lower()
        host = up.urlparse(u).netloc
        lab = f" {label}" if label else ""
        print(
            f"[dbg] dl attempt={attempt} host={host} status={r.status_code} ct={ct} len={len(r.content)}{lab}"
        )
        return r

    MAX_ATTEMPTS = 5
    RETRY_STATUSES = {403, 422, 429, 503}

    for attempt in range(1, MAX_ATTEMPTS + 1):
        r = try_get(url, attempt)

        # Non-HTML: treat as ZIP payload
        ct = (r.headers.get("Content-Type") or "").lower()
        if 200 <= r.status_code < 300 and "text/html" not in ct:
            return r.content

        # HTML landing: try extract .zip and fetch
        if "text/html" in ct and r.text:
            zurl = _extract_zip_from_html(r.text, url)
            if zurl:
                zr = try_get(zurl, attempt, label="(extracted)")
                zct = (zr.headers.get("Content-Type") or "").lower()
                if 200 <= zr.status_code < 300 and "text/html" not in zct:
                    return zr.content
                # host swap for extracted link
                parsed = up.urlparse(zurl)
                alt_host = (
                    "chat.openai.com"
                    if parsed.netloc == "chatgpt.com"
                    else ("chatgpt.com" if parsed.netloc == "chat.openai.com" else None)
                )
                if alt_host:
                    zurl2 = up.urlunparse(parsed._replace(netloc=alt_host))
                    zr2 = try_get(zurl2, attempt, label="(extracted-alt-host)")
                    zct2 = (zr2.headers.get("Content-Type") or "").lower()
                    if 200 <= zr2.status_code < 300 and "text/html" not in zct2:
                        return zr2.content

        # Direct host fallback (original URL)
        need_alt = (r.status_code in RETRY_STATUSES) or ("text/html" in ct)
        parsed = up.urlparse(url)
        if need_alt and parsed.netloc in ("chatgpt.com", "chat.openai.com"):
            alt = up.urlunparse(
                parsed._replace(
                    netloc=(
                        "chat.openai.com"
                        if parsed.netloc == "chatgpt.com"
                        else "chatgpt.com"
                    )
                )
            )
            r2 = try_get(alt, attempt, label="(alt-host)")
            ct2 = (r2.headers.get("Content-Type") or "").lower()
            if 200 <= r2.status_code < 300 and "text/html" not in ct2:
                return r2.content
            if "text/html" in ct2 and r2.text:
                zurl = _extract_zip_from_html(r2.text, alt)
                if zurl:
                    zr = try_get(zurl, attempt, label="(alt-host-extracted)")
                    zct = (zr.headers.get("Content-Type") or "").lower()
                    if 200 <= zr.status_code < 300 and "text/html" not in zct:
                        return zr.content

        time.sleep(1.25 * attempt)

    # LAST CHANCE: authenticated browser download with Playwright
    tmp_path = "/data/pw_downloads/export_tmp.zip"
    print("[dbg] falling back to Playwright browser download...")
    download_via_browser(url, tmp_path)
    with open(tmp_path, "rb") as f:
        blob = f.read()
    if not blob or len(blob) < 1024:
        raise RuntimeError("Downloaded file is empty or too small.")
    return blob


def upload_export_and_log(content: bytes, message_id: Optional[str]) -> str:
    iso = dt.datetime.utcnow().replace(microsecond=0).isoformat().replace(":", "-")
    key = f"chat_history/chatgpt_export_{iso}Z.zip"
    supa.storage.from_(BUCKET).upload(
        key, content, file_options={"contentType": "application/zip"}
    )
    digest = hashlib.sha256(content).hexdigest()
    ins = (
        supa.table("chatgpt_exports")
        .insert(
            {
                "email_message_id": message_id,
                "storage_bucket": BUCKET,
                "storage_path": key,
                "size_bytes": len(content),
                "sha256": digest,
                "notes": "auto-import from gmail",
            }
        )
        .execute()
    )
    export_id = ins.data[0]["id"]
    try:
        supa.table("notifications_log").insert(
            {
                "id": __import__("uuid").uuid4().hex,
                "channel": "system",
                "title": "ChatGPT export stored",
                "text": f"Saved {key} ({len(content)} bytes)",
            }
        ).execute()
    except Exception:
        pass
    return export_id


def parse_conversations_from_zip(zbytes: bytes) -> List[dict]:
    convs: List[dict] = []
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        names = z.namelist()
        # conversations.json (array)
        for name in names:
            if name.lower().endswith("conversations.json"):
                try:
                    data = json.loads(z.read(name).decode("utf-8", "ignore"))
                    if isinstance(data, list):
                        convs.extend(data)
                except Exception:
                    pass
        # conversations/*.json (one per file)
        for name in names:
            low = name.lower()
            if (
                low.endswith(".json")
                and "/conversations/" in low
                and not low.endswith("conversations.json")
            ):
                try:
                    obj = json.loads(z.read(name).decode("utf-8", "ignore"))
                    if isinstance(obj, dict):
                        convs.append(obj)
                    elif isinstance(obj, list):
                        convs.extend(obj)
                except Exception:
                    pass
        # jsonl (rare)
        for name in names:
            low = name.lower()
            if low.endswith(".jsonl"):
                try:
                    for line in z.read(name).splitlines():
                        line = line.decode("utf-8", "ignore").strip()
                        if line:
                            convs.append(json.loads(line))
                except Exception:
                    pass
    # de-dup
    seen = set()
    uniq = []
    for c in convs:
        cid = c.get("id") or c.get("conversation_id")
        key = (
            cid
            or hashlib.sha1(
                json.dumps(c, sort_keys=True, default=str).encode()
            ).hexdigest()
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def upsert_conversations(export_id: str, convs: List[dict]) -> int:
    rows = []
    for c in convs:
        conv_id = (
            c.get("id")
            or c.get("conversation_id")
            or hashlib.sha1(
                json.dumps(c, sort_keys=True, default=str).encode()
            ).hexdigest()
        )
        rows.append(
            {
                "export_id": export_id,
                "conv_id": conv_id,
                "title": c.get("title"),
                "raw": c,
            }
        )
    CHUNK = 500
    total = 0
    for i in range(0, len(rows), CHUNK):
        supa.table("chatgpt_conversations").upsert(
            rows[i : i + CHUNK], on_conflict="export_id,conv_id"
        ).execute()
        total += len(rows[i : i + CHUNK])
    return total


def cleanup_old_exports(days: int):
    try:
        cutoff = (dt.datetime.utcnow() - dt.timedelta(days=days)).isoformat()
        rows = (
            supa.table("chatgpt_exports")
            .select("id,storage_path")
            .lt("created_at", cutoff)
            .limit(1000)
            .execute()
            .data
            or []
        )
        for r in rows:
            try:
                supa.storage.from_(BUCKET).remove([r["storage_path"]])
                supa.table("chatgpt_exports").delete().eq("id", r["id"]).execute()
                print(f"[cleanup] removed {r['storage_path']}")
            except Exception as e:
                print(f"[cleanup][err] {r['storage_path']}: {e}")
    except Exception as e:
        print("[cleanup][err] listing:", e)


def process_message(mail, num) -> bool:
    typ, data = mail.fetch(num, "(RFC822)")
    if typ != "OK":
        return False
    raw = data[0][1]
    msg = email.message_from_bytes(raw)

    from_addr = email.utils.parseaddr(msg.get("From", ""))[1].lower()
    subj = decode_subj(msg.get("Subject", "")).lower()
    message_id = (
        (msg.get("Message-Id") or msg.get("Message-ID") or "").strip().strip("<>")
    )

    if SENDER not in from_addr:
        return False
    if SUBJECT_PHRASE and SUBJECT_PHRASE not in subj:
        print(f"[info] subject didn't match hint (subj='{subj}') — continuing")

    url = find_download_url_from_msg(msg)
    if not url:
        print("[warn] no URL found in export email")
        return False

    try:
        blob = download_zip(url)
        export_id = upload_export_and_log(blob, message_id or None)
        convs = parse_conversations_from_zip(blob)
        count = upsert_conversations(export_id, convs)
        print(f"[ok] stored export {export_id}; conversations upserted={count}")
        mail.store(num, "+FLAGS", "\\Seen")
        if DELETE_EMAIL_AFTER_SUCCESS:
            mail.store(num, "+FLAGS", "\\Deleted")
            mail.expunge()
        return True
    except Exception as e:
        print("[err] processing failed:", e)
        return False


def main():
    loop = 0
    while True:
        try:
            M = connect_imap()
            # UNSEEN first to avoid reprocessing
            typ, data = M.search(None, "UNSEEN", "FROM", f'"{SENDER}"')
            ids = data[0].split() if (typ == "OK" and data and data[0]) else []
            if ids:
                print(f"[imap] candidate messages: {len(ids)} (tail): {ids[-5:]}")
                for num in ids:
                    try:
                        process_message(M, num)
                    except Exception as e:
                        print("[err] per-message:", e)
            M.logout()
        except Exception as e:
            print("[err] imap loop:", e)

        loop += 1
        if CLEANUP_DAYS > 0 and loop % max(CLEANUP_EVERY_N_LOOPS, 1) == 0:
            cleanup_old_exports(CLEANUP_DAYS)
        time.sleep(POLL_SECS)


if __name__ == "__main__":
    main()
