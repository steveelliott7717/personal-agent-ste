# backend/workers/email_export_watcher.py
# Robust Gmail/IMAP watcher for ChatGPT export emails:
# - Finds latest unread export email from OpenAI
# - Extracts the actual ZIP download URL from HTML (handles meta-refresh, JS redirects, relative hrefs)
# - Downloads ZIP with host fallbacks (chatgpt.com -> chat.openai.com)
# - Stores ZIP in Supabase Storage and upserts conversations into a table
# - Marks the email as read (and optionally deletes) on success

import os, re, time, ssl, email, imaplib, datetime as dt, requests, io, zipfile, json, hashlib
from email.header import decode_header
from typing import Optional, List
from supabase import create_client
import html
import string
import urllib.parse as up

# ---------- Config from env ----------
IMAP_HOST = os.environ["IMAP_HOST"]
IMAP_PORT = int(os.environ.get("IMAP_PORT", "993"))
IMAP_SSL = os.environ.get("IMAP_SSL", "true").lower() == "true"
IMAP_USER = os.environ["IMAP_USER"]
IMAP_PASS = os.environ["IMAP_PASS"]
IMAP_FOLDER = os.environ.get("IMAP_FOLDER", "INBOX")
POLL_SECS = int(os.environ.get("POLL_SECS", "120"))

# Be permissive: default to substring 'openai.com' so it matches noreply@tm.openai.com too.
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


def connect_imap():
    if IMAP_SSL:
        M = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    else:
        M = imaplib.IMAP4(IMAP_HOST, IMAP_PORT)
        M.starttls(ssl_context=ssl.create_default_context())
    M.login(IMAP_USER, IMAP_PASS)
    # Try desired folder; if it fails (Gmail naming oddities), attempt common alternates.
    typ, _ = M.select(IMAP_FOLDER)
    if typ != "OK":
        for alt in (
            "INBOX",
            "[Gmail]/All Mail",
            "[Gmail]/All Mail",
            "[Gmail]/All Mail",
        ):
            typ, _ = M.select(alt)
            if typ == "OK":
                print(f"[info] fallback-selected folder: {alt}")
                break
        else:
            raise RuntimeError(f"Could not select IMAP folder: '{IMAP_FOLDER}'")
    return M


def extract_urls(payload, charset: Optional[str]) -> List[str]:
    if isinstance(payload, (bytes, bytearray)):
        try:
            text = payload.decode(charset or "utf-8", "ignore")
        except Exception:
            text = ""
    else:
        text = payload or ""

    # Find raw candidates
    candidates = [u for u in URL_REGEX.findall(text) if u.startswith("https://")]

    cleaned: List[str] = []
    for u in candidates:
        # 1) Unescape HTML entities (e.g., &amp;)
        u = html.unescape(u)
        # 2) Strip trailing junk
        u = u.rstrip(")\"' ;" + string.whitespace)
        cleaned.append(u)

    # Prefer likely export links (contain .zip and estuary/content)
    cleaned.sort(
        key=lambda u: (".zip" in u, "estuary/content" in u, "openai" in u, len(u)),
        reverse=True,
    )
    return cleaned


def _normalize_relative(base_url: str, candidate: str) -> str:
    """
    Turn '/backend-api/estuary/content?...' into 'https://host/backend-api/estuary/content?...'
    while preserving the candidate's own query string.
    """
    base = up.urlparse(base_url)
    cand = up.urlparse(candidate)
    # Join with base while preserving candidate query
    return up.urlunparse((base.scheme, base.netloc, cand.path, "", cand.query, ""))


def _extract_zip_from_html(page_html: str, base_url: str) -> Optional[str]:
    """
    Try several strategies to extract a real ZIP link from an HTML landing page.
    """
    body = html.unescape(page_html)

    # 1) <a href="...zip...">
    m = re.search(r'href=["\']([^"\']+?\.zip[^"\']*)["\']', body, re.IGNORECASE)
    if m:
        href = m.group(1).strip()
        return _normalize_relative(base_url, href) if href.startswith("/") else href

    # 2) <meta http-equiv="refresh" content="0;url=...zip...">
    m = re.search(
        r'http-equiv=["\']refresh["\'][^>]*content=["\'][^"\']*url=([^"\']+?\.zip[^"\']*)["\']',
        body,
        re.IGNORECASE,
    )
    if m:
        href = m.group(1).strip()
        return _normalize_relative(base_url, href) if href.startswith("/") else href

    # 3) JS redirects: location.href = "....zip...." or location.assign("....zip....")
    m = re.search(
        r'location(?:\.href|\.assign)\((["\'])([^"\']+?\.zip[^"\']*)\1\)',
        body,
        re.IGNORECASE,
    )
    if m:
        href = m.group(2).strip()
        return _normalize_relative(base_url, href) if href.startswith("/") else href

    # 4) Any estuary content link (absolute), even if '.zip' is omitted (some variants)
    m = re.search(
        r'["\'](https?://[^"\']*?/backend-api/estuary/content[^"\']*?)["\']',
        body,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    # 5) Any estuary content link (relative)
    m = re.search(
        r'["\'](/backend-api/estuary/content[^"\']*?)["\']', body, re.IGNORECASE
    )
    if m:
        return _normalize_relative(base_url, m.group(1).strip())

    return None


def find_download_url_from_msg(msg) -> Optional[str]:
    """
    Returns the first plausible URL from the message body; we will
    still treat HTML landing pages by parsing out the real ZIP URL.
    """

    def pick(urls: List[str]) -> Optional[str]:
        # Prefer explicit .zip but fall back to first candidate
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
    """
    Download the export ZIP from chatgpt.com estuary.
    - Follows redirects
    - If the response is HTML, parse out the real .zip link (handles relative, meta-refresh, JS redirects)
    - Falls back to chat.openai.com host if chatgpt.com refuses us
    """
    session = requests.Session()
    session.headers.update(BASE_HEADERS)

    def try_get(u: str, attempt: int, label: str = "") -> requests.Response:
        r = session.get(u, timeout=180, allow_redirects=True)
        ct = (r.headers.get("Content-Type") or "").lower()
        host = up.urlparse(u).netloc
        lab = f" {label}" if label else ""
        print(
            f"[dbg] dl attempt={attempt} host={host} status={r.status_code} ct={ct} len={len(r.content)}{lab}"
        )
        return r

    MAX_ATTEMPTS = 6
    RETRY_STATUSES = {403, 422, 429, 503}

    for attempt in range(1, MAX_ATTEMPTS + 1):
        r = try_get(url, attempt)

        # Fast path: got a non-HTML payload
        ct = (r.headers.get("Content-Type") or "").lower()
        if 200 <= r.status_code < 300 and "text/html" not in ct:
            return r.content

        # HTML page: try to extract a .zip or estuary link
        if "text/html" in ct and r.text:
            zurl = _extract_zip_from_html(r.text, url)
            if zurl:
                zr = try_get(zurl, attempt, label="(extracted)")
                zct = (zr.headers.get("Content-Type") or "").lower()
                if 200 <= zr.status_code < 300 and "text/html" not in zct:
                    return zr.content

                # host swap for the extracted URL too
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

        # Host fallback (original URL), on specific transient statuses or persistent HTML
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

        # Backoff
        time.sleep(1.25 * attempt)

    raise RuntimeError(
        "Failed to download export ZIP after parsing HTML and host fallbacks. "
        "If the email link is old, trigger a fresh export and try again."
    )


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
    # optional notification
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
    """Return a list of conversation dicts from typical ChatGPT export zips."""
    convs: List[dict] = []
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        names = z.namelist()

        # 1) conversations.json (array)
        for name in names:
            if name.lower().endswith("conversations.json"):
                try:
                    data = json.loads(z.read(name).decode("utf-8", "ignore"))
                    if isinstance(data, list):
                        convs.extend(data)
                except Exception:
                    pass

        # 2) conversations/*.json (one per file)
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

        # 3) jsonl (rare)
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

    # De-dup simple
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
    """Upsert conversations into public.chatgpt_conversations."""
    rows = []
    for c in convs:
        conv_id = (
            c.get("id")
            or c.get("conversation_id")
            or hashlib.sha1(
                json.dumps(c, sort_keys=True, default=str).encode()
            ).hexdigest()
        )
        title = c.get("title")
        rows.append(
            {"export_id": export_id, "conv_id": conv_id, "title": title, "raw": c}
        )

    # Batch in chunks (Supabase REST limit ~ 1000 rows)
    CHUNK = 500
    total = 0
    for i in range(0, len(rows), CHUNK):
        batch = rows[i : i + CHUNK]
        supa.table("chatgpt_conversations").upsert(
            batch, on_conflict="export_id,conv_id"
        ).execute()
        total += len(batch)
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

    # Sender / subject checks (permissive by design)
    if SENDER not in from_addr:
        return False
    if SUBJECT_PHRASE and SUBJECT_PHRASE not in subj:
        print(f"[info] subject didn't match hint (subj='{subj}') — continuing")

    # Find download link and pull the ZIP
    url = find_download_url_from_msg(msg)
    if not url:
        print("[warn] no URL found in export email")
        return False

    try:
        blob = download_zip(url)
        export_id = upload_export_and_log(blob, message_id or None)

        # Parse conversations and upsert
        convs = parse_conversations_from_zip(blob)
        count = upsert_conversations(export_id, convs)
        print(f"[ok] stored export {export_id}; conversations upserted={count}")

        # Mark email read (and optionally delete)
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
            # Only unread messages from the sender (so we don't reprocess)
            typ, data = M.search(None, "UNSEEN", "FROM", f'"{SENDER}"')
            if typ == "OK":
                ids = data[0].split() if data and data[0] else []
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
