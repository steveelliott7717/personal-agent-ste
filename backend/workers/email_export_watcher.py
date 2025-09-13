# backend/workers/email_export_watcher.py
import os, re, time, ssl, email, imaplib, datetime as dt, requests, io, zipfile, json, hashlib
from email.header import decode_header
from typing import Optional, List
from supabase import create_client
import html
import string

# ---------- Config from env ----------
IMAP_HOST = os.environ["IMAP_HOST"]
IMAP_PORT = int(os.environ.get("IMAP_PORT", "993"))
IMAP_SSL = os.environ.get("IMAP_SSL", "true").lower() == "true"
IMAP_USER = os.environ["IMAP_USER"]
IMAP_PASS = os.environ["IMAP_PASS"]
IMAP_FOLDER = os.environ.get("IMAP_FOLDER", "INBOX")
POLL_SECS = int(os.environ.get("POLL_SECS", "120"))

# Search behavior
IMAP_ONLY_UNSEEN = os.environ.get("IMAP_ONLY_UNSEEN", "true").lower() == "true"
FALLBACK_SEARCH_SINCE = (
    os.environ.get("FALLBACK_SEARCH_SINCE", "true").lower() == "true"
)
SEARCH_SINCE_DAYS = int(os.environ.get("SEARCH_SINCE_DAYS", "3"))

SENDER = os.environ.get(
    "EXPORT_SENDER", "noreply@openai.com"
)  # set to 'openai.com' to loosen
SUBJECT_PHRASE = os.environ.get("EXPORT_SUBJECT_PHRASE", "export")
URL_REGEX = re.compile(r'https?://[^\s">]+', re.IGNORECASE)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE"]
BUCKET = os.environ.get("SUPABASE_BUCKET", "chatgpt_exports")

DELETE_EMAIL_AFTER_SUCCESS = (
    os.environ.get("DELETE_EMAIL_AFTER_SUCCESS", "false").lower() == "true"
)
CLEANUP_DAYS = int(os.environ.get("CLEANUP_DAYS", "0"))
CLEANUP_EVERY_N_LOOPS = int(os.environ.get("CLEANUP_EVERY_N_LOOPS", "3600"))

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


def _try_select(mail, box: str) -> bool:
    try:
        typ, _ = mail.select(box)
        if typ == "OK":
            print(f"[imap] selected mailbox: {box}")
            return True
    except Exception as e:
        print(f"[imap] select '{box}' failed: {e}")
    return False


def connect_imap():
    if IMAP_SSL:
        M = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    else:
        M = imaplib.IMAP4(IMAP_HOST, IMAP_PORT)
        M.starttls(ssl_context=ssl.create_default_context())
    M.login(IMAP_USER, IMAP_PASS)

    # Build a robust candidate list of mailboxes to try.
    candidates: List[str] = []
    if IMAP_FOLDER:
        candidates.append(IMAP_FOLDER)
    # Common Gmail variants
    candidates += ["[Gmail]/All Mail", "[Gmail]/All Mail", "All Mail", "INBOX"]

    # Also inspect server mailboxes for localized "All Mail" equivalents
    try:
        typ, boxes = M.list()
        if typ == "OK" and boxes:
            for b in boxes:
                line = b.decode(errors="ignore")
                # Typical format: (<flags>) "<delim>" "Box Name"
                # Get the last quoted token as the name
                name = line.split(' "/" ')[-1].strip().strip('"')
                if "all" in name.lower() and "mail" in name.lower():
                    candidates.insert(1, name)
    except Exception as e:
        print(f"[imap] list failed (non-fatal): {e}")

    # Deduplicate, preserve order
    seen = set()
    ordered = [c for c in candidates if not (c in seen or seen.add(c))]

    for box in ordered:
        if _try_select(M, box):
            return M

    # Last resort: INBOX
    if _try_select(M, "INBOX"):
        return M

    raise RuntimeError(f"Could not select any mailbox from candidates: {ordered}")


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
        # Decode any HTML entities (&amp; → &), then strip trailing junk
        u = html.unescape(u)
        u = u.rstrip(")\"' ;" + string.whitespace)
        cleaned.append(u)

    # Prefer likely export links (contain .zip and estuary/content)
    cleaned.sort(
        key=lambda u: (".zip" in u, "estuary/content" in u, "openai" in u, len(u)),
        reverse=True,
    )
    return cleaned


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
    """
    Download the export ZIP from chatgpt.com estuary.
    If the response is HTML, parse out the real .zip link and fetch that.
    Falls back to chat.openai.com host if chatgpt.com refuses us.
    """
    import urllib.parse as up

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "application/zip,application/octet-stream,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://chatgpt.com/",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    HREF_RE = re.compile(r'href=["\']([^"\']+\.zip[^"\']*)["\']', re.IGNORECASE)

    def try_get(u: str, attempt: int):
        r = requests.get(u, timeout=180, allow_redirects=True, headers=HEADERS)
        ct = (r.headers.get("Content-Type") or "").lower()
        print(
            f"[dbg] dl attempt={attempt} host={up.urlparse(u).netloc} status={r.status_code} ct={ct} len={len(r.content)}"
        )
        return r

    def extract_zip_from_html(body: str) -> Optional[str]:
        body = html.unescape(body or "")
        m = HREF_RE.search(body)
        if not m:
            return None
        candidate = m.group(1).strip()
        if candidate.startswith("/"):
            parsed = up.urlparse(url)
            candidate = f"{parsed.scheme}://{parsed.netloc}{candidate}"
        return candidate

    for attempt in range(1, 6):
        r = try_get(url, attempt)
        ct = (r.headers.get("Content-Type") or "").lower()

        if 200 <= r.status_code < 300:
            if "text/html" not in ct:
                return r.content
            # HTML page; try to pull out the .zip href and fetch it
            zurl = extract_zip_from_html(r.text)
            if zurl:
                zr = try_get(zurl, attempt)
                zct = (zr.headers.get("Content-Type") or "").lower()
                if 200 <= zr.status_code < 300 and "text/html" not in zct:
                    return zr.content
                # Try host fallback for zurl
                parsed = up.urlparse(zurl)
                if parsed.netloc == "chatgpt.com":
                    zurl2 = up.urlunparse(parsed._replace(netloc="chat.openai.com"))
                    zr2 = try_get(zurl2, attempt)
                    zct2 = (zr2.headers.get("Content-Type") or "").lower()
                    if 200 <= zr2.status_code < 300 and "text/html" not in zct2:
                        return zr2.content

        # On transient statuses, try host fallback then backoff
        if r.status_code in (403, 422, 429, 503):
            parsed = up.urlparse(url)
            if parsed.netloc == "chatgpt.com":
                alt = up.urlunparse(parsed._replace(netloc="chat.openai.com"))
                r2 = try_get(alt, attempt)
                ct2 = (r2.headers.get("Content-Type") or "").lower()
                if 200 <= r2.status_code < 300 and "text/html" not in ct2:
                    return r2.content
                if "text/html" in ct2:
                    zurl = extract_zip_from_html(r2.text)
                    if zurl:
                        zr = try_get(zurl, attempt)
                        zct = (zr.headers.get("Content-Type") or "").lower()
                        if 200 <= zr.status_code < 300 and "text/html" not in zct:
                            return zr.content
            time.sleep(1.5 * attempt)
            continue

        r.raise_for_status()

    raise RuntimeError(
        "Failed to download export ZIP after parsing HTML and host fallbacks. If the email link is old, trigger a fresh export and try again."
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
    # Optional notification
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


def _already_ingested(message_id: Optional[str]) -> bool:
    if not message_id:
        return False
    try:
        res = (
            supa.table("chatgpt_exports")
            .select("id")
            .eq("email_message_id", message_id)
            .limit(1)
            .execute()
        )
        return bool(res.data)
    except Exception as e:
        print("[warn] precheck for duplicate failed:", e)
        return False


def process_message(mail, num) -> bool:
    typ, data = mail.fetch(num, "(RFC822)")
    if typ != "OK" or not data or not data[0]:
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

    # Avoid duplicate ingestion if we've already stored this message_id
    if _already_ingested(message_id):
        print(
            f"[info] already ingested message_id={message_id}; marking read and skipping"
        )
        try:
            mail.store(num, "+FLAGS", "\\Seen")
        except Exception:
            pass
        return True

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
        try:
            mail.store(num, "+FLAGS", "\\Seen")
        except Exception:
            pass
        if DELETE_EMAIL_AFTER_SUCCESS:
            try:
                mail.store(num, "+FLAGS", "\\Deleted")
                mail.expunge()
            except Exception:
                pass
        return True
    except Exception as e:
        print("[err] processing failed:", e)
        return False


def _search_messages(mail):
    # Prefer UNSEEN to avoid reprocessing, then fall back to SINCE N days if nothing is unseen
    ids: List[bytes] = []
    if IMAP_ONLY_UNSEEN:
        try:
            typ, data = mail.search(None, "UNSEEN", "FROM", f'"{SENDER}"')
            if typ == "OK":
                ids = data[0].split() if data and data[0] else []
        except Exception as e:
            print("[imap] UNSEEN search failed:", e)

    if not ids and FALLBACK_SEARCH_SINCE:
        since = (dt.datetime.utcnow() - dt.timedelta(days=SEARCH_SINCE_DAYS)).strftime(
            "%d-%b-%Y"
        )
        try:
            typ, data = mail.search(None, "SINCE", since, "FROM", f'"{SENDER}"')
            if typ == "OK":
                ids = data[0].split() if data and data[0] else []
        except Exception as e:
            print("[imap] SINCE search failed:", e)

    # Last resort: FROM only (can be noisy)
    if not ids and not IMAP_ONLY_UNSEEN and not FALLBACK_SEARCH_SINCE:
        try:
            typ, data = mail.search(None, "FROM", f'"{SENDER}"')
            if typ == "OK":
                ids = data[0].split() if data and data[0] else []
        except Exception as e:
            print("[imap] FROM-only search failed:", e)

    return ids


def main():
    loop = 0
    while True:
        try:
            M = connect_imap()
            ids = _search_messages(M)
            if ids:
                print(
                    f"[imap] candidate messages: {len(ids)} (showing last 5): {ids[-5:]}"
                )
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
