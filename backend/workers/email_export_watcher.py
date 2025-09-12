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

SENDER = os.environ.get(
    "EXPORT_SENDER", "noreply@openai.com"
)  # you can set to 'openai.com' to loosen
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


def connect_imap():
    if IMAP_SSL:
        M = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    else:
        M = imaplib.IMAP4(IMAP_HOST, IMAP_PORT)
        M.starttls(ssl_context=ssl.create_default_context())
    M.login(IMAP_USER, IMAP_PASS)
    M.select(IMAP_FOLDER)
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

        # 2) Strip trailing junk that sometimes sneaks in (quotes, ); etc.)
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
    Download the export ZIP. Some chatgpt.com links return 4xx to bare clients.
    We retry with browser-like headers and short backoff.
    """
    import time

    HEADERS = {
        # Chrome-like UA
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        # Helps some CDNs on chatgpt.com
        "Referer": "https://chatgpt.com/",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    for attempt in range(1, 6):
        try:
            r = requests.get(url, timeout=180, allow_redirects=True, headers=HEADERS)
            ct = (r.headers.get("Content-Type") or "").lower()
            print(
                f"[dbg] download attempt={attempt} status={r.status_code} ct={ct} len={len(r.content)}"
            )

            # Success path: any 2xx with non-html/octet-stream-ish is OK
            if 200 <= r.status_code < 300:
                # Sometimes it's application/octet-stream; that's fine.
                if "text/html" in ct and len(r.content) < 1024:
                    # tiny HTML often means error page; keep trying
                    raise requests.HTTPError(f"html small body: {len(r.content)}")
                return r.content

            # 403/422 transient? Short backoff and retry
            if r.status_code in (403, 422, 429, 503):
                time.sleep(1.5 * attempt)
                continue

            # Any other 4xx/5xx -> raise
            r.raise_for_status()

        except Exception as e:
            print(f"[dbg] download error attempt={attempt}: {e}")
            time.sleep(1.0 * attempt)

    # If we failed all attempts, surface a helpful error
    raise RuntimeError(
        "Failed to download export ZIP after headered retries. "
        "If the link is old, request a fresh export email and try again."
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
                        if not line:
                            continue
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
            # broadened: FROM "SENDER" where SENDER can be 'openai.com' or 'noreply@openai.com'
            typ, data = M.search(None, "FROM", f'"{SENDER}"')
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
