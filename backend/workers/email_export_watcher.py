# backend/workers/email_export_watcher.py
import os
import re
import time
import ssl
import io
import email
import imaplib
import datetime as dt
import requests
from email.header import decode_header
from hashlib import sha256
from typing import Optional, List

from supabase import create_client

# ---------- Config (env) ----------
IMAP_HOST = os.environ["IMAP_HOST"]  # e.g. 127.0.0.1 (Proton Bridge) or imap.gmail.com
IMAP_PORT = int(
    os.environ.get("IMAP_PORT", "1143")
)  # Proton Bridge STARTTLS default (993 for implicit SSL)
IMAP_SSL = os.environ.get("IMAP_SSL", "false").lower() == "true"
IMAP_USER = os.environ[
    "IMAP_USER"
]  # Proton Bridge "IMAP username" (NOT your Proton login)
IMAP_PASS = os.environ["IMAP_PASS"]  # Proton Bridge "IMAP password"
IMAP_FOLDER = os.environ.get("IMAP_FOLDER", "INBOX")
POLL_SECS = int(os.environ.get("POLL_SECS", "120"))

SENDER = os.environ.get("EXPORT_SENDER", "noreply@openai.com")
SUBJECT_PHRASE = os.environ.get(
    "EXPORT_SUBJECT_PHRASE", "data export"
)  # permissive match
URL_REGEX = re.compile(r'https?://[^\s">]+', re.IGNORECASE)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE"]  # Service role
BUCKET = os.environ.get("SUPABASE_BUCKET", "chatgpt_exports")

DELETE_EMAIL_AFTER_SUCCESS = (
    os.environ.get("DELETE_EMAIL_AFTER_SUCCESS", "false").lower() == "true"
)

# Cleanup (optional)
CLEANUP_DAYS = int(os.environ.get("CLEANUP_DAYS", "0"))  # 0 disables cleanup
CLEANUP_EVERY_N_LOOPS = int(
    os.environ.get("CLEANUP_EVERY_N_LOOPS", "3600")
)  # big number ≈ “rarely”
# ---------------------------------

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
    """Connect using SSL or STARTTLS (Proton Bridge defaults to STARTTLS on 1143)."""
    if IMAP_SSL:
        M = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    else:
        M = imaplib.IMAP4(IMAP_HOST, IMAP_PORT)
        M.starttls(ssl_context=ssl.create_default_context())
    M.login(IMAP_USER, IMAP_PASS)
    M.select(IMAP_FOLDER)
    return M


def extract_urls_from_payload(
    payload: bytes | str, charset: Optional[str]
) -> List[str]:
    if isinstance(payload, (bytes, bytearray)):
        try:
            text = payload.decode(charset or "utf-8", "ignore")
        except Exception:
            text = ""
    else:
        text = payload or ""
    urls = [u for u in URL_REGEX.findall(text) if u.startswith("https://")]
    # Heuristic: prefer links that look relevant
    urls.sort(
        key=lambda u: (("export" in u) or ("download" in u) or ("openai" in u), len(u)),
        reverse=True,
    )
    return urls


def find_download_url_from_msg(msg) -> Optional[str]:
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype in ("text/html", "text/plain"):
                urls = extract_urls_from_payload(
                    part.get_payload(decode=True), part.get_content_charset()
                )
                if urls:
                    return urls[0]
    else:
        urls = extract_urls_from_payload(
            msg.get_payload(decode=True), msg.get_content_charset()
        )
        if urls:
            return urls[0]
    return None


def download_zip(url: str) -> bytes:
    r = requests.get(url, timeout=180, allow_redirects=True)
    r.raise_for_status()
    return r.content  # may be application/zip or octet-stream


def already_ingested(message_id: Optional[str]) -> bool:
    """Check idempotency by email Message-Id."""
    if not message_id:
        return False
    res = (
        supa.table("chatgpt_exports")
        .select("id")
        .eq("email_message_id", message_id)
        .limit(1)
        .execute()
    )
    return bool(res.data)


def upload_to_supabase_and_log(
    content: bytes,
    message_id: Optional[str],
    note: str = "auto-import from noreply@openai.com",
) -> str:
    # key: timestamped path
    iso = dt.datetime.utcnow().replace(microsecond=0).isoformat().replace(":", "-")
    key = f"chat_history/chatgpt_export_{iso}Z.zip"

    # 1) storage upload (private bucket)
    supa.storage.from_(BUCKET).upload(
        key, content, file_options={"contentType": "application/zip", "upsert": False}
    )

    # 2) checksum
    digest = sha256(content).hexdigest()

    # 3) metadata row
    supa.table("chatgpt_exports").insert(
        {
            "email_message_id": message_id,
            "storage_bucket": BUCKET,
            "storage_path": key,
            "size_bytes": len(content),
            "sha256": digest,
            "notes": note,
        }
    ).execute()

    # 4) optional: notify (uses your existing log table)
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

    return key


def cleanup_old_exports(days: int):
    """Delete old blobs + rows via Supabase APIs (only if CLEANUP_DAYS > 0)."""
    try:
        cutoff = (dt.datetime.utcnow() - dt.timedelta(days=days)).isoformat()
        rows = (
            supa.table("chatgpt_exports")
            .select("id,storage_path,created_at")
            .lt("created_at", cutoff)
            .limit(1000)
            .execute()
            .data
            or []
        )
        for r in rows:
            path = r["storage_path"]
            try:
                supa.storage.from_(BUCKET).remove([path])
                supa.table("chatgpt_exports").delete().eq("id", r["id"]).execute()
                print(f"[cleanup] removed {path}")
            except Exception as e:
                print(f"[cleanup][err] {path}: {e}")
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

    # Subject hint helps, but don't block if it changes
    if SUBJECT_PHRASE and SUBJECT_PHRASE not in subj:
        print(f"[info] from matches, subject didn't (subj='{subj}') — continuing")

    # Idempotency: skip if this Message-Id is already ingested
    if message_id and already_ingested(message_id):
        print(f"[skip] already ingested Message-Id={message_id}")
        mail.store(num, "+FLAGS", "\\Seen")
        return True

    url = find_download_url_from_msg(msg)
    if not url:
        print("[warn] no URL found in export email")
        return False

    try:
        blob = download_zip(url)
        key = upload_to_supabase_and_log(blob, message_id or None)
        print(f"[ok] uploaded export → supabase://{BUCKET}/{key}")

        # Mark seen (and optionally delete) to keep mailbox clean
        mail.store(num, "+FLAGS", "\\Seen")
        if DELETE_EMAIL_AFTER_SUCCESS:
            mail.store(num, "+FLAGS", "\\Deleted")
            mail.expunge()
        return True
    except Exception as e:
        print("[err] download/upload failed:", e)
        return False


def main():
    loop = 0
    while True:
        try:
            mail = connect_imap()
            # Only UNSEEN messages from the sender
            typ, data = mail.search(None, "UNSEEN", f'FROM "{SENDER}"')
            if typ == "OK":
                ids = data[0].split()
                for num in ids:
                    try:
                        process_message(mail, num)
                    except Exception as e:
                        print("[err] per-message:", e)
            mail.logout()
        except Exception as e:
            print("[err] imap loop:", e)

        loop += 1
        if CLEANUP_DAYS > 0 and loop % max(CLEANUP_EVERY_N_LOOPS, 1) == 0:
            cleanup_old_exports(CLEANUP_DAYS)

        time.sleep(POLL_SECS)


if __name__ == "__main__":
    main()
