# backend/services/conversation.py
from __future__ import annotations
import os
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional

DB_PATH = os.getenv("RMS_CONV_DB", "data/conversations.sqlite3")
_DEFAULT_N = int(os.getenv("RMS_THREAD_N", "20") or 20)

def _ensure_db() -> None:
    p = Path(DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ts DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_session_ts ON conversations(session, ts)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conv_config (
                session TEXT PRIMARY KEY,
                thread_n INTEGER
            )
        """)
        conn.commit()
    finally:
        conn.close()

_ensure_db()

def default_n() -> int:
    return _DEFAULT_N

def append_message(session: str, role: str, content: str) -> None:
    if not session or not role:
        return
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO conversations(session, role, content) VALUES(?,?,?)",
            (session, role, content),
        )
        conn.commit()
    finally:
        conn.close()

def get_session_n(session: Optional[str]) -> Optional[int]:
    if not session:
        return None
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT thread_n FROM conv_config WHERE session=?", (session,))
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else None
    finally:
        conn.close()

def set_session_n(session: str, n: int) -> None:
    if not session:
        return
    n = max(0, int(n))
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO conv_config(session, thread_n) VALUES(?,?)
            ON CONFLICT(session) DO UPDATE SET thread_n=excluded.thread_n
            """,
            (session, n),
        )
        conn.commit()
    finally:
        conn.close()

def get_messages(session: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    if not session:
        return []
    if limit is None:
        limit = get_session_n(session) or default_n()
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT role, content
            FROM conversations
            WHERE session=?
            ORDER BY ts DESC, id DESC
            LIMIT ?
            """,
            (session, int(limit)),
        )
        rows = cur.fetchall() or []
        rows.reverse()  # chronological
        return [{"role": r[0], "content": r[1]} for r in rows]
    finally:
        conn.close()

# alias for clarity in callers
def get_last_messages(session: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    return get_messages(session, limit)

def clear_session(session: str) -> int:
    if not session:
        return 0
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM conversations WHERE session=?", (session,))
        deleted = cur.rowcount or 0
        conn.commit()
        return deleted
    finally:
        conn.close()

def export_messages(session: str, limit: int = 50) -> List[Dict[str, str]]:
    return get_messages(session, limit)
