# backend/services/conversation.py
from __future__ import annotations
import os
from typing import List, Dict, Optional

from backend.services.supabase_service import supabase

DEFAULT_THREAD_N = int(os.getenv("RMS_THREAD_N", "20") or 20)

def _safe_int(v, fallback: int) -> int:
    try:
        return int(v)
    except Exception:
        return fallback

# --- Writes ---

def append_message(session: str, role: str, content: str) -> None:
    """Append one chat turn."""
    if not session or not role or content is None:
        return
    supabase.table("conversations").insert({
        "session": session,
        "role": role,
        "content": content,
    }).execute()

def set_thread_n(session: str, n: int) -> None:
    """Set per-session history length (upsert)."""
    if not session:
        return
    n = max(0, _safe_int(n, DEFAULT_THREAD_N))
    supabase.table("conv_config").upsert({
        "session": session,
        "thread_n": n,
    }).execute()

# --- Reads ---

def get_thread_n(session: Optional[str]) -> int:
    """Read per-session N; fall back to env default."""
    if not session:
        return DEFAULT_THREAD_N
    res = supabase.table("conv_config").select("thread_n").eq("session", session).limit(1).execute()
    rows = getattr(res, "data", None) or []
    if rows and "thread_n" in rows[0]:
        return _safe_int(rows[0]["thread_n"], DEFAULT_THREAD_N)
    return DEFAULT_THREAD_N

def get_messages(session: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Return last N messages (chronological)."""
    if not session:
        return []
    if limit is None:
        limit = get_thread_n(session)
    limit = max(0, _safe_int(limit, DEFAULT_THREAD_N))
    if limit == 0:
        return []
    res = (supabase.table("conversations")
           .select("role,content,created_at")
           .eq("session", session)
           .order("created_at", desc=True)
           .limit(limit)
           .execute())
    rows = getattr(res, "data", None) or []
    rows.reverse()  # chronological
    return [{"role": r["role"], "content": r["content"]} for r in rows]

def export_messages(session: str, limit: int = 50) -> List[Dict[str, str]]:
    return get_messages(session, limit=limit)

# --- Back-compat shims for repo_agent imports ---

def get_session_n(session: str) -> int:
    """Alias of get_thread_n(session)."""
    try:
        return get_thread_n(session)  # if you have this already
    except NameError:
        # If this file uses a different name, fall back to DEFAULT_THREAD_N
        return DEFAULT_THREAD_N

def default_n() -> int:
    """Return the default limit (alias of DEFAULT_THREAD_N)."""
    try:
        return DEFAULT_THREAD_N
    except NameError:
        # Reasonable hardcoded fallback if constant not found for some reason
        return 20


def clear_session(session: str) -> int:
    """Delete all messages for a session; return count deleted."""
    if not session:
        return 0
    # Count first (PostgREST delete usually doesn't return deleted rows)
    cnt_res = supabase.table("conversations").select("id", count="exact").eq("session", session).execute()
    total = getattr(cnt_res, "count", None) or 0
    supabase.table("conversations").delete().eq("session", session).execute()
    return int(total)
