# backend/services/supabase_service.py
from __future__ import annotations

import os
from typing import Any, Optional

from dotenv import load_dotenv, find_dotenv


# --- Load .env from repo root even when uvicorn cwd varies ---
# We don't override existing env so container/CI secrets still win.
load_dotenv(find_dotenv(usecwd=True), override=False)

# Support either name; service-role preferred
SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
SUPABASE_KEY: Optional[str] = os.getenv("SUPABASE_SERVICE_ROLE") or os.getenv(
    "SUPABASE_KEY"
)

# Lazy client: created on first access so imports don't crash without env
_client = None


def _create_client():
    """
    Create and cache the Supabase client on first use.
    Raises at call-time (not import-time) if credentials are missing.
    """
    global _client
    if _client is not None:
        return _client

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            "Supabase credentials missing. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE "
            "or SUPABASE_KEY in environment or .env at the repo root."
        )

    # Import here to avoid failing import of this module when keys are absent.
    from supabase import create_client  # type: ignore

    _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


class _SupabaseProxy:
    """
    Transparent proxy so existing code can keep doing:
        from backend.services.supabase_service import supabase
        supabase.table("...").select("*").execute()
    The underlying client is initialized on first attribute access.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(_create_client(), name)


# Public handle used throughout the codebase
supabase = _SupabaseProxy()


# --- Optional helpers (useful for diagnostics) ---


def get_client():
    """Return the real Supabase client (initializing it if needed)."""
    return _create_client()


def assert_supabase_ready() -> None:
    """
    Quick readiness check to fail fast at runtime (not import time).
    Call this from startup hooks or health checks if you want.
    """
    _ = _create_client()
