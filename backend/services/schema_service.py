from __future__ import annotations
from typing import List, Dict, Any, Optional
import os, requests

# Reads (metadata) via PostgREST RPCs we created in Supabase SQL editor:
#   - public.list_tables(schema_name text default 'public')
#   - public.list_columns(schema_name text default 'public', table_name text default null)
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

_HDRS = {
    "apikey": SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    "Content-Type": "application/json",
}


def _rpc(name: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise RuntimeError("SUPABASE_URL or SUPABASE_ANON_KEY missing")
    url = f"{SUPABASE_URL}/rest/v1/rpc/{name}"
    r = requests.post(url, headers=_HDRS, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def list_tables(schema: str = "public") -> List[Dict[str, Any]]:
    return _rpc("list_tables", {"schema_name": schema})


def list_columns(
    schema: str = "public", table: Optional[str] = None
) -> List[Dict[str, Any]]:
    payload: Dict[str, Any] = {"schema_name": schema}
    if table:
        payload["table_name"] = table
    return _rpc("list_columns", payload)
