# backend/services/agent_settings.py
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

from backend.services.supabase_service import supabase


def _coerce_lines(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list) and all(isinstance(x, str) for x in val):
        return val
    if isinstance(val, str):
        return [ln for ln in val.splitlines() if ln.strip()]
    return []


@lru_cache(maxsize=256)
def get_settings(slug: str) -> Dict[str, Any]:
    """Return {key: value} for an agent slug from public.agent_settings."""
    res = (
        supabase.table("agent_settings")
        .select("key,value")
        .eq("agent_slug", slug)
        .execute()
    )
    out: Dict[str, Any] = {}
    for row in res.data or []:
        out[row["key"]] = row["value"]
    return out


def load_agent_prompt(slug: str) -> str:
    """
    Merge includes (default: ['jobs_common_rules']) + this agent's system_prompt,
    returning one newline-joined string for the LLM.
    """
    s = get_settings(slug)
    includes = s.get("includes") or ["jobs_common_rules"]

    lines: List[str] = []
    for inc in includes:
        inc_val = get_settings(inc).get("system_prompt")
        lines.extend(_coerce_lines(inc_val))

    lines.extend(_coerce_lines(s.get("system_prompt")))
    return "\n".join(lines)


def verbs_allowlist(slug: str) -> List[str]:
    v = get_settings(slug).get("verbs_allowlist")
    return list(v) if isinstance(v, list) else []


def clear_cache() -> None:
    get_settings.cache_clear()  # type: ignore[attr-defined]
