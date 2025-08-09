# backend/utils/agent_registry.py
from __future__ import annotations
from typing import Dict, Any, List
import logging

from services.supabase_service import supabase
logger = logging.getLogger("agent-registry")

def register_agent(meta: Dict[str, Any]) -> str:
    """
    Idempotently register/update an agent in Supabase.
    meta keys:
      slug, title, description, handler_key, namespaces(list[str]),
      capabilities(list[str]), keywords(list[str]), status('enabled'|'disabled')
    Returns agent_id.
    """
    slug = meta["slug"].strip().lower()
    payload = {
        "slug": slug,
        "title": meta["title"],
        "description": meta["description"],
        "handler_key": meta["handler_key"],
        "namespaces": meta.get("namespaces") or [],
        "status": meta.get("status", "enabled"),
    }

    # Upsert agent row
    res = supabase.table("agents").upsert(payload, on_conflict="slug").execute()
    # Not all PostgREST versions return the row on upsert; ensure we have it
    if res.data and len(res.data) == 1 and res.data[0].get("id"):
        agent_id = res.data[0]["id"]
    else:
        agent_id = (
            supabase.table("agents").select("id").eq("slug", slug).single().execute().data["id"]
        )

    # Replace capabilities (sync model)
    caps: List[str] = meta.get("capabilities") or []
    supabase.table("agent_capabilities").delete().eq("agent_id", agent_id).execute()
    if caps:
        supabase.table("agent_capabilities").insert(
            [{"agent_id": agent_id, "capability": c} for c in caps]
        ).execute()

    # Replace keywords (sync model)
    kws: List[str] = meta.get("keywords") or []
    supabase.table("agent_keywords").delete().eq("agent_id", agent_id).execute()
    if kws:
        supabase.table("agent_keywords").insert(
            [{"agent_id": agent_id, "keyword": k} for k in kws]
        ).execute()

    logger.info("registered agent '%s' (%s) with %d caps, %d keywords",
                slug, agent_id, len(caps), len(kws))
    return agent_id
