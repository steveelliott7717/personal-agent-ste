# backend/semantics/backfill_capabilities.py
from __future__ import annotations
from typing import List, Dict, Any

from backend.services.supabase_service import supabase
from backend.semantics.embeddings import embed_batch


def run() -> None:
    """
    Populate/refresh public.agent_capabilities by embedding each enabled agent's description.
    Expects an 'agents' table with (slug, title, description, status).
    """
    res = supabase.table("agents").select("slug, title, description").eq("status", "enabled").execute()
    rows: List[Dict[str, Any]] = getattr(res, "data", None) or []
    if not rows:
        print("[cap-backfill] no agents found")
        return

    keys: List[str] = []
    descs: List[str] = []
    for r in rows:
        slug = (r.get("slug") or "").strip().lower()
        title = (r.get("title") or "").strip()
        description = (r.get("description") or "").strip()
        text = f"{title}: {description}" if (title or description) else slug
        keys.append(slug)
        descs.append(text)

    embs = embed_batch(descs)
    for slug, text, emb in zip(keys, descs, embs):
        supabase.table("agent_capabilities").upsert(
            {"agent_slug": slug, "description": text, "param_schema": {}, "embedding": emb},
            on_conflict="agent_slug",
        ).execute()

    print(f"[cap-backfill] upserted {len(keys)} capability embeddings")


if __name__ == "__main__":
    run()
