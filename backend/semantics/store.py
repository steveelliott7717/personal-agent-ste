# backend/semantics/store.py
from __future__ import annotations
from typing import Dict, Any, Optional

from backend.services.supabase_service import supabase
from backend.semantics.embeddings import embed_text


def upsert(
    namespace: str,
    doc_id: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    kind: Optional[str] = None,
    ref: Optional[str] = None,
) -> None:
    """
    Embeds text and upserts it into the agent_embeddings table.
    """
    if not all([namespace, doc_id, text]):
        raise ValueError("namespace, doc_id, and text are required for upsert.")

    embedding = embed_text(text)

    payload = {
        "namespace": namespace,
        "doc_id": doc_id,
        "text": text,
        "embedding": embedding,
        "metadata": metadata or {},
        "kind": kind,
        "ref": ref,
    }

    # Use on_conflict to handle upsert logic
    supabase.table("agent_embeddings").upsert(
        payload, on_conflict="namespace,doc_id"
    ).execute()
