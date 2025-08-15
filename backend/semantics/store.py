# backend/semantics/store.py
from __future__ import annotations
from typing import Any, Dict, Optional

from backend.semantics.embeddings import embed_text
from backend.services.supabase_service import supabase


def upsert(
    namespace: str,
    doc_id: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    kind: Optional[str] = None,
    ref: Optional[str] = None,
) -> None:
    """
    Upsert a single semantic record into public.agent_embeddings using namespace+doc_id as the conflict key.
    Embeds the provided text with Cohere (1024-d) under the hood.

    Parameters
    ----------
    namespace : logical grouping of vectors (e.g., "routing", "meals", "training")
    doc_id    : unique id within the namespace (we set a UNIQUE (namespace, doc_id) in SQL)
    text      : raw text to embed
    metadata  : optional JSON-serializable dict stored in metadata jsonb
    kind      : 'utterance' or 'capability' (default is handled server-side if None)
    ref       : optional pointer (e.g., agent_slug) to help router attribution
    """
    emb = embed_text(text)

    payload: Dict[str, Any] = {
        "namespace": namespace,
        "doc_id": doc_id,
        "text": text,
        "embedding": emb,  # vector(1024) on the DB side
        "metadata": metadata or {},  # jsonb
    }
    if kind is not None:
        payload["kind"] = kind
    if ref is not None:
        payload["ref"] = ref

    # Uses UNIQUE (namespace, doc_id) set in the migration
    supabase.table("agent_embeddings").upsert(
        payload,
        on_conflict="namespace,doc_id",
    ).execute()
