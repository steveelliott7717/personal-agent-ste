from typing import Any, Dict
try:
    from semantics.embedding import embed_text
except Exception:
    from semantics.embeddings import embed_text

from services.supabase_service import supabase

def upsert(namespace: str, doc_id: str, text: str, metadata: Dict[str, Any] | None = None):
    emb = embed_text(text)
    payload = {
        "namespace": namespace,
        "doc_id": doc_id,
        "text": text,
        "embedding": emb,
        "metadata": metadata or {}
    }
    supabase.table("agent_embeddings").upsert(payload, on_conflict="namespace,doc_id").execute()
