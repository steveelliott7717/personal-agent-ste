from typing import List, Dict, Any
from backend.services.supabase_service import supabase
from backend.semantics.store import embed_text  # embed_text lives in store.py right now

def search(namespace: str, query: str, k: int = 10) -> List[Dict[str, Any]]:
    qemb = embed_text(query)
    # Try server-side RPC first
    try:
        resp = supabase.rpc("semantic_search_agent_embeddings", {
            "p_namespace": namespace,
            "p_query_embedding": qemb,
            "p_match_count": k
        }).execute()
        if hasattr(resp, "data") and resp.data:
            return resp.data
    except Exception:
        pass
    # Fallback: client-side cosine
    rows = supabase.table("agent_embeddings").select("*").eq("namespace", namespace).execute().data or []
    import math
    def cos(a, b):
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a)) or 1.0
        nb = math.sqrt(sum(x*x for x in b)) or 1.0
        return dot/(na*nb)
    for r in rows:
        emb = r.get("embedding") or []
        r["score"] = cos(qemb, emb) if emb else 0.0
    rows.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    return rows[:k]
