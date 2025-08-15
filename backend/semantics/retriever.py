# backend/semantics/retriever.py
from __future__ import annotations
from typing import List, Dict, Any, Iterable
import json, math

from backend.services.supabase_service import supabase
from backend.semantics.store import embed_text


def _to_vec(v: Any) -> List[float]:
    """
    Normalize a Supabase 'vector' field into a List[float].
    Handles: already-a-list, JSON-strings like "[...]", pgvector text like "{...}" or "(...)".
    Returns [] if unusable.
    """
    if v is None:
        return []
    # Already a list/tuple
    if isinstance(v, (list, tuple)):
        out: List[float] = []
        for x in v:
            try:
                out.append(float(x))
            except Exception:
                return []
        return out
    # String cases
    if isinstance(v, str):
        s = v.strip()
        # JSON-like
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                return [float(x) for x in arr]
            except Exception:
                return []
        # pgvector text forms "(...)" or "{...}"
        if (s.startswith("(") and s.endswith(")")) or (
            s.startswith("{") and s.endswith("}")
        ):
            s2 = s[1:-1]  # strip parens/braces
            try:
                parts = [p.strip() for p in s2.split(",")]
                return [float(p) for p in parts if p]
            except Exception:
                return []
    # Unknown
    return []


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def search(namespace: str, query: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Simple local top-k cosine search over agent_embeddings for a given namespace.
    Used as a generic fallback elsewhere.
    """
    qemb = embed_text(query)
    rows = (
        supabase.table("agent_embeddings")
        .select("*")
        .eq("namespace", namespace)
        .execute()
        .data
        or []
    )
    for r in rows:
        emb = _to_vec(r.get("embedding"))
        r["score"] = _cosine(qemb, emb) if emb else 0.0
    rows.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    return rows[:k]


def search_router(query: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Blend capability matches (agent_capabilities) with historical routing utterances (namespace='routing').

    final_score = 0.6 * capability_score + 0.4 * utterance_score
    """
    qemb = embed_text(query)

    # 1) Capabilities (client-side cosine)
    caps = (
        supabase.table("agent_capabilities")
        .select("agent_slug, description, embedding")
        .execute()
        .data
        or []
    )
    cap_rows: Dict[str, Dict[str, Any]] = {}
    for c in caps:
        emb = _to_vec(c.get("embedding"))
        s = _cosine(qemb, emb) if emb else 0.0
        cap_rows[c["agent_slug"]] = {
            "type": "capability",
            "agent_slug": c["agent_slug"],
            "text": c.get("description") or "",
            "score_cap": s,
            "score_utt": 0.0,
        }

    # 2) Historical utterances via RPC, fallback to local cosine
    try:
        resp = supabase.rpc(
            "semantic_search_agent_embeddings",
            {
                "p_namespace": "routing",
                "p_query_embedding": qemb,
                "p_match_count": k,
            },
        ).execute()
        utt = getattr(resp, "data", None) or []
        # RPC rows: (doc_id, text, score, metadata)
        for u in utt:
            ref = None
            md = u.get("metadata")
            if isinstance(md, dict):
                ref = md.get("ref") or md.get("agent") or None
            if not ref and "ref" in u:
                ref = u.get("ref")
            if not ref:
                continue
            row = cap_rows.setdefault(
                ref,
                {
                    "type": "capability",
                    "agent_slug": ref,
                    "text": "",
                    "score_cap": 0.0,
                    "score_utt": 0.0,
                },
            )
            row["score_utt"] = max(
                row.get("score_utt", 0.0), float(u.get("score") or 0.0)
            )
    except Exception:
        rows = (
            supabase.table("agent_embeddings")
            .select("*")
            .eq("namespace", "routing")
            .execute()
            .data
            or []
        )
        for r in rows:
            emb = _to_vec(r.get("embedding"))
            s = _cosine(qemb, emb) if emb else 0.0
            ref = r.get("ref") or (r.get("metadata") or {}).get("ref")
            if not ref:
                continue
            row = cap_rows.setdefault(
                ref,
                {
                    "type": "capability",
                    "agent_slug": ref,
                    "text": "",
                    "score_cap": 0.0,
                    "score_utt": 0.0,
                },
            )
            row["score_utt"] = max(row["score_utt"], s)

    # 3) Blend and rank
    blended: List[Dict[str, Any]] = []
    for row in cap_rows.values():
        row["score"] = 0.6 * float(row.get("score_cap", 0.0)) + 0.4 * float(
            row.get("score_utt", 0.0)
        )
        blended.append(row)

    blended.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    return blended[:k]
