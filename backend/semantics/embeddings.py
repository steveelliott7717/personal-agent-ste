# backend/semantics/embeddings.py
from __future__ import annotations
from typing import List
import os

try:
    import cohere
except Exception:
    cohere = None

# ---- Config ----
# You can override these with env vars in Fly:
#   COHERE_API_KEY, COHERE_EMBED_MODEL
_API_KEY = os.getenv("COHERE_API_KEY")
_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
# embed-english-v3.0 returns 1024-d vectors
_DIM_FALLBACK = int(os.getenv("COHERE_EMBED_DIM", "1024"))

_client_cache = None


def _client():
    """Return a singleton Cohere client instance."""
    global _client_cache
    if _client_cache is not None:
        return _client_cache
    if cohere is None:
        raise RuntimeError("cohere package not installed")
    if not _API_KEY:
        raise RuntimeError("COHERE_API_KEY is not set")
    _client_cache = cohere.Client(_API_KEY)
    return _client_cache


def _zeros(n: int) -> List[float]:
    return [0.0] * n


def embed_text(text: str) -> List[float]:
    """
    Embed a single text. Returns a list[float].
    On any error, returns a zero vector so the app never 500s.
    """
    t = (text or "").strip()
    if not t:
        return _zeros(_DIM_FALLBACK)
    try:
        resp = _client().embed(texts=[t], model=_MODEL, input_type="search_query")
        # Cohere returns .embeddings: List[List[float]]
        return list(map(float, resp.embeddings[0]))
    except Exception:
        return _zeros(_DIM_FALLBACK)


def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Embed many texts. Returns a list of vectors.
    On any error, returns zero vectors for each input.
    """
    items = [(x or "").strip() for x in (texts or [])]
    if not items:
        return []
    try:
        resp = _client().embed(texts=items, model=_MODEL, input_type="search_query")
        return [list(map(float, v)) for v in resp.embeddings]
    except Exception:
        return [_zeros(_DIM_FALLBACK) for _ in items]
