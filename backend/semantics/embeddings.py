# backend/semantics/embeddings.py
from __future__ import annotations
import os
from typing import List
from openai import OpenAI

# Centralized configuration for embedding
_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
_EMBED_DIMENSIONS = int(os.getenv("EMBED_DIMENSIONS", "1024"))

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed_text(text: str) -> List[float]:
    """Generates an embedding for a single text string with configured dimensions."""
    if not text:
        return []
    res = _client.embeddings.create(
        model=_EMBED_MODEL, input=[text], dimensions=_EMBED_DIMENSIONS
    )
    return res.data[0].embedding


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Generates embeddings for a batch of texts with configured dimensions."""
    if not texts:
        return []
    res = _client.embeddings.create(
        model=_EMBED_MODEL, input=texts, dimensions=_EMBED_DIMENSIONS
    )
    return [d.embedding for d in res.data]
