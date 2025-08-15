# backend/semantics/embeddings.py
from __future__ import annotations
import os
from typing import List

# ENV:
#   EMBED_PROVIDER = "openai" | "cohere" (default: openai)
#   EMBED_MODEL    = OpenAI embedding model name (default: text-embedding-3-small)
#   RMS_FORCE_DIMS = optional int; if set, we re-embed with OpenAI to match

_provider = os.getenv("EMBED_PROVIDER", "openai").strip().lower()
_openai_model = os.getenv("EMBED_MODEL", "text-embedding-3-small").strip()


def _embed_openai(text: str) -> List[float]:
    from openai import OpenAI

    client = OpenAI()
    resp = client.embeddings.create(model=_openai_model, input=[text])
    return resp.data[0].embedding  # list[float], 1536 for text-embedding-3-small


def _embed_cohere(text: str) -> List[float]:
    # Only if you really need it; cohere often returns 1024 dims.
    import cohere

    api_key = os.environ["COHERE_API_KEY"]
    co = cohere.ClientV2(api_key)
    out = co.embed(texts=[text], model="embed-english-v3.0")
    return list(out.embeddings[0].float)  # adapt to your cohere SDK version


def embed_text(text: str) -> List[float]:
    # First pass: selected provider
    if _provider == "cohere":
        vec = _embed_cohere(text)
    else:
        vec = _embed_openai(text)

    # If we forced dims, ensure compliance
    want = os.getenv("RMS_FORCE_DIMS")
    if want and want.isdigit():
        need = int(want)
        if len(vec) != need:
            # Re-embed with OpenAI 1536 if mismatch (don’t try to pad/truncate)
            vec = _embed_openai(text)
            if len(vec) != need:
                raise ValueError(
                    f"Embedding dims mismatch: want {need}, got {len(vec)}"
                )

    return vec
