# backend/llm/multi_model_clients.py
from typing import Dict, Any, List
import os
import cohere
import logging

# Initialize Cohere client
try:
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    if not os.getenv("COHERE_API_KEY"):
        co = None  # Don't initialize if key is missing
except Exception:
    co = None


def call_gemini_flash(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {}


def call_gemini_pro(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {}


def call_reranker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Reranks a list of change spans against a task query using Cohere.
    Falls back to a simple slice if the API is unavailable or fails.
    """
    spans = payload.get("change_spans", [])
    if not co:
        logging.getLogger("reranker").warning(
            "Cohere client not available, using fallback."
        )
        return spans[:5]

    query = payload.get("task")
    if not query or not spans:
        return spans

    # The documents to be reranked are the 'intent' strings from each span
    documents = [s.get("intent", "") for s in spans]

    try:
        reranked_results = co.rerank(
            query=query, documents=documents, model="rerank-english-v3.0"
        )
        # Map the reranked results back to the original span objects
        return [spans[r.index] for r in reranked_results.results]
    except Exception as e:
        logging.getLogger("reranker").error(f"Cohere API call failed: {e}")
        # Fallback to original order on error
        return spans
