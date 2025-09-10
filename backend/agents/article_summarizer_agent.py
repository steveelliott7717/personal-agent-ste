# backend/agents/article_summarizer_agent.py
from __future__ import annotations

import logging
import json
import re
from typing import Any, Dict

from backend.registry.capability_registry import CapabilityRegistry
from backend.llm.llm_runner import run_llm_agent

logger = logging.getLogger(__name__)
registry = CapabilityRegistry()


def handle_article_summary(query: str) -> Dict[str, Any]:
    """
    Handles a request to summarize an article from a URL.
    Accepts:
      - raw text containing a URL
      - a JSON string with {"url": "..."} or {"task": "... https://..."}
      - a dict (already-parsed) with the same keys (when the router passes structured input)
    """

    def _extract_url_from_text(text: str) -> str | None:
        if not isinstance(text, str):
            return None
        m = re.search(r'https?://[^\s"\'<>]+', text)
        return m.group(0) if m else None

    url: str | None = None
    payload: Dict[str, Any] | None = None

    # If router already passed a dict, use it
    if isinstance(query, dict):
        payload = query
    else:
        try:
            payload = json.loads(query) if isinstance(query, str) else None
        except Exception:
            payload = None

    # Extract URL from structured payload
    if isinstance(payload, dict):
        maybe_url = payload.get("url")
        if isinstance(maybe_url, str) and maybe_url.strip():
            url = maybe_url.strip()
        if not url and isinstance(payload.get("task"), str):
            url = _extract_url_from_text(payload["task"])

    # Fallback: extract URL directly from raw string
    if not url:
        url = _extract_url_from_text(query if isinstance(query, str) else "")

    if not url:
        return {
            "ok": False,
            "error": "No URL found in the request.",
            "details": "Please provide a URL to summarize.",
        }

    # Use the capability registry to fetch and parse the article content
    logger.info(f"[article_summarizer] Fetching content from {url}")
    fetch_result = registry.dispatch("web.smart_get", {"url": url}, {})

    if not fetch_result.get("ok"):
        error_details = fetch_result.get("error", {})
        logger.error(
            f"[article_summarizer] Failed to fetch content from {url}: {error_details}"
        )
        return {
            "ok": False,
            "error": "Failed to fetch article content.",
            "details": error_details,
        }

    # >>> Flatten the nested shape: { ok, result: { ok?, result: {...} } }
    outer = fetch_result.get("result") or {}
    content_result = outer.get("result") or outer  # handle both shapes

    article_text = (content_result.get("text") or "").strip()
    article_title = content_result.get("title")

    if not article_text:
        logger.warning(f"[article_summarizer] Could not extract text from {url}")
        return {
            "ok": False,
            "error": "Could not extract text content from the URL.",
            "details": "The page might be empty or require JavaScript.",
        }

    logger.info(f"[article_summarizer] Summarizing content for {url}")
    summary_result = run_llm_agent(
        agent_slug="article_summarizer",
        user_text=article_text,
    )

    if not summary_result.ok:
        logger.error(
            f"[article_summarizer] Failed to generate summary for {url}: {summary_result.error}"
        )
        return {
            "ok": False,
            "error": "Failed to generate summary.",
            "details": summary_result.error,
        }

    return {
        "ok": True,
        "summary": summary_result.response_text,
        "source_url": url,
        "title": article_title,
        "content_char_count": len(article_text),
    }
