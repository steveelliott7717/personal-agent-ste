# backend/agents/article_summarizer_agent.py
from __future__ import annotations

import logging
import json
from typing import Any, Dict
import re

from backend.registry.capability_registry import CapabilityRegistry
from backend.llm.llm_runner import run_llm_agent

logger = logging.getLogger(__name__)
registry = CapabilityRegistry()


def handle_article_summary(query: str) -> Dict[str, Any]:
    """
    Handles a request to summarize an article from a URL.
    It can extract a URL from a plain text query or parse a JSON string with a "url" key.
    """
    url = None
    # The router should provide a JSON string with a "url" key.
    try:
        payload = json.loads(query)
        url = payload.get("url")
        # Add a fallback to check for a URL inside a "task" key
        if not url and isinstance(payload.get("task"), str):
            match = re.search(r'https?://[^\s"\'`]+', payload["task"])
            if match:
                url = match.group(0)
    except json.JSONDecodeError:
        # Fallback for plain text queries that are just a URL
        if query.strip().startswith("http"):
            url = query.strip()

    if not url:
        return {
            "ok": False,
            "error": "Failed to fetch article content.",
            "details": {"code": "BadRequest", "message": "url is required"},
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

    # The result from web.smart_get is nested
    content_result = fetch_result.get("result", {})
    article_text = content_result.get("text")
    article_title = content_result.get("title")

    if not article_text:
        logger.warning(f"[article_summarizer] Could not extract text from {url}")
        return {
            "ok": False,
            "error": "Could not extract text content from the URL.",
            "details": "The page might be empty or require JavaScript.",
        }

    # Use the LLM runner to perform the summarization
    logger.info(f"[article_summarizer] Summarizing content for {url}")
    summary_result = run_llm_agent(
        agent_slug="article_summarizer",  # Pass slug to use the correct model from settings
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
