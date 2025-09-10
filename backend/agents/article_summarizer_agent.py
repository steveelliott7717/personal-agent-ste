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
    import json, re, logging

    logger = logging.getLogger(__name__)

    def _extract_url_from_text(text: str) -> str | None:
        if not isinstance(text, str):
            return None
        m = re.search(r'https?://[^\s"\'<>]+', text)
        return m.group(0) if m else None

    url = None
    payload = None

    if isinstance(query, dict):
        payload = query
    else:
        try:
            payload = json.loads(query) if isinstance(query, str) else None
        except Exception:
            payload = None

    if isinstance(payload, dict):
        if isinstance(payload.get("url"), str):
            url = payload["url"].strip()
        if not url and isinstance(payload.get("task"), str):
            url = _extract_url_from_text(payload["task"])

    if not url:
        url = _extract_url_from_text(query if isinstance(query, str) else "")

    if not url:
        return {
            "ok": False,
            "error": "No URL found in the request.",
            "details": "Please provide a URL.",
        }

    logger.info(f"[article_summarizer] Fetching content from {url}")
    fetch_result = registry.dispatch("web.smart_get", {"url": url}, {})

    if not fetch_result.get("ok"):
        return {
            "ok": False,
            "error": "Failed to fetch article content.",
            "details": fetch_result.get("error"),
        }

    content_result = fetch_result.get("result") or {}
    article_text = content_result.get("text") or ""
    article_title = content_result.get("title")

    if not article_text.strip():
        return {
            "ok": False,
            "error": "Could not extract text content from the URL.",
            "details": "The page might be empty or require JavaScript.",
        }

    summary_result = run_llm_agent(
        agent_slug="article_summarizer", user_text=article_text
    )

    if not summary_result.ok:
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
