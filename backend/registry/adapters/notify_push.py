from __future__ import annotations
from typing import Any, Dict, Optional
import json
import os
import time
import urllib
from urllib import request, error


def notify_push_adapter(args: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a simple message via channel=slack using an incoming webhook.

    args:
      channel: "slack" (only supported for now)
      text: str (required)
      webhook_url: str (optional; falls back to env SLACK_WEBHOOK_URL)

    returns:
      { channel, status, body (truncated) }
    """
    channel = (args.get("channel") or "slack").lower()
    text = args.get("text")
    if not text or not isinstance(text, str):
        raise ValueError("args.text (str) is required")

    if channel != "slack":
        raise ValueError(f"unsupported channel '{channel}' (only 'slack' supported)")

    webhook_url = args.get("webhook_url") or os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        raise ValueError("SLACK_WEBHOOK_URL not set and args.webhook_url missing")

    payload = {"text": text}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            status = resp.getcode()
            body = resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as e:
        status = e.code
        body = e.read().decode("utf-8", errors="ignore")
    except Exception as e:
        raise RuntimeError(f"notify.push failed: {type(e).__name__}: {e}")

    # Slack returns "ok" on success for classic webhooks; keep body short
    return {"channel": channel, "status": int(status), "body": body[:500]}
