from __future__ import annotations
from typing import Any, Dict
import json, os, time, urllib
from urllib import request, error


def notify_push_adapter(args: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tiny notifier (MVP).
    Supports:
      - provider: "slack.webhook"
      - channel:  "slack"            (back-compat)
    Args (normalized):
      provider     : str  ("slack.webhook")
      message      : str  (or 'text' back-compat)
      level        : str  (optional: info|warn|error)
      meta         : dict (optional key/values)
      url          : str  (or 'webhook_url' back-compat) – incoming webhook URL
      timeout_ms   : int  (optional, default 5000)

    Returns:
      { message_id, provider, status, echoed:{...} }
    """
    # -------- normalize inputs --------
    provider = (args.get("provider") or "").strip().lower()
    channel = (args.get("channel") or "").strip().lower()  # back-compat
    if not provider:
        # allow older shape: channel="slack"
        if channel == "slack":
            provider = "slack.webhook"
        else:
            provider = "slack.webhook"  # sensible default for MVP

    message = args.get("message")
    if message is None:
        message = args.get("text")  # back-compat
    if not isinstance(message, str) or not message.strip():
        raise ValueError("notify.push: 'message' (or legacy 'text') is required")

    url = args.get("url") or args.get("webhook_url") or os.getenv("SLACK_WEBHOOK_URL")
    if provider == "slack.webhook" and not url:
        raise ValueError(
            "notify.push: missing webhook URL (args.url / args.webhook_url / SLACK_WEBHOOK_URL)"
        )

    level = (args.get("level") or "info").lower()
    extra = args.get("meta") or {}
    timeout_ms = int(args.get("timeout_ms") or 5000)

    # -------- dispatch by provider --------
    if provider == "slack.webhook":
        payload = {
            "text": (
                message
                if not extra
                else f"{message}\n```meta={json.dumps(extra, ensure_ascii=False)}```"
            )
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_ms / 1000.0) as resp:
                status = int(resp.getcode() or 0)
                body = resp.read().decode("utf-8", errors="ignore")[:500]
        except urllib.error.HTTPError as e:
            status = int(e.code or 0)
            body = e.read().decode("utf-8", errors="ignore")[:500]
        except Exception as e:
            # bubble to registry to wrap as AdapterError
            raise RuntimeError(
                f"notify.push(slack.webhook) failed: {type(e).__name__}: {e}"
            )

        # Slack classic webhooks don’t return a message id → synthesize one
        synthetic_id = f"slack-{int(time.time()*1000)}"
        return {
            "message_id": synthetic_id,
            "provider": provider,
            "status": status,
            "echoed": {"level": level, "meta": extra},
            "body": body,
        }

    raise ValueError(
        f"notify.push: unsupported provider '{provider}' (try 'slack.webhook')"
    )
