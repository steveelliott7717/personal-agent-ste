# backend/agents/orchestrator.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from backend.registry.Archived_capability_registry_ import CapabilityRegistry
from backend.services.supabase_service import supabase


class Orchestrator:
    """
    Thin meta-agent wrapper.
    For now: call a single verb via CapabilityRegistry with IDs passed through.
    Later: add policy checks, multi-step plans, idempotency cache, trace graphs.
    """

    def __init__(self, registry: Optional[CapabilityRegistry] = None) -> None:
        self.registry = registry or CapabilityRegistry()
        # Simple in-memory idempotency (optional; extend later)
        self._idem_cache: Dict[str, Dict[str, Any]] = {}

        # --- simple policy knobs (env-driven) ---

    def _allowed_write_tables(self) -> set[str]:
        """
        Comma-separated env var, e.g.:
          ALLOW_DB_WRITE=events,agent_decisions,training_log
        Default empty (no writes) unless explicitly allowed.
        """
        raw = os.getenv("ALLOW_DB_WRITE", "")
        return {t.strip() for t in raw.split(",") if t.strip()}

    def _allowed_notify_channels(self) -> set[str]:
        """
        Comma-separated env var, e.g.:
          ALLOW_NOTIFY_CHANNELS=slack
        Defaults to {'slack'}.
        """
        raw = os.getenv("ALLOW_NOTIFY_CHANNELS", "slack")
        return {t.strip().lower() for t in raw.split(",") if t.strip()}

    def _log_event(
        self,
        topic: str,
        payload: dict[str, Any],
        *,
        correlation_id: str,
        idempotency_key: str,
        latency_ms: Optional[int] = None,
    ) -> None:
        """Best-effort event log; never fail the main call."""
        try:
            row = {
                "topic": topic,
                "payload": payload,
                "source_agent": "orchestrator.meta",
                "correlation_id": correlation_id,
                "idempotency_key": idempotency_key,
            }
            if latency_ms is not None:
                row["latency_ms"] = latency_ms
            supabase.table("events").insert([row]).execute()
        except Exception:
            pass

    def call_verb(
        self,
        verb: str,
        args: Dict[str, Any],
        meta: Dict[str, Any],
        *,
        correlation_id: str,
        idempotency_key: str,
        actor: Optional[Dict[str, Any]] = None,
        policy_ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Idempotency: return cached result for identical key
        # Idempotency: return cached result for identical key
        if idempotency_key in self._idem_cache:
            cached = self._idem_cache[idempotency_key]
            cached.setdefault("correlation_id", correlation_id)
            cached.setdefault("idempotency_key", idempotency_key)
            # audit cache hit
            self._log_event(
                "verb.cache.hit",
                {"verb": verb, "args": args},
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
            )
            return cached

        # ----- policy checks -----
        # Example: allow db.read; restrict db.write to an allowlist of tables.
        if verb == "db.write":
            table = (args or {}).get("table")
            allowed = self._allowed_write_tables()
            if not table or table not in allowed:
                # audit reject
                self._log_event(
                    "verb.policy.block",
                    {
                        "verb": verb,
                        "args": args,
                        "reason": f"write to table '{table}' not allowed",
                        "allowed": sorted(allowed),
                    },
                    correlation_id=correlation_id,
                    idempotency_key=idempotency_key,
                )
                return {
                    "ok": False,
                    "result": {
                        "error": "PolicyDenied",
                        "message": f"writes to '{table}' are not allowed",
                    },
                    "latency_ms": 0,
                    "correlation_id": correlation_id,
                    "idempotency_key": idempotency_key,
                }

        # Restrict notifications to allowed channels
        if verb == "notify.push":
            allowed = self._allowed_notify_channels()
            channel = ((args or {}).get("channel") or "slack").lower()
            if channel not in allowed:
                self._log_event(
                    "verb.policy.block",
                    {
                        "verb": verb,
                        "args": {"channel": channel},
                        "reason": "channel not allowed",
                        "allowed": sorted(allowed),
                    },
                    correlation_id=correlation_id,
                    idempotency_key=idempotency_key,
                )
                return {
                    "ok": False,
                    "result": {
                        "error": "PolicyDenied",
                        "message": f"notify channel '{channel}' is not allowed",
                    },
                    "latency_ms": 0,
                    "correlation_id": correlation_id,
                    "idempotency_key": idempotency_key,
                }

        # Merge meta and attach ids/actor/policy
        _meta = dict(meta or {})
        if actor:
            _meta["actor"] = actor
        if policy_ctx:
            _meta["policy_ctx"] = policy_ctx
        _meta["correlation_id"] = correlation_id

        # audit call start
        self._log_event(
            "verb.call",
            {"verb": verb, "args": args},
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )

        out = self.registry.dispatch(verb, args or {}, _meta)
        out["correlation_id"] = correlation_id
        out["idempotency_key"] = idempotency_key

        if verb == "notify.push":
            # lightweight audit row for notifications
            status = None
            try:
                status = int(out.get("result", {}).get("status"))
            except Exception:
                pass
            self._log_event(
                "notify.sent",
                {
                    "channel": (args or {}).get("channel", "slack"),
                    "status": status,
                },
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
            )

        # audit result (lightweight payload)
        self._log_event(
            "verb.result",
            {
                "verb": verb,
                "ok": bool(out.get("ok")),
                "latency_ms": int(out.get("latency_ms", 0)),
            },
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
            latency_ms=int(out.get("latency_ms", 0)),
        )

        # Cache only successful calls
        if out.get("ok"):
            self._idem_cache[idempotency_key] = dict(out)
        return out
