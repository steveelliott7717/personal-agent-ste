# backend/agents/orchestrator.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Set

from backend.registry.capability_registry import CapabilityRegistry
from backend.services.supabase_service import supabase


class Orchestrator:
    """
    Thin meta-agent wrapper around the capability registry.

    Responsibilities
    ---------------
    - Dispatch a single verb to the capability registry.
    - Provide lightweight idempotency (per idempotency_key, success-only).
    - Enforce minimal, env-driven policy checks before dispatch (db.write table gate,
      notify channel allowlist).
    - Emit best-effort audit events to the `events` table (never fail the main path).

    Notes
    -----
    - Error and policy semantics are preserved exactly as before:
      * db.write table denies return:
        {
          "ok": False,
          "result": {"error":"PolicyDenied","message":"writes to '<table>' are not allowed"},
          ...
        }
      * notify channel denies return:
        {
          "ok": False,
          "result": {"error":"PolicyDenied","message":"notify channel '<channel>' is not allowed"},
          ...
        }
    """

    def __init__(self, registry: Optional[CapabilityRegistry] = None) -> None:
        self.registry = registry or CapabilityRegistry()
        # In-memory idempotency cache: idempotency_key -> full envelope
        self._idem_cache: Dict[str, Dict[str, Any]] = {}
        # Simple cache metrics
        self._idem_cache_hits: int = 0
        self._idem_cache_misses: int = 0

    # -------------------------
    # Env / policy helpers
    # -------------------------
    @staticmethod
    def _norm_table_name(s: Optional[str]) -> str:
        """Normalize a table identifier: trim, lowercase, drop 'public.' prefix."""
        s = (s or "").strip().lower()
        return s[7:] if s.startswith("public.") else s

    @staticmethod
    def _env_csv(name: str) -> Set[str]:
        """Parse a comma-separated env var into a set of trimmed, non-empty values."""
        raw = os.getenv(name, "") or ""
        return {x.strip() for x in raw.split(",") if x.strip()}

    def _allowed_write_tables(self) -> Set[str]:
        """
        Tables allowed for db.write.

        Source of truth: DBWRITE_TABLE_ALLOWLIST (normalized).
        Back-compat:    ALLOW_DB_WRITE (legacy), if the primary is unset/empty.

        Semantics: if allowlist is empty → allow all tables (adapter-compatible).
        """
        allow = {
            self._norm_table_name(t) for t in self._env_csv("DBWRITE_TABLE_ALLOWLIST")
        }
        if not allow:
            legacy = {self._norm_table_name(t) for t in self._env_csv("ALLOW_DB_WRITE")}
            allow = legacy
        return allow

    @staticmethod
    def _allowed_notify_channels() -> Set[str]:
        """
        Allowed notify channels (csv). Defaults to {'slack'} if unset.
        Example: ALLOW_NOTIFY_CHANNELS=slack,discord
        """
        raw = os.getenv("ALLOW_NOTIFY_CHANNELS", "slack")
        return {t.strip().lower() for t in raw.split(",") if t.strip()}

    # -------------------------
    # Audit logging
    # -------------------------
    def _log_event(
        self,
        topic: str,
        payload: Dict[str, Any],
        *,
        correlation_id: str,
        idempotency_key: str,
        latency_ms: Optional[int] = None,
    ) -> None:
        """Best-effort audit row to Supabase `events` (never raises)."""
        try:
            row: Dict[str, Any] = {
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
            # Do not let logging interfere with verb execution.
            pass

    # -------------------------
    # Public surface
    # -------------------------
    def list_verbs(self) -> list[str]:
        """
        Optional discovery hook used by /api/agents/verbs.
        If CapabilityRegistry exposes introspection, return it; else empty list.
        """
        try:
            if hasattr(self.registry, "list_verbs"):
                # type: ignore[attr-defined]
                verbs = self.registry.list_verbs()  # expected: Iterable[str]
                return sorted(set(verbs))
        except Exception:
            pass
        return []

    def get_cache_stats(self) -> Dict[str, Any]:
        """Snapshot of the idempotency cache: size, hits/misses, and a small key sample."""
        keys = list(self._idem_cache.keys())
        return {
            "size": len(self._idem_cache),
            "hits": self._idem_cache_hits,
            "misses": self._idem_cache_misses,
            "keys_sample": keys[:20],
        }

    def clear_cache(self) -> Dict[str, Any]:
        """Clear idempotency cache and reset counters. Returns stats after reset."""
        self._idem_cache.clear()
        self._idem_cache_hits = 0
        self._idem_cache_misses = 0
        return self.get_cache_stats()

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
        """
        Dispatch a single verb with policy guards, idempotency, and audit logs.

        Returns an envelope like:
        {
          "ok": bool,
          "result": {...} | null,
          "error": {...} | null,
          "latency_ms": int,
          "correlation_id": str,
          "idempotency_key": str
        }
        """
        # ---------- Idempotency (success-only cache) ----------
        cached = self._idem_cache.get(idempotency_key)
        if cached is not None:
            # Annotate and audit the cache hit
            cached.setdefault("correlation_id", correlation_id)
            cached.setdefault("idempotency_key", idempotency_key)
            self._idem_cache_hits += 1
            self._log_event(
                "verb.cache.hit",
                {"verb": verb, "args": args},
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
            )
            return cached
        else:
            self._idem_cache_misses += 1

        # ---------- Policy checks (pre-dispatch) ----------
        if verb == "db.write":
            table = (args or {}).get("table")
            tnorm = self._norm_table_name(table)

            # Honor adapter’s kill switch: DBWRITE_DISABLE_TABLE_GUARD=1
            if os.getenv("DBWRITE_DISABLE_TABLE_GUARD", "0") != "1":
                allowed = self._allowed_write_tables()
                # Adapter semantics: enforce only when allowlist is non-empty.
                if allowed and (not tnorm or tnorm not in allowed):
                    # Audit reject and return policy error envelope.
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
            # else: bypass table gate here; adapter may still enforce *column* allowlists.

        if verb == "notify.push":
            allowed_channels = self._allowed_notify_channels()
            channel = ((args or {}).get("channel") or "slack").lower()
            if channel not in allowed_channels:
                self._log_event(
                    "verb.policy.block",
                    {
                        "verb": verb,
                        "args": {"channel": channel},
                        "reason": "channel not allowed",
                        "allowed": sorted(allowed_channels),
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

        # ---------- Merge meta & audit call ----------
        _meta = dict(meta or {})
        if actor:
            _meta["actor"] = actor
        if policy_ctx:
            _meta["policy_ctx"] = policy_ctx
        _meta["correlation_id"] = correlation_id

        self._log_event(
            "verb.call",
            {"verb": verb, "args": args},
            correlation_id=correlation_id,
            idempotency_key=idempotency_key,
        )

        # ---------- Dispatch ----------
        out = self.registry.dispatch(verb, args or {}, _meta)

        # Attach IDs for uniform envelopes
        out["correlation_id"] = correlation_id
        out["idempotency_key"] = idempotency_key

        # ---------- Post-dispatch audits ----------
        if verb == "notify.push":
            # Lightweight audit row for notifications
            status = None
            try:
                status = int(out.get("result", {}).get("status"))
            except Exception:
                # tolerate missing/invalid status
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

        # ---------- Idempotency (cache only successes) ----------
        if out.get("ok"):
            # store a shallow copy to avoid accidental later mutation
            self._idem_cache[idempotency_key] = dict(out)

        return out
