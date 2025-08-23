from __future__ import annotations
from typing import Any, Callable, Dict, Literal, TypedDict
from datetime import datetime, timezone
import time

# Import only adapter entrypoints here
from backend.registry.adapters.http_fetch import http_fetch_adapter
from backend.registry.adapters.db_read import db_read_adapter
from backend.registry.adapters.db_write import db_write_adapter
from backend.registry.adapters.notify_push import notify_push_adapter


class Envelope(TypedDict, total=False):
    ok: bool
    result: Dict[str, Any]
    latency_ms: int
    correlation_id: str


MetaDict = Dict[str, Any]


class CapabilityRegistry:
    def __init__(self) -> None:
        self._adapters: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Any]] = {}
        self._register()

    def register(
        self, verb: str, fn: Callable[[Dict[str, Any], Dict[str, Any]], Any]
    ) -> None:
        self._adapters[verb] = fn

    def has(self, verb: str) -> bool:
        return verb in self._adapters

    def dispatch(
        self, verb: str, args: Dict[str, Any], meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            fn = self._adapters[verb]
            data = fn(args or {}, meta or {})
            ok, result = True, (data if isinstance(data, dict) else {"data": data})
        except KeyError:
            ok, result = False, {"error": f"unknown verb '{verb}'"}
        except Exception as e:
            ok, result = False, {"error": type(e).__name__, "message": str(e)}
        return {
            "ok": ok,
            "result": result,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "correlation_id": (meta or {}).get("correlation_id", ""),
        }

    def _register(self) -> None:
        self.register("health.echo", lambda a, m: {"echo": {"args": a, "meta": m}})
        self.register(
            "time.now", lambda _a, _m: {"iso": datetime.now(timezone.utc).isoformat()}
        )
        self.register("db.read", db_read_adapter)
        self.register("db.write", db_write_adapter)
        self.register("notify.push", notify_push_adapter)
        self.register("http.fetch", http_fetch_adapter)
