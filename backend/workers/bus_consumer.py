from __future__ import annotations
from typing import Dict, Callable
from importlib import import_module
from backend.service.supabase_service import supabase

CONSUMER = "nutrition"  # set per process/env

def _get_map() -> Dict[str, str]:
    """topic -> handler dotted path, stored in agent_settings for this consumer name."""
    try:
        res = supabase.table("agent_settings").select("key,value").eq("agent_slug", CONSUMER).execute()
        kv = {r["key"]: r["value"] for r in (res.data or [])}
        return kv.get("bus.consumer_map", {}) or {}
    except Exception:
        return {}

def _handler(dotted: str) -> Callable[[dict], None]:
    mod, fn = dotted.rsplit(":", 1)
    return getattr(import_module(mod), fn)

def _get_offset():
    res = supabase.table("event_offsets").select("*").eq("consumer", CONSUMER).limit(1).execute().data
    if res: return res[0]["last_event_time"]
    supabase.table("event_offsets").insert({"consumer": CONSUMER}).execute()
    return "epoch"

def _set_offset(ts_iso):
    supabase.table("event_offsets").upsert({"consumer": CONSUMER, "last_event_time": ts_iso}).execute()

def run_once(limit=200):
    topic_map = _get_map()  # {"meal.done":"plugins.nutrition.handlers:handle_meal_done", ...}
    if not topic_map:
        return

    last = _get_offset()
    rows = (supabase.table("events")
        .select("*")
        .gt("created_at", last)
        .order("created_at", asc=True)
        .limit(limit)
        .execute().data or [])

    max_ts = None
    for ev in rows:
        dotted = topic_map.get(ev["topic"])
        if dotted:
            try:
                _handler(dotted)(ev)  # pass the full event row {topic,payload,...}
            except Exception:
                # log and continue; don't block the stream
                pass
        ts = ev["created_at"]
        if not max_ts or ts > max_ts:
            max_ts = ts

    if max_ts:
        _set_offset(max_ts)

if __name__ == "__main__":
    run_once()
