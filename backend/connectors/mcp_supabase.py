# backend/connectors/mcp_supabase.py
import os, json, asyncio, time
from typing import Any, Dict
from fastapi import APIRouter, Request, Header, Response
from fastapi.responses import StreamingResponse, JSONResponse

# Import your registry dispatcher (adjust path if needed)
from backend.registry.capability_registry import execute_capability

# ENV
CONNECTOR_TOKEN = os.getenv("SUPABASE_CONNECTOR_TOKEN")  # simple bearer for /sse

router = APIRouter(prefix="/connectors/supabase", tags=["connectors:supabase"])

# Advertised tools (loose schemas so you can pass your existing args verbatim)
TOOLS = [
    {
        "name": "db.read",
        "description": "Read rows using repo verbs",
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "select": {"type": ["string", "array", "null"]},
                "where": {"type": ["object", "null"]},
                "order": {"type": ["array", "null"]},
                "limit": {"type": ["integer", "null"]},
            },
            "required": ["table"],
        },
    },
    {
        "name": "db.write",
        "description": "Write rows using repo verbs",
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string"},
                "mode": {"type": "string", "enum": ["insert", "upsert", "update"]},
                "on_conflict": {"type": ["string", "null"]},
                "rows": {"type": ["array", "object", "null"]},
                "where": {"type": ["object", "null"]},
                "returning": {"type": ["string", "null"]},
            },
            "required": ["table", "mode"],
        },
    },
]


async def _call_repo_verb(verb: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fast path: directly use your CapabilityRegistry dispatcher.
    This preserves logging, idempotency, retries, RLS, etc.
    """
    # If your execute_capability is sync, wrap in threadpool; if it’s async, await it.
    res = await execute_capability(verb, args)  # your repo returns a dict {ok:..., ...}
    # Normalize a little for MCP client ergonomics
    if not isinstance(res, dict):
        return {"ok": False, "error": "unexpected return type from execute_capability"}
    return res


@router.get("/sse")
async def mcp_sse(request: Request, authorization: str | None = Header(None)):
    # Simple bearer guard so the connector is not public
    if CONNECTOR_TOKEN and authorization != f"Bearer {CONNECTOR_TOKEN}":
        return Response(status_code=401)

    async def event_stream():
        # 1) Advertise tools once on connect
        header = {"type": "mcp", "event": "tools", "data": {"tools": TOOLS}}
        yield f"data: {json.dumps(header)}\n\n"

        # 2) Minimal MCP loop:
        #    Your ChatGPT client will POST tool invocations back over a secondary channel
        #    or you can keep this endpoint as "announce only" and use the REST helpers below.
        #    To keep it simple and reliable, we also expose REST tool shims:
        #    POST /connectors/supabase/tool/db.read and /tool/db.write (see below).
        t0 = time.time()
        while not await request.is_disconnected():
            # Keep-alive tick (some MCP clients like periodic SSE)
            tick = {
                "type": "mcp",
                "event": "ping",
                "data": {"uptime_s": int(time.time() - t0)},
            }
            yield f"data: {json.dumps(tick)}\n\n"
            await asyncio.sleep(10)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Convenience REST shims (used by ChatGPT custom connector “Actions” or by you for smoke tests)
@router.post("/tool/db.read")
async def tool_read(body: Dict[str, Any]):
    try:
        res = await _call_repo_verb("db.read", body or {})
        return JSONResponse(res)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@router.post("/tool/db.write")
async def tool_write(body: Dict[str, Any]):
    try:
        res = await _call_repo_verb("db.write", body or {})
        return JSONResponse(res)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
