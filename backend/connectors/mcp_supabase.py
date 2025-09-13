# backend/connectors/mcp_supabase.py
# Minimal MCP server that exposes ONLY the required tools: "search" and "fetch".
# Endpoints:
#   - POST /connectors/supabase/sse  : JSON-RPC 2.0 ("initialize", "tools/list", "tools/call")
#   - GET  /connectors/supabase/sse  : SSE keepalive (for clients that open an event stream)
#   - HEAD/OPTIONS on /sse           : liveness/CORS
#   - GET  /connectors/supabase/health

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional, Union

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ValidationError

logger = logging.getLogger("mcp.min")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )

router = APIRouter(prefix="/connectors/supabase", tags=["mcp"])

# ---------- JSON-RPC types ----------


class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    id: Optional[Union[int, str]] = None
    params: Optional[Dict[str, Any]] = None


def rpc_ok(id_: Optional[Union[int, str]], result: Any) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": id_, "result": result})


def rpc_err(
    id_: Optional[Union[int, str]], code: int, message: str, data: Any = None
) -> JSONResponse:
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    # Keep HTTP 200 so the client parses JSON-RPC errors
    return JSONResponse({"jsonrpc": "2.0", "id": id_, "error": err}, status_code=200)


# ---------- Tools (search + fetch only) ----------


def tools_payload() -> Dict[str, Any]:
    return {
        "tools": [
            {
                "name": "search",
                "description": 'Keyword search placeholder. Args: {"query": string}',
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "fetch",
                "description": 'Fetch by id placeholder. Args: {"id": string}',
                "input_schema": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                    "additionalProperties": False,
                },
            },
        ]
    }


async def tool_search(args: Dict[str, Any]) -> Dict[str, Any]:
    query = str(args.get("query", "")).strip()
    # Return an empty-but-valid result set; validator just needs shape.
    payload = {"query": query, "results": []}
    return {"content": [{"type": "text", "text": json.dumps(payload)}]}


async def tool_fetch(args: Dict[str, Any]) -> Dict[str, Any]:
    the_id = str(args.get("id", "")).strip()
    payload = {"id": the_id, "content": None}
    return {"content": [{"type": "text", "text": json.dumps(payload)}]}


# ---------- Health ----------


@router.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "server": "mcp-supabase-minimal"}


# ---------- SSE (keepalive only) ----------


@router.head("/sse")
async def sse_head() -> Response:
    return Response(
        status_code=200,
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, HEAD, OPTIONS",
            "Access-Control-Allow-Headers": "Authorization, Content-Type, Accept",
        },
    )


@router.options("/sse")
async def sse_options() -> Response:
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, HEAD, OPTIONS",
            "Access-Control-Allow-Headers": "Authorization, Content-Type, Accept",
            "Access-Control-Max-Age": "86400",
        },
    )


@router.get("/sse")
async def sse_get(request: Request) -> StreamingResponse:
    async def event_stream():
        # Emit a one-time tools snapshot (some clients look at SSE too)
        yield "event: tools\n"
        yield "data: " + json.dumps(tools_payload(), separators=(",", ":")) + "\n\n"
        # Heartbeats
        started = time.time()
        while not await request.is_disconnected():
            await asyncio.sleep(15)
            yield "event: ping\n"
            yield "data: " + json.dumps(
                {"uptime_s": int(time.time() - started)}
            ) + "\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ---------- JSON-RPC on POST /sse ----------


@router.post("/sse")
async def mcp_sse_post(request: Request) -> JSONResponse:
    # Log headers & body for quick triage (redact auth if present)
    try:
        hdr = {k.lower(): v for k, v in request.headers.items()}
        hdr.pop("authorization", None)
        logger.info("mcp.headers: %s", json.dumps(hdr))
        body_text = (await request.body()).decode("utf-8", errors="replace")
        logger.info("mcp.body: %s", body_text)
        jr = JsonRpcRequest.parse_raw(body_text)
    except ValidationError as ve:
        return rpc_err(None, -32700, "Parse error", ve.errors())
    except Exception as e:
        return rpc_err(None, -32700, f"Parse error: {e!r}")

    method = jr.method
    params = jr.params or {}

    if method == "initialize":
        proto = params.get("protocolVersion") or "2025-06-18"
        return rpc_ok(
            jr.id,
            {
                "protocolVersion": str(proto),
                "capabilities": {},
                "serverInfo": {"name": "mcp-supabase-minimal", "version": "0.1.0"},
            },
        )

    if method == "tools/list":
        return rpc_ok(jr.id, tools_payload())

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "search":
            return rpc_ok(jr.id, await tool_search(args))
        if name == "fetch":
            return rpc_ok(jr.id, await tool_fetch(args))
        return rpc_err(jr.id, -32601, f"Unknown tool '{name}'")

    if method == "ping":
        return rpc_ok(jr.id, {"ok": True})

    return rpc_err(jr.id, -32601, f"Unknown method '{method}'")
