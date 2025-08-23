# backend/registry/adapters/http_fetch.py
from typing import Any, Dict
from backend.registry.http.client import http_request  # the plumbing
from backend.registry.http.auth import apply_auth_headers
from backend.registry.http.caching import apply_cache_headers
from backend.registry.http.pagination import paginate_response
from backend.registry.http.headers import choose_parse_body
from backend.registry.net.safety import is_disallowed_ip_host
from backend.registry.net.ports import host_port_allowed
from backend.registry.util.encode import clamp_bytes


def http_fetch_adapter(args: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    # Build request
    req = http_request.build(args)

    # Allow/deny + port checks
    host_ok, why = host_port_allowed(req.url)
    if not host_ok:
        return {"status": 0, "error": "PortNotAllowed", "message": why}
    if is_disallowed_ip_host(req.host):
        return {
            "status": 0,
            "error": "DeniedIP",
            "message": f"Disallowed IP for host: {req.host}",
        }

    # Apply auth/cache headers (no-op if not provided)
    apply_auth_headers(req, args.get("auth") or {})
    apply_cache_headers(req, args.get("cache") or {})

    # Execute
    resp = req.send()

    # Parse + clamp
    body = clamp_bytes(resp.body, max_bytes=args.get("max_bytes"))
    text, json_data = choose_parse_body(resp.headers, body)

    # First page as dict
    first = {
        "status": resp.status,
        "url": req.url,
        "final_url": resp.final_url,
        "headers": resp.headers,
        "bytes_len": len(body),
        "elapsed_ms": resp.elapsed_ms,
        "json": json_data,
        "text": text,
        "truncated": resp.truncated,
        "not_modified": (resp.status == 304),
    }

    # Optional pagination
    paginated = None
    paginate_cfg = args.get("paginate") or {}
    if paginate_cfg:

        def _fetch_next(next_url: str) -> Dict[str, Any]:
            next_req = http_request.build(
                {
                    "url": next_url,
                    "method": "GET",
                    "headers": req.headers,  # reuse headers/auth
                    "timeout": args.get("timeout"),
                    "allow_redirects": args.get("allow_redirects", True),
                    "max_bytes": args.get("max_bytes", 2_000_000),
                }
            )
            next_resp = next_req.send()
            next_body = clamp_bytes(next_resp.body, max_bytes=args.get("max_bytes"))
            next_text, next_json = choose_parse_body(next_resp.headers, next_body)
            return {
                "status": next_resp.status,
                "url": next_url,
                "final_url": next_resp.final_url,
                "headers": next_resp.headers,
                "bytes_len": len(next_body),
                "elapsed_ms": next_resp.elapsed_ms,
                "json": next_json,
                "text": next_text,
            }

        cfg = dict(paginate_cfg)
        cfg["fetch"] = _fetch_next
        paginated = paginate_response(first, args, cfg)

    out = {
        "status": resp.status,
        "url": req.url,
        "final_url": resp.final_url,
        "headers": resp.headers,
        "bytes_len": len(body),
        "elapsed_ms": resp.elapsed_ms,
        "json": json_data,
        "text": text,
        "truncated": resp.truncated,
        "not_modified": (resp.status == 304),
    }
    if paginated:
        out["pagination"] = paginated
    if args.get("trace"):
        out["trace"] = {k: v for k, v in (args.get("trace") or {}).items() if v}
    return out
