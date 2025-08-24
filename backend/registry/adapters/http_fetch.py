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
import time
from backend.registry.http.client import compute_retry_wait, build_request_body
import os
import base64
import urllib.request
from backend.services.supabase_service import supabase
import json
import hashlib
import re
from pathlib import Path
import hashlib
import os

# Root for on-disk saves (override on Fly with: HTTP_FETCH_SAVE_ROOT=/tmp)
SAVE_ROOT = os.getenv("HTTP_FETCH_SAVE_ROOT", "/tmp")


def _save_stream_to_path(stream_iter, dest_path: str) -> Dict[str, Any]:
    """
    Streams bytes to a local file safely under SAVE_ROOT.
    - If dest_path is relative, it is resolved under SAVE_ROOT.
    - If absolute, it must resolve within SAVE_ROOT (prevents path traversal).
    Returns { path, bytes, sha256 }.
    """
    root = Path(SAVE_ROOT).resolve()
    p = Path(dest_path)
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    # ensure p is inside root
    if not (p == root or root in p.parents):
        raise ValueError(f"save_to path must be within {root}")

    p.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    h = hashlib.sha256()
    with open(p, "wb") as f:
        for chunk in stream_iter:
            if not chunk:
                continue
            f.write(chunk)
            total += len(chunk)
            h.update(chunk)

    return {"path": str(p), "bytes": total, "sha256": h.hexdigest()}


def _upload_to_github(dest: Dict[str, Any], content_bytes: bytes) -> Dict[str, Any]:
    """
    dest: {
      "type":"github",
      "owner": "...",
      "repo": "...",
      "path": "dir/file.bin",
      "branch": "main" (opt),
      "token": "ghp_..." (opt) OR "token_env": "GITHUB_TOKEN"
    }
    """
    owner = dest["owner"]
    repo = dest["repo"]
    path = dest["path"].lstrip("/")
    branch = dest.get("branch", "main")
    token = dest.get("token")
    if not token and dest.get("token_env"):
        token = os.getenv(dest["token_env"])
    if not token:
        token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError(
            "GitHub token not provided (set args.destination.token or token_env or GITHUB_TOKEN)"
        )

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    payload = {
        "message": f"http.fetch upload {path}",
        "content": base64.b64encode(content_bytes).decode("ascii"),
        "branch": branch,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=data,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "personal-agent/1.0 (+registry-http.fetch)",
        },
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            status = resp.getcode()
            body = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        status = e.code
        body = e.read().decode("utf-8", "replace")
    return {
        "status": status,
        "body": body,
        "path": path,
        "branch": branch,
        "repo": f"{owner}/{repo}",
    }


def _upload_to_supabase_storage(
    dest: Dict[str, Any], content_bytes: bytes
) -> Dict[str, Any]:
    """
    dest: {
      "type":"supabase_storage",
      "bucket":"agent-artifacts",
      "path":"downloads/test-fly.bin",
      "upsert": bool (opt, default True),
      "content_type": "application/octet-stream" (opt),
      "cache_control": "3600" (opt)
    }
    """
    bucket = dest["bucket"]
    path = dest["path"].lstrip("/")
    upsert = bool(dest.get("upsert", True))
    ctype = dest.get("content_type") or "application/octet-stream"
    cache = dest.get("cache_control", None)

    # Supabase client expects camelCase keys and header-safe string values.
    options: Dict[str, Any] = {
        "upsert": "true" if upsert else "false",
        "contentType": str(ctype),
    }
    if cache is not None:
        options["cacheControl"] = str(cache)

    # Upload raw bytes (not a BytesIO)
    res = supabase.storage.from_(bucket).upload(path, content_bytes, options)
    return {"bucket": bucket, "path": path, "result": getattr(res, "data", res)}


def _insert_into_supabase_table(
    dest: Dict[str, Any], content_bytes: bytes
) -> Dict[str, Any]:
    """
    dest: { "type":"supabase_table", "table":"files", "columns": {"name":"...", "mime":"...", "content":"bytea|base64", ... } }
    Default stores base64 into column named by 'content' (or 'content_b64' if omitted).
    """
    table = dest["table"]
    cols = dict(dest.get("columns") or {})
    content_col = cols.pop("content", "content_b64")
    row = cols.copy()
    row[content_col] = base64.b64encode(content_bytes).decode("ascii")
    res = supabase.table(table).insert([row]).execute()
    return {"table": table, "row": getattr(res, "data", res)}


def _storage_public_url(bucket: str, path: str) -> str | None:
    """
    Returns a public URL for (bucket, path) if your bucket policy allows public access.
    Otherwise returns None.
    """
    try:
        res = supabase.storage.from_(bucket).get_public_url(path)
        # supabase-py may return dict or an object with ".data"
        data = getattr(res, "data", res) or {}
        # accept "publicUrl" or "public_url"
        return data.get("publicUrl") or data.get("public_url")
    except Exception:
        return None


def _render_template(value: Any, ctx: Dict[str, Any]) -> Any:
    """
    Minimal templating: replaces {{key}} in strings using ctx[key] if present.
    Works recursively for dicts/lists. Non-strings return as-is.
    """
    if isinstance(value, dict):
        return {k: _render_template(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [_render_template(v, ctx) for v in value]
    if isinstance(value, str):
        # replace occurrences of {{token}}
        def repl(m: re.Match[str]) -> str:
            key = m.group(1).strip()
            v = ctx.get(key)
            return str(v) if v is not None else m.group(0)

        return re.sub(r"\{\{\s*([^\}]+)\s*\}\}", repl, value)
    return value


def _run_destination_chain(
    chain: list[Dict[str, Any]],
    content: bytes,
    base_ctx: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Executes a list of destinations in order. Supports:
      - {"type":"supabase_storage", ...}
      - {"type":"supabase_table", ...}
    Context tokens available for templating after a storage step:
      {{bucket}}, {{path}}, {{public_url}}, {{bytes_len}}, {{sha256}}, {{content_type}}
    """
    ctx = dict(base_ctx)
    results: list[Dict[str, Any]] = []

    for step in chain:
        stype = (step.get("type") or "").lower()
        if stype == "supabase_storage":
            up = _upload_to_supabase_storage(step, content)
            # enrich context with storage info
            bucket = up.get("bucket")
            path = up.get("path")
            public_url = _storage_public_url(bucket, path) if bucket and path else None
            ctx.update(
                {
                    "bucket": bucket,
                    "path": path,
                    "public_url": public_url,
                }
            )
            results.append({"backend": "supabase_storage", **up})

        elif stype == "supabase_table":
            # allow templated columns using current ctx
            columns = _render_template(step.get("columns") or {}, ctx)
            ins = _insert_into_supabase_table(
                {"type": "supabase_table", "table": step["table"], "columns": columns},
                content,
            )
            results.append({"backend": "supabase_table", **ins})

        else:
            raise ValueError(f"Unsupported destination_chain type: {stype}")

    return {"saved_chain": results}


def http_fetch_adapter(args: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    # Build body (files/json/form/body) and merge headers before constructing the request
    payload, hdrs_add = build_request_body(args)

    headers = dict(args.get("headers") or {})
    # Respect caller's explicit Content-Type; only set if absent
    if (
        "Content-Type" not in {k.title(): v for k, v in headers.items()}
        and "Content-Type" in hdrs_add
    ):
        headers["Content-Type"] = hdrs_add["Content-Type"]
    # Ask servers to compress unless caller already specified
    if "Accept-Encoding" not in {k.title(): v for k, v in headers.items()}:
        headers["Accept-Encoding"] = "gzip, deflate"

    args["headers"] = headers
    if payload is not None:
        args["body"] = payload

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

    # Execute (destination-aware) with Retry-After/backoff
    retry_cfg = args.get("retry") or {}
    max_retries = int(retry_cfg.get("max", 0))
    retry_on = set(retry_cfg.get("on") or [429, 500, 502, 503, 504])
    backoff_ms = int(retry_cfg.get("backoff_ms", 100))
    jitter = bool(retry_cfg.get("jitter", True))

    # --- save_to (stream to local file) ---
    save_to = args.get("save_to")
    if isinstance(save_to, str) and save_to.strip():
        attempt = 0
        while True:
            req = http_request.build(args)
            meta, stream = req.send_stream(chunk_size=64 * 1024)
            info = _save_stream_to_path(stream, save_to)
            if meta.status in retry_on and attempt < max_retries:
                attempt += 1
                try:
                    os.remove(info["path"])
                except Exception:
                    pass
                wait_s = compute_retry_wait(
                    meta.headers,
                    attempt=attempt,
                    backoff_ms=backoff_ms,
                    jitter=jitter,
                    cap_seconds=60.0,
                )
                time.sleep(wait_s)
                continue
            return {
                "status": meta.status,
                "url": req.url,
                "final_url": meta.final_url,
                "headers": meta.headers,
                "bytes_len": info["bytes"],
                "elapsed_ms": meta.elapsed_ms,
                "json": None,
                "text": None,
                "truncated": False,
                "not_modified": (meta.status == 304),
                "saved": {"backend": "file", **info},
            }

    destination_chain = args.get("destination_chain")
    destination = args.get("destination")

    if destination_chain:
        # Single stream for entire chain
        attempt = 0
        collected = bytearray()
        while True:
            req = http_request.build(args)
            meta, stream = req.send_stream(chunk_size=64 * 1024)
            for chunk in stream:
                collected.extend(chunk)

            if meta.status in retry_on and attempt < max_retries:
                attempt += 1
                collected = bytearray()
                wait_s = compute_retry_wait(
                    meta.headers,
                    attempt=attempt,
                    backoff_ms=backoff_ms,
                    jitter=jitter,
                    cap_seconds=60.0,
                )
                time.sleep(wait_s)
                continue

            content = bytes(collected)
            # context for templating
            sha256 = hashlib.sha256(content).hexdigest()
            content_type = (
                (meta.headers.get("Content-Type") or "").split(";")[0].strip()
            )
            base_ctx = {
                "bytes_len": len(content),
                "sha256": sha256,
                "content_type": content_type,
            }
            chain_out = _run_destination_chain(destination_chain, content, base_ctx)

            return {
                "status": meta.status,
                "url": req.url,
                "final_url": meta.final_url,
                "headers": meta.headers,
                "bytes_len": len(content),
                "elapsed_ms": meta.elapsed_ms,
                "json": None,
                "text": None,
                "truncated": False,
                "not_modified": (meta.status == 304),
                **chain_out,
            }

    elif destination:
        # Stream → single destination (existing behavior)
        attempt = 0
        collected = bytearray()
        while True:
            req = http_request.build(args)
            meta, stream = req.send_stream(chunk_size=64 * 1024)
            for chunk in stream:
                collected.extend(chunk)

            if meta.status in retry_on and attempt < max_retries:
                attempt += 1
                collected = bytearray()
                wait_s = compute_retry_wait(
                    meta.headers,
                    attempt=attempt,
                    backoff_ms=backoff_ms,
                    jitter=jitter,
                    cap_seconds=60.0,
                )
                time.sleep(wait_s)
                continue

            content = bytes(collected)
            dtype = (destination.get("type") or "").lower()
            if dtype == "github":
                up = _upload_to_github(destination, content)
                return {
                    "status": meta.status,
                    "url": req.url,
                    "final_url": meta.final_url,
                    "headers": meta.headers,
                    "bytes_len": len(content),
                    "elapsed_ms": meta.elapsed_ms,
                    "json": None,
                    "text": None,
                    "truncated": False,
                    "not_modified": (meta.status == 304),
                    "saved": {"backend": "github", **up},
                }
            elif dtype == "supabase_storage":
                up = _upload_to_supabase_storage(destination, content)
                return {
                    "status": meta.status,
                    "url": req.url,
                    "final_url": meta.final_url,
                    "headers": meta.headers,
                    "bytes_len": len(content),
                    "elapsed_ms": meta.elapsed_ms,
                    "json": None,
                    "text": None,
                    "truncated": False,
                    "not_modified": (meta.status == 304),
                    "saved": {"backend": "supabase_storage", **up},
                }
            elif dtype == "supabase_table":
                up = _insert_into_supabase_table(destination, content)
                return {
                    "status": meta.status,
                    "url": req.url,
                    "final_url": meta.final_url,
                    "headers": meta.headers,
                    "bytes_len": len(content),
                    "elapsed_ms": meta.elapsed_ms,
                    "json": None,
                    "text": None,
                    "truncated": False,
                    "not_modified": (meta.status == 304),
                    "saved": {"backend": "supabase_table", **up},
                }
            else:
                raise ValueError(
                    "destination.type must be one of: github | supabase_storage | supabase_table"
                )
    else:
        # Normal in-memory path (existing behavior)
        attempt = 0  # completed retries so far
        while True:
            req = http_request.build(args)
            resp = req.send()

            # stop if not retryable OR we already used up all retries
            if resp.status not in retry_on or attempt >= max_retries:
                break

            # wait according to Retry-After (if present) or backoff policy, then retry
            attempt += 1
            wait_s = compute_retry_wait(
                resp.headers,
                attempt=attempt,
                backoff_ms=backoff_ms,
                jitter=jitter,
                cap_seconds=60.0,
            )
            time.sleep(wait_s)

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
            # retry for next pages too
            _attempt = 0
            while True:
                next_resp = next_req.send()
                if next_resp.status not in retry_on or _attempt >= max_retries:
                    break
                _attempt += 1
                _wait_s = compute_retry_wait(
                    next_resp.headers,
                    attempt=_attempt,
                    backoff_ms=backoff_ms,
                    jitter=jitter,
                    cap_seconds=60.0,
                )
                time.sleep(_wait_s)

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
