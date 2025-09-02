# backend/registry/adapters/http_fetch.py
import os
import re
import json
import time
import base64
import hashlib
import urllib.request
import urllib.error
from typing import Any, Dict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from backend.services.supabase_service import supabase
from backend.registry.http.client import (
    http_request,
    compute_retry_wait,
    build_request_body,
    _decompress_if_needed,
)
from backend.registry.http.auth import apply_auth_headers
from backend.registry.http.caching import apply_cache_headers
from backend.registry.http.pagination import paginate_response
from backend.registry.http.headers import choose_parse_body
from backend.registry.net.safety import is_disallowed_ip_host
from backend.registry.net.ports import host_port_allowed
from backend.registry.util.encode import clamp_bytes
from urllib.parse import urlparse
from typing import Iterable


import threading

# Root for on-disk saves (override on Fly with: HTTP_FETCH_SAVE_ROOT=/tmp)
SAVE_ROOT = os.getenv("HTTP_FETCH_SAVE_ROOT", "/tmp")


def _save_stream_to_path(stream, dest_path: str) -> Dict[str, Any]:
    """
    Streams to a temp file in the target directory, fsyncs, then atomically renames.
    Returns {"path": final_path, "bytes": n}
    """
    import os, tempfile

    dest_dir = os.path.dirname(dest_path) or "."
    os.makedirs(dest_dir, exist_ok=True)

    total = 0
    # create temp file in same directory for atomic rename on all platforms
    fd, tmp_path = tempfile.mkstemp(prefix=".pa_tmp_", dir=dest_dir)
    try:
        with os.fdopen(fd, "wb", buffering=0) as f:
            for chunk in stream:
                if not chunk:
                    continue
                f.write(chunk)
                total += len(chunk)
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass
        # atomic move
        os.replace(tmp_path, dest_path)
        return {"path": dest_path, "bytes": total}
    except Exception:
        # best-effort cleanup
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


def _normalize_host_patterns(val: Any) -> list[str]:
    """
    Accepts list/tuple/set or comma-separated string.
    Returns all-lowercase patterns with whitespace trimmed.
    Supported forms: "example.com", "*.example.com", ".example.com", "*"
    """
    if val is None:
        return []
    if isinstance(val, (list, tuple, set)):
        parts = [str(p).strip() for p in val if str(p).strip()]
    elif isinstance(val, str):
        parts = [p.strip() for p in val.split(",") if p.strip()]
    else:
        return []
    return [p.lower() for p in parts]


def _host_matches(host: str, pattern: str) -> bool:
    """
    Suffix wildcard semantics:
      - "*.example.com" → matches "api.example.com" (and exact "example.com")
      - ".example.com"  → matches subdomains only (not the apex)
      - "example.com"   → exact or any subdomain (safe suffix w/ dot boundary)
      - "*"             → matches anything
    """
    h = (host or "").lower()
    p = (pattern or "").lower()
    if not h or not p:
        return False
    if p == "*":
        return True
    if p.startswith("*."):
        suf = p[2:]
        return h == suf or h.endswith("." + suf)
    if p.startswith("."):
        suf = p[1:]
        return h.endswith("." + suf)  # subdomains only, not apex
    # plain domain: exact or subdomain suffix
    return h == p or h.endswith("." + p)


def _enforce_host_policies(
    host: str, allow: Iterable[str], deny: Iterable[str]
) -> tuple[bool, str | None, str | None]:
    """
    Returns (ok, code, message). 'code' is a short error code string or None.
    Precedence: allow-list first (if present), then deny-list.
    """
    allow = list(allow or [])
    deny = list(deny or [])
    if allow and not any(_host_matches(host, pat) for pat in allow):
        return False, "HostNotAllowed", f"host '{host}' not in allow_hosts"
    if deny and any(_host_matches(host, pat) for pat in deny):
        return False, "HostDenied", f"host '{host}' matches deny_hosts"
    return True, None, None


# --- redirect policy helpers -------------------------------------------------
def _normalize_redirect_scope(val: str | None) -> str:
    """
    Normalize to one of: 'any' (default), 'same_host', 'same_site', 'allow_hosts_only'
    """
    v = (val or "").strip().lower()
    mapping = {
        "": "any",
        "any": "any",
        "all": "any",
        "true": "any",
        "same-host": "same_host",
        "same_host": "same_host",
        "samehost": "same_host",
        "same-site": "same_site",
        "same_site": "same_site",
        "samesite": "same_site",
        "allow-hosts-only": "allow_hosts_only",
        "allow_hosts_only": "allow_hosts_only",
        "allowhosts": "allow_hosts_only",
    }
    return mapping.get(v, "any")


def _site_key(host: str) -> str:
    """
    Best-effort eTLD+1. Handles common multi-part TLDs; extend via env HTTP_FETCH_PSL_EXTRA.
    """
    h = (host or "").lower()
    parts = [p for p in h.split(".") if p]
    if len(parts) <= 2:
        return h
    common_multi = {
        "co.uk",
        "org.uk",
        "ac.uk",
        "gov.uk",
        "co.jp",
        "com.au",
        "com.br",
        "com.cn",
        "com.sg",
        "com.hk",
        "co.in",
        "co.kr",
        "com.mx",
        "com.ar",
    }
    extra = os.getenv("HTTP_FETCH_PSL_EXTRA", "")
    if extra.strip():
        for s in extra.split(","):
            s = s.strip().lower()
            if s:
                common_multi.add(s)
    tail2 = ".".join(parts[-2:])
    if tail2 in common_multi and len(parts) >= 3:
        return ".".join(parts[-3:])
    return tail2


def _redirect_allowed(
    origin_host: str, final_host: str, scope: str, allow_hosts
) -> bool:
    if scope == "any":
        return True
    if scope == "same_host":
        return (final_host or "").lower() == (origin_host or "").lower()
    if scope == "same_site":
        return _site_key(final_host) == _site_key(origin_host)
    if scope == "allow_hosts_only":
        # If allow list empty, be permissive (treat like 'any')
        return (not allow_hosts) or any(
            _host_matches(final_host, p) for p in allow_hosts
        )
    return True


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


def _parse_retry_after(header_val: str | None) -> int | None:
    """
    Returns milliseconds to wait, or None if unparseable.
    Accepts either delta-seconds (e.g., '120') or HTTP-date.
    """
    if not header_val:
        return None
    s = header_val.strip()
    # delta-seconds
    if s.isdigit():
        try:
            return int(s) * 1000
        except Exception:
            return None
    # HTTP date
    try:
        dt = parsedate_to_datetime(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        ms = int((dt - now).total_seconds() * 1000)
        return ms if ms > 0 else 0
    except Exception:
        return None


def _sleep_ms(ms: int):
    if ms and ms > 0:
        time.sleep(ms / 1000.0)


# ---- per-host token bucket (process-local) ----
_TOKEN_BUCKETS: dict[str, dict] = {}
_TOKEN_LOCK = threading.Lock()


def _tb_take(host: str, capacity: int, refill_per_sec: float, max_wait_ms: int) -> None:
    """
    Blocks until a token is available or max_wait_ms exceeded.
    Raises TimeoutError if couldn't take a token in time.
    """
    now = time.monotonic()
    with _TOKEN_LOCK:
        b = _TOKEN_BUCKETS.get(host)
        if not b:
            b = {"tokens": capacity, "last": now}
            _TOKEN_BUCKETS[host] = b

    deadline = now + max(0, max_wait_ms) / 1000.0

    while True:
        with _TOKEN_LOCK:
            # refill
            now = time.monotonic()
            elapsed = max(0.0, now - b["last"])
            if refill_per_sec > 0:
                b["tokens"] = min(
                    capacity, b["tokens"] + elapsed * float(refill_per_sec)
                )
            b["last"] = now

            if b["tokens"] >= 1.0:
                b["tokens"] -= 1.0
                return  # got a token

        # no token; check deadline then sleep briefly
        if time.monotonic() >= deadline:
            raise TimeoutError(f"rate limit exceeded for host '{host}'")
        time.sleep(0.02)


def http_fetch_adapter(args: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Registry verb: http.fetch
    Supports:
      - JSON/text/binary fetch with gzip/deflate decoding
      - Retries honoring Retry-After (bounded) + exponential backoff + jitter
      - Streaming to local file (save_to), or piping into destinations (github/supabase)
      - Pagination modes via paginate_response (cursor/header_link/page)
      - Per-host token-bucket rate limiting
      - Input validation + header sanitization + timeout_ms + max_bytes clamping
    """
    # --------------------
    # Basic argument guard
    # --------------------
    if not isinstance(args, dict):
        return {"status": 0, "error": "BadRequest", "message": "args must be an object"}

    url = (args.get("url") or "").strip()
    if not url:
        return {"status": 0, "error": "BadRequest", "message": "args.url is required"}

    # Method whitelist (default GET)
    method = (args.get("method") or "GET").upper()
    if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
        return {
            "status": 0,
            "error": "BadRequest",
            "message": f"unsupported method: {method}",
        }
    args["method"] = method

    # basic shapes
    if (
        "headers" in args
        and args["headers"] is not None
        and not isinstance(args["headers"], dict)
    ):
        return {
            "status": 0,
            "error": "BadRequest",
            "message": "headers must be an object",
        }
    if (
        "retry" in args
        and args["retry"] is not None
        and not isinstance(args["retry"], dict)
    ):
        return {
            "status": 0,
            "error": "BadRequest",
            "message": "retry must be an object",
        }
    if (
        "rate_limit" in args
        and args["rate_limit"] is not None
        and not isinstance(args["rate_limit"], dict)
    ):
        return {
            "status": 0,
            "error": "BadRequest",
            "message": "rate_limit must be an object",
        }

    # validate rate_limit.per_host numbers (optional)
    rl_cfg = (args.get("rate_limit") or {}).get("per_host") or {}
    try:
        if rl_cfg:
            _ = int(rl_cfg.get("capacity", 5))
            _ = float(rl_cfg.get("refill_per_sec", 2.0))
            _ = int(rl_cfg.get("max_wait_ms", 2000))
    except Exception:
        return {
            "status": 0,
            "error": "BadRequest",
            "message": "rate_limit.per_host fields must be numbers",
        }

        # --- allow/deny host lists (args or env) ---
    env_allow = os.getenv("HTTP_FETCH_ALLOW_HOSTS")
    env_deny = os.getenv("HTTP_FETCH_DENY_HOSTS")
    allow_hosts = _normalize_host_patterns(args.get("allow_hosts", env_allow))
    deny_hosts = _normalize_host_patterns(args.get("deny_hosts", env_deny))

    # --- redirect clamp (default + bounds via env) ---
    try:
        _redir_default = int(os.getenv("HTTP_FETCH_MAX_REDIRECTS_DEFAULT", "5"))
    except Exception:
        _redir_default = 5
    try:
        _redir_cap = int(os.getenv("HTTP_FETCH_MAX_REDIRECTS_MAX", "10"))
    except Exception:
        _redir_cap = 10

    def _clamp_redirects(val) -> int:
        try:
            v = int(val) if val is not None else _redir_default
        except Exception:
            v = _redir_default
        if v < 0:
            v = 0
        if v > _redir_cap:
            v = _redir_cap
        return v

    max_redirects = _clamp_redirects(args.get("max_redirects"))
    args["max_redirects"] = max_redirects
    if "allow_redirects" not in args:
        args["allow_redirects"] = bool(max_redirects > 0)

    # --- initial host safety + policy (before building/sending) ---
    url = (args.get("url") or "").strip()  # already validated earlier
    host0 = (urlparse(url).hostname or "").lower()
    if not host0:
        return {"status": 0, "error": "BadRequest", "message": "invalid url host"}

    # port/scheme allow-list (existing)
    # We'll still run the canonical checks on the request object below.
    # Here we preflight policy checks early for clearer errors.
    okp, whyp = host_port_allowed(url)
    if not okp:
        return {"status": 0, "error": "PortNotAllowed", "message": whyp}
    if is_disallowed_ip_host(host0):
        return {
            "status": 0,
            "error": "DeniedIP",
            "message": f"Disallowed IP for host: {host0}",
        }

    okh, codeh, whyh = _enforce_host_policies(host0, allow_hosts, deny_hosts)
    if not okh:
        return {"status": 0, "error": codeh or "DeniedHost", "message": whyh}

    # destination (single) shape
    if (
        "destination" in args
        and args["destination"] is not None
        and not isinstance(args["destination"], dict)
    ):
        return {
            "status": 0,
            "error": "BadRequest",
            "message": "destination must be an object",
        }
    # destination_chain shape
    if "destination_chain" in args and args["destination_chain"] is not None:
        if not isinstance(args["destination_chain"], list):
            return {
                "status": 0,
                "error": "BadRequest",
                "message": "destination_chain must be a list",
            }
        for i, step in enumerate(args["destination_chain"]):
            if not isinstance(step, dict) or "type" not in step:
                return {
                    "status": 0,
                    "error": "BadRequest",
                    "message": f"destination_chain[{i}] must be an object with 'type'",
                }

    # --- add: timeout_ms support (alias for timeout in seconds) ---
    if "timeout_ms" in args and args.get("timeout_ms") is not None:
        try:
            t_ms = max(50, int(args["timeout_ms"]))  # enforce 50 ms minimum
        except Exception:
            return {
                "status": 0,
                "error": "BadRequest",
                "message": "timeout_ms must be an integer",
            }
        args["timeout"] = max(0.05, float(t_ms) / 1000.0)  # convert ms → seconds
        del args["timeout_ms"]  # normalize

    # max_bytes: normalize and clamp lower bound (avoid 0/negative). Upper bound left to infra limits.
    mb_default = 2_000_000  # 2 MB
    try:
        max_bytes = int(args.get("max_bytes", mb_default))
    except Exception:
        return {
            "status": 0,
            "error": "BadRequest",
            "message": "max_bytes must be an integer",
        }
    if max_bytes < 1024:  # keep at least 1 KiB
        max_bytes = 1024
    args["max_bytes"] = max_bytes

    # ---- build body and baseline headers ----
    payload, hdrs_add = build_request_body(
        args
    )  # may set Content-Type when encoding form/json/files
    headers = dict(args.get("headers") or {})

    # case-insensitive header checks
    def _has(h: Dict[str, Any], key: str) -> bool:
        lk = key.lower()
        return any(k.lower() == lk for k in h.keys())

    # default UA and Accept-Encoding; don't stomp caller's values
    headers.setdefault("User-Agent", "personal-agent/1.0 (+registry-http.fetch)")
    import random

    ua_pool = args.get("user_agent_pool") or []
    if ua_pool:
        headers["User-Agent"] = random.choice(ua_pool)
    al_pool = args.get("accept_language_pool") or []
    if al_pool and not any(k.lower() == "accept-language" for k in headers):
        headers["Accept-Language"] = random.choice(al_pool)

    if not _has(headers, "Accept-Encoding"):
        headers["Accept-Encoding"] = "gzip, deflate"

    # respect caller's explicit Content-Type; only add if absent and we encoded a body
    if not _has(headers, "Content-Type") and "Content-Type" in (hdrs_add or {}):
        headers["Content-Type"] = hdrs_add["Content-Type"]

    # ---- sanitize unsafe hop-by-hop headers (http client will set correct ones) ----
    for bad in (
        "Host",
        "host",
        "Content-Length",
        "content-length",
        "Connection",
        "connection",
        "Transfer-Encoding",
        "transfer-encoding",
        "Upgrade",
        "upgrade",
    ):
        headers.pop(bad, None)

    # apply headers/body back to args
    args["headers"] = headers
    if payload is not None:
        args["body"] = payload

    # --- redirect scope (args or env) ---
    redirect_scope = _normalize_redirect_scope(
        args.get("redirect_scope") or os.getenv("HTTP_FETCH_REDIRECT_SCOPE")
    )

    # --- early host policy preflight on initial URL ---
    url0 = (args.get("url") or "").strip()
    try:
        host0 = (urlparse(url0).hostname or "").lower()
    except Exception:
        host0 = ""
    if not host0:
        return {"status": 0, "error": "BadRequest", "message": "invalid url host"}

    okh, codeh, whyh = _enforce_host_policies(host0, allow_hosts, deny_hosts)
    if not okh:
        return {"status": 0, "error": codeh or "DeniedHost", "message": whyh}

    # ---- build request & safety checks (host/port/IP) ----
    req = http_request.build(args)
    # set per-request timeout on the request object
    tsec = args.get("timeout")
    if isinstance(tsec, (int, float)) and tsec > 0:
        setattr(req, "timeout", float(tsec))

    host_ok, why = host_port_allowed(req.url)
    if not host_ok:
        return {"status": 0, "error": "PortNotAllowed", "message": why}

    if is_disallowed_ip_host(req.host):
        return {
            "status": 0,
            "error": "DeniedIP",
            "message": f"Disallowed IP for host: {req.host}",
        }

    # ---- auth + cache validators (no-op if not provided) ----
    apply_auth_headers(req, args.get("auth") or {})
    apply_cache_headers(req, args.get("cache") or {})

    # ---- retry config ----
    retry_cfg = args.get("retry") or {}
    max_retries = int(
        retry_cfg.get("max", 0)
    )  # number of retry attempts (0 = no retry)
    retry_on = set(int(s) for s in (retry_cfg.get("on") or [429, 500, 502, 503, 504]))
    backoff_ms = int(retry_cfg.get("backoff_ms", 100))
    jitter = bool(retry_cfg.get("jitter", True))

    # ---- rate limit (per-host) ----
    rl_cfg = (args.get("rate_limit") or {}).get("per_host") or {}
    rl_capacity = int(rl_cfg.get("capacity", 5))
    rl_refill = float(rl_cfg.get("refill_per_sec", 2.0))
    rl_maxwait = int(rl_cfg.get("max_wait_ms", 2000))

    # ---- streaming to file (save_to) ----
    save_to = args.get("save_to")
    if isinstance(save_to, str) and save_to.strip():
        # If relative, keep downloads under SAVE_ROOT to be safe in containers
        if not os.path.isabs(save_to):
            save_to = os.path.join(SAVE_ROOT, save_to)
        attempt = 0
        while True:
            req = http_request.build(
                args
            )  # re-build to include any per-attempt header changes
            try:
                _tb_take(req.host, rl_capacity, rl_refill, rl_maxwait)
                # set per-request timeout
                tsec = args.get("timeout")
                if isinstance(tsec, (int, float)) and tsec > 0:
                    setattr(req, "timeout", float(tsec))
                meta_r, stream = req.send_stream(
                    chunk_size=64 * 1024
                )  # no timeout kwarg

            except TimeoutError as e:
                return {
                    "status": 0,
                    "url": req.url,
                    "final_url": req.url,
                    "headers": {},
                    "bytes_len": 0,
                    "elapsed_ms": 0,
                    "json": None,
                    "text": None,
                    "truncated": False,
                    "not_modified": False,
                    "error": "RateLimited",
                    "message": str(e),
                }

            info = _save_stream_to_path(stream, save_to)

            if meta_r.status in retry_on and attempt < max_retries:
                attempt += 1
                try:
                    os.remove(info["path"])
                except Exception:
                    pass
                time.sleep(
                    compute_retry_wait(
                        meta_r.headers,
                        attempt=attempt,
                        backoff_ms=backoff_ms,
                        jitter=jitter,
                        cap_seconds=60.0,
                    )
                )
                continue

            return {
                "status": meta_r.status,
                "url": req.url,
                "final_url": meta_r.final_url,
                "headers": meta_r.headers,
                "bytes_len": info["bytes"],
                "elapsed_ms": meta_r.elapsed_ms,
                "json": None,
                "text": None,
                "truncated": False,
                "not_modified": (meta_r.status == 304),
                "saved": {"backend": "file", **info},
            }

    # ---- destination chain / single destination (stream, then handle) ----
    destination_chain = args.get("destination_chain")
    destination = args.get("destination")

    if destination_chain:
        attempt = 0
        collected = bytearray()
        while True:
            req = http_request.build(
                args
            )  # re-build to include any per-attempt header changes
            # set per-request timeout
            tsec = args.get("timeout")
            if isinstance(tsec, (int, float)) and tsec > 0:
                setattr(req, "timeout", float(tsec))
            try:
                _tb_take(req.host, rl_capacity, rl_refill, rl_maxwait)
                meta_r, stream = req.send_stream(
                    chunk_size=64 * 1024
                )  # no timeout kwarg

            except TimeoutError as e:
                return {
                    "status": 0,
                    "url": req.url,
                    "final_url": req.url,
                    "headers": {},
                    "bytes_len": 0,
                    "elapsed_ms": 0,
                    "json": None,
                    "text": None,
                    "truncated": False,
                    "not_modified": False,
                    "error": "RateLimited",
                    "message": str(e),
                }

            for chunk in stream:
                collected.extend(chunk)

            if meta_r.status in retry_on and attempt < max_retries:
                attempt += 1
                collected = bytearray()
                time.sleep(
                    compute_retry_wait(
                        meta_r.headers,
                        attempt=attempt,
                        backoff_ms=backoff_ms,
                        jitter=jitter,
                        cap_seconds=60.0,
                    )
                )
                continue

            content = bytes(collected)
            sha256 = hashlib.sha256(content).hexdigest()
            content_type = (
                (meta_r.headers.get("Content-Type") or "").split(";")[0].strip()
            )
            base_ctx = {
                "bytes_len": len(content),
                "sha256": sha256,
                "content_type": content_type,
            }
            chain_out = _run_destination_chain(destination_chain, content, base_ctx)

            return {
                "status": meta_r.status,
                "url": req.url,
                "final_url": meta_r.final_url,
                "headers": meta_r.headers,
                "bytes_len": len(content),
                "elapsed_ms": meta_r.elapsed_ms,
                "json": None,
                "text": None,
                "truncated": False,
                "not_modified": (meta_r.status == 304),
                **chain_out,
            }

    if destination:
        attempt = 0
        collected = bytearray()
        while True:
            req = http_request.build(
                args
            )  # re-build to include any per-attempt header changes
            # set per-request timeout
            tsec = args.get("timeout")
            if isinstance(tsec, (int, float)) and tsec > 0:
                setattr(req, "timeout", float(tsec))
            try:
                _tb_take(req.host, rl_capacity, rl_refill, rl_maxwait)
                meta_r, stream = req.send_stream(
                    chunk_size=64 * 1024
                )  # no timeout kwarg

            except TimeoutError as e:
                return {
                    "status": 0,
                    "url": req.url,
                    "final_url": req.url,
                    "headers": {},
                    "bytes_len": 0,
                    "elapsed_ms": 0,
                    "json": None,
                    "text": None,
                    "truncated": False,
                    "not_modified": False,
                    "error": "RateLimited",
                    "message": str(e),
                }

            for chunk in stream:
                collected.extend(chunk)

            if meta_r.status in retry_on and attempt < max_retries:
                attempt += 1
                collected = bytearray()
                time.sleep(
                    compute_retry_wait(
                        meta_r.headers,
                        attempt=attempt,
                        backoff_ms=backoff_ms,
                        jitter=jitter,
                        cap_seconds=60.0,
                    )
                )
                continue

            content = bytes(collected)
            dtype = (destination.get("type") or "").lower()
            if dtype == "github":
                up = _upload_to_github(destination, content)
                backend_tag = {"backend": "github", **up}
            elif dtype == "supabase_storage":
                up = _upload_to_supabase_storage(destination, content)
                backend_tag = {"backend": "supabase_storage", **up}
            elif dtype == "supabase_table":
                up = _insert_into_supabase_table(destination, content)
                backend_tag = {"backend": "supabase_table", **up}
            else:
                raise ValueError(
                    "destination.type must be one of: github | supabase_storage | supabase_table"
                )

            return {
                "status": meta_r.status,
                "url": req.url,
                "final_url": meta_r.final_url,
                "headers": meta_r.headers,
                "bytes_len": len(content),
                "elapsed_ms": meta_r.elapsed_ms,
                "json": None,
                "text": None,
                "truncated": False,
                "not_modified": (meta_r.status == 304),
                "saved": backend_tag,
            }

    # ---- normal in-memory path (with retries + rate limit) ----
    attempt = 0
    last_rate_error: str | None = None
    while True:
        req = http_request.build(
            args
        )  # re-build to include any per-attempt header changes
        # set per-request timeout
        tsec = args.get("timeout")
        if isinstance(tsec, (int, float)) and tsec > 0:
            setattr(req, "timeout", float(tsec))
        try:
            _tb_take(req.host, rl_capacity, rl_refill, rl_maxwait)
            resp = (
                req.send()
            )  # non-streaming call; returns response with .body, .headers, etc.
        except TimeoutError as e:
            last_rate_error = str(e)
            if attempt >= max_retries:
                return {
                    "status": 0,
                    "url": req.url,
                    "final_url": req.url,
                    "headers": {},
                    "bytes_len": 0,
                    "elapsed_ms": 0,
                    "json": None,
                    "text": None,
                    "truncated": False,
                    "not_modified": False,
                    "error": "RateLimited",
                    "message": last_rate_error,
                }
            attempt += 1
            time.sleep(
                compute_retry_wait(
                    {},
                    attempt=attempt,
                    backoff_ms=backoff_ms,
                    jitter=jitter,
                    cap_seconds=5.0,
                )
            )
            continue

        if resp.status not in retry_on or attempt >= max_retries:
            break

        attempt += 1
        time.sleep(
            compute_retry_wait(
                resp.headers,
                attempt=attempt,
                backoff_ms=backoff_ms,
                jitter=jitter,
                cap_seconds=60.0,
            )
        )

    # ---- decompress -> clamp -> parse (FIRST PAGE) ----
    content_encoding = (resp.headers.get("Content-Encoding") or "").lower()
    decoded = _decompress_if_needed(resp.body, content_encoding)
    body = clamp_bytes(decoded, max_bytes=args.get("max_bytes"))
    text, json_data = choose_parse_body(resp.headers, body)

    # --- redirect policy (first response) ---
    origin_host = (req.host or "").lower()
    final_host = (
        urlparse(getattr(resp, "final_url", "") or req.url).hostname or ""
    ).lower()
    if args.get("allow_redirects", True) and final_host and final_host != origin_host:
        if not _redirect_allowed(origin_host, final_host, redirect_scope, allow_hosts):
            return {
                "status": 0,
                "url": req.url,
                "final_url": resp.final_url,
                "headers": resp.headers,
                "bytes_len": 0,
                "elapsed_ms": resp.elapsed_ms,
                "json": None,
                "text": None,
                "truncated": False,
                "error": "RedirectOutOfScope",
                "message": f"Redirect from {origin_host} to {final_host} blocked by redirect_scope={redirect_scope}",
            }

    first = {
        "status": resp.status,
        "url": req.url,
        "final_url": resp.final_url,
        "headers": resp.headers,
        "bytes_len": len(body),
        "elapsed_ms": resp.elapsed_ms,
        "json": json_data,
        "text": text,
        "truncated": resp.truncated or (len(decoded) > len(body)),
        "not_modified": (resp.status == 304),
    }

    # ---- pagination ----
    paginated = None
    paginate_cfg = args.get("paginate") or {}
    if paginate_cfg:

        def _fetch_next(next_url: str) -> Dict[str, Any]:
            # port/scheme + IP guard
            okp, whyp = host_port_allowed(next_url)
            if not okp:
                return {"status": 0, "error": "PortNotAllowed", "message": whyp}

            h2 = (urlparse(next_url).hostname or "").lower()
            if is_disallowed_ip_host(h2):
                return {
                    "status": 0,
                    "error": "DeniedIP",
                    "message": f"Disallowed IP for host: {h2}",
                }

            # allow/deny patterns for the next page
            okh, codeh, whyh = _enforce_host_policies(h2, allow_hosts, deny_hosts)
            if not okh:
                return {"status": 0, "error": codeh or "DeniedHost", "message": whyh}

            next_args = {
                "url": next_url,
                "method": "GET",
                "headers": req.headers,  # reuse auth/UA
                "timeout": args.get("timeout"),
                "allow_redirects": args.get("allow_redirects", bool(max_redirects > 0)),
                "max_redirects": max_redirects,
                "max_bytes": args.get("max_bytes", 2_000_000),
            }
            next_req = http_request.build(next_args)
            tsec = args.get("timeout")
            if isinstance(tsec, (int, float)) and tsec > 0:
                setattr(next_req, "timeout", float(tsec))

            _attempt = 0
            while True:
                try:
                    _tb_take(next_req.host, rl_capacity, rl_refill, rl_maxwait)
                    next_resp = next_req.send()
                except TimeoutError as e:
                    if _attempt >= max_retries:
                        return {
                            "status": 0,
                            "url": next_url,
                            "final_url": next_url,
                            "headers": {},
                            "bytes_len": 0,
                            "elapsed_ms": 0,
                            "json": None,
                            "text": None,
                            "truncated": False,
                            "error": "RateLimited",
                            "message": str(e),
                        }
                    _attempt += 1
                    time.sleep(
                        compute_retry_wait(
                            {},
                            attempt=_attempt,
                            backoff_ms=backoff_ms,
                            jitter=jitter,
                            cap_seconds=5.0,
                        )
                    )
                    continue

                if next_resp.status not in retry_on or _attempt >= max_retries:
                    break
                _attempt += 1
                time.sleep(
                    compute_retry_wait(
                        next_resp.headers,
                        attempt=_attempt,
                        backoff_ms=backoff_ms,
                        jitter=jitter,
                        cap_seconds=60.0,
                    )
                )

            # Redirect policy on paginated hop
            origin2 = (next_req.host or "").lower()
            final2 = (
                urlparse(getattr(next_resp, "final_url", "") or next_url).hostname or ""
            ).lower()
            if args.get("allow_redirects", True) and final2 and final2 != origin2:
                if not _redirect_allowed(origin2, final2, redirect_scope, allow_hosts):
                    return {
                        "status": 0,
                        "url": next_url,
                        "final_url": next_resp.final_url,
                        "headers": next_resp.headers,
                        "bytes_len": 0,
                        "elapsed_ms": next_resp.elapsed_ms,
                        "json": None,
                        "text": None,
                        "truncated": False,
                        "error": "RedirectOutOfScope",
                        "message": f"Redirect from {origin2} to {final2} blocked by redirect_scope={redirect_scope}",
                    }

            enc_n = (next_resp.headers.get("Content-Encoding") or "").lower()
            decoded_n = _decompress_if_needed(next_resp.body, enc_n)
            next_body = clamp_bytes(decoded_n, max_bytes=args.get("max_bytes"))
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
                "truncated": next_resp.truncated or (len(decoded_n) > len(next_body)),
            }

        cfg = dict(paginate_cfg)
        cfg["fetch"] = _fetch_next
        paginated = paginate_response(first, args, cfg)

    # ---- assemble final output ----
    out = dict(first)
    if paginated:
        out["pagination"] = paginated
    if args.get("trace"):
        out["trace"] = {k: v for k, v in (args.get("trace") or {}).items() if v}
    return out
