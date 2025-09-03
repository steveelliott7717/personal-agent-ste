# ports.py
from __future__ import annotations

import os
from typing import Tuple
from urllib.parse import urlparse

# ENV knobs (same spirit as host allow/deny)
#   HTTP_FETCH_ALLOW_PORTS="80,443,8080,1024-65535"
#   HTTP_FETCH_DENY_PORTS="25,3306,0-1023"
#
# Behavior:
# - If an allowlist is set, ONLY those ports are allowed (denylist still respected).
# - If no allowlist is set, default allow is: 80/443 + any port not explicitly denied.
# - Port is derived from scheme when not provided (http->80, https->443).
#
# Return:
#   (ok: bool, why: str)
#     ok=True  => port allowed
#     ok=False => reason in `why`

_DEFAULT_SCHEME_PORTS = {"http": 80, "https": 443}


def _parse_port_ranges(spec: str) -> set[int]:
    """Parse comma-separated list like '80,443,1024-65535' -> set of ints."""
    out: set[int] = set()
    for part in (spec or "").split(","):
        p = part.strip()
        if not p:
            continue
        if "-" in p:
            lo_s, hi_s = p.split("-", 1)
            try:
                lo = int(lo_s)
                hi = int(hi_s)
                if 0 <= lo <= 65535 and 0 <= hi <= 65535 and lo <= hi:
                    out.update(range(lo, hi + 1))
            except ValueError:
                # Ignore invalid ranges
                continue
        else:
            try:
                v = int(p)
                if 0 <= v <= 65535:
                    out.add(v)
            except ValueError:
                continue
    return out


def _port_from_parsed(parsed) -> int:
    """Resolve effective port from a parsed URL (scheme default if missing)."""
    if parsed.port is not None:
        return parsed.port
    return _DEFAULT_SCHEME_PORTS.get((parsed.scheme or "").lower(), -1)


def host_port_allowed(parsed_url_or_str) -> Tuple[bool, str]:
    """
    Enforce ports against env allow/deny; return (ok, why).

    Env:
      HTTP_FETCH_ALLOW_PORTS
      HTTP_FETCH_DENY_PORTS
    """
    parsed = (
        urlparse(parsed_url_or_str)
        if isinstance(parsed_url_or_str, str)
        else parsed_url_or_str
    )
    port = _port_from_parsed(parsed)

    if port < 0 or port > 65535:
        return (False, f"InvalidPort: {port}")

    allow_env = os.getenv("HTTP_FETCH_ALLOW_PORTS", "").strip()
    deny_env = os.getenv("HTTP_FETCH_DENY_PORTS", "").strip()

    allow_set = _parse_port_ranges(allow_env)
    deny_set = _parse_port_ranges(deny_env)

    # If an allowlist exists, only it is valid
    if allow_set:
        if port not in allow_set:
            return (False, f"Port {port} not in allowlist")
        if port in deny_set:
            return (False, f"Port {port} explicitly denied")
        return (True, "ok")

    # Default: allow 80/443 unless denied; allow others unless denied
    if port in deny_set:
        return (False, f"Port {port} explicitly denied")

    # Be permissive by default if no allowlist; many APIs use 8443/8080/etc.
    return (True, "ok")


# Back-compat aliases so existing imports continue working
_host_port_allowed = host_port_allowed  # noqa: N816 (match legacy name)
