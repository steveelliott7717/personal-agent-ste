# safety.py
from __future__ import annotations

import ipaddress
import socket
from typing import Iterable

# Blocks:
#  - Loopback (127.0.0.0/8, ::1/128)
#  - Private RFC1918 (10/8, 172.16/12, 192.168/16)
#  - Link-local (169.254/16, fe80::/10)
#  - CGNAT (100.64/10)
#  - Multicast (224/4, ff00::/8)
#  - Reserved/unspecified (0.0.0.0/8, ::/128, 240.0.0.0/4 etc.)
#  - Common metadata/IMDS endpoints (169.254.169.254, fd00:ec2::254) treated as link-local/ULA anyway

_IMDS_V4 = ipaddress.ip_address(
    "169.254.169.254"
)  # falls in link-local, but keep explicit


def _iter_addrs(host: str) -> Iterable[ipaddress._BaseAddress]:
    """Resolve host → ipaddress objects (both v4/v6) safely."""
    try:
        # If already an IP, ip_address will succeed
        yield ipaddress.ip_address(host)
        return
    except ValueError:
        pass

    # DNS resolution (AF_UNSPEC: both v4 & v6)
    try:
        infos = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except Exception:
        # If DNS fails, fail safe: treat as disallowed? No—return nothing; caller decides.
        infos = []

    for fam, _, _, _, sockaddr in infos:
        try:
            if fam == socket.AF_INET:
                yield ipaddress.ip_address(sockaddr[0])
            elif fam == socket.AF_INET6:
                yield ipaddress.ip_address(sockaddr[0])
        except Exception:
            continue


def _ip_is_sensitive(ip: ipaddress._BaseAddress) -> bool:
    # Standard sensitivity checks
    if ip.is_loopback:
        return True
    if ip.is_link_local:
        return True
    if ip.is_private:
        return True
    if ip.is_multicast:
        return True
    if ip.is_unspecified:
        return True
    # Some reserved/bogon ranges are not covered by helpers (esp. IPv4 240/4)
    if isinstance(ip, ipaddress.IPv4Address):
        if ip in ipaddress.ip_network("0.0.0.0/8"):
            return True
        if ip in ipaddress.ip_network("240.0.0.0/4"):
            return True
        if ip in ipaddress.ip_network("100.64.0.0/10"):  # CGNAT
            return True
    else:
        # Unique-local (fc00::/7) — treat as private
        if ip in ipaddress.ip_network("fc00::/7"):
            return True
    # Explicitly block IMDS v4 (already link-local, but keep explicit)
    if ip == _IMDS_V4:
        return True
    return False


def is_disallowed_ip_host(host: str) -> bool:
    """
    True if host is (or resolves to) a loopback/private/link-local/multicast/etc. address.
    """
    for ip in _iter_addrs(host):
        if _ip_is_sensitive(ip):
            return True
    return False


# Back-compat alias for legacy imports
_is_disallowed_ip_host = is_disallowed_ip_host  # noqa: N816 (match legacy name)
