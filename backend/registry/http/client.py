# backend/registry/http/client.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import urllib.request
import urllib.parse
import ssl
import time
import json
import zlib

# ---- small, local helpers (keep this file self-contained) ----


def _merge_params(url: str, params: Dict[str, Any] | None) -> str:
    if not params:
        return url
    parsed = urllib.parse.urlparse(url)
    q = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    q.update(params)
    new_q = urllib.parse.urlencode(q, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=new_q))


def _clamp(b: bytes, max_bytes: int | None) -> Tuple[bytes, bool]:
    if max_bytes is None or len(b) <= max_bytes:
        return b, False
    return b[:max_bytes], True


def _decompress_if_needed(raw: bytes, content_encoding: str | None) -> bytes:
    enc = (content_encoding or "").lower()
    if not raw:
        return raw
    if enc == "gzip":
        try:
            import gzip

            return gzip.decompress(raw)
        except Exception:
            return raw
    if enc == "deflate":
        try:
            return zlib.decompress(raw)
        except Exception:
            return raw
    return raw


def _guess_json_or_text(
    raw: bytes, content_type: str | None
) -> Tuple[Optional[dict], Optional[str]]:
    ctype = (content_type or "").lower()
    text = None
    j = None
    if "application/json" in ctype or ctype.endswith("+json"):
        try:
            text = raw.decode("utf-8", errors="ignore")
            j = json.loads(text) if text else None
            return j, text
        except Exception:
            return None, raw.decode("utf-8", errors="ignore")
    if ctype.startswith("text/") or "xml" in ctype or "html" in ctype:
        return None, raw.decode("utf-8", errors="ignore")
    if raw and raw.lstrip()[:1] in (b"{", b"["):
        try:
            text = raw.decode("utf-8", errors="ignore")
            j = json.loads(text)
            return j, text
        except Exception:
            pass
    return None, None


# ---- response model ----


@dataclass
class HttpResponse:
    status: int
    url: str
    final_url: str
    headers: Dict[str, str]
    body: bytes
    elapsed_ms: int
    truncated: bool


# ---- request model + executor ----


class http_request:
    """
    Minimal request builder/executor used by http_fetch_adapter:
      req = http_request.build(args)
      resp = req.send()
    """

    def __init__(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[bytes] = None,
        timeout: Optional[float] = 15.0,
        max_bytes: int = 2_000_000,
        allow_redirects: bool = True,
    ) -> None:
        self.url = url
        self._parsed = urllib.parse.urlparse(self.url)
        self.scheme = self._parsed.scheme
        self.host = self._parsed.hostname or ""
        self.port = self._parsed.port or (443 if self.scheme == "https" else 80)

        self.method = method.upper()
        self.headers = dict(headers or {})
        self.data = data
        self.timeout = None if timeout is None else float(timeout)
        self.max_bytes = int(max_bytes)
        self.allow_redirects = bool(allow_redirects)

        # default UA (adapter can override)
        self.headers.setdefault(
            "User-Agent", "personal-agent/1.0 (+registry-http.fetch)"
        )

        # build a standard HTTPS context
        self._ssl_ctx = ssl.create_default_context()

    @property
    def ssl_context(self) -> ssl.SSLContext:
        return self._ssl_ctx

    # classmethod used by the adapter to construct a request from args
    @classmethod
    def build(cls, args: Dict[str, Any]) -> "http_request":
        if not isinstance(args, dict):
            raise ValueError("http_request.build expects a dict of args")

        url = args.get("url")
        if not url or not isinstance(url, str):
            raise ValueError("args.url (str) is required")

        method = str(args.get("method", "GET")).upper()
        if method not in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"):
            raise ValueError(f"Unsupported HTTP method: {method}")

        params = args.get("params") or {}
        if params:
            url = _merge_params(url, params)

        headers = dict(args.get("headers") or {})

        # body handling
        body = args.get("body", None)
        data: Optional[bytes] = None
        if body is None:
            data = None
        elif isinstance(body, (dict, list)):
            headers.setdefault("Content-Type", "application/json")
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            headers.setdefault("Content-Type", "text/plain; charset=utf-8")
            data = body.encode("utf-8")
        elif isinstance(body, (bytes, bytearray)):
            data = bytes(body)
        else:
            raise ValueError("Unsupported 'body' type; expected dict/list/str/bytes")

        timeout = args.get("timeout", 15.0)
        timeout = None if timeout is None else float(timeout)
        max_bytes = int(args.get("max_bytes", 2_000_000))
        allow_redirects = bool(args.get("allow_redirects", True))

        return cls(
            url=url,
            method=method,
            headers=headers,
            data=data,
            timeout=timeout,
            max_bytes=max_bytes,
            allow_redirects=allow_redirects,
        )

    def to_urllib(self) -> urllib.request.Request:
        """Build a urllib Request with headers/body/method."""
        req = urllib.request.Request(url=self.url, data=self.data, method=self.method)
        for k, v in (self.headers or {}).items():
            # urllib sets Host/Content-Length; don’t override hop-by-hop
            lk = k.lower()
            if lk in {"host", "content-length", "transfer-encoding", "connection"}:
                continue
            req.add_header(k, v)
        return req

    def send(self) -> "HttpResponse":
        req = self.to_urllib()
        t0 = time.time()
        try:
            with urllib.request.urlopen(
                req,
                timeout=float(self.timeout) if self.timeout is not None else 15.0,
                context=self.ssl_context,
            ) as resp:
                raw = resp.read(self.max_bytes + 1)
                truncated = len(raw) > self.max_bytes
                if truncated:
                    raw = raw[: self.max_bytes]
                return HttpResponse(
                    status=getattr(resp, "status", resp.getcode()),
                    url=self.url,
                    final_url=resp.geturl() or self.url,
                    headers=dict(resp.headers.items()),
                    body=raw,
                    elapsed_ms=int((time.time() - t0) * 1000),
                    truncated=truncated,
                )
        except urllib.error.HTTPError as e:
            # Return an HttpResponse instead of raising, so retry logic can handle it.
            try:
                body = e.read() or b""
            except Exception:
                body = b""
            return HttpResponse(
                status=getattr(e, "code", 0),
                url=self.url,
                final_url=getattr(e, "url", self.url),
                headers=dict(getattr(e, "headers", {}) or {}),
                body=body,
                elapsed_ms=int((time.time() - t0) * 1000),
                truncated=False,
            )

    # convenience parse for adapters that want decoded content
    @staticmethod
    def parse_body(
        body: bytes, headers: Dict[str, str]
    ) -> Tuple[Optional[dict], Optional[str]]:
        # kept for completeness; adapter currently parses in headers.choose_parse_body
        ctype = (headers.get("Content-Type") or "").lower()
        return _guess_json_or_text(body, ctype)
