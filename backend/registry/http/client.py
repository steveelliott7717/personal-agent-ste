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
import gzip, zlib, io
from backend.registry.http.headers import detect_charset
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional, Dict
import io
import json
import uuid
import base64
from urllib.parse import urlencode
from typing import Any, Dict, Optional, Tuple, Iterator


def decode_response_bytes(
    raw: bytes, headers: dict[str, str]
) -> tuple[str | None, str | None]:
    """
    Returns (text, charset). If non-text, returns (None, None).
    """
    ctype = headers.get("Content-Type")
    charset = detect_charset(ctype)
    if not charset:
        return None, None
    try:
        return raw.decode(charset, errors="replace"), charset
    except Exception:
        # last resort
        return raw.decode("utf-8", errors="replace"), "utf-8"


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
    if not raw or not enc:
        return raw

    if "gzip" in enc:
        try:
            import gzip

            return gzip.decompress(raw)
        except Exception:
            return raw

    if "deflate" in enc:
        try:
            return zlib.decompress(raw)
        except Exception:
            try:
                return zlib.decompress(raw, -zlib.MAX_WBITS)
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
        # Request compressed responses by default, unless caller specified
        if "Accept-Encoding" not in {k.title(): v for k, v in headers.items()}:
            headers["Accept-Encoding"] = "gzip, deflate"

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
                # Read up to max_bytes+1 compressed bytes, then decompress, then clamp
                raw = resp.read(self.max_bytes + 1)
                headers_map = dict(resp.headers.items())
                raw = _decompress_if_needed(raw, headers_map.get("Content-Encoding"))
                truncated = len(raw) > self.max_bytes
                if truncated:
                    raw = raw[: self.max_bytes]

                # Now clamp the decompressed body
                raw, truncated = _clamp(raw, self.max_bytes)

                return HttpResponse(
                    status=getattr(resp, "status", resp.getcode()),
                    url=self.url,
                    final_url=resp.geturl() or self.url,
                    headers=headers_map,
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

    def send_stream(
        self, chunk_size: int = 64 * 1024
    ) -> Tuple["HttpResponse", Iterator[bytes]]:
        """
        Opens the request and yields decompressed bytes chunks (if Content-Encoding set).
        The returned HttpResponse contains status/headers/final_url but body is empty.
        """
        req = self.to_urllib()
        t0 = time.time()
        resp = urllib.request.urlopen(
            req,
            timeout=float(self.timeout) if self.timeout is not None else 15.0,
            context=self.ssl_context,
        )
        headers_map = dict(resp.headers.items())
        status = getattr(resp, "status", resp.getcode())
        final_url = resp.geturl() or self.url

        dec, flush = _decompress_stream(headers_map.get("Content-Encoding"))

        def _iter():
            try:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out = dec(chunk)
                    if out:
                        yield out
                tail = flush()
                if tail:
                    yield tail
            finally:
                try:
                    resp.close()
                except Exception:
                    pass

        meta = HttpResponse(
            status=status,
            url=self.url,
            final_url=final_url,
            headers=headers_map,
            body=b"",  # consumer reads from iterator
            elapsed_ms=int((time.time() - t0) * 1000),
            truncated=False,
        )
        return meta, _iter()

    # convenience parse for adapters that want decoded content
    @staticmethod
    def parse_body(
        body: bytes, headers: Dict[str, str]
    ) -> Tuple[Optional[dict], Optional[str]]:
        # kept for completeness; adapter currently parses in headers.choose_parse_body
        ctype = (headers.get("Content-Type") or "").lower()
        return _guess_json_or_text(body, ctype)


def retry_after_seconds(
    headers: Dict[str, str], now_ts: float | None = None, cap_seconds: float = 60.0
) -> Optional[float]:
    """
    Parses Retry-After header (seconds or HTTP-date) and returns a bounded wait in seconds,
    or None if header is absent/invalid.
    """
    v = headers.get("Retry-After")
    if not v:
        return None
    v = v.strip()

    # "Retry-After: <seconds>"
    if v.isdigit():
        try:
            secs = float(v)
            return max(0.0, min(secs, cap_seconds))
        except Exception:
            return None

    # "Retry-After: <http-date>"
    try:
        dt = parsedate_to_datetime(v)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = (
            datetime.fromtimestamp(now_ts, tz=timezone.utc)
            if now_ts
            else datetime.now(timezone.utc)
        )
        delta = (dt - now).total_seconds()
        return max(0.0, min(delta, cap_seconds))
    except Exception:
        return None


def compute_retry_wait(
    headers: Dict[str, str],
    attempt: int,
    backoff_ms: int = 100,
    jitter: bool = True,
    cap_seconds: float = 60.0,
) -> float:
    """
    attempt: 1-based attempt index (1 = first retry)
    Returns wait seconds. Prefers Retry-After header when present; otherwise exponential backoff.
    """
    ra = retry_after_seconds(headers, cap_seconds=cap_seconds)
    if ra is not None:
        return ra

    # Exponential backoff: base * 2^(attempt-1)
    base = (backoff_ms / 1000.0) * (2 ** max(0, attempt - 1))
    if jitter:
        base += random.uniform(0, base * 0.1)
    return min(base, cap_seconds)


def _coerce_file_bytes(value: Any) -> bytes:
    """
    Accepts:
      - bytes -> returned as-is
      - str   -> tries base64 (strict). If that fails, encodes as UTF-8 text bytes.
    """
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        # Try strict base64 first (most JSON uploads will use base64)
        try:
            return base64.b64decode(value, validate=True)
        except Exception:
            return value.encode("utf-8")
    raise TypeError(f"Unsupported file content type: {type(value).__name__}")


def build_request_body(args: Dict[str, Any]) -> Tuple[Optional[bytes], Dict[str, str]]:
    """
    Builds the outbound request body and any Content-Type header needed.

    Supported shapes (first match wins):
      - files: List[ {name, filename?, content|content_b64, content_type?} ]  -> multipart/form-data
        * optional: form: Dict[str, str|List[str]] -> extra simple fields in same multipart
      - json: Any                                     -> application/json
      - form: Dict[str, str|List[str]]                -> application/x-www-form-urlencoded
      - body: str|bytes                               -> caller-provided; no Content-Type set here

    Returns: (payload_bytes or None, headers_to_add)
    """
    headers: Dict[str, str] = {}

    files = args.get("files")
    if files:
        boundary = f"----registry-{uuid.uuid4().hex}"
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        buf = io.BytesIO()

        # optional simple fields alongside files
        form = args.get("form") or {}
        for k, v in form.items() if isinstance(form, dict) else []:
            values = v if isinstance(v, list) else [v]
            for sv in values:
                buf.write(f"--{boundary}\r\n".encode())
                buf.write(
                    f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode()
                )
                buf.write(str(sv).encode())
                buf.write(b"\r\n")

        for part in files:
            if not isinstance(part, dict) or "name" not in part:
                raise ValueError(
                    "each files[] item must be a dict with at least 'name'"
                )
            name = part["name"]
            filename = part.get("filename") or "upload.bin"
            ctype = part.get("content_type") or "application/octet-stream"
            # allow either 'content' or 'content_b64'
            raw = part.get("content", None)
            if raw is None and "content_b64" in part:
                raw = part["content_b64"]
            content_bytes = _coerce_file_bytes(raw)

            buf.write(f"--{boundary}\r\n".encode())
            buf.write(
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
            )
            buf.write(f"Content-Type: {ctype}\r\n\r\n".encode())
            buf.write(content_bytes)
            buf.write(b"\r\n")

        buf.write(f"--{boundary}--\r\n".encode())
        return buf.getvalue(), headers

    if "json" in args and args["json"] is not None:
        payload = json.dumps(args["json"]).encode("utf-8")
        headers["Content-Type"] = "application/json"
        return payload, headers

    if "form" in args and args["form"] is not None:
        form = args["form"]
        if not isinstance(form, dict):
            raise ValueError("args.form must be a dict")
        payload = urlencode(form, doseq=True).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        return payload, headers

    if "body" in args and args["body"] is not None:
        data = args["body"]
        if isinstance(data, bytes):
            return data, headers
        if isinstance(data, str):
            return data.encode("utf-8"), headers
        raise ValueError("args.body must be str or bytes")

    return None, headers


def _decompress_stream(enc: str | None):
    """
    Returns a (decompress, flush) pair for streaming gzip/deflate; identity if none.
    """
    enc = (enc or "").lower()
    if "gzip" in enc:
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        return d.decompress, d.flush
    if "deflate" in enc:
        d = zlib.decompressobj()

        def _dec(b: bytes) -> bytes:
            try:
                return d.decompress(b)
            except zlib.error:
                d2 = zlib.decompressobj(-zlib.MAX_WBITS)
                return d2.decompress(b)

        return _dec, d.flush
    return (lambda b: b), (lambda: b"")
