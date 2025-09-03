import gzip
import zlib
import re

_TEXT_CT_RE = re.compile(
    r"^(?:text/|application/(?:json|xml|x-www-form-urlencoded))(?:[;].*)?$",
    re.I,
)


def detect_charset(content_type: str | None) -> str | None:
    """
    Best-effort charset detection from Content-Type header.
    Returns codec name (e.g., 'utf-8') or None if not clearly text.
    """
    if not content_type:
        return None
    m = re.search(r"charset=([^\s;]+)", content_type, flags=re.I)
    if m:
        return m.group(1).strip('"').strip("'")
    if _TEXT_CT_RE.match(content_type):
        return "utf-8"
    return None


def _maybe_decompress(data: bytes, headers: dict[str, str]) -> bytes:
    enc = (headers.get("Content-Encoding") or "").lower()
    if "gzip" in enc:
        return gzip.decompress(data)
    if "deflate" in enc:
        # raw deflate stream (zlib default handles both)
        try:
            return zlib.decompress(data)
        except zlib.error:
            return zlib.decompress(data, -zlib.MAX_WBITS)
    return data


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


def get_ci(headers: dict, name: str):
    ln = name.lower()
    for k, v in headers.items():
        if k.lower() == ln:
            return v
    return None


def parse_link_next(link_value: str, want_rel: str = "next") -> str | None:
    try:
        parts = [p.strip() for p in link_value.split(",")]
        for part in parts:
            if ";" not in part:
                continue
            url_part, *params = [x.strip() for x in part.split(";")]
            if not (url_part.startswith("<") and url_part.endswith(">")):
                continue
            url_str = url_part[1:-1]
            rel_val = None
            for p in params:
                if p.lower().startswith("rel="):
                    rel_val = p.split("=", 1)[1].strip().strip('"').strip("'")
                    break
            if rel_val == want_rel:
                return url_str
    except Exception:
        pass
    return None


def choose_parse_body(headers: dict, raw: bytes) -> tuple[str | None, object | None]:
    ctype = headers.get("Content-Type")
    ctype_l = (ctype or "").lower()
    cs = detect_charset(ctype)

    def _decode(b: bytes) -> str | None:
        if cs:
            try:
                return b.decode(cs, errors="replace")
            except Exception:
                pass
        try:
            return b.decode("utf-8", errors="replace")
        except Exception:
            return None

    text = None
    j = None

    if "application/json" in ctype_l or ctype_l.endswith("+json"):
        text = _decode(raw)
        if text:
            try:
                import json

                j = json.loads(text)
            except Exception:
                j = None
        return text, j

    if ctype_l.startswith("text/") or "xml" in ctype_l or "html" in ctype_l:
        text = _decode(raw)
        return text, None

    if raw and raw.lstrip()[:1] in (b"{", b"["):
        text = _decode(raw)
        if text:
            try:
                import json

                j = json.loads(text)
            except Exception:
                j = None
        return text, j

    return None, None
