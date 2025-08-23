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
    ctype = (headers.get("Content-Type") or "").lower()
    text = None
    j = None
    if "application/json" in ctype or ctype.endswith("+json"):
        try:
            text = raw.decode("utf-8", errors="ignore")
            import json

            j = json.loads(text) if text else None
        except Exception:
            j = None
    elif ctype.startswith("text/") or "xml" in ctype or "html" in ctype:
        try:
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            text = None
    else:
        if raw and raw.lstrip()[:1] in (b"{", b"["):
            try:
                text = raw.decode("utf-8", errors="ignore")
                import json

                j = json.loads(text)
            except Exception:
                pass
    return text, j
