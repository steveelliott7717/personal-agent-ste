def clamp_bytes(raw: bytes, max_bytes: int | None) -> bytes:
    if not isinstance(raw, (bytes, bytearray)):
        return b""
    if not max_bytes:
        return bytes(raw)
    raw = bytes(raw)
    return raw if len(raw) <= max_bytes else raw[:max_bytes]
