def json_pointer_get(doc, pointer: str):
    if pointer in ("", "/"):
        return doc
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        return None
    cur = doc
    for raw in pointer.split("/")[1:]:
        token = raw.replace("~1", "/").replace("~0", "~")
        try:
            cur = (
                cur[int(token)]
                if isinstance(cur, list)
                else (cur.get(token) if isinstance(cur, dict) else None)
            )
        except Exception:
            return None
    return cur
