def apply_cache_headers(req, cache: dict):
    if not isinstance(cache, dict):
        return
    if cache.get("etag"):
        req.headers.setdefault("If-None-Match", str(cache["etag"]))
    if cache.get("last_modified"):
        req.headers.setdefault("If-Modified-Since", str(cache["last_modified"]))
