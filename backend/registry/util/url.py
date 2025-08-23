from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse


def url_with_param(base_url: str, key: str, value: str) -> str:
    p = urlparse(base_url)
    q = dict(parse_qsl(p.query, keep_blank_values=True))
    q[key] = value
    return urlunparse(p._replace(query=urlencode(q, doseq=True)))


def looks_like_url(s: object) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))


def merge_params(url: str, params: dict) -> str:
    if not params:
        return url
    return url_with_param(
        url, next(iter(params.keys())), next(iter(params.values()))
    )  # simple if you only need one; expand if needed
