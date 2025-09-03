import os, requests

BASE = os.getenv("AGENTS_BASE", "http://localhost:8000")
URL = f"{BASE}/app/api/agents/verb"


def call(payload: dict):
    r = requests.post(URL, json=payload, timeout=15)
    r.raise_for_status()
    return r.json()


def test_http_headers_shape_error():
    out = call(
        {
            "verb": "http.fetch",
            "args": {"url": "https://example.com", "headers": ["not", "an", "object"]},
        }
    )
    assert out["ok"] is False
    assert out["error"]["code"] in ("BadRequest", "AdapterError")
    assert "headers must be an object" in out["error"]["message"]


def test_http_rate_limit_numbers_error():
    out = call(
        {
            "verb": "http.fetch",
            "args": {
                "url": "https://example.com",
                "rate_limit": {
                    "per_host": {
                        "capacity": "x",
                        "refill_per_sec": "nope",
                        "max_wait_ms": "z",
                    }
                },
            },
        }
    )
    assert out["ok"] is False
    assert out["error"]["code"] in ("BadRequest", "AdapterError")
    assert "rate_limit.per_host fields must be numbers" in out["error"]["message"]


def test_http_unsupported_method():
    out = call(
        {
            "verb": "http.fetch",
            "args": {"url": "https://example.com", "method": "TRACE"},
        }
    )
    assert out["ok"] is False
    assert out["error"]["code"] in ("BadRequest", "AdapterError")
    assert "unsupported method" in out["error"]["message"].lower()


def test_http_happy_path_gzip_json():
    out = call({"verb": "http.fetch", "args": {"url": "https://httpbin.org/gzip"}})
    assert out["ok"] is True
    res = out["result"]
    assert res["status"] == 200
    # body is parsed into json or text
    assert res["json"] or res["text"]
