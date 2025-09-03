#!/usr/bin/env python3
import json
import urllib.request

BASE = "http://localhost:8000/app/api/agents/verb"
HEADERS = {"Content-Type": "application/json"}


def post(p):
    req = urllib.request.Request(
        BASE, data=json.dumps(p).encode(), headers=HEADERS, method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


def run(name, args, expect_ok=True, expect_meta=True):
    res = post({"verb": "db.read", "args": args})
    ok = res.get("ok")
    meta = (res.get("result", {}) or {}).get("meta")
    rows = (res.get("result", {}) or {}).get("rows") or []
    passed = (ok is expect_ok) and ((meta is not None) if expect_meta else True)
    print(
        f"[{'PASS' if passed else 'FAIL'}] {name} -> ok={ok} rows={len(rows)} meta={meta}"
    )
    if not passed:
        print(json.dumps(res, indent=2))
    return passed


if __name__ == "__main__":
    total = 0
    total += run(
        "Count only (no rows)",
        {
            "table": "events",
            "where": {"topic": {"op": "ilike", "value": "plan.%"}},
            "aggregate": {"count": "*"},
            "limit": 0,
        },
    )
    total += run(
        "Rows + count",
        {
            "table": "events",
            "where": {"topic": {"op": "ilike", "value": "plan.%"}},
            "aggregate": {"count": "*"},
            "order": {"field": "created_at", "desc": True},
            "limit": 3,
        },
    )
    print(f"\n{total}/2 tests passed")
