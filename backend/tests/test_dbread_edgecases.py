#!/usr/bin/env python3
"""
Edge & guardrail tests (expect some failures by design)
Run:  python test_dbread_edgecases.py
"""
import json, time, urllib.request

BASE = "http://localhost:8000/app/api/agents/verb"
HEADERS = {"Content-Type": "application/json"}


def post(payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE, data=data, headers=HEADERS, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return {"ok": False, "result": {"error": "HTTPError", "message": str(e)}}


def run_case(name, args, expect_ok=True):
    res = post({"verb": "db.read", "args": args})
    ok = res.get("ok", False)
    rows = (res.get("result", {}) or {}).get("rows")
    n = len(rows) if isinstance(rows, list) else None
    passed = ok is expect_ok
    print(f"[{'PASS' if passed else 'FAIL'}] {name} -> ok={ok} rows={n}")
    if not passed:
        print(json.dumps(res, indent=2))
    return passed


tests = [
    # happy-path edges
    ("Empty where == no filter", {"table": "events", "where": {}, "limit": 2}, True),
    (
        "Select as string + order asc",
        {
            "table": "events",
            "select": "id,topic,created_at",
            "where": {},
            "order": {"field": "created_at", "desc": False},
            "limit": 3,
        },
        True,
    ),
    (
        "Pagination: huge offset returns []",
        {
            "table": "events",
            "where": {},
            "order": {"field": "created_at", "desc": True},
            "limit": 3,
            "offset": 99999,
        },
        True,
    ),
    (
        "OR→IN fold mixed shapes",
        {
            "table": "events",
            "where": {
                "or": [
                    {"topic": "plan.start"},
                    {"topic": {"op": "eq", "value": "plan.end"}},
                ]
            },
            "limit": 3,
        },
        True,
    ),
    # guardrails (expect ok:false)
    (
        "ERROR: IN value must be list",
        {
            "table": "events",
            "where": {"topic": {"op": "in", "value": "not-a-list"}},
            "limit": 1,
        },
        False,
    ),
    (
        "ERROR: BETWEEN needs 2 values",
        {
            "table": "events",
            "where": {"latency_ms": {"op": "between", "value": [100]}},
            "limit": 1,
        },
        False,
    ),
    (
        "ERROR: where must be dict",
        {"table": "events", "where": ["oops"], "limit": 1},
        False,
    ),
    (
        "ERROR: empty OR list (define policy to reject)",
        {"table": "events", "where": {"op": "or", "conditions": []}, "limit": 1},
        False,
    ),
]

if __name__ == "__main__":
    passed = 0
    for name, args, expect_ok in tests:
        if run_case(name, args, expect_ok):
            passed += 1
        time.sleep(0.05)
    print(f"\n{passed}/{len(tests)} tests passed")
