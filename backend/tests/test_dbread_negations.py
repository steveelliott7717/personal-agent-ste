#!/usr/bin/env python3
"""
Negation-focused smoke tests (NOT on leaves and simple groups)
Run:  python test_dbread_negations.py
"""
import json
import time
import urllib.request

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
    (
        "NOT EQ: NOT(topic='smoke')",
        {
            "table": "events",
            "where": {"topic": {"op": "not", "value": {"op": "eq", "value": "smoke"}}},
            "limit": 5,
        },
        True,
    ),
    (
        "NOT ILIKE: NOT(topic ILIKE 'plan.%')",
        {
            "table": "events",
            "where": {
                "topic": {"op": "not", "value": {"op": "ilike", "value": "plan.%"}}
            },
            "limit": 5,
        },
        True,
    ),
    (
        "NOT CONTAINS: NOT(topic LIKE '%plan%')",
        {
            "table": "events",
            "where": {
                "topic": {"op": "not", "value": {"op": "contains", "value": "plan"}}
            },
            "limit": 5,
        },
        True,
    ),
    (
        "NOT IS NULL (aka IS NOT NULL)",
        {
            "table": "events",
            "where": {
                "latency_ms": {"op": "not", "value": {"op": "is", "value": None}}
            },
            "limit": 5,
        },
        True,
    ),
    (
        "NOT on one branch of OR",
        {
            "table": "events",
            "where": {
                "op": "or",
                "conditions": [
                    {
                        "field": "topic",
                        "op": "not",
                        "value": {"op": "ilike", "value": "plan.%"},
                    },
                    {"field": "source_agent", "op": "eq", "value": "manual"},
                ],
            },
            "limit": 5,
        },
        True,
    ),
]

if __name__ == "__main__":
    passed = 0
    for name, args, expect_ok in tests:
        if run_case(name, args, expect_ok):
            passed += 1
        time.sleep(0.05)
    print(f"\n{passed}/{len(tests)} tests passed")
