#!/usr/bin/env python3
"""
Edge/guardrail smoke tests for db.read (stdlib only).
Run:  python smoke_dbread_edges.py
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
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
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
        "Shorthand equality (back-compat)",
        {"table": "events", "where": {"topic": "smoke"}, "limit": 2},
        True,
    ),
    (
        "Top-level empty where behaves like no filters",
        {"table": "events", "where": {}, "limit": 2},
        True,
    ),
    (
        "Order by multiple fields (list form, asc/desc mix)",
        {
            "table": "events",
            "where": {"topic": {"op": "ilike", "value": "plan.%"}},
            "order": [
                {"field": "created_at", "desc": True},
                {"field": "id", "desc": False},
            ],
            "limit": 5,
        },
        True,
    ),
    (
        "Pagination: big offset returns []",
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
        "Select projection as string",
        {
            "table": "events",
            "select": "id,topic,created_at",
            "where": {"topic": {"op": "ilike", "value": "plan.%"}},
            "limit": 2,
        },
        True,
    ),
    (
        "OR→IN fold (mixed scalar + op-dict)",
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
    (
        "Text helpers inside OR group",
        {
            "table": "events",
            "where": {
                "op": "or",
                "conditions": [
                    {"field": "topic", "op": "contains", "value": "plan"},
                    {"field": "topic", "op": "ends_with", "value": "result"},
                ],
            },
            "limit": 5,
        },
        True,
    ),
    (
        "Case sanity ILIKE",
        {
            "table": "events",
            "where": {"topic": {"op": "ilike", "value": "PLAN.%"}},
            "limit": 2,
        },
        True,
    ),
    (
        "NOT + IS NULL on a leaf",
        {
            "table": "events",
            "where": {
                "latency_ms": {"op": "not", "value": {"op": "is", "value": None}}
            },
            "limit": 2,
        },
        True,
    ),
    # --- Guardrails expecting errors ---
    (
        "Guardrail: IN with non-list (expect error)",
        {
            "table": "events",
            "where": {"topic": {"op": "in", "value": "not-a-list"}},
            "limit": 1,
        },
        False,
    ),
    (
        "Guardrail: BETWEEN wrong arity (expect error)",
        {
            "table": "events",
            "where": {"latency_ms": {"op": "between", "value": [100]}},
            "limit": 1,
        },
        False,
    ),
    (
        "Guardrail: where must be dict (expect error)",
        {"table": "events", "where": ["oops"], "limit": 1},
        False,
    ),
    (
        "NOT inside OR group",
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
    (
        "Mixed types in one group",
        {
            "table": "events",
            "where": {
                "op": "and",
                "conditions": [
                    {"field": "latency_ms", "op": "gte", "value": 0},
                    {"field": "topic", "op": "icontains", "value": "plan"},
                ],
            },
            "limit": 5,
        },
        True,
    ),
    (
        "Ascending order sanity",
        {
            "table": "events",
            "where": {},
            "order": {"field": "created_at", "desc": False},
            "limit": 3,
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
