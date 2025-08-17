#!/usr/bin/env python3
"""
Comprehensive grouped-combos smoke test for db.read.
- Uses stdlib only (urllib).
- Validates grouped AND/OR, NOT, text helpers, ranges, IN, null checks, and ordering.
Run:  python smoke_dbread_groups_full.py
"""

import json, time, urllib.request

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
        "Simple AND: topic=smoke AND latency_ms>10",
        {
            "table": "events",
            "where": {
                "op": "and",
                "conditions": [
                    {"field": "topic", "op": "eq", "value": "smoke"},
                    {"field": "latency_ms", "op": "gt", "value": 10},
                ],
            },
            "limit": 3,
        },
        True,
    ),
    (
        "Simple OR: topic ilike plan.% OR source_agent=manual",
        {
            "table": "events",
            "where": {
                "op": "or",
                "conditions": [
                    {"field": "topic", "op": "ilike", "value": "plan.%"},
                    {"field": "source_agent", "op": "eq", "value": "manual"},
                ],
            },
            "limit": 3,
        },
        True,
    ),
    (
        "Nested OR(AND,...): OR( AND(topic=smoke, latency_ms>10), source_agent=manual )",
        {
            "table": "events",
            "where": {
                "op": "or",
                "conditions": [
                    {
                        "op": "and",
                        "conditions": [
                            {"field": "topic", "op": "eq", "value": "smoke"},
                            {"field": "latency_ms", "op": "gt", "value": 10},
                        ],
                    },
                    {"field": "source_agent", "op": "eq", "value": "manual"},
                ],
            },
            "limit": 3,
        },
        True,
    ),
    (
        "Double-nesting: AND( OR(starts_with plan., topic=smoke), latency_ms<1000 )",
        {
            "table": "events",
            "where": {
                "op": "and",
                "conditions": [
                    {
                        "op": "or",
                        "conditions": [
                            {"field": "topic", "op": "starts_with", "value": "plan."},
                            {"field": "topic", "op": "eq", "value": "smoke"},
                        ],
                    },
                    {"field": "latency_ms", "op": "lt", "value": 1000},
                ],
            },
            "limit": 3,
        },
        True,
    ),
    (
        "AND( ilike plan.%, OR(lte 200, is_null) )",
        {
            "table": "events",
            "where": {
                "op": "and",
                "conditions": [
                    {"field": "topic", "op": "ilike", "value": "plan.%"},
                    {
                        "op": "or",
                        "conditions": [
                            {"field": "latency_ms", "op": "lte", "value": 200},
                            {"field": "latency_ms", "op": "is_null"},
                        ],
                    },
                ],
            },
            "limit": 5,
        },
        True,
    ),
    (
        "OR across columns + not_null",
        {
            "table": "events",
            "where": {
                "op": "and",
                "conditions": [
                    {
                        "op": "or",
                        "conditions": [
                            {"field": "topic", "op": "starts_with", "value": "plan."},
                            {"field": "topic", "op": "ends_with", "value": "result"},
                        ],
                    },
                    {"field": "source_agent", "op": "not_null"},
                ],
            },
            "limit": 5,
        },
        True,
    ),
    (
        "AND( IN(plan.start,plan.end), between [0,1000], contains 'plan' )",
        {
            "table": "events",
            "where": {
                "op": "and",
                "conditions": [
                    {"field": "topic", "op": "in", "value": ["plan.start", "plan.end"]},
                    {"field": "latency_ms", "op": "between", "value": [0, 1000]},
                    {"field": "topic", "op": "contains", "value": "plan"},
                ],
            },
            "limit": 5,
        },
        True,
    ),
    (
        "AND( NOT ilike plan.%, OR(payload is null, latency_ms >= 100) )",
        {
            "table": "events",
            "where": {
                "op": "and",
                "conditions": [
                    {
                        "field": "topic",
                        "op": "not",
                        "value": {"op": "ilike", "value": "plan.%"},
                    },
                    {
                        "op": "or",
                        "conditions": [
                            {"field": "payload", "op": "is_null"},
                            {"field": "latency_ms", "op": "gte", "value": 100},
                        ],
                    },
                ],
            },
            "limit": 5,
        },
        True,
    ),
    (
        "AND( OR(eq 'plan.start', eq 'plan.end')->IN, latency_ms is_null )",
        {
            "table": "events",
            "where": {
                "op": "and",
                "conditions": [
                    {
                        "op": "or",
                        "conditions": [
                            {"field": "topic", "op": "eq", "value": "plan.start"},
                            {"field": "topic", "op": "eq", "value": "plan.end"},
                        ],
                    },
                    {"field": "latency_ms", "op": "is_null"},
                ],
            },
            "order": {"field": "created_at", "desc": True},
            "limit": 3,
        },
        True,
    ),
    (
        "AND( between [50,300], created_at >= date )",
        {
            "table": "events",
            "where": {
                "op": "and",
                "conditions": [
                    {"field": "latency_ms", "op": "between", "value": [50, 300]},
                    {
                        "field": "created_at",
                        "op": "gte",
                        "value": "2025-08-16T00:00:00Z",
                    },
                ],
            },
            "limit": 5,
        },
        True,
    ),
    (
        "OR( contains 'plan', icontains 'result' )",
        {
            "table": "events",
            "where": {
                "op": "or",
                "conditions": [
                    {"field": "topic", "op": "contains", "value": "plan"},
                    {"field": "topic", "op": "icontains", "value": "result"},
                ],
            },
            "limit": 5,
        },
        True,
    ),
    (
        "AND( NOT eq 'smoke', NOT ilike 'plan.%' )",
        {
            "table": "events",
            "where": {
                "op": "and",
                "conditions": [
                    {
                        "field": "topic",
                        "op": "not",
                        "value": {"op": "eq", "value": "smoke"},
                    },
                    {
                        "field": "topic",
                        "op": "not",
                        "value": {"op": "ilike", "value": "plan.%"},
                    },
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
