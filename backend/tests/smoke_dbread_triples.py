#!/usr/bin/env python3
"""
Triple-nesting / deep-groups stress tests for db.read (stdlib only).
Run:  python smoke_dbread_triples.py
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
        "Triple: AND( OR(starts_with, AND(gt,lt)), not_null )",
        {
            "table": "events",
            "where": {
                "op": "and",
                "conditions": [
                    {
                        "op": "or",
                        "conditions": [
                            {"field": "topic", "op": "starts_with", "value": "plan."},
                            {
                                "op": "and",
                                "conditions": [
                                    {"field": "latency_ms", "op": "gt", "value": 50},
                                    {"field": "latency_ms", "op": "lt", "value": 500},
                                ],
                            },
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
        "Triple: OR( AND(ilike, OR(is_null,gte)), eq )",
        {
            "table": "events",
            "where": {
                "op": "or",
                "conditions": [
                    {
                        "op": "and",
                        "conditions": [
                            {"field": "topic", "op": "ilike", "value": "plan.%"},
                            {
                                "op": "or",
                                "conditions": [
                                    {"field": "latency_ms", "op": "is_null"},
                                    {"field": "latency_ms", "op": "gte", "value": 100},
                                ],
                            },
                        ],
                    },
                    {"field": "topic", "op": "eq", "value": "smoke"},
                ],
            },
            "limit": 5,
        },
        True,
    ),
    (
        "Triple: AND( OR(contains, icontains), OR(in, ends_with) )",
        {
            "table": "events",
            "where": {
                "op": "and",
                "conditions": [
                    {
                        "op": "or",
                        "conditions": [
                            {"field": "topic", "op": "contains", "value": "plan"},
                            {"field": "topic", "op": "icontains", "value": "RESULT"},
                        ],
                    },
                    {
                        "op": "or",
                        "conditions": [
                            {
                                "field": "topic",
                                "op": "in",
                                "value": ["plan.start", "plan.end"],
                            },
                            {"field": "topic", "op": "ends_with", "value": "end"},
                        ],
                    },
                ],
            },
            "limit": 5,
        },
        True,
    ),
    (
        "Triple: AND( OR(eq,eq), OR(not eq, not ilike) )",
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
                    {
                        "op": "or",
                        "conditions": [
                            {
                                "field": "topic",
                                "op": "not",
                                "value": {"op": "eq", "value": "smoke"},
                            },
                            {
                                "field": "topic",
                                "op": "not",
                                "value": {"op": "ilike", "value": "plan.step.%"},
                            },
                        ],
                    },
                ],
            },
            "limit": 5,
        },
        True,
    ),
    (
        "Serializer: OR( contains, starts_with, ends_with, ilike )",
        {
            "table": "events",
            "where": {
                "op": "or",
                "conditions": [
                    {"field": "topic", "op": "contains", "value": "plan"},
                    {"field": "topic", "op": "starts_with", "value": "plan."},
                    {"field": "topic", "op": "ends_with", "value": "result"},
                    {"field": "topic", "op": "ilike", "value": "PLAN.%"},
                ],
            },
            "limit": 5,
        },
        True,
    ),
    (
        "Fold: AND( OR(eq,eq,eq)->IN, created_at>=date )",
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
                            {"field": "topic", "op": "eq", "value": "plan.step.end"},
                        ],
                    },
                    {
                        "field": "created_at",
                        "op": "gte",
                        "value": "2025-08-01T00:00:00Z",
                    },
                ],
            },
            "order": {"field": "created_at", "desc": True},
            "limit": 5,
        },
        True,
    ),
    (
        "Triple + select/order: AND( ilike, OR(lte,is_null) )",
        {
            "table": "events",
            "select": ["id", "topic", "created_at", "latency_ms"],
            "where": {
                "op": "and",
                "conditions": [
                    {"field": "topic", "op": "ilike", "value": "plan.%"},
                    {
                        "op": "or",
                        "conditions": [
                            {"field": "latency_ms", "op": "lte", "value": 300},
                            {"field": "latency_ms", "op": "is_null"},
                        ],
                    },
                ],
            },
            "order": {"field": "created_at", "desc": True},
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
