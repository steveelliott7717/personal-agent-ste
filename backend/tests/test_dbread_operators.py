#!/usr/bin/env python3
"""
Operator-focused smoke tests for db.read (diagnostic version)
- No third-party deps (urllib only)
- Unbuffered prints + short timeouts so you always see output
"""

import json
import os
import sys
import time
import urllib.request

BASE = "http://localhost:8000/app/api/agents/verb"
HEADERS = {"Content-Type": "application/json"}


def say(msg):
    print(msg, flush=True)


def post(payload, timeout=6):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE, data=data, headers=HEADERS, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except Exception as e:
        return {"ok": False, "result": {"error": "HTTPError", "message": str(e)}}


def run_case(name, args, expect_ok=True):
    payload = {"verb": "db.read", "args": args}
    res = post(payload)
    ok = res.get("ok", False)
    rows = (res.get("result", {}) or {}).get("rows")
    n = len(rows) if isinstance(rows, list) else None
    passed = ok is expect_ok
    status = "PASS" if passed else "FAIL"
    say(f"[{status}] {name} -> ok={ok} rows={n}")
    if not passed:
        say(json.dumps(res, indent=2))
    return passed


tests = [
    (
        "EQ: topic = 'smoke'",
        {
            "table": "events",
            "where": {"topic": {"op": "eq", "value": "smoke"}},
            "limit": 2,
        },
        True,
    ),
    (
        "NEQ: topic != 'smoke'",
        {
            "table": "events",
            "where": {"topic": {"op": "neq", "value": "smoke"}},
            "limit": 2,
        },
        True,
    ),
    (
        "GT: latency_ms > 50",
        {
            "table": "events",
            "where": {"latency_ms": {"op": "gt", "value": 50}},
            "limit": 2,
        },
        True,
    ),
    (
        "GTE: latency_ms >= 50",
        {
            "table": "events",
            "where": {"latency_ms": {"op": "gte", "value": 50}},
            "limit": 2,
        },
        True,
    ),
    (
        "LT: latency_ms < 200",
        {
            "table": "events",
            "where": {"latency_ms": {"op": "lt", "value": 200}},
            "limit": 2,
        },
        True,
    ),
    (
        "LTE: latency_ms <= 200",
        {
            "table": "events",
            "where": {"latency_ms": {"op": "lte", "value": 200}},
            "limit": 2,
        },
        True,
    ),
    (
        "LIKE: topic LIKE 'plan.%'",
        {
            "table": "events",
            "where": {"topic": {"op": "like", "value": "plan.%"}},
            "limit": 2,
        },
        True,
    ),
    (
        "ILIKE: topic ILIKE 'PLAN.%'",
        {
            "table": "events",
            "where": {"topic": {"op": "ilike", "value": "PLAN.%"}},
            "limit": 2,
        },
        True,
    ),
    (
        "CONTAINS: topic LIKE '%%plan%%'",
        {
            "table": "events",
            "where": {"topic": {"op": "contains", "value": "plan"}},
            "limit": 2,
        },
        True,
    ),
    (
        "ICONTAINS: topic ILIKE '%%PLAN%%'",
        {
            "table": "events",
            "where": {"topic": {"op": "icontains", "value": "PLAN"}},
            "limit": 2,
        },
        True,
    ),
    (
        "STARTS_WITH: topic LIKE 'plan.%'",
        {
            "table": "events",
            "where": {"topic": {"op": "starts_with", "value": "plan."}},
            "limit": 2,
        },
        True,
    ),
    (
        "ENDS_WITH: topic LIKE '%%end'",
        {
            "table": "events",
            "where": {"topic": {"op": "ends_with", "value": "end"}},
            "limit": 2,
        },
        True,
    ),
    (
        "IN: topic IN ['plan.start','plan.end']",
        {
            "table": "events",
            "where": {"topic": {"op": "in", "value": ["plan.start", "plan.end"]}},
            "limit": 3,
        },
        True,
    ),
    (
        "BETWEEN: latency_ms between [0,1000]",
        {
            "table": "events",
            "where": {"latency_ms": {"op": "between", "value": [0, 1000]}},
            "limit": 3,
        },
        True,
    ),
    (
        "IS NULL: latency_ms is null",
        {"table": "events", "where": {"latency_ms": {"op": "is_null"}}, "limit": 2},
        True,
    ),
    (
        "NOT NULL: source_agent not null",
        {"table": "events", "where": {"source_agent": {"op": "not_null"}}, "limit": 2},
        True,
    ),
    (
        "NOT(EQ): NOT(topic = 'smoke')",
        {
            "table": "events",
            "where": {"topic": {"op": "not", "value": {"op": "eq", "value": "smoke"}}},
            "limit": 3,
        },
        True,
    ),
    (
        "NOT(ILIKE): NOT(topic ILIKE 'plan.%')",
        {
            "table": "events",
            "where": {
                "topic": {"op": "not", "value": {"op": "ilike", "value": "plan.%"}}
            },
            "limit": 3,
        },
        True,
    ),
    (
        "SELECT(list)+ORDER(desc): topic ilike plan.%",
        {
            "table": "events",
            "select": ["id", "topic", "created_at"],
            "where": {"topic": {"op": "ilike", "value": "plan.%"}},
            "order": {"field": "created_at", "desc": True},
            "limit": 3,
        },
        True,
    ),
    (
        "Back-compat scalar equals: topic='smoke'",
        {"table": "events", "where": {"topic": "smoke"}, "limit": 2},
        True,
    ),
    # Guardrails (expect ok:false)
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
]

if __name__ == "__main__":
    say("=" * 72)
    say(f"Running: {os.path.abspath(__file__)}")
    say(f"Python : {sys.version.split()[0]}")
    say(f"Target : {BASE}")
    say("=" * 72)

    passed = 0
    for name, args, expect_ok in tests:
        if run_case(name, args, expect_ok):
            passed += 1
        time.sleep(0.05)
    total = len(tests)
    say(f"\n{passed}/{total} tests passed")

if __name__ == "__main__":
    import unittest

    unittest.main(verbosity=2)
