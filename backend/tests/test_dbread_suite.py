#!/usr/bin/env python3
import json, sys, urllib.request

BASE = "http://localhost:8000/app/api/agents/verb"
HEADERS = {"Content-Type": "application/json"}


def post(payload, timeout=15):
    req = urllib.request.Request(
        BASE, data=json.dumps(payload).encode(), headers=HEADERS, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def run(name, args, expect_ok=True):
    res = post({"verb": "db.read", "args": args})
    ok = bool(res.get("ok"))
    rows = (res.get("result", {}) or {}).get("rows") or []
    meta = (res.get("result", {}) or {}).get("meta") or {}
    passed = ok is expect_ok
    print(
        f"[{'PASS' if passed else 'FAIL'}] {name} -> ok={ok} rows={len(rows)} meta_keys={list(meta.keys())}"
    )
    if not passed:
        print(json.dumps(res, indent=2))
    return passed, rows, meta, res


def main():
    total = 0
    fails = 0

    # 0) Smoke: Count only + Rows+Count  (keeps parity with test_dbread_count.py)
    p, *_ = run(
        "Count only (no rows)",
        {
            "table": "events",
            "where": {"topic": {"op": "ilike", "value": "plan.%"}},
            "aggregate": {"count": "*"},
            "limit": 0,
        },
    )
    total += 1
    fails += 0 if p else 1

    p, *_ = run(
        "Rows + count",
        {
            "table": "events",
            "where": {"topic": {"op": "ilike", "value": "plan.%"}},
            "aggregate": {"count": "*"},
            "order": {"field": "created_at", "desc": True},
            "limit": 3,
        },
    )
    total += 1
    fails += 0 if p else 1

    # 1) Offset pagination (deterministic order)
    p, rows1, _, _ = run(
        "Offset page 1 (id ASC)",
        {
            "table": "events",
            "order": {"field": "id", "desc": False},
            "limit": 3,
            "offset": 0,
        },
    )
    total += 1
    fails += 0 if p else 1

    p, rows2, _, _ = run(
        "Offset page 2 (id ASC)",
        {
            "table": "events",
            "order": {"field": "id", "desc": False},
            "limit": 3,
            "offset": 3,
        },
    )
    total += 1
    fails += 0 if p else 1

    # 2) Keyset forward (created_at DESC, id DESC) + next_cursor chaining
    p, rowsA, metaA, resA = run(
        "Keyset page A",
        {
            "table": "events",
            "order": [
                {"field": "created_at", "desc": True},
                {"field": "id", "desc": True},
            ],
            "limit": 5,
        },
    )
    total += 1
    fails += 0 if p else 1

    # Build next request using meta.next_cursor if present; otherwise fall back to last row
    next_cursor = metaA.get("next_cursor") or {}
    if not next_cursor and rowsA:
        # Fallback: derive cursor from last row (compatible with your registry)
        last = rowsA[-1]
        next_cursor = {
            "mode": "keyset",
            "after": {"created_at": last.get("created_at"), "id": last.get("id")},
        }

    p, rowsB, metaB, resB = run(
        "Keyset page B (after A)",
        {
            "table": "events",
            "order": [
                {"field": "created_at", "desc": True},
                {"field": "id", "desc": True},
            ],
            "cursor": next_cursor,
            "limit": 5,
        },
    )
    total += 1
    fails += 0 if p else 1

    # 3) Keyset reverse (before) using tail of A
    before_cursor = {}
    if rowsA:
        tail = rowsA[-1]
        before_cursor = {
            "mode": "keyset",
            "before": {"created_at": tail.get("created_at"), "id": tail.get("id")},
        }

    p, rowsRev, metaRev, resRev = run(
        "Keyset reverse (before tail of A)",
        {
            "table": "events",
            "order": [
                {"field": "created_at", "desc": True},
                {"field": "id", "desc": True},
            ],
            "cursor": before_cursor,
            "limit": 5,
        },
    )
    total += 1
    fails += 0 if p else 1

    # 4) Order w/ NULLS LAST on latency_ms
    p, *_ = run(
        "Order latency_ms ASC NULLS LAST",
        {
            "table": "events",
            "order": [
                {"field": "latency_ms", "desc": False, "nulls": "last"},
                {"field": "id", "desc": False},
            ],
            "limit": 5,
        },
    )
    total += 1
    fails += 0 if p else 1

    # 5) NOT IN (flat path)
    p, *_ = run(
        "NOT IN (flat)",
        {
            "table": "events",
            "where": {
                "topic": {"op": "not_in", "value": ["plan.step.start", "plan.step.end"]}
            },
            "limit": 3,
        },
    )
    total += 1
    fails += 0 if p else 1

    # 6) Serializer special chars in grouped OR (contains dot/percent/comma)
    p, *_ = run(
        "Grouped OR with special chars",
        {
            "table": "events",
            "where": {
                "op": "or",
                "conditions": [
                    {"field": "topic", "op": "contains", "value": "plan.step"},
                    {"field": "topic", "op": "eq", "value": "a,b(c)%"},
                ],
            },
            "limit": 2,
        },
    )
    total += 1
    fails += 0 if p else 1

    # 7) DISTINCT (boolean)
    p, *_ = run(
        "DISTINCT topics",
        {"table": "events", "select": ["topic"], "distinct": True, "limit": 5},
    )
    total += 1
    fails += 0 if p else 1

    print(f"\n{total - fails}/{total} tests passed")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
