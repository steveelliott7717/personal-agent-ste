#!/usr/bin/env python3
import json, urllib.request, time

BASE = "http://localhost:8000/app/api/agents/verb"
HEADERS = {"Content-Type": "application/json"}


def post(p):
    req = urllib.request.Request(
        BASE, data=json.dumps(p).encode(), headers=HEADERS, method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


def run(name, args, expect_ok=True, require_rows=True):
    res = post({"verb": "db.read", "args": args})
    ok = res.get("ok")
    rows = (res.get("result", {}) or {}).get("rows") or []
    passed = (ok is expect_ok) and (len(rows) > 0 if require_rows else True)
    print(f"[{'PASS' if passed else 'FAIL'}] {name} -> ok={ok} rows={len(rows)}")
    if not passed:
        print(json.dumps(res, indent=2))
    return passed


if __name__ == "__main__":
    total = 0
    # show embed objects; relies on your FK name + the one good test row you inserted
    total += run(
        "Embed agent via FK name",
        {
            "table": "events",
            "select": "id,topic,created_at,agent:agents!events_agent_id_fkey(slug,title)",
            "order": {"field": "created_at", "desc": True},
            "limit": 3,
        },
        expect_ok=True,
        require_rows=True,
    )
    print(f"\n{total}/1 tests passed")
