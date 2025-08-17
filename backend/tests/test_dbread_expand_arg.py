#!/usr/bin/env python3
import json, urllib.request

BASE = "http://localhost:8000/app/api/agents/verb"
HDR = {"Content-Type": "application/json"}


def post(p):
    r = urllib.request.Request(
        BASE, data=json.dumps(p).encode(), headers=HDR, method="POST"
    )
    with urllib.request.urlopen(r, timeout=10) as resp:
        return json.loads(resp.read().decode())


def run(name, args):
    res = post({"verb": "db.read", "args": args})
    ok = res.get("ok") is True
    rows = (res.get("result") or {}).get("rows") or []
    print(f"[{'PASS' if ok else 'FAIL'}] {name} -> rows={len(rows)}")
    if not ok:
        print(json.dumps(res, indent=2))
    return ok


if __name__ == "__main__":
    total = 0
    total += run(
        "expand via explicit FK",
        {
            "table": "events",
            "select": ["id", "topic", "created_at"],
            "expand": {"agents!events_agent_id_fkey": ["slug", "title"]},
            "limit": 3,
        },
    )
    print(f"\n{total}/1 tests passed")
