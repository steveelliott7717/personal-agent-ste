# Simple micro-plan registry
MICRO_PLANS = {
    "quick.health_check": [
        {"verb": "time.now"},
        {"verb": "notify.push", "args": {"channel": "health", "message": "ping"}},
    ],
    "db.ping": [
        {"verb": "db.read", "args": {"table": "agents", "limit": 1}},
    ],
}
