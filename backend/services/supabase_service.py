import os
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

TABLES = {
    "finance_ledger": [
        {"name": "description", "type": "text"},
        {"name": "amount", "type": "numeric"}
    ],
    "workout_log": [
        {"name": "description", "type": "text"}
    ],
    "meals_log": [
        {"name": "description", "type": "text"}
    ],
    "grooming_log": [
        {"name": "description", "type": "text"}
    ],
    "recurring_orders": [
        {"name": "description", "type": "text"}
    ],
    "notifications": [
        {"name": "message", "type": "text"}
    ],
    "router_logs": [
        {"name": "query", "type": "text"},
        {"name": "agent", "type": "text"},
        {"name": "result", "type": "text"},
        {"name": "source", "type": "text"}
    ]
}

def init_tables():
    # In Supabase, you'd run SQL migrations — placeholder here
    pass

def insert_router_log(query, agent, result, source):
    supabase.table("router_logs").insert({
        "query": query,
        "agent": agent,
        "result": str(result),
        "source": source
    }).execute()
