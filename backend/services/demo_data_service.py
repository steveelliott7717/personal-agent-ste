# backend/services/demo_data_service.py
from typing import List, Dict
from postgrest import APIError
from backend.services.supabase_service import supabase


def _table_count(table: str) -> int:
    """
    Returns total rows in a table (fast HEAD request with exact count).
    If table doesn't exist yet, return 0 and let init_tables() handle creation.
    """
    try:
        res = supabase.table(table).select("id", count="exact", head=True).execute()
        # The Python client surfaces count on the response object
        return getattr(res, "count", 0) or 0
    except APIError as e:
        # If relation doesn’t exist, treat as empty (init_tables should create it)
        if "does not exist" in str(e):
            return 0
        raise


def _ensure_seed(table: str, rows: List[Dict]):
    """
    Seed rows only if the table is empty.
    """
    if _table_count(table) == 0 and rows:
        supabase.table(table).insert(rows).execute()
        print(f"Seeded demo rows into {table} ({len(rows)} rows)")
    else:
        print(f"Skipped seeding {table}: already has data")


def seed_demo_data_once():
    """
    Idempotent seeding: runs every startup but only inserts when tables are empty.
    Safe to call repeatedly.
    """
    # Demo rows you can tweak any time
    finance_rows = [
        {"description": "Demo Grocery Purchase", "amount": 45.67},
        {"description": "Demo Coffee Shop", "amount": 4.50},
        {"description": "Demo Ride Share", "amount": 12.80},
    ]
    meals_rows = [
        {"description": "Demo Breakfast - Oatmeal"},
        {"description": "Demo Lunch - Chicken Salad"},
        {"description": "Demo Dinner - Salmon and Rice"},
    ]
    workouts_rows = [
        {"description": "Demo Workout - 5km Run"},
        {"description": "Demo Workout - Push/Pull/Legs"},
    ]
    grooming_rows = [
        {"description": "Demo Grooming - Sunscreen"},
        {"description": "Demo Grooming - Moisturizer"},
    ]
    recurring_rows = [
        {"description": "Demo Coffee Subscription"},
        {"description": "Demo Protein Powder Subscription"},
    ]

    # Only seed if table is empty
    _ensure_seed("finance_ledger", finance_rows)
    _ensure_seed("meals_log", meals_rows)
    _ensure_seed("workout_log", workouts_rows)
    _ensure_seed("grooming_log", grooming_rows)
    _ensure_seed("recurring_orders", recurring_rows)
