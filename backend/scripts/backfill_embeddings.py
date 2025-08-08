import os
from typing import Iterable, Dict, Any
from services.supabase_service import supabase
from semantics.store import upsert

def fetch_finance() -> Iterable[Dict[str, Any]]:
    rows = (supabase.table("finance_ledger")
            .select("id, description, amount, created_at")
            .order("created_at", desc=False).execute().data) or []
    for r in rows:
        doc_id = f"finance:{r['id']}"
        text = f"ITEM: {r.get('description','').strip()} | PRICE: {r.get('amount',0)} | DATE: {r.get('created_at','')}"
        meta = {"amount": r.get("amount"), "created_at": r.get("created_at")}
        yield {"namespace": "expenses", "doc_id": doc_id, "text": text, "metadata": meta}

def fetch_meals():
    rows = (supabase.table("meals_log")
            .select("id, description, created_at")
            .order("created_at", desc=False).execute().data) or []
    for r in rows:
        doc_id = f"meals:{r['id']}"
        text = f"MEAL: {r.get('description','').strip()} | DATE: {r.get('created_at','')}"
        meta = {"created_at": r.get("created_at")}
        yield {"namespace": "meals", "doc_id": doc_id, "text": text, "metadata": meta}

def fetch_workouts():
    rows = (supabase.table("workout_log")
            .select("id, description, created_at")
            .order("created_at", desc=False).execute().data) or []
    for r in rows:
        doc_id = f"workouts:{r['id']}"
        text = f"WORKOUT: {r.get('description','').strip()} | DATE: {r.get('created_at','')}"
        meta = {"created_at": r.get("created_at")}
        yield {"namespace": "workouts", "doc_id": doc_id, "text": text, "metadata": meta}

def fetch_grooming():
    rows = (supabase.table("grooming_log")
            .select("id, description, created_at")
            .order("created_at", desc=False).execute().data) or []
    for r in rows:
        doc_id = f"grooming:{r['id']}"
        text = f"GROOMING: {r.get('description','').strip()} | DATE: {r.get('created_at','')}"
        meta = {"created_at": r.get("created_at")}
        yield {"namespace": "grooming", "doc_id": doc_id, "text": text, "metadata": meta}

SOURCES = [fetch_finance, fetch_meals, fetch_workouts, fetch_grooming]

def main():
    total = 0
    for src in SOURCES:
        for row in src():
            upsert(row["namespace"], row["doc_id"], row["text"], row["metadata"])
            total += 1
    print(f"Backfill complete. Upserted {total} embedding rows.")

if __name__ == "__main__":
    for k in ("SUPABASE_URL", "SUPABASE_KEY", "COHERE_API_KEY"):
        if not os.getenv(k):
            print(f"WARNING: {k} not set")
    main()
