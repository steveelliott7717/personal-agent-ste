from services.supabase_service import supabase

def handle_bill_payer(query: str):
    if "pay" in query or "approve" in query:
        supabase.table("finance_ledger").insert({"description": query, "amount": 0}).execute()
        return "Bill payment logged."
    elif "pending" in query:
        return "No pending bills."
    return "Bill payer agent ready."
