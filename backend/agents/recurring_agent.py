from backend.services.supabase_service import supabase

def handle_recurring(query: str):
    if "add" in query or "log" in query:
        supabase.table("recurring_orders").insert({"description": query}).execute()
        return "Recurring order logged."
    elif "list" in query or "show" in query:
        data = supabase.table("recurring_orders").select("*").execute().data
        return data
    return "Recurring orders agent ready."
