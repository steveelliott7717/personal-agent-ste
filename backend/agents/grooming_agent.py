from services.supabase_service import supabase

def handle_grooming(query: str):
    if "log" in query or "done" in query or "complete" in query:
        supabase.table("grooming_log").insert({"description": query}).execute()
        return "Grooming task logged."
    elif "show" in query or "list" in query or "today" in query:
        data = supabase.table("grooming_log").select("*").execute().data
        return data
    return "Grooming agent ready."
