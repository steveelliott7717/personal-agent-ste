from services.supabase_service import supabase

def handle_meals(query: str):
    if "log" in query:
        supabase.table("meals_log").insert({"description": query}).execute()
        return "Meal logged."
    elif "show" in query or "list" in query or "planned" in query:
        data = supabase.table("meals_log").select("*").execute().data
        return data
    return "Meals agent ready."
