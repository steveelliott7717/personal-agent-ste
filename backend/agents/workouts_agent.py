from services.supabase_service import supabase

def handle_workouts(query: str):
    if "log" in query or "add" in query:
        supabase.table("workout_log").insert({"description": query}).execute()
        return "Workout logged."
    elif "show" in query or "list" in query or "history" in query:
        data = supabase.table("workout_log").select("*").execute().data
        return data
    return "Workouts agent ready."
