from services.firebase_service import send_push_notification

def handle_notifications(query: str):
    if "send" in query or "alert" in query:
        send_push_notification("Notification", query)
        return "Notification sent."
    return "Notifications agent ready."
