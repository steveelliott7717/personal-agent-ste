import os
import requests

FCM_SERVER_KEY = os.getenv("FCM_SERVER_KEY")


def send_push_notification(title, body):
    headers = {
        "Authorization": f"key={FCM_SERVER_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"to": "/topics/all", "notification": {"title": title, "body": body}}
    r = requests.post(
        "https://fcm.googleapis.com/fcm/send", json=payload, headers=headers
    )
    return r.json()
