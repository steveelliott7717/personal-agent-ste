# backend/routers/shortcuts_ingest.py
from __future__ import annotations
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
import os
from backend.services.supabase_service import supabase

router = APIRouter(prefix="/shortcuts", tags=["shortcuts"])

# Token to authenticate Shortcuts requests (set via Fly secrets)
INGEST_TOKEN = os.getenv("HEALTHKIT_INGEST_TOKEN")
# Supabase table name; defaults to the same one used by health_service.py
HEALTH_TABLE = os.getenv("HEALTH_TABLE_NAME", "health_metrics")


class StepsPayload(BaseModel):
    date: str  # "YYYY-MM-DD"
    steps: int
    device_id: str | None = None  # optional


@router.post("/steps")
def ingest_steps(
    payload: StepsPayload, authorization: str | None = Header(default=None)
):
    # simple bearer-token check
    if not INGEST_TOKEN:
        raise HTTPException(status_code=500, detail="Server ingest token not set")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = authorization.split(" ", 1)[1]
    if token != INGEST_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

    row = {"date": payload.date, "steps": payload.steps, "device_id": payload.device_id}
    # upsert into Supabase (create table beforehand)
    resp = supabase.table(HEALTH_TABLE).upsert(row, on_conflict="date").execute()
    return {"ok": True, "rows": getattr(resp, "data", [])}
