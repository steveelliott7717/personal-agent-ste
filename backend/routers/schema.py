from __future__ import annotations
from fastapi import APIRouter, Query
from pydantic import BaseModel
from backend.services import schema_service
from backend.agents.schema_agent import ask_schema

router = APIRouter(prefix="/app/api", tags=["schema"])


@router.get("/db/tables")
def http_list_tables(schema: str = Query(default="public")):
    return {"schema": schema, "tables": schema_service.list_tables(schema)}


@router.get("/db/columns")
def http_list_columns(schema: str = Query(default="public"), table: str | None = None):
    return {
        "schema": schema,
        "table": table,
        "columns": schema_service.list_columns(schema, table),
    }


class AskBody(BaseModel):
    question: str


@router.post("/schema/ask")
def http_schema_ask(body: AskBody):
    answer = ask_schema(body.question.strip())
    return {"question": body.question, "answer": answer}
