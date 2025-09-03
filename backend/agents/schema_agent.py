from __future__ import annotations
import json
import os
from openai import OpenAI
from backend.services import schema_service

_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_SYSTEM = (
    "You can answer questions about my Supabase schema by calling tools:\n"
    "- list_tables(schema='public')\n"
    "- list_columns(schema='public', table=None)\n"
    "Rules:\n"
    "- 'What tables exist?' → list_tables\n"
    "- 'What columns are in X?' → list_columns(schema='public', table='X')\n"
    "- 'Show all tables and their columns' → list_tables, then list_columns for each table.\n"
    "After tools return, summarize concisely in markdown tables."
)

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "List base tables in a schema",
            "parameters": {
                "type": "object",
                "properties": {"schema": {"type": "string", "default": "public"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_columns",
            "description": "List columns for a table or all tables in a schema",
            "parameters": {
                "type": "object",
                "properties": {
                    "schema": {"type": "string", "default": "public"},
                    "table": {"type": "string"},
                },
                "required": [],
            },
        },
    },
]


def ask_schema(question: str) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": question},
    ]

    resp = _client.chat.completions.create(
        model=_MODEL,
        messages=messages,
        tools=_TOOLS,
        tool_choice="auto",
    )
    msg = resp.choices[0].message

    # Handle tool calls (function calling)
    while getattr(msg, "tool_calls", None):
        for call in msg.tool_calls:
            fn = call.function.name
            args = json.loads(call.function.arguments or "{}")
            if fn == "list_tables":
                data = schema_service.list_tables(schema=args.get("schema", "public"))
            elif fn == "list_columns":
                data = schema_service.list_columns(
                    schema=args.get("schema", "public"), table=args.get("table")
                )
            else:
                data = {"error": f"unknown tool {fn}"}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": fn,
                    "content": json.dumps(data),
                }
            )

        resp = _client.chat.completions.create(
            model=_MODEL,
            messages=messages,
        )
        msg = resp.choices[0].message

    return msg.content or ""
