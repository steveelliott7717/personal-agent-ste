# tools/schema_dump_from_app.py
import requests, pathlib

BASE = "https://personal-agent-ste.fly.dev/app/api"  # your live app base


def get_tables():
    r = requests.get(f"{BASE}/db/tables", timeout=30)
    r.raise_for_status()
    return [t["table_name"] for t in r.json()["tables"]]


def get_columns(table):
    r = requests.get(f"{BASE}/db/columns", params={"table": table}, timeout=30)
    r.raise_for_status()
    return r.json()["columns"]


def main():
    lines = ["# Database Schema (public)\n"]
    for t in get_tables():
        lines.append(f"## {t}")
        cols = get_columns(t)
        if not cols:
            lines.append("_No columns returned_")
        else:
            for c in cols:
                nn = "" if c.get("is_nullable") == "YES" else " — not null"
                default = c.get("column_default")
                dftxt = "" if (default in (None, "")) else f" (default: `{default}`)"
                lines.append(f"- **{c['column_name']}** ({c['data_type']}){nn}{dftxt}")
        lines.append("")  # blank line after each table
    out = pathlib.Path("schema_with_columns.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
