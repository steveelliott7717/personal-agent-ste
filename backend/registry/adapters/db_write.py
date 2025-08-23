from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict

from backend.services.supabase_service import supabase


def _env_csv(name: str) -> set[str]:
    v = os.getenv(name, "") or ""
    return {s.strip() for s in v.split(",") if s.strip()}


_DBWRITE_TABLE_ALLOW = _env_csv("DBWRITE_TABLE_ALLOWLIST")


def _dbwrite_cols_allow(table: str) -> set[str]:
    # e.g., DBWRITE_COL_ALLOWLIST_workout_plan
    return _env_csv(f"DBWRITE_COL_ALLOWLIST_{table}")


def db_write_adapter(args: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    args:
      table: str (required)
      mode: "insert" | "upsert" | "update" | "delete" (required)
      rows: list[dict] (required for insert/upsert; optional single-row for update)
      values: dict (optional for update; preferred for inc/dec)
      where: dict[str, Any] target selector (required for update/delete)
      returning: bool = True (optional)
      pk: str = "id" (optional)  # primary key used for row-by-row updates
    """
    table = args.get("table")
    mode = args.get("mode")
    rows = args.get("rows")
    where = args.get("where", {}) or {}
    returning = bool(args.get("returning", True))

    if not table or not isinstance(table, str):
        raise ValueError("args.table (str) is required")
    if mode not in {"insert", "upsert", "update", "delete"}:
        raise ValueError("args.mode must be one of insert|upsert|update|delete")

    def _collect_write_columns(mode: str, rows, values) -> set[str]:
        cols: set[str] = set()
        if mode in {"insert", "upsert"}:
            if isinstance(rows, list):
                for r in rows:
                    if isinstance(r, dict):
                        cols |= set(r.keys())
        elif mode == "update":
            if isinstance(values, dict) and values:
                cols |= set(values.keys())
            elif (
                isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict)
            ):
                cols |= set(rows[0].keys())
        # deletes don’t set columns
        return cols

    # --- enforce write allowlists ---
    if _DBWRITE_TABLE_ALLOW and table not in _DBWRITE_TABLE_ALLOW:
        raise ValueError(f"PolicyDenied: writes to '{table}' are not allowed")

    # Figure out which columns this write would touch
    write_cols = _collect_write_columns(mode, rows, args.get("values"))
    allowed_cols = _dbwrite_cols_allow(table)
    if allowed_cols:
        # Ignore meta/selector fields that aren't actually written
        # (pk/where keys aren't in write_cols because we only collect set columns)
        bad = {c for c in write_cols if c not in allowed_cols}
        if bad:
            raise ValueError(
                f"PolicyDenied: writing columns {sorted(bad)} on '{table}' is not allowed; "
                f"allowed: {sorted(allowed_cols)}"
            )

    tbl = supabase.table(table)

    # ---------- INSERT / UPSERT ----------
    if mode in {"insert", "upsert"}:
        if not isinstance(rows, list) or not rows:
            raise ValueError(
                "args.rows (non-empty list[dict]) is required for insert/upsert"
            )
        op = tbl.upsert(rows) if mode == "upsert" else tbl.insert(rows)
        if not returning:
            op = op.select("*", count="exact").limit(0)  # cheap no-return
        res = op.execute()
        return {"rows": getattr(res, "data", res)}

    # ---------- UPDATE (supports inc/dec) ----------
    if mode == "update":
        if not isinstance(where, dict) or not where:
            raise ValueError("args.where (dict) is required for update")

        # Accept either rows=[{...}] or values={...}
        base_vals: Dict[str, Any] = {}
        if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
            base_vals = rows[0]
        if isinstance(args.get("values"), dict):
            # values takes precedence if both provided
            base_vals = args["values"]

        if not base_vals:
            raise ValueError(
                "update requires either args.values (dict) or args.rows=[{...}]"
            )

        # ---- helpers for math updates ----
        def _split_math_updates(values: dict[str, object]) -> tuple[dict, dict, dict]:
            """
            Split values into (assignments, increments, decrements).
            assignments: {"col": scalar}
            increments:  {"col": +delta}
            decrements:  {"col": -delta}
            """
            assigns, incs, decs = {}, {}, {}
            for k, v in (values or {}).items():
                if isinstance(v, dict) and "op" in v:
                    op = (v.get("op") or "").lower()
                    delta = v.get("value")
                    if op == "inc":
                        if not isinstance(delta, (int, float)):
                            raise ValueError(f"inc for {k} requires numeric value")
                        incs[k] = float(delta)
                    elif op == "dec":
                        if not isinstance(delta, (int, float)):
                            raise ValueError(f"dec for {k} requires numeric value")
                        decs[k] = float(delta)
                    else:
                        raise ValueError(f"unsupported math op for {k}: {op}")
                else:
                    assigns[k] = v
            return assigns, incs, decs

        def _apply_math(assigns: dict, incs: dict, decs: dict, row: dict) -> dict:
            """
            Apply inc/dec onto a row snapshot; direct assigns win last.
            """
            out = dict(row)  # shallow copy
            for col, d in incs.items():
                cur = out.get(col)
                if cur is None:
                    out[col] = d
                elif isinstance(cur, (int, float)):
                    out[col] = float(cur) + d
                else:
                    raise ValueError(f"cannot inc non-numeric column {col}: {cur!r}")
            for col, d in decs.items():
                cur = out.get(col)
                if cur is None:
                    out[col] = -d
                elif isinstance(cur, (int, float)):
                    out[col] = float(cur) - d
                else:
                    raise ValueError(f"cannot dec non-numeric column {col}: {cur!r}")
            out.update(assigns)
            return out

        def _guess_pk(a: dict) -> str:
            pk = a.get("pk")
            return str(pk) if pk else "id"

        assigns, incs, decs = _split_math_updates(base_vals)

        # Fast path: no math → your old update logic
        if not incs and not decs:
            q = tbl.update(assigns)
            for col, val in where.items():
                q = q.eq(col, val)
            if not returning:
                q = q.select("*", count="exact").limit(0)
            res = q.execute()
            return {"rows": getattr(res, "data", res)}

        # Math present: read-modify-write per row, optimistic lock on updated_at when present
        pk_col = _guess_pk(args)

        # 1) Read candidates
        #    We need pk, updated_at (if exists), and all touched columns
        read_cols = {pk_col, "updated_at", *assigns.keys(), *incs.keys(), *decs.keys()}
        select_list = ",".join(sorted(c for c in read_cols if c))
        rq = tbl.select(select_list)
        for col, val in where.items():
            rq = rq.eq(col, val)
        rres = rq.execute()
        rows_in = getattr(rres, "data", rres) or []
        if not rows_in:
            return {"updated": 0, "rows": []} if returning else {"updated": 0}

        updated = 0
        out_rows: list[dict] = []

        for r in rows_in:
            # compute final assignments for this row
            computed = _apply_math(assigns, incs, decs, r)

            patch: Dict[str, Any] = {}
            for k in set(assigns.keys()) | set(incs.keys()) | set(decs.keys()):
                if k in (pk_col, "updated_at"):
                    continue
                if k in computed:
                    patch[k] = computed[k]

            if not patch:
                continue

            uq = supabase.table(table)
            uq = uq.eq(pk_col, r[pk_col])
            if "updated_at" in r and r["updated_at"] is not None:
                uq = uq.eq("updated_at", r["updated_at"])

            ures = uq.update(patch, returning=True).execute()
            udata = getattr(ures, "data", ures) or []
            if udata:
                updated += 1
                out_rows.extend(udata)
            else:
                # optimistic lock failed or row vanished; skip (or collect conflicts)
                pass

        if returning:
            return {"updated": updated, "rows": out_rows}
        return {"updated": updated}

    # ---------- DELETE ----------
    if mode == "delete":
        if not isinstance(where, dict) or not where:
            raise ValueError("args.where (dict) is required for delete")
        q = tbl.delete()
        for col, val in where.items():
            q = q.eq(col, val)
        if not returning:
            q = q.select("*", count="exact").limit(0)
        res = q.execute()
        return {"rows": getattr(res, "data", res)}
