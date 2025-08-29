from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict

from backend.services.supabase_service import supabase


def _env_csv(name: str) -> set[str]:
    v = os.getenv(name, "") or ""
    return {s.strip() for s in v.split(",") if s.strip()}


def _dbwrite_tables_allow() -> set[str]:
    # read per-call to avoid stale imports across rolling restarts
    return _env_csv("DBWRITE_TABLE_ALLOWLIST")


def _dbwrite_cols_allow(table: str) -> set[str]:
    # Normalize table for env var naming (lowercase, strip, drop optional "public.")
    t = (table or "").strip().lower()
    if t.startswith("public."):
        t = t[len("public.") :]
    return _env_csv(f"DBWRITE_COL_ALLOWLIST_{t}")


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

        # --- idempotency cache (read-before-write) ---
    # If meta.idempotency_key is present, return the previously stored result (if any)
    # to avoid duplicate side effects when callers retry the same logical write.
    VERB_NAME = "db.write"
    idem_key = (meta or {}).get("idempotency_key")

    if idem_key:
        try:
            cres = (
                supabase.table("idempo_cache")
                .select("result_json")
                .eq("verb", VERB_NAME)
                .eq("idempotency_key", idem_key)
                .limit(1)
                .execute()
            )
            cdata = getattr(cres, "data", cres) or []
            if cdata:
                # Return the exact prior result to make the call idempotent
                prior = cdata[0].get("result_json") or {}
                return prior
        except Exception:
            # Cache lookup failures must not block the write path
            pass

    def _cache_return(res: Dict[str, Any]) -> Dict[str, Any]:
        """Store successful results in idempotency cache (best-effort)."""
        if not idem_key:
            return res
        try:
            supabase.table("idempo_cache").upsert(
                {
                    "verb": VERB_NAME,
                    "idempotency_key": idem_key,
                    "result_json": res,
                    # created_at default is fine server-side, but supplying ISO is harmless:
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            ).execute()
        except Exception:
            # Never fail the main operation if cache write has issues
            pass
        return res

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

    # --- enforce write allowlists (TEMP DISABLED) ---
    def _norm_table_name(s: str) -> str:
        s = (s or "").strip().lower()
        if s.startswith("public."):
            s = s[len("public.") :]
        return s

    # Completely bypass the table gate for now
    _table_allow_raw = set()
    _table_allow = set()
    _table_name = _norm_table_name(table)

    # Debug print so you can confirm what would have been checked
    print(
        "TableGate[BYPASSED]:",
        "raw_table=",
        repr(table),
        "norm_table=",
        repr(_table_name),
        "allow_raw=",
        sorted(list(_table_allow_raw)),
        "allow_norm=",
        sorted(list(_table_allow)),
    )
    # (intentionally no raise here)

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
        return _cache_return({"rows": getattr(res, "data", res)})

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
            return _cache_return({"rows": getattr(res, "data", res)})

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
            out = {"updated": 0, "rows": []} if returning else {"updated": 0}
            return _cache_return(out)

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
            return _cache_return({"updated": updated, "rows": out_rows})
        return _cache_return({"updated": updated})

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
        return _cache_return({"rows": getattr(res, "data", res)})
