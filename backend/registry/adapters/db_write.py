from __future__ import annotations

import os
import time
import logging
import json
from datetime import datetime, timezone
from typing import Any, Dict, Tuple, List, Optional

from backend.services.supabase_service import supabase

# ------------------------------------------------------------------------------
# Optional hardening toggles
# ------------------------------------------------------------------------------
REQUIRE_WHERE = os.getenv("DBWRITE_REQUIRE_WHERE_FOR_UPDATE_DELETE", "0") == "1"

# ------------------------------------------------------------------------------
# Try both module layouts (monorepo vs local); fall back to shims if both fail.
# ------------------------------------------------------------------------------
_READ_APPLY_WHERE = _READ_APPLY_ORDER = _READ_APPLY_LIMIT = None
try:
    from backend.registry.adapters.db_read import (
        _apply_where as _READ_APPLY_WHERE,
        _apply_order as _READ_APPLY_ORDER,
        _apply_limit as _READ_APPLY_LIMIT,
    )
except Exception:
    try:
        from backend.registry.adapters.db_read import (
            _apply_where as _READ_APPLY_WHERE,
            _apply_order as _READ_APPLY_ORDER,
            _apply_limit as _READ_APPLY_LIMIT,
        )
    except Exception:
        pass  # keep local fallbacks


# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logger = logging.getLogger("db.write")


def _log_event(action: str, table: str, mode: str, meta: dict, extra: dict):
    payload = {
        "action": action,
        "table": table,
        "mode": mode,
        "correlation_id": (meta or {}).get("correlation_id"),
        "idempotency_key": (meta or {}).get("idempotency_key"),
        **(extra or {}),
    }
    logger.info("db.write", extra={"event": payload})


# ------------------------------------------------------------------------------
# helpers: parsing / normalization / safety / projection
# ------------------------------------------------------------------------------
_ALLOWED_MODES = {"insert", "upsert", "update", "delete"}
_ALLOWED_RETURNING = {"none", "minimal", "representation"}
_ALLOWED_COUNTS = {None, "exact", "planned", "estimated"}


def _project_rows(rows: list[dict] | None, select_csv: str | None) -> list[dict]:
    if not rows:
        return []
    if not select_csv or select_csv.strip() == "*":
        return rows
    keep = [c.strip() for c in select_csv.split(",") if c.strip()]
    if not keep:
        return rows
    out = []
    for r in rows:
        out.append({k: r.get(k) for k in keep})
    return out


def _needs_embedded_projection(select_csv: Optional[str]) -> bool:
    # detect PostgREST relation syntax like fk(*), alias:rel(*), nested(...)
    return bool(select_csv and "(" in select_csv)


def _embedded_projection_by_pk(
    table: str, select_csv: str, pk_col: str, rows: list[dict]
) -> list[dict]:
    ids = [r.get(pk_col) for r in (rows or []) if r.get(pk_col) is not None]
    if not ids:
        return _project_rows(rows, select_csv)
    q = supabase.table(table).select(select_csv).in_(pk_col, ids)
    resp = _exec(q)
    data = getattr(resp, "data", None) or []
    by_id = {r.get(pk_col): r for r in data if r.get(pk_col) is not None}
    return [by_id[i] for i in ids if i in by_id]


def _normalize_returning(v: Any) -> str:
    if isinstance(v, bool):
        return "representation" if v else "minimal"
    s = v or "representation"
    if isinstance(s, str):
        s = s.strip().lower()
    if s not in _ALLOWED_RETURNING:
        raise ValueError("args.returning must be one of none|minimal|representation")
    return s


def _normalize_select(sel: Any) -> Optional[str]:
    if sel is None:
        return None
    if isinstance(sel, str):
        return sel.strip() or None
    if isinstance(sel, (list, tuple)):
        flat: List[str] = []
        for it in sel:
            if not isinstance(it, str):
                raise ValueError("args.select must be a CSV string or list[str]")
            it = it.strip()
            if it:
                flat.append(it)
        return ",".join(flat) if flat else None
    raise ValueError("args.select must be a CSV string or list[str]")


def _coerce_rows_for_write(
    mode: str, rows: Any
) -> Tuple[Optional[List[dict]], Optional[dict]]:
    if mode in ("insert", "upsert"):
        if rows is None:
            raise ValueError("args.rows is required for insert/upsert")
        if isinstance(rows, dict):
            return [rows], None
        if isinstance(rows, list) and all(isinstance(x, dict) for x in rows):
            if not rows:
                raise ValueError("args.rows must be non-empty for insert/upsert")
            return rows, None
        raise ValueError(
            "args.rows must be an object or array of objects for insert/upsert"
        )
    if mode == "update":
        if rows is None:
            raise ValueError("args.rows (patch) is required for update")
        if isinstance(rows, dict):
            if not rows:
                raise ValueError("args.rows (patch) must not be empty for update")
            return None, rows
        raise ValueError("args.rows must be a single object (patch) for update")
    return None, None  # delete


def _env_csv(name: str) -> set[str]:
    v = os.getenv(name, "") or ""
    return {s.strip() for s in v.split(",") if s.strip()}


def _dbwrite_tables_allow() -> set[str]:
    return _env_csv("DBWRITE_TABLE_ALLOWLIST")


def _dbwrite_cols_allow(table: str) -> set[str]:
    t = (table or "").strip().lower()
    if t.startswith("public."):
        t = t[len("public.") :]
    return _env_csv(f"DB_WRITE_COL_ALLOWLIST_{t}") or _env_csv(
        f"DBWRITE_COL_ALLOWLIST_{t}"
    )


def _enforce_table_allowlist(table: str) -> None:
    allow = _dbwrite_tables_allow()
    if allow:
        t = (table or "").strip().lower()
        if t.startswith("public."):
            t = t[len("public.") :]
        if t not in allow:
            raise ValueError(f"table '{table}' not allowed (DBWRITE_TABLE_ALLOWLIST)")


def _enforce_column_allowlist(
    table: str, rows_list: Optional[List[dict]], patch: Optional[dict]
) -> None:
    allow = _dbwrite_cols_allow(table)
    if not allow:
        return

    def check_obj(d: dict):
        illegal = [k for k in d.keys() if k not in allow]
        if illegal:
            raise ValueError(
                f"columns {illegal} not allowed for table '{table}' "
                f"(env DB_WRITE_COL_ALLOWLIST_{(table or '').strip().lower()})"
            )

    if rows_list:
        for r in rows_list:
            check_obj(r)
    if patch:
        check_obj(patch)


def _normalize_count(v: Any) -> Optional[str]:
    s = v or None
    if s is None:
        return None
    if isinstance(s, str):
        s = s.strip().lower()
    if s not in _ALLOWED_COUNTS:
        raise ValueError("args.count must be one of exact|planned|estimated")
    return s


def _require_on_conflict_if_upsert(mode: str, on_conflict: Any) -> Optional[List[str]]:
    if mode != "upsert":
        return None
    if (
        not on_conflict
        or not isinstance(on_conflict, list)
        or not all(isinstance(c, str) and c.strip() for c in on_conflict)
    ):
        raise ValueError("args.on_conflict (list[str]) is required for upsert")
    return [c.strip() for c in on_conflict]


# ------------------------------------------------------------------------------
# local fallbacks for where/order/limit if db_read helpers aren't importable
# ------------------------------------------------------------------------------
def _fallback_apply_where(qh, where: Any):
    if not where:
        return qh

    def apply_cond(q, cond):
        if "field" not in cond:
            for k, v in cond.items():
                q = q.eq(k, v)
            return q
        field = cond["field"]
        op = str(cond.get("op", "eq")).lower()
        val = cond.get("value")
        if op == "eq":
            return q.eq(field, val)
        if op == "in":
            return q.in_(field, val if isinstance(val, (list, tuple)) else [val])
        if op == "is":
            return q.is_(field, val)
        return q.eq(field, val)

    if isinstance(where, dict) and "op" not in where and "conditions" not in where:
        for k, v in where.items():
            qh = qh.eq(k, v)
        return qh
    if isinstance(where, dict) and "op" in where and "conditions" in where:
        conds = where.get("conditions") or []
        # Safe approximation: apply all as AND if we don't have real .or_ serialization
        for c in conds:
            qh = apply_cond(qh, c if isinstance(c, dict) else {})
        return qh
    raise ValueError(
        "complex where requires db.read serializer; import failed in db_write.py"
    )


def _fallback_apply_order(qh, order: Any):
    if not order:
        return qh
    seq = order if isinstance(order, list) else [order]
    for item in seq:
        if not isinstance(item, dict) or "field" not in item:
            continue
        field = item["field"]
        desc = bool(item.get("desc", False))
        nulls = item.get("nulls", None)
        try:
            if nulls is not None:
                qh = qh.order(field, desc=desc, nulls=nulls)
            else:
                qh = qh.order(field, desc=desc)
        except TypeError:
            qh = qh.order(field, desc=desc)
    return qh


def _fallback_apply_limit(qh, limit: Any):
    if isinstance(limit, int) and limit > 0:
        qh = qh.limit(limit)
    return qh


def _apply_where(qh, where: Any):
    if _READ_APPLY_WHERE:
        return _READ_APPLY_WHERE(qh, where)  # type: ignore
    return _fallback_apply_where(qh, where)


def _apply_order(qh, order: Any):
    if _READ_APPLY_ORDER:
        return _READ_APPLY_ORDER(qh, order)  # type: ignore
    return _fallback_apply_order(qh, order)


def _apply_limit(qh, limit: Any):
    if _READ_APPLY_LIMIT:
        return _READ_APPLY_LIMIT(qh, limit)  # type: ignore
    return _fallback_apply_limit(qh, limit)


# ------------------------------------------------------------------------------
# math update helpers
# ------------------------------------------------------------------------------
def _split_math_updates(values: dict[str, object]) -> tuple[dict, dict, dict]:
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


# ------------------------------------------------------------------------------
# Friendly execute wrapper (maps common Postgres errors)
# ------------------------------------------------------------------------------
try:
    from postgrest.exceptions import APIError  # type: ignore
except Exception:  # pragma: no cover
    APIError = Exception  # fallback


def _exec(q):
    try:
        return q.execute()
    except APIError as e:  # supabase-py wraps PostgREST errors
        # Normalize error payload to a dict (can be dict or string)
        first = e.args[0] if getattr(e, "args", None) else None
        if isinstance(first, dict):
            err_obj = first
        elif isinstance(first, str):
            try:
                parsed = json.loads(first)
                err_obj = parsed if isinstance(parsed, dict) else {"message": first}
            except Exception:
                err_obj = {"message": first}
        else:
            err_obj = {"message": str(e)}

        code = err_obj.get("code")
        msg = err_obj.get("message") or str(e)

        # Friendly mappings
        if code == "23505":  # unique_violation
            raise ValueError(f"unique violation: {msg}")
        if code == "23503":  # foreign_key_violation
            raise ValueError(f"foreign key violation: {msg}")
        if code == "23502":  # not_null_violation
            raise ValueError(f"not-null violation: {msg}")
        if code == "22P02":  # invalid_text_representation
            raise ValueError(f"invalid input: {msg}")
        if code == "42703":  # undefined_column
            raise ValueError(f"invalid column: {msg}")
        if code == "21000":  # WHERE required on write
            raise ValueError(f"unsafe write: {msg}")
        if code == "23514":  # check_violation
            raise ValueError(f"check violation: {msg}")

        # Unknown shape/code → let the registry envelope capture it
        raise


# ------------------------------------------------------------------------------
# Adapter
# ------------------------------------------------------------------------------
def db_write_adapter(args: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parity write adapter with modes: insert|upsert|update|delete, on_conflict for upsert,
    where/order/limit for update/delete (reusing db.read serializer), returning policy
    none|minimal|representation (+optional select), count hint, table/column allowlists,
    idempotency cache, embedded-select follow-up, and math update helpers.
    """
    t0 = time.time()

    # ---- required & basic ----
    table = args.get("table")
    if not table or not isinstance(table, str):
        raise ValueError("args.table (str) is required")
    table = table.strip()

    mode_in = args.get("mode") or "insert"
    mode = mode_in.strip().lower() if isinstance(mode_in, str) else mode_in
    if mode not in _ALLOWED_MODES:
        raise ValueError("args.mode must be one of insert|upsert|update|delete")

    # rows/patch normalization per mode
    rows_list, patch = _coerce_rows_for_write(mode, args.get("rows"))

    # returning / select / count
    returning = _normalize_returning(args.get("returning"))
    select_csv = _normalize_select(args.get("select"))
    count_mode = _normalize_count(args.get("count"))

    # where/order/limit (raw; passed to db.read serializers)
    where = args.get("where") or {}
    order = args.get("order")
    limit = args.get("limit")

    # upsert extras
    on_conflict = _require_on_conflict_if_upsert(mode, args.get("on_conflict"))
    ignore_duplicates = bool(args.get("ignore_duplicates", False))

    # ---- safety rails ----
    if os.getenv("DBWRITE_DISABLE_TABLE_GUARD", "0") != "1":
        _enforce_table_allowlist(table)
        _enforce_column_allowlist(table, rows_list, patch)

    if REQUIRE_WHERE and mode in {"update", "delete"}:
        if not where or (isinstance(where, dict) and not where):
            raise ValueError(
                "unsafe write: update/delete requires a non-empty 'where' (DBWRITE_REQUIRE_WHERE_FOR_UPDATE_DELETE=1)"
            )

    # ---- idempotency (read-before-write) ----
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
            )
            cres = _exec(cres)
            cdata = getattr(cres, "data", cres) or []
            if cdata:
                prior = cdata[0].get("result_json") or {}
                return prior
        except Exception:
            pass  # never block on cache failure

    def _cache_return(res: Dict[str, Any]) -> Dict[str, Any]:
        if not idem_key:
            return res
        try:
            _exec(
                supabase.table("idempo_cache").upsert(
                    {
                        "verb": VERB_NAME,
                        "idempotency_key": idem_key,
                        "result_json": res,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
            )
        except Exception:
            pass
        return res

    def _done(
        payload: Dict[str, Any],
        *,
        action: str,
        affected: Optional[int] = None,
        two_phase: bool = False,
    ):
        duration_ms = int((time.time() - t0) * 1000)
        _log_event(
            action,
            table,
            mode,
            meta,
            {
                "affected": affected,
                "two_phase": bool(two_phase),
                "duration_ms": duration_ms,
                "returning": returning,
                "count_mode": count_mode,
            },
        )
        return _cache_return(payload)

    # ---- count helper ----
    def _maybe_apply_count_meta(result_payload: Dict[str, Any], resp_obj):
        if count_mode:
            count_val = getattr(resp_obj, "count", None)
            result_payload.setdefault("meta", {})["count"] = count_val
        return result_payload

    # ------------------------------------------------------------------------------
    # INSERT / UPSERT
    # ------------------------------------------------------------------------------
    if mode in {"insert", "upsert"}:
        if not rows_list:
            raise ValueError(
                "args.rows (non-empty list[dict]) is required for insert/upsert"
            )

        ret = "minimal" if returning in {"none", "minimal"} else "representation"

        if mode == "upsert":
            if not on_conflict:
                raise ValueError("args.on_conflict (list[str]) is required for upsert")
            qh = supabase.table(table).upsert(
                rows_list,
                on_conflict=",".join(on_conflict),
                ignore_duplicates=ignore_duplicates,
                returning=ret,
                count=count_mode,
            )
        else:
            qh = supabase.table(table).insert(
                rows_list,
                returning=ret,
                count=count_mode,
            )

        resp = _exec(qh)
        rows = getattr(resp, "data", None)

        if ret == "minimal":
            out_rows = []
        else:
            if _needs_embedded_projection(select_csv):
                pk_col = _guess_pk(args)
                out_rows = _embedded_projection_by_pk(
                    table, select_csv or "*", pk_col, rows or []
                )
            else:
                out_rows = _project_rows(rows, select_csv)

        out = {"rows": out_rows}
        _maybe_apply_count_meta(out, resp)
        return _done(
            out,
            action="success",
            affected=(
                len(out_rows)
                if returning == "representation"
                else getattr(resp, "count", None)
            ),
        )

    # ------------------------------------------------------------------------------
    # UPDATE
    # ------------------------------------------------------------------------------
    if mode == "update":
        # Accept either patch (from rows) or args.values/patch alias
        values = args.get("values")
        patch_arg = args.get("patch")
        base_patch: Dict[str, Any] = (
            patch or patch_arg or (values if isinstance(values, dict) else {})
        )
        if not base_patch:
            raise ValueError(
                "update requires args.rows (patch object) or args.values (dict)"
            )

        # Split into math + assigns
        assigns, incs, decs = _split_math_updates(base_patch)

        # If no math operations, single-statement update (two-phase if order/limit present)
        if not incs and not decs:
            ret = "minimal" if returning in {"none", "minimal"} else "representation"
            pk_col = _guess_pk(args)

            if order or limit:
                # Phase 1: read candidate PKs
                rq = supabase.table(table).select(pk_col)
                rq = _apply_where(rq, where)
                rq = _apply_order(rq, order)
                rq = _apply_limit(rq, limit)
                rresp = _exec(rq)
                rrows = getattr(rresp, "data", None) or []
                ids = [r.get(pk_col) for r in rrows if r.get(pk_col) is not None]
                if not ids:
                    out = {"rows": []}
                    if count_mode:
                        out.setdefault("meta", {})["count"] = 0
                    return _done(out, action="success", affected=0, two_phase=True)

                # Phase 2: constrained write
                wq = (
                    supabase.table(table)
                    .update(assigns, returning=ret, count=None)
                    .in_(pk_col, ids)
                )
                wresp = _exec(wq)
                wrows = getattr(wresp, "data", None)

                if ret == "minimal":
                    out = {"rows": []}
                    if count_mode:
                        out.setdefault("meta", {})["count"] = len(ids)
                    return _done(
                        out, action="success", affected=len(ids), two_phase=True
                    )
                else:
                    if _needs_embedded_projection(select_csv):
                        out_rows = _embedded_projection_by_pk(
                            table, select_csv or "*", pk_col, wrows or []
                        )
                    else:
                        out_rows = _project_rows(wrows, select_csv)
                    out = {"rows": out_rows}
                    if count_mode:
                        out.setdefault("meta", {})["count"] = len(ids)
                    return _done(
                        out, action="success", affected=len(ids), two_phase=True
                    )

            # No order/limit: direct filtered update
            qh = supabase.table(table).update(assigns, returning=ret, count=count_mode)
            qh = _apply_where(qh, where)
            wresp = _exec(qh)
            wrows = getattr(wresp, "data", None)

            if ret == "minimal":
                out = {"rows": []}
                _maybe_apply_count_meta(out, wresp)
                return _done(
                    out, action="success", affected=getattr(wresp, "count", None)
                )
            else:
                if _needs_embedded_projection(select_csv):
                    out_rows = _embedded_projection_by_pk(
                        table, select_csv or "*", pk_col, wrows or []
                    )
                else:
                    out_rows = _project_rows(wrows, select_csv)
                out = {"rows": out_rows}
                _maybe_apply_count_meta(out, wresp)
                return _done(
                    out,
                    action="success",
                    affected=(
                        len(out_rows)
                        if returning == "representation"
                        else getattr(wresp, "count", None)
                    ),
                )

        # Math present: read-modify-write per row (optimistic lock if updated_at exists).
        pk_col = _guess_pk(args)
        # Do not assume 'updated_at' exists; include only changed columns + pk
        read_cols = {pk_col, *assigns.keys(), *incs.keys(), *decs.keys()}
        select_list = ",".join(sorted(c for c in read_cols if c))
        rq = supabase.table(table).select(select_list)

        rq = _apply_where(rq, where)
        rq = _apply_order(rq, order)  # order honored
        rq = _apply_limit(rq, limit)  # limit honored
        resp_read = _exec(rq)

        input_rows = getattr(resp_read, "data", resp_read) or []
        if not input_rows:
            out = (
                {"updated": 0, "rows": []}
                if returning == "representation"
                else {"updated": 0}
            )
            return _done(out, action="success", affected=0)

        updated = 0
        out_rows: List[dict] = []

        for r in input_rows:
            computed = _apply_math(assigns, incs, decs, r)
            per_row_patch: Dict[str, Any] = {}
            for k in set(assigns.keys()) | set(incs.keys()) | set(decs.keys()):
                if k in (pk_col, "updated_at"):
                    continue
                if k in computed:
                    per_row_patch[k] = computed[k]
            if not per_row_patch:
                continue

            uq = supabase.table(table).eq(pk_col, r[pk_col])
            if "updated_at" in r and r["updated_at"] is not None:
                uq = uq.eq("updated_at", r["updated_at"])

            ret = "minimal" if returning in {"none", "minimal"} else "representation"
            ures = _exec(uq.update(per_row_patch, returning=ret, count=None))
            if ret != "minimal":
                udata = getattr(ures, "data", None) or []
                out_rows.extend(udata)
            updated += 1

        if returning in {"none", "minimal"}:
            out = {"updated": updated}
            if count_mode:
                out.setdefault("meta", {})["count"] = updated
            return _done(out, action="success", affected=updated)
        else:
            if _needs_embedded_projection(select_csv):
                out_rows = _embedded_projection_by_pk(
                    table, select_csv or "*", pk_col, out_rows
                )
            else:
                out_rows = _project_rows(out_rows, select_csv)
            out = {"updated": updated, "rows": out_rows}
            if count_mode:
                out.setdefault("meta", {})["count"] = updated
            return _done(out, action="success", affected=updated)

    # ------------------------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------------------------
    if mode == "delete":
        ret = "minimal" if returning in {"none", "minimal"} else "representation"
        pk_col = _guess_pk(args)

        if order or limit:
            # Phase 1: read candidate PKs
            rq = supabase.table(table).select(pk_col)
            rq = _apply_where(rq, where)
            rq = _apply_order(rq, order)
            rq = _apply_limit(rq, limit)
            rresp = _exec(rq)
            rrows = getattr(rresp, "data", None) or []
            ids = [r.get(pk_col) for r in rrows if r.get(pk_col) is not None]
            if not ids:
                out = {"rows": []}
                if count_mode:
                    out.setdefault("meta", {})["count"] = 0
                return _done(out, action="success", affected=0, two_phase=True)

            # Phase 2: constrained delete
            wq = (
                supabase.table(table).delete(returning=ret, count=None).in_(pk_col, ids)
            )
            wresp = _exec(wq)
            wrows = getattr(wresp, "data", None)

            if ret == "minimal":
                out = {"rows": []}
                if count_mode:
                    out.setdefault("meta", {})["count"] = len(ids)
                return _done(out, action="success", affected=len(ids), two_phase=True)
            else:
                if _needs_embedded_projection(select_csv):
                    out_rows = _embedded_projection_by_pk(
                        table, select_csv or "*", pk_col, wrows or []
                    )
                else:
                    out_rows = _project_rows(wrows, select_csv)
                out = {"rows": out_rows}
                if count_mode:
                    out.setdefault("meta", {})["count"] = len(ids)
                return _done(out, action="success", affected=len(ids), two_phase=True)

        # No order/limit: direct filtered delete
        qh = supabase.table(table).delete(returning=ret, count=count_mode)
        qh = _apply_where(qh, where)
        wresp = _exec(qh)
        wrows = getattr(wresp, "data", None)

        if ret == "minimal":
            out = {"rows": []}
            _maybe_apply_count_meta(out, wresp)
            return _done(out, action="success", affected=getattr(wresp, "count", None))
        else:
            if _needs_embedded_projection(select_csv):
                out_rows = _embedded_projection_by_pk(
                    table, select_csv or "*", pk_col, wrows or []
                )
            else:
                out_rows = _project_rows(wrows, select_csv)
            out = {"rows": out_rows}
            _maybe_apply_count_meta(out, wresp)
            return _done(
                out,
                action="success",
                affected=(
                    len(out_rows)
                    if returning == "representation"
                    else getattr(wresp, "count", None)
                ),
            )

    # Fallback
    raise ValueError(f"unsupported mode: {mode!r}")
