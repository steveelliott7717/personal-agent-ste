from __future__ import annotations
import re
import json
import os
import time
import urllib.error
import urllib.request

from datetime import datetime, timezone
from typing import Any, Callable, Dict

from backend.services.supabase_service import supabase


class CapabilityRegistry:
    """
    Minimal registry that maps verb -> adapter( args:dict, meta:dict ) -> Any
    Returns a normalized envelope: {ok, result, latency_ms}.
    """

    def __init__(self) -> None:
        self._adapters: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Any]] = {}
        self._register_builtin_adapters()

    # ---- public API ----
    def register(
        self, verb: str, fn: Callable[[Dict[str, Any], Dict[str, Any]], Any]
    ) -> None:
        self._adapters[verb] = fn

    def has(self, verb: str) -> bool:
        return verb in self._adapters

    def dispatch(
        self, verb: str, args: Dict[str, Any], meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            fn = self._adapters[verb]
        except KeyError:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return {
                "ok": False,
                "result": {"error": f"unknown verb '{verb}'"},
                "latency_ms": latency_ms,
            }

        try:
            data = fn(args or {}, meta or {})
            ok = True
            result = data if isinstance(data, dict) else {"data": data}
        except Exception as e:
            ok = False
            result = {"error": type(e).__name__, "message": str(e)}
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {"ok": ok, "result": result, "latency_ms": latency_ms}

    # ---- built-ins ----
    def _register_builtin_adapters(self) -> None:
        # health.echo – echoes back inputs (useful for smoke tests)
        self.register("health.echo", lambda a, m: {"echo": {"args": a, "meta": m}})

        # time.now – UTC ISO timestamp
        def time_now(_a, _m):
            return {"iso": datetime.now(timezone.utc).isoformat()}

        self.register("time.now", time_now)

        # math.add – sum a list of numbers under args.values
        def math_add(a, _m):
            values = a.get("values", [])
            if not isinstance(values, list) or any(
                not isinstance(x, (int, float)) for x in values
            ):
                raise ValueError("args.values must be a list of numbers")
            return {"sum": float(sum(values))}

        self.register("math.add", math_add)

        # ---- DB verbs ----
        self.register("db.read", _db_read_adapter)
        self.register("db.write", _db_write_adapter)
        self.register("notify.push", _notify_push_adapter)


# --- DB adapters ---


def _db_read_adapter(args: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    args:
      table: str (required)
      select: str | list[str] = "*" (optional)
      aggregate: {"count": "*"} (optional)
        # Returns total row count in result.meta.count
        # Example: {"aggregate": {"count": "*"}}  (use with limit:0 for count-only)

      expand: dict[str, list[str] | "*"] (optional)
        # lightweight relations via PostgREST embedded selects
        # {"expand":{"users":["name","email"]}} -> "...,users(name,email)"

      where: dict (optional)
        # equality (shortcut):
        #   {"status": "ok", "user_id": "u1"}
        # rich ops per column:
        #   {"created_at": {"op":"gte","value":"2025-08-01"},
        #    "topic": {"op":"in","value":["verb.call","verb.result"]},
        #    "slug": {"op":"ilike","value":"%planner%"}}
        # supported ops (flat): eq, neq, gt, gte, lt, lte, in, like, ilike,
        # contains, icontains, starts_with, istarts_with, ends_with, iends_with,
        # is_null, not_null, between, not{eq|neq|...}
        #
        # Nested boolean groups (new):
        #   {"op":"or","conditions":[
        #       {"field":"topic","op":"eq","value":"verb.result"},
        #       {"op":"and","conditions":[
        #           {"field":"topic","op":"eq","value":"plan.step.end"},
        #           {"field":"latency_ms","op":"gt","value":0}
        #       ]}
        #   ]}
      order: list[{"field": str, "desc": bool}] | {"field": str, "desc": bool} (optional)
      limit: int (optional)
      offset: int (optional)
    """
    table = args.get("table")
    if not table or not isinstance(table, str):
        raise ValueError("args.table (str) is required")

    sel = args.get("select", "*")
    if isinstance(sel, list):
        sel = ",".join(sel)

    # --- expand: lightweight embedded relations -> PostgREST select syntax ---
    # Supports keys like:
    #   "agents"                              -> agents(...)
    #   "agent:agents"                        -> agent:agents(...)
    #   "agents!events_agent_id_fkey"         -> agents!events_agent_id_fkey(...)
    #   "agent:agents!events_agent_id_fkey"   -> agent:agents!events_agent_id_fkey(...)
    # Values can be list[str] of columns, or "*" / None for all columns.
    expand = args.get("expand")
    if isinstance(expand, dict) and expand:
        base_parts = []
        # keep caller's select if provided; default to "*" otherwise
        if isinstance(sel, str) and sel and sel.strip() != "*":
            base_parts.append(sel)
        else:
            base_parts.append("*")

        def _build_rel_segment(key: str, cols) -> str:
            # parse alias/table/fk from key
            # forms:
            #   alias:table!fk
            #   alias:table
            #   table!fk
            #   table
            alias = None
            table_fk = key
            if ":" in key:
                alias, table_fk = key.split(":", 1)
                alias = alias.strip() or None

            fk = None
            table = table_fk
            if "!" in table_fk:
                table, fk = table_fk.split("!", 1)
                table, fk = table.strip(), fk.strip()
            else:
                table = table_fk.strip()

            # columns
            if cols == "*" or cols is None:
                col_list = "*"
            elif isinstance(cols, list) and cols:
                col_list = ",".join(cols)
            else:
                raise ValueError(f"expand.{key} must be a list of columns or '*'")

            rel = table if not fk else f"{table}!{fk}"
            if alias:
                rel = f"{alias}:{rel}"
            return f"{rel}({col_list})"

        for k, v in expand.items():
            base_parts.append(_build_rel_segment(str(k), v))

        # join parts; "*,rel(...)" is valid in PostgREST; keep as-is
        sel = ",".join(p for p in base_parts if p) or "*"

    # optional guardrail (configurable)
    MAX_EXPANDS = int(os.getenv("DBREAD_MAX_EXPANDS", "8"))
    if isinstance(expand, dict) and len(expand) > MAX_EXPANDS:
        raise ValueError(f"expand has {len(expand)} relations; max is {MAX_EXPANDS}")

    where = args.get("where", {}) or {}
    order = args.get("order")
    limit = args.get("limit")
    offset = args.get("offset")
    tiebreaker = args.get("tiebreaker")
    distinct = args.get("distinct")  # bool | list[str] | None

    # Ensure order fields are in the base projection (for keyset cursor + next_cursor)
    # If caller used "*", nothing to do. If they enumerated columns, append any missing
    # order fields so we can read them back and build next_cursor safely.
    if isinstance(order, dict):
        ord_fields = [order.get("field")]
    elif isinstance(order, list):
        ord_fields = [o.get("field") for o in order if isinstance(o, dict)]
    else:
        ord_fields = []
    ord_fields = [f for f in ord_fields if f]

    if ord_fields and isinstance(sel, str) and sel.strip() != "*":
        current = [s.strip() for s in sel.split(",") if s and s.strip()]
        missing = [f for f in ord_fields if f not in current]
        if missing:
            sel = ",".join(current + missing)

    # --- aggregates ---
    aggregate = args.get("aggregate") or {}
    count_mode = None
    if isinstance(aggregate, dict) and aggregate.get("count"):
        # PostgREST modes: exact | planned | estimated
        count_mode = str(aggregate.get("mode") or "exact").lower()
        if count_mode not in {"exact", "planned", "estimated"}:
            count_mode = "exact"

        # ----- column expression helper (shared by flat + grouped filters) -----

    def _colexpr(col: str, *, text_mode: bool = True) -> str:
        """
        Normalize a column reference; supports JSON path via dot notation.
        Example: "payload.status" -> "payload->>status" for text-mode comparisons.
                 If you later add JSON (non-text) ops, call with text_mode=False
                 to use 'payload->status' instead of '->>'.
        """
        if isinstance(col, str) and "." in col:
            left, right = col.split(".", 1)
            # Add other JSONB roots here as needed
            if left in ("payload",):
                return f"{left}->>{right}" if text_mode else f"{left}->{right}"
        return col

    if count_mode:
        q = supabase.table(table).select(sel, count=count_mode)
    else:
        q = supabase.table(table).select(sel)

    # ---- fold simple OR into a single IN for the same column ----
    # Supports shapes like:
    #   {"or": [{"topic": "plan.step.start"}, {"topic": "plan.step.end"}]}
    #   {"or": [{"topic": {"op": "eq", "value": "plan.step.start"}},
    #           {"topic": {"op": "eq", "value": "plan.step.end"}}]}
    or_list = where.get("or")
    if isinstance(or_list, list) and or_list:
        same_col = None
        values = []
        can_fold = True
        for item in or_list:
            # Each OR item must be a single-field dict
            if not isinstance(item, dict) or len(item) != 1:
                can_fold = False
                break
            col, spec = next(iter(item.items()))
            if same_col is None:
                same_col = col
            elif col != same_col:
                # Different columns in OR — skip folding
                can_fold = False
                break

            # Accept either scalar equality, or {"op":"eq","value":...}
            if isinstance(spec, dict):
                iop = (spec.get("op") or "eq").lower()
                if iop != "eq":
                    can_fold = False
                    break
                val = spec.get("value")
            else:
                val = spec
            values.append(val)

        if can_fold and same_col is not None:
            # Build a new 'where' without the top-level OR
            new_where = {k: v for k, v in where.items() if k != "or"}
            existing = new_where.get(same_col)

            # Merge with any existing constraint on the same column
            if isinstance(existing, dict):
                if "in" in existing and isinstance(existing["in"], list):
                    existing["in"].extend(values)
                elif (existing.get("op") == "in") and isinstance(
                    existing.get("value"), list
                ):
                    existing["value"].extend(values)
                else:
                    # turn existing into IN together with new values
                    prev = existing.get("value") if "value" in existing else existing
                    new_where[same_col] = (
                        {"in": [prev] + values} if prev is not None else {"in": values}
                    )
            elif existing is not None:
                new_where[same_col] = {"in": [existing] + values}
            else:
                new_where[same_col] = {"in": values}

            where = new_where

    # ------------------------------------------------------------------
    # Normalize top-level boolean shorthand and order keys
    # (Place this AFTER the simple OR→IN fold and BEFORE group helpers)
    # ------------------------------------------------------------------

    # 1) where: accept {"and":[...]} / {"or":[...]} at top-level
    if isinstance(where, dict):
        # If a simple 'or' could not be folded above (different columns, non-eq, etc.),
        # convert it to the explicit group form so _apply_group() can handle it.
        if "and" in where and isinstance(where["and"], list):
            where = {"op": "and", "conditions": where["and"]}
        elif "or" in where and isinstance(where["or"], list):
            where = {"op": "or", "conditions": where["or"]}

    # 2) order: tolerate {"column","ascending"} and map to {"field","desc"}
    def _normalize_order(o):
        if isinstance(o, dict):
            # Support alt keys
            if "column" in o and "field" not in o:
                field = o.get("column")
                ascending = bool(o.get("ascending", True))
                return {"field": field, "desc": (not ascending)}
            # Ensure canonical shape and bool default
            return {"field": o.get("field"), "desc": bool(o.get("desc", False))}
        if isinstance(o, list):
            out = []
            for item in o:
                if isinstance(item, dict):
                    out.append(_normalize_order(item))
            return out
        return o

    order = _normalize_order(order)

    # =========================
    # Nested AND/OR groups
    # =========================

    def _is_group(node: object) -> bool:
        return (
            isinstance(node, dict)
            and isinstance(node.get("op"), str)
            and isinstance(node.get("conditions"), list)
        )

    def _normalize_leaf(node: object) -> tuple[str, str, object]:
        """
        Accept as leaf:
          - {"field":"topic","op":"eq","value":"..."}
          - {"topic":{"op":"eq","value":"..."}}
          - {"topic":"literal"}  (scalar eq)
        Return (field, op, value) with op lowercased; default op is 'eq'.
        """
        if not isinstance(node, dict):
            raise ValueError(f"Invalid leaf node: {node}")
        if "field" in node:
            field = node["field"]
            op = (node.get("op") or "eq").lower()
            value = node.get("value")
            return field, op, value
        if len(node) == 1:
            field, spec = next(iter(node.items()))
            if isinstance(spec, dict) and "op" in spec:
                return field, (spec.get("op") or "eq").lower(), spec.get("value")
            # scalar eq
            return field, "eq", spec
        raise ValueError(f"Unrecognized leaf shape: {node}")

    def _apply_leaf(qh, field: str, op: str, value: object):
        """
        Apply one leaf predicate to the query handle.
        Fast-path eq/neq/in on top-level columns; use .filter(expr, op, val) for others/JSON.
        """
        expr = _colexpr(field, text_mode=True)

        # IN on real columns
        if op == "in":
            values = value or []
            if not isinstance(values, list):
                raise ValueError(f"{field}.op=in requires list 'value'")
            if expr == field:
                return qh.in_(field, values)
            # If IN targets a JSON path, prefer expressing it as an OR group externally.
            raise ValueError("IN on JSON path not supported directly; use an OR group.")

        if op == "not_in":
            values = value or []
            if not isinstance(values, list):
                raise ValueError(f"{field}.op=not_in requires list 'value'")
            if expr == field:
                # PostgREST: NOT IN via filter("not.in", "(a,b,...)")
                return qh.filter(field, "not.in", f"({','.join(map(str, values))})")
            raise ValueError(
                "NOT IN on JSON path not supported directly; use a NOT/OR group."
            )

        # eq / neq fast paths on real columns
        if op in ("eq", "=") and expr == field:
            return qh.eq(field, value)
        if op in ("ne", "neq", "!=") and expr == field:
            return qh.neq(field, value)

        # other ops (or JSON) via .filter
        if op in ("eq", "="):
            return qh.filter(expr, "eq", str(value))
        if op in ("ne", "neq", "!="):
            return qh.filter(expr, "neq", str(value))
        if op in ("gt", ">"):
            return qh.filter(expr, "gt", str(value))
        if op in ("gte", ">="):
            return qh.filter(expr, "gte", str(value))
        if op in ("lt", "<"):
            return qh.filter(expr, "lt", str(value))
        if op in ("lte", "<="):
            return qh.filter(expr, "lte", str(value))
        if op == "like":
            return qh.filter(expr, "like", str(value))
        if op == "ilike":
            return qh.filter(expr, "ilike", str(value))
        if op in ("contains", "icontains"):
            pat = f"%{value}%"
            return qh.filter(expr, "ilike" if op == "icontains" else "like", pat)
        if op in ("starts_with", "istarts_with"):
            pat = f"{value}%"
            return qh.filter(expr, "ilike" if op == "istarts_with" else "like", pat)
        if op in ("ends_with", "iends_with"):
            pat = f"%{value}"
            return qh.filter(expr, "ilike" if op == "iends_with" else "like", pat)
        if op in ("is_null", "isnull", "is-null"):
            return qh.filter(expr, "is", "null")
        if op in ("not_null", "notnull", "not-null"):
            return qh.filter(expr, "not.is", "null")
        if op == "between":
            if not (isinstance(value, (list, tuple)) and len(value) == 2):
                raise ValueError(f"{field}.between requires [low, high]")
            low, high = value
            return qh.filter(expr, "gte", str(low)).filter(expr, "lte", str(high))
        if op == "not":
            # value should be a nested op dict, e.g. {"op":"ilike","value":"plan.%"}
            if not isinstance(value, dict) or "op" not in value:
                raise ValueError(f"{field}.not requires nested {{op,value}}")
            iop = (value.get("op") or "eq").lower()
            ival = value.get("value")

            # Pattern-style NOTs first: contains / icontains / starts_with / istarts_with / ends_with / iends_with
            if iop in (
                "contains",
                "icontains",
                "starts_with",
                "istarts_with",
                "ends_with",
                "iends_with",
            ):
                base = "" if ival is None else str(ival)
                if iop in ("contains", "icontains"):
                    patt = f"%{base}%"
                elif iop in ("starts_with", "istarts_with"):
                    patt = f"{base}%"
                else:  # ends_with / iends_with
                    patt = f"%{base}"
                operator = (
                    "not.ilike"
                    if iop in ("icontains", "istarts_with", "iends_with")
                    else "not.like"
                )
                return qh.filter(expr, operator, patt)

            # Map inner op to PostgREST operator string used by .filter
            op_map = {
                "eq": "eq",
                "neq": "neq",
                "ne": "neq",
                "gt": "gt",
                "gte": "gte",
                "lt": "lt",
                "lte": "lte",
                "like": "like",
                "ilike": "ilike",
                "is": "is",
            }
            if iop not in op_map:
                raise ValueError(f"Unsupported inner op for not: {iop}")

            # Use PostgREST "not.<op>" via .filter
            val_str = "null" if iop == "is" and ival in (None, "null") else str(ival)
            return qh.filter(expr, f"not.{op_map[iop]}", val_str)

        raise ValueError(f"Unsupported operator: {op}")

        # --- shared helpers -------------------------------------------------

    def _pg_encode(val: object) -> str:
        """
        Encode just the characters that break PostgREST boolean expr parsing
        when used inside or= / and= CSVs. Keep ISO timestamps, UUIDs, '%' etc.
        """
        s = "" if val is None else str(val)

        # Allow common typed literals unmodified
        if re.match(
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$", s
        ):  # ISO 8601 Zulu
            return s
        if re.match(r"^[0-9a-fA-F-]{36}$", s):  # UUID v4-ish
            return s

        # For everything else, encode only commas and parentheses.
        # Keep '%' (LIKE), ':' (time), '-' (dates), '.' etc. intact.
        return s.replace("(", "%28").replace(")", "%29").replace(",", "%2C")

    def _map_simple_op(iop: str) -> str:
        """
        Shared map for simple ops used by NOT handling (both flat & grouped).
        Keeps the two code paths in sync.
        """
        iop = (iop or "eq").lower()
        op_map = {
            "eq": "eq",
            "neq": "neq",
            "ne": "neq",
            "gt": "gt",
            "gte": "gte",
            "lt": "lt",
            "lte": "lte",
            "like": "like",
            "ilike": "ilike",
            "is": "is",
        }
        if iop not in op_map:
            raise ValueError(f"Unsupported inner op for not: {iop}")
        return op_map[iop]

    def _serialize_node_to_postgrest(node: dict) -> str:
        """
        Build a PostgREST boolean expression string for .or_(...) when we cannot fold to IN.
        Supports nested and/or with leaf conditions.
        """
        if _is_group(node):
            op = node["op"].lower()
            parts = [_serialize_node_to_postgrest(c) for c in node["conditions"]]
            fn = "and" if op == "and" else "or"
            inner = ",".join(parts)
            return f"{fn}({inner})"

        # leaf
        field, op, value = _normalize_leaf(node)
        col = _colexpr(field, text_mode=True)

        if op in ("eq", "="):
            return f"{col}.eq.{_pg_encode(value)}"
        if op in ("ne", "neq", "!="):
            return f"{col}.neq.{_pg_encode(value)}"
        if op in ("gt", ">"):
            return f"{col}.gt.{_pg_encode(value)}"
        if op in ("gte", ">="):
            return f"{col}.gte.{_pg_encode(value)}"
        if op in ("lt", "<"):
            return f"{col}.lt.{_pg_encode(value)}"
        if op in ("lte", "<="):
            return f"{col}.lte.{_pg_encode(value)}"
        if op == "like":
            return f"{col}.like.{_pg_encode(value)}"
        if op == "ilike":
            return f"{col}.ilike.{_pg_encode(value)}"
        if op == "in":
            if not isinstance(value, list):
                raise ValueError(f"{field}.op=in requires list 'value'")
            items = ",".join(_pg_encode(v) for v in value)
            return f"{col}.in.({items})"
        if op in (
            "starts_with",
            "istarts_with",
            "ends_with",
            "iends_with",
            "contains",
            "icontains",
        ):
            base = "" if value is None else str(value)
            if op in ("starts_with", "istarts_with"):
                patt = f"{base}%"
            elif op in ("ends_with", "iends_with"):
                patt = f"%{base}"
            else:  # contains / icontains
                patt = f"%{base}%"
            operator = (
                "ilike" if op in ("istarts_with", "iends_with", "icontains") else "like"
            )
            return f"{col}.{operator}.{_pg_encode(patt)}"

        if op == "not":
            if not isinstance(value, dict) or "op" not in value:
                raise ValueError(f"{field}.not requires nested {{op,value}}")
            iop = (value.get("op") or "eq").lower()
            ival = value.get("value")

            # Pattern-style NOTs first
            if iop in (
                "contains",
                "icontains",
                "starts_with",
                "istarts_with",
                "ends_with",
                "iends_with",
            ):
                base = "" if ival is None else str(ival)
                if iop in ("contains", "icontains"):
                    patt = f"%{base}%"
                elif iop in ("starts_with", "istarts_with"):
                    patt = f"{base}%"
                else:
                    patt = f"%{base}"
                operator = (
                    "not.ilike"
                    if iop in ("icontains", "istarts_with", "iends_with")
                    else "not.like"
                )
                return f"{col}.{operator}.{_pg_encode(patt)}"

            # Simple NOTs via shared map
            op_str = _map_simple_op(iop)
            val_str = (
                "null"
                if op_str == "is" and ival in (None, "null")
                else _pg_encode(ival)
            )
            return f"{col}.not.{op_str}.{val_str}"

        if op == "not_in":
            if not isinstance(value, list):
                raise ValueError(f"{field}.op=not_in requires list 'value'")
            items = ",".join(_pg_encode(v) for v in value)
            return f"{col}.not.in.({items})"

        if op in ("is_null", "isnull", "is-null"):
            return f"{col}.is.null"
        if op in ("not_null", "notnull", "not-null"):
            return f"{col}.not.is.null"
        if op == "between":
            if not (isinstance(value, (list, tuple)) and len(value) == 2):
                raise ValueError(f"{field}.between requires [low, high]")
            low, high = value
            return f"and({col}.gte.{_pg_encode(low)},{col}.lte.{_pg_encode(high)})"

        raise ValueError(f"Unsupported op in serializer: {op}")

    def _apply_group(qh, node: dict):
        """
        Apply a group node to the query:
          - AND: chain each child (AND semantics are natural in PostgREST)
          - OR: fold same-col EQs to IN; otherwise emit one .or_(expr) string
        """
        if not _is_group(node):
            fld, op, val = _normalize_leaf(node)
            return _apply_leaf(qh, fld, op, val)

        op = node["op"].lower()
        conds = node["conditions"]

        if op == "and":
            for child in conds:
                qh = _apply_group(qh, child)
            return qh

        # op == "or"
        # fast path: OR of same-column equals → IN
        same_col = None
        values = []
        all_eq_same_col = True
        for child in conds:
            if _is_group(child):
                all_eq_same_col = False
                break
            fld, cop, val = _normalize_leaf(child)
            if cop not in ("eq", "="):
                all_eq_same_col = False
                break
            if same_col is None:
                same_col = fld
            elif fld != same_col:
                all_eq_same_col = False
                break
            values.append(val)

        if all_eq_same_col and same_col is not None:
            return _apply_leaf(qh, same_col, "in", values)

        # general OR: one composite .or_(...) expression (inner CSV only)
        parts = [_serialize_node_to_postgrest(c) for c in conds]
        inner = ",".join(parts)
        return qh.or_(inner)

    # If the top-level 'where' is a structured group, apply it and SKIP the flat loop.
    if _is_group(where):
        q = _apply_group(q, where)
    else:
        # --------------------------
        # Flat filters (existing path)
        # --------------------------
        # --------------------------
        # Flat filters (unified with _colexpr)
        # --------------------------
        def _apply_pred(col: str, spec: Any):
            nonlocal q
            expr = _colexpr(col, text_mode=True)

            # equality shortcut (scalar)
            if not isinstance(spec, dict):
                q = q.eq(col, spec) if expr == col else q.filter(expr, "eq", str(spec))
                return

            op = (spec.get("op") or "eq").lower()
            val = spec.get("value")

            # IN
            if ("in" in spec) or op == "in":
                values = spec.get("in") if "in" in spec else val
                if not isinstance(values, list):
                    raise ValueError(f"{col}.in must be a list")
                if expr != col:
                    # Guardrail: IN on JSON path can’t be expressed cleanly; ask caller to OR
                    raise ValueError(
                        "IN on JSON path not supported directly; use an OR group."
                    )
                q = q.in_(col, values)
                return

            # NOT IN (flat path)
            if op == "not_in":
                if not isinstance(val, (list, tuple)) or len(val) == 0:
                    raise ValueError(
                        f"where.{col}.value must be a non-empty list for op 'not_in'"
                    )
                # PostgREST expects a parenthesized CSV for not.in
                csv = ",".join(map(str, val))
                q = q.filter(expr, "not.in", f"({csv})")
                return

            # simple comparisons
            if op in ("eq", "="):
                q = q.eq(col, val) if expr == col else q.filter(expr, "eq", str(val))
                return
            if op in ("ne", "neq", "!="):
                q = q.neq(col, val) if expr == col else q.filter(expr, "neq", str(val))
                return
            if op in ("gt", ">"):
                q = q.gt(col, val) if expr == col else q.filter(expr, "gt", str(val))
                return
            if op in ("gte", ">="):
                q = q.gte(col, val) if expr == col else q.filter(expr, "gte", str(val))
                return
            if op in ("lt", "<"):
                q = q.lt(col, val) if expr == col else q.filter(expr, "lt", str(val))
                return
            if op in ("lte", "<="):
                q = q.lte(col, val) if expr == col else q.filter(expr, "lte", str(val))
                return

            # patterns
            if op == "like":
                q = q.filter(expr, "like", str(val))
                return
            if op == "ilike":
                q = q.filter(expr, "ilike", str(val))
                return
            if op in ("contains", "icontains"):
                patt = f"%{val}%"
                q = q.filter(expr, "ilike" if op == "icontains" else "like", patt)
                return
            if op in ("starts_with", "istarts_with"):
                patt = f"{val}%"
                q = q.filter(expr, "ilike" if op == "istarts_with" else "like", patt)
                return
            if op in ("ends_with", "iends_with"):
                patt = f"%{val}"
                q = q.filter(expr, "ilike" if op == "iends_with" else "like", patt)
                return

            # null checks
            if op in ("is_null", "isnull", "is-null"):
                q = q.filter(expr, "is", "null")
                return
            if op in ("not_null", "notnull", "not-null"):
                q = q.filter(expr, "not.is", "null")
                return
            if op == "is":
                # explicit IS (mostly for JSON paths)
                q = (
                    q.is_(col, val)
                    if expr == col
                    else q.filter(
                        expr, "is", "null" if val in (None, "null") else str(val)
                    )
                )
                return

            # between
            if op == "between":
                if not (isinstance(val, (list, tuple)) and len(val) == 2):
                    raise ValueError(f"{col}.between requires [low, high]")
                lo, hi = val
                if expr == col:
                    q = q.gte(col, lo).lte(col, hi)
                else:
                    q = q.filter(expr, "gte", str(lo)).filter(expr, "lte", str(hi))
                return

            # NOT (covers simple + pattern NOTs with expr)
            if op == "not":
                inner = val if isinstance(val, dict) else {"op": "eq", "value": val}
                iop = (inner.get("op") or "eq").lower()
                ival = inner.get("value")

                # pattern NOTs
                if iop in (
                    "contains",
                    "icontains",
                    "starts_with",
                    "istarts_with",
                    "ends_with",
                    "iends_with",
                ):
                    base = "" if ival is None else str(ival)
                    patt = (
                        f"%{base}%"
                        if iop in ("contains", "icontains")
                        else (
                            f"{base}%"
                            if iop in ("starts_with", "istarts_with")
                            else f"%{base}"
                        )
                    )
                    oper = (
                        "not.ilike"
                        if iop in ("icontains", "istarts_with", "iends_with")
                        else "not.like"
                    )
                    q = q.filter(expr, oper, patt)
                    return

                # simple NOTs via PostgREST not.<op>
                # Regular NOTs (shared map)
                op_str = _map_simple_op(iop)
                crit = (
                    "null" if (op_str == "is" and ival in (None, "null")) else str(ival)
                )
                q = q.filter(expr, f"not.{op_str}", crit)
                return

            raise ValueError(f"unsupported op '{op}' for column '{col}'")

        # apply flat where (unchanged behavior)
        if isinstance(where, dict):
            for col, cond in where.items():
                _apply_pred(col, cond)

    def _parse_nulls_first(v) -> object:
        """
        Returns True for 'first', False for 'last', or None when unspecified.
        """
        if v is None:
            return None
        s = str(v).strip().lower()
        if s == "first":
            return True
        if s == "last":
            return False
        raise ValueError("order.nulls must be 'first' or 'last' if provided")

    def _apply_keyset_cursor(qh, order_spec, cursor):
        after = cursor.get("after")
        before = cursor.get("before")
        if not isinstance(order_spec, list):
            order_spec = [order_spec]
        target = after or before
        if not isinstance(target, dict) or not target:
            return qh
        reverse = bool(before)  # if before, flip the inequality

        disjuncts = []
        for i in range(len(order_spec)):
            clauses = []
            complete = True
            for j in range(i):
                fj = order_spec[j].get("field")
                if not fj or fj not in target:
                    complete = False
                    break
                clauses.append({"field": fj, "op": "eq", "value": target[fj]})
            if not complete:
                continue
            fi_item = order_spec[i]
            fi = fi_item.get("field")
            if not fi or fi not in target:
                continue
            desc = bool(fi_item.get("desc", False))
            # normal: desc→lt, asc→gt ; reversed: swap
            op = ("gt" if desc else "lt") if reverse else ("lt" if desc else "gt")
            clauses.append({"field": fi, "op": op, "value": target[fi]})
            node = {"op": "and", "conditions": clauses}
            disjuncts.append(_serialize_node_to_postgrest(node))

        if not disjuncts:
            return qh
        return qh.or_(",".join(disjuncts))

    def _build_cursor_from_row(row: dict, ord_list: list[dict]) -> dict:
        """
        Given a row and the normalized order spec, return {"after": {...}}
        using only fields present in the order. Skips if fields missing.
        """
        if not isinstance(row, dict) or not ord_list:
            return {}
        after = {}
        for item in ord_list:
            f = item.get("field")
            if f and (f in row):
                after[f] = row[f]
        return {"after": after} if after else {}

    # ---- ordering (single or list) ----
    def _apply_order(qh, item):
        field = item.get("field")
        if not field:
            return qh
        desc = bool(item.get("desc", False))
        nf = _parse_nulls_first(item.get("nulls")) if "nulls" in item else None
        return (
            qh.order(field, desc=desc)
            if nf is None
            else qh.order(field, desc=desc, nullsfirst=bool(nf))
        )

    applied_any_order = False
    normalized_order = []

    if isinstance(order, dict):
        q = _apply_order(q, order)
        applied_any_order = True
        normalized_order = [order]
    elif isinstance(order, list):
        for item in order:
            if isinstance(item, dict) and item.get("field"):
                q = _apply_order(q, item)
                applied_any_order = True
                normalized_order.append(item)

    # ---- keyset cursor (optional) ----
    cursor = args.get("cursor")
    if isinstance(cursor, dict) and cursor.get("mode") == "keyset" and normalized_order:
        q = _apply_keyset_cursor(q, normalized_order, cursor)

    # If caller didn’t pass order but gave a tiebreaker, apply it.
    if (
        not applied_any_order
        and isinstance(tiebreaker, dict)
        and tiebreaker.get("field")
    ):
        q = _apply_order(q, tiebreaker)

    # If still no order was applied, fall back to a stable tiebreaker
    if not applied_any_order:
        q = _apply_order(q, {"field": "id", "desc": False})
        normalized_order = [{"field": "id", "desc": False}]

    if isinstance(limit, int) and limit > 0:
        q = q.limit(limit)
    if isinstance(offset, int) and offset >= 0:
        q = q.range(
            offset, (offset + (limit or 100)) - 1
        )  # supabase uses inclusive ranges

    # --- execute and shape result ---
    resp = q.execute() if hasattr(q, "execute") else q

    rows = None
    cnt = None

    # Try common response shapes
    if hasattr(resp, "data"):
        rows = resp.data
        cnt = getattr(resp, "count", None)
    elif isinstance(resp, dict) and "data" in resp:
        rows = resp.get("data")
        cnt = resp.get("count")
    else:
        # fallback: assume resp already is the rows
        rows = resp

    # ----- client-side DISTINCT (temporary, until client exposes 'distinct' param) -----
    # Supports: distinct: true  -> distinct over selected columns
    #           distinct: ["col1","col2"] -> distinct over these keys
    distinct = args.get("distinct")
    if distinct:
        if rows is None:
            rows = []
        if not isinstance(rows, list):
            rows = list(rows)

        # Determine which keys to use for distinct
        if isinstance(distinct, list) and distinct:
            keys = [str(k) for k in distinct]
        else:
            # distinct: true -> use explicit select columns (not "*")
            if not isinstance(sel, str) or sel.strip() in ("", "*"):
                raise ValueError(
                    "distinct:true requires explicit select columns (not '*')"
                )
            # take base columns only (ignore any expand segments like rel(...))
            keys = [c.strip() for c in sel.split(",") if "(" not in c and c.strip()]

        seen = set()
        deduped = []
        for r in rows:
            if not isinstance(r, dict):
                deduped.append(r)
                continue
            key_tuple = tuple(r.get(k) for k in keys)
            if key_tuple in seen:
                continue
            seen.add(key_tuple)
            deduped.append(r)
        rows = deduped

    result = {"rows": rows}

    # ---- meta: count/paging info ----
    meta = {}

    if isinstance(cursor, dict):
        result.setdefault("meta", {}).update(
            {"cursor": {"after": cursor.get("after"), "mode": cursor.get("mode")}}
        )

    # Include total count if aggregate was requested
    if count_mode is not None:
        meta["count"] = cnt

    # Echo paging knobs when provided
    if isinstance(limit, int) and limit > 0:
        meta["limit"] = limit
    if isinstance(offset, int) and offset >= 0:
        meta["offset"] = offset

        # If caller set offset without limit, note the implicit default (100)
        if not isinstance(limit, int):
            meta["default_limit_used"] = (
                100  # matches q.range(..., (limit or 100)) above  :contentReference[oaicite:1]{index=1}
            )

    # Convenience: has_more / next_offset when we know count + paging knobs
    if "count" in meta and "limit" in meta and "offset" in meta:
        has_more = (meta["offset"] + meta["limit"]) < meta["count"]
        meta["has_more"] = has_more
        if has_more:
            meta["next_offset"] = meta["offset"] + meta["limit"]

    if meta:
        # merge into any prior meta (e.g., cursor)
        merged = result.get("meta", {}).copy()
        merged.update(meta)
        result["meta"] = merged

    # Keyset: include next_cursor if possible
    if isinstance(rows, list) and rows and normalized_order:
        tail = rows[-1]
        nc = _build_cursor_from_row(tail, normalized_order)
        if nc:
            result.setdefault("meta", {})["next_cursor"] = {"mode": "keyset", **nc}

    return result


def _db_write_adapter(args: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    args:
      table: str (required)
      mode: "insert" | "upsert" | "update" | "delete" (required)
      rows: list[dict] (required for insert/upsert; single-row for update)
      where: dict[str, Any] for update/delete target (required for update/delete)
      returning: bool = True (optional)
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

    tbl = supabase.table(table)

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

    if mode == "update":
        if (
            not isinstance(rows, list)
            or len(rows) != 1
            or not isinstance(rows[0], dict)
        ):
            raise ValueError("args.rows must be a single-row list[dict] for update")
        if not isinstance(where, dict) or not where:
            raise ValueError("args.where (dict) is required for update")
        q = tbl.update(rows[0])
        for col, val in where.items():
            q = q.eq(col, val)
        if not returning:
            q = q.select("*", count="exact").limit(0)
        res = q.execute()
        return {"rows": getattr(res, "data", res)}

    # mode == "delete"
    if not isinstance(where, dict) or not where:
        raise ValueError("args.where (dict) is required for delete")
    q = tbl.delete()
    for col, val in where.items():
        q = q.eq(col, val)
    if not returning:
        q = q.select("*", count="exact").limit(0)
    res = q.execute()
    return {"rows": getattr(res, "data", res)}


# --- Notification adapter (Slack webhook) ---


def _notify_push_adapter(args: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a simple message via channel=slack using an incoming webhook.

    args:
      channel: "slack" (only supported for now)
      text: str (required)
      webhook_url: str (optional; falls back to env SLACK_WEBHOOK_URL)

    returns:
      { channel, status, body (truncated) }
    """
    channel = (args.get("channel") or "slack").lower()
    text = args.get("text")
    if not text or not isinstance(text, str):
        raise ValueError("args.text (str) is required")

    if channel != "slack":
        raise ValueError(f"unsupported channel '{channel}' (only 'slack' supported)")

    webhook_url = args.get("webhook_url") or os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        raise ValueError("SLACK_WEBHOOK_URL not set and args.webhook_url missing")

    payload = {"text": text}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            status = resp.getcode()
            body = resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as e:
        status = e.code
        body = e.read().decode("utf-8", errors="ignore")
    except Exception as e:
        raise RuntimeError(f"notify.push failed: {type(e).__name__}: {e}")

    # Slack returns "ok" on success for classic webhooks; keep body short
    return {"channel": channel, "status": int(status), "body": body[:500]}
