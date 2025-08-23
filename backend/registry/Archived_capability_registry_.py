from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import ssl
import time
import urllib
from datetime import datetime, timezone
from typing import Any, Callable, Dict
from urllib import error, parse, request

from backend.services.supabase_service import supabase
import socket
import ipaddress
import random
from typing import Optional, Tuple
from typing import Any, Callable, Dict, Optional, Literal, TypedDict
from typing import Any, Callable, Dict, Optional, Literal, TypedDict, overload
import socket
import ipaddress
from typing import Optional, Tuple

import gzip
import zlib
import io
import re


# HTTP defaults (centralize tunables)
HTTP_DEFAULT_TIMEOUT_S = 15
HTTP_MAX_BYTES = 2_000_000
HTTP_DEFAULT_ALLOWED_PORTS = {80, 443, 8443}  # can be overridden by env


def _env_csv(name: str) -> set[str]:
    v = os.getenv(name, "") or ""
    return {s.strip() for s in v.split(",") if s.strip()}


_DBWRITE_TABLE_ALLOW = _env_csv("DBWRITE_TABLE_ALLOWLIST")


def _dbwrite_cols_allow(table: str) -> set[str]:
    # e.g., DBWRITE_COL_ALLOWLIST_workout_plan
    return _env_csv(f"DBWRITE_COL_ALLOWLIST_{table}")


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

        # ---- typed overloads for dispatch ----

    @overload
    def dispatch(
        self, verb: Literal["http.fetch"], args: HttpFetchArgs, meta: MetaDict
    ) -> Envelope: ...
    @overload
    def dispatch(
        self, verb: str, args: Dict[str, Any], meta: Dict[str, Any]
    ) -> Envelope: ...

    def dispatch(
        self, verb: str, args: Dict[str, Any], meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            fn = self._adapters[verb]
        except KeyError:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            t = (args or {}).get("trace")
            corr = (meta or {}).get("correlation_id") or (
                t.get("correlation_id") if isinstance(t, dict) else None
            )

            return {
                "ok": False,
                "result": {"error": f"unknown verb '{verb}'"},
                "latency_ms": latency_ms,
                "correlation_id": corr or "",
            }

        try:
            data = fn(args or {}, meta or {})
            ok = True
            result = data if isinstance(data, dict) else {"data": data}
        except Exception as e:
            ok = False
            result = {"error": type(e).__name__, "message": str(e)}

        latency_ms = int((time.perf_counter() - t0) * 1000)
        t = (args or {}).get("trace")
        corr = (meta or {}).get("correlation_id") or (
            t.get("correlation_id") if isinstance(t, dict) else None
        )

        return {
            "ok": ok,
            "result": result,
            "latency_ms": latency_ms,
            "correlation_id": corr or "",
        }

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
        self.register("http.fetch", _http_fetch)


class HttpRetryCfg(TypedDict, total=False):
    max: int
    backoff_ms: int
    jitter: bool
    on: list[int]


class HttpTraceCfg(TypedDict, total=False):
    correlation_id: str
    idempotency_key: str


class HttpAuthBearer(TypedDict, total=False):
    type: Literal["bearer"]
    token: str


class HttpAuthBasic(TypedDict, total=False):
    type: Literal["basic"]
    username: str
    password: str


class HttpAuthHmac(TypedDict, total=False):
    type: Literal["hmac"]
    secret_env: str
    algo: Literal["sha256", "sha1", "md5"]
    header: str


HttpAuth = HttpAuthBearer | HttpAuthBasic | HttpAuthHmac


class HttpFetchArgs(TypedDict, total=False):
    url: str
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
    headers: Dict[str, str]
    params: Dict[str, Any]
    body: str | bytes | dict | list
    timeout: int | float
    allow_redirects: bool
    max_redirects: int
    max_bytes: int
    retry: HttpRetryCfg
    auth: HttpAuth
    cache: Dict[str, str]
    trace: HttpTraceCfg
    paginate: PaginateCfg


# Meta you pass around; keep loose but documented
class MetaDict(TypedDict, total=False):
    correlation_id: str
    idempotency_key: str
    actor: str
    # add anything else you commonly pass (ip, user_id, etc.)


# Result envelope your registry returns
class Envelope(TypedDict, total=False):
    ok: bool
    result: Dict[str, Any]
    latency_ms: int
    correlation_id: str  # we’ll add this in dispatch()


class HttpFetchResult(TypedDict, total=False):
    status: int
    url: str
    final_url: str
    headers: Dict[str, str]
    bytes_len: int
    elapsed_ms: int
    json: Any | None
    text: str | None
    truncated: bool
    not_modified: bool


# ADD/UPDATE this where PaginateCfg is defined
class PaginateCfg(TypedDict, total=False):
    # mode
    mode: Literal["cursor", "header_link", "page"]

    # cursor mode (existing)
    json_pointer: str
    cursor_param: str
    max_pages: int
    items_pointer: str
    concat_json: bool

    # header_link mode
    header: str  # default "Link"
    rel: str  # default "next" (used when header == "Link")

    # page mode
    page_param: str  # default "page"
    start_page: int  # default 1


# --- DB adapters ---


def _db_read_adapter(args: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    args:
      table: str (required)
      select: str | list[str] = "*" (optional)
      aggregate: {"count": "*"} (optional)
    """
    # ---- debug flag (echo the final query parts) ----
    debug_explain = bool((args.get("debug") or {}).get("explain"))
    _debug_meta: dict = {}
    """
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

        # ---- optional safety allowlists (env-driven) ----
    # Format examples:
    #   DBREAD_TABLE_ALLOWLIST="events,agents"
    #   DBREAD_COL_ALLOWLIST_events="id,topic,created_at,source_agent,latency_ms,payload"
    tbl_allow = {
        t.strip()
        for t in (os.getenv("DBREAD_TABLE_ALLOWLIST") or "").split(",")
        if t.strip()
    }
    if tbl_allow and table not in tbl_allow:
        raise ValueError(f"table '{table}' is not allowed")

    # Gather a column allowlist for this table if set
    col_allow_env = f"DBREAD_COL_ALLOWLIST_{table}"
    col_allow = {
        c.strip() for c in (os.getenv(col_allow_env) or "").split(",") if c.strip()
    }

    def _assert_allowed_column(colname: str) -> None:
        if not col_allow:
            return
        # Allow JSON roots e.g. payload.something — we validate the root
        root = colname.split(".", 1)[0]
        if root not in col_allow:
            raise ValueError(f"column '{root}' is not allowed for table '{table}'")

    if debug_explain:
        _debug_meta["request"] = {
            "table": table,
            "select": sel,
            "limit": args.get("limit"),
            "offset": args.get("offset"),
        }

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

    # debug.explain support
    debug_cfg = args.get("debug") or {}
    # ---- debug flag (echo the final query parts) ----
    debug_explain = bool((args.get("debug") or {}).get("explain"))
    _debug_meta: dict = {
        "enabled": debug_explain,
        "notes": [],
    }

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

    # ---- aggregate bundle ----
    agg = args.get("aggregate")
    agg_items: list[str] = []
    count_mode: str | None = None

    # capture caller's base select (before we append aggregates)
    _base_select = (args.get("select") or "").strip()

    def _alias(col: str, fn: str) -> str:
        # Use PostgREST-friendly projection with alias:
        #   min_created_at:created_at.min()
        return f"{fn}_{col}:{col}.{fn}()"

    _seen = set()

    def _push(col: str, fn: str):
        expr = _alias(col, fn)
        if expr not in _seen:
            _seen.add(expr)
            agg_items.append(expr)

    def _apply_count(mode: str | None):
        nonlocal count_mode
        explicit = (mode or "").strip().lower() if mode else None
        override = (args.get("aggregate_count_mode") or "").strip().lower()
        count_mode = explicit or override or "planned"

    def _parse_fn_col_token(token: str):
        # Accept "min:created_at", "max:created_at", "avg:latency_ms", "sum:latency_ms"
        t = token.strip()
        tl = t.lower()
        if tl == "count":
            _apply_count(None)
            return
        if tl.startswith("count:"):
            _, m = tl.split(":", 1)
            _apply_count(m)
            return
        if ":" not in t:
            # require a column for aggregates with PostgREST
            return
        fn, col = t.split(":", 1)
        fn = fn.strip().lower()
        col = col.strip()
        if fn in ("min", "max", "avg", "sum") and col:
            _push(col, fn)

    if isinstance(agg, list):
        for token in agg:
            if isinstance(token, str) and token.strip():
                _parse_fn_col_token(token)

    elif isinstance(agg, dict):
        # {"count":"planned"} | {"count":true} | {"count":{"mode":"planned"}}
        if "count" in agg:
            v = agg.get("count")
            if isinstance(v, dict) and "mode" in v:
                _apply_count(str(v.get("mode")))
            elif isinstance(v, str) and v.strip().lower() in (
                "exact",
                "planned",
                "estimated",
            ):
                _apply_count(v)
            elif v is True:
                _apply_count(None)
            else:
                _apply_count(str(agg.get("mode")) if agg.get("mode") else None)

        for fn in ("min", "max", "avg", "sum"):
            if fn in agg:
                col = str(agg[fn]).strip()
                if col:
                    _push(col, fn)

    elif isinstance(agg, str):
        _parse_fn_col_token(agg)

    # Append aggregates to select
    if agg_items:
        sel = f"{sel},{','.join(agg_items)}" if sel else ",".join(agg_items)

    # Prefer header for count (PostgREST way)
    if count_mode:
        prefer_headers = prefer_headers if "prefer_headers" in locals() else []
        prefer_headers.append(f"count={count_mode}")

    # --- PURE-AGG REGIME: if projection has ONLY aggregate outputs, disable ordering & force limit(0)
    # determine non-aggregate columns from the *original* caller select
    # determine non-aggregate columns from the *original* caller select
    _non_agg_cols = [c.strip() for c in _base_select.split(",") if c.strip()]
    pure_agg = bool(agg_items) and not _non_agg_cols

    if pure_agg:
        # keep only aggregate projections (strip anything else that may have been appended)
        sel = ",".join(agg_items)

        # block any ordering later
        args["order"] = None
        args["_force_no_order"] = True

        # no pagination for pure aggregates; keeps the request side-effect-free
        args["limit"] = 0

        try:
            order = None
        except Exception:
            pass

    if debug_explain:
        _debug_meta["aggregate"] = {
            "select_funcs": agg_items,
            "count_mode": count_mode,
            "pure_agg": pure_agg,
            "non_agg_cols": _non_agg_cols,
            "notes": (
                ["disabled order + forced limit(0) for pure-aggregate"]
                if pure_agg
                else []
            ),
        }

        # ----- column expression helper (shared by flat + grouped filters) -----

    def _colexpr(
        col: str,
        *,
        op: str | None = None,
        cast: str | None = None,
        text_mode: bool = True,
    ) -> str:
        """
        Normalize a column reference; supports JSON path via dot notation.
        IMPORTANT: PostgREST filter keys do not accept SQL casts. For JSON paths,
        we return text extraction (->>) without ::cast, because adding :: will be
        treated as an identifier and fail.
        """
        # Plain column: return as-is
        if not (isinstance(col, str) and "." in col):
            return col

        left, right = col.split(".", 1)

        # JSON roots we support
        if left in ("payload",):
            # Always use text extraction for filters
            return f"{left}->>{right}"

        # Dotted but not known JSON root — return as-is
        return col

    if count_mode:
        q = supabase.table(table).select(sel, count=count_mode)
    else:
        q = supabase.table(table).select(sel)

    if debug_explain:
        _debug_meta["select"] = sel

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

    # 1) where: accept {"and":[...]}, {"or":[...]}, and {"not": {...}|[...]} at top-level
    if isinstance(where, dict):
        if "and" in where and isinstance(where["and"], list):
            where = {"op": "and", "conditions": where["and"]}
        elif "or" in where and isinstance(where["or"], list):
            where = {"op": "or", "conditions": where["or"]}
        elif "not" in where and where["not"] is not None:
            conds = where["not"]
            if isinstance(conds, dict):
                conds = [conds]
            elif not isinstance(conds, list):
                raise ValueError("where.not must be an object or a list")
            where = {"op": "not", "conditions": conds}

    def _parse_order_string(s: str) -> list[dict]:
        items = []
        for part in (s or "").split(","):
            p = part.strip()
            if not p:
                continue
            toks = p.split()
            field = toks[0]
            dir_ = toks[1].lower() if len(toks) > 1 else "asc"
            if dir_ not in ("asc", "desc"):
                raise ValueError(f"invalid order direction for '{field}': {dir_}")
            items.append({"field": field, "desc": (dir_ == "desc")})
        return items

    # 2) order: tolerate {"column","ascending"} and map to {"field","desc"}
    def _normalize_order(o):
        if isinstance(o, str):
            return _parse_order_string(o)
        if isinstance(o, dict):
            if "column" in o and "field" not in o:
                field = o.get("column")
                ascending = bool(o.get("ascending", True))
                return {"field": field, "desc": (not ascending)}
            return {"field": o.get("field"), "desc": bool(o.get("desc", False))}
        if isinstance(o, list):
            out = []
            for item in o:
                if isinstance(item, dict):
                    out.append(_normalize_order(item))
                elif isinstance(item, str):
                    out.extend(_parse_order_string(item))
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
            and node["op"].lower() in {"and", "or", "not"}
        )

    def _normalize_leaf(node: object) -> tuple[str, str, object, str | None]:
        """
        Accept as leaf:
        - {"field":"topic","op":"eq","value":"...","cast":"int"}
        - {"topic":{"op":"eq","value":"...","cast":"int"}}
        - {"topic":"literal"}  (scalar eq)
        Return (field, op, value, cast) with op lowercased; default op is 'eq'.
        """
        if not isinstance(node, dict):
            raise ValueError(f"Invalid leaf node: {node}")
        if "field" in node:
            field = node["field"]
            op = (node.get("op") or "eq").lower()
            value = node.get("value")
            cast = node.get("cast")
            return field, op, value, cast
        if len(node) == 1:
            field, spec = next(iter(node.items()))
            if isinstance(spec, dict) and "op" in spec:
                return (
                    field,
                    (spec.get("op") or "eq").lower(),
                    spec.get("value"),
                    spec.get("cast"),
                )
            # scalar eq
            return field, "eq", spec, None
        raise ValueError(f"Unrecognized leaf shape: {node}")

    def _apply_leaf(qh, field: str, op: str, value: object, cast: str | None = None):
        """
        Apply one leaf predicate to the query handle.
        Fast-path eq/neq/in on top-level columns; use .filter(expr, op, val) for others/JSON.
        """
        expr = _colexpr(field, op=op, cast=cast, text_mode=True)

        # IN on real columns
        if op == "in":
            values = value or []
            if not isinstance(values, list):
                raise ValueError(f"{field}.op=in requires list 'value'")
            if expr == field:
                return qh.in_(field, values)
            raise ValueError("IN on JSON path not supported directly; use an OR group.")

        if op == "not_in":
            values = value or []
            if not isinstance(values, list):
                raise ValueError(f"{field}.op=not_in requires list 'value'")
            if expr == field:
                return qh.filter(field, "not.in", f"({','.join(map(str, values))})")
            raise ValueError(
                "NOT IN on JSON path not supported directly; use a NOT/OR group."
            )

        # eq / neq fast paths on real columns
        if op in ("eq", "=") and expr == field:
            return qh.eq(field, value)
        if op in ("ne", "neq", "!=") and expr == field:
            return qh.neq(field, value)

            # ---- pattern/regex ops (PostgREST docs: match=~, imatch=~*) ----
        if op == "match":  # regex, case-sensitive
            return q.filter(
                _colexpr(field, op=op, cast=cast, text_mode=True), "match", value
            )
        if op == "imatch":  # regex, case-insensitive
            return q.filter(
                _colexpr(field, op=op, cast=cast, text_mode=True), "imatch", value
            )

        # ---- "exists" for JSON paths (pragmatic version) ----
        # Treat as "json path not null" by using -> or ->> expression and NOT IS NULL.
        # NOTE: This checks presence of a key with a non-null value. If a key exists
        # but its value is JSON null, this will read as absent (which is usually OK).
        if op in ("exists", "has_key"):
            expr = _colexpr(
                field, op=op, cast=cast, text_mode=False
            )  # use -> for JSON, not text only
            # PostgREST "not.is.null" is the portable way to assert non-null
            return q.filter(expr, "not.is", "null")

        # ---- inclusive between (you already have "between" most likely) ----
        if op == "between_exclusive":
            # strict bounds: a < x < b
            if not (isinstance(value, (list, tuple)) and len(value) == 2):
                raise ValueError("between_exclusive expects a 2-item [low, high] array")
            low, high = value
            qh = q.gt(_colexpr(field, op=op, cast=cast, text_mode=False), low)
            return qh.lt(_colexpr(field, op=op, cast=cast, text_mode=False), high)

        if op == "not_between":
            # logical NOT of inclusive between: NOT(a <= x AND x <= b)
            if not (isinstance(value, (list, tuple)) and len(value) == 2):
                raise ValueError("not_between expects a 2-item [low, high] array")
            low, high = value
            col = _colexpr(field, op=op, cast=cast, text_mode=False)
            # Use a single boolean-expression param. See PostgREST logical operators.
            token = f"{col}.not.and({col}.gte.{_pg_encode(low)},{col}.lte.{_pg_encode(high)})"
            return q.or_(token)

        # other ops (or JSON) via .filter with the casted expr
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
        if op == "is":
            return qh.filter(
                expr, "is", "null" if value in (None, "null") else str(value)
            )
        if op == "not":
            if not isinstance(value, dict) or "op" not in value:
                raise ValueError(f"{field}.not requires nested {{op,value}}")
            iop = (value.get("op") or "eq").lower()
            ival = value.get("value")

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
                return qh.filter(expr, oper, patt)

            op_str = _map_simple_op(iop)
            val_str = "null" if op_str == "is" and ival in (None, "null") else str(ival)
            return qh.filter(expr, f"not.{op_str}", val_str)

            # --- regex ops (PostgREST ~ / ~*) ---
        if op == "match":  # case-sensitive regex
            expr = _colexpr(field, op=op, cast=cast, text_mode=True)
            return q.filter(expr, "match", value)
        if op == "imatch":  # case-insensitive regex
            expr = _colexpr(field, op=op, cast=cast, text_mode=True)
            return q.filter(expr, "imatch", value)

        # --- JSON key existence (pragmatic: "key present AND value not null") ---
        # Works for dotted JSON paths like payload.latency_ms
        if op in ("exists", "has_key"):
            # use JSON mode (->) so PostgREST treats it as JSON, then "not.is null"
            expr = _colexpr(field, op=op, cast=cast, text_mode=False)
            return q.filter(expr, "not.is", "null")

            # --- JSONB key existence family ---
        # NOTE: For dotted JSON paths (payload.foo.bar) these check the LAST segment
        #       as a key of the JSON object at the parent path.
        # --- JSON key present (practical): parent->>key is not null ---
        if op in ("has_key", "exists", "exists_strict"):
            coltxt = _colexpr(field, op=op, cast=cast, text_mode=True)
            return q.filter(coltxt, "not.is", "null")

        if op in ("has_keys_any", "exists_any"):
            # any of the keys present
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError("has_keys_any expects a non-empty array of keys")
            parent_expr, _ = _split_json_path(field)
            keys = "{" + ",".join(_pg_encode(k) for k in value) + "}"
            token = f"{parent_expr}.ov.{keys}"  # ov => overlaps
            return q.or_(token)

        if op in ("has_keys_all", "exists_all"):
            # all keys present
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError("has_keys_all expects a non-empty array of keys")
            parent_expr, _ = _split_json_path(field)
            keys = "{" + ",".join(_pg_encode(k) for k in value) + "}"
            token = f"{parent_expr}.cs.{keys}"  # cs => contains
            return q.or_(token)

            # --- array containment on ARRAY-typed columns ---
        # contains_all  -> col.cs.{a,b}     (array contains ALL of the given values)
        # contains_any  -> col.ov.{a,b}     (array overlaps ANY of the given values)
        # contained_by  -> col.cd.{a,b}     (array is contained BY the given set)
        if op in ("contains_all", "contains_any", "contained_by"):
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError(f"{field}.{op} expects a non-empty array value")

            # These operators are for *array* columns (NOT JSON paths).
            # If a dotted JSON path was provided, suggest reshaping the data or
            # querying via a view with a real array column.
            if "." in field:
                raise ValueError(
                    f"{field}.{op} is only supported on array columns (not JSON paths). "
                    "Consider a view that exposes a real array column."
                )

            # PostgREST array literal: {v1,v2,...}
            arr_lit = "{" + ",".join(str(v) for v in value) + "}"

            oper = {"contains_all": "cs", "contains_any": "ov", "contained_by": "cd"}[
                op
            ]
            qh = q.filter(field, oper, arr_lit)
            return qh

        # --- pattern ANY sugar (fold to one boolean token) ---
        # starts_with_any / ends_with_any / contains_any
        if op in (
            "starts_with_any",
            "istarts_with_any",
            "ends_with_any",
            "iends_with_any",
            "contains_any",
            "icontains_any",
        ):
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError(f"{op} expects a non-empty array of strings")
            case_ins = op.startswith(("i",)) or "i" in op.split("_")[0]
            kind = (
                "starts" if "starts" in op else ("ends" if "ends" in op else "contains")
            )
            oper = "ilike" if ("i" in op) else "like"
            col = _colexpr(field, op=oper, cast=cast, text_mode=True)

            patt_list = []
            for v in value:
                if kind == "starts":
                    patt_list.append(f"{v}%")
                elif kind == "ends":
                    patt_list.append(f"%{v}")
                else:
                    patt_list.append(f"%{v}%")

            # Build one token or(p1,p2,...) with not needed here
            parts = [f"{col}.{oper}.{_pg_encode(p)}" for p in patt_list]
            token = f"or({','.join(parts)})"
            return q.or_(token)

        # --- strict range: a < x < b ---
        if op == "between_exclusive":
            pair = value if isinstance(value, (list, tuple)) else None
            if pair is None:
                # support {values:[...]} too
                pair = (
                    node.get("values")
                    if isinstance(node := {"value": value}, dict)
                    else None
                )  # harmless fallback
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                raise ValueError(
                    "between_exclusive expects a 2-item [low, high] array as 'value' or 'values'"
                )
            low, high = pair
            col = _colexpr(field, op=op, cast=cast, text_mode=False)
            qh2 = q.gt(col, low)
            return qh2.lt(col, high)

        # --- NOT of inclusive between: NOT(a <= x AND x <= b) ---
        if op == "not_between":
            pair = value if isinstance(value, (list, tuple)) else None
            if pair is None:
                # support {values:[...]} too
                pair = (
                    node.get("values")
                    if isinstance(node := {"value": value}, dict)
                    else None
                )  # harmless fallback
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                raise ValueError(
                    "not_between expects a 2-item [low, high] array as 'value' or 'values'"
                )
            low, high = pair
            col = _colexpr(field, op=op, cast=cast, text_mode=False)
            # Build ONE boolean token using PostgREST logical operator grammar:
            # not.and(col.gte.low,col.lte.high)
            token = f"not.and({col}.gte.{_pg_encode(low)},{col}.lte.{_pg_encode(high)})"
            return q.or_(token)

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
        if re.match(r"^[0-9a-fA-F-]{36}$", s):  # UUID-like
            return s

        # Encode only the problematic characters (plus spaces as a safety net)
        return (
            s.replace("(", "%28")
            .replace(")", "%29")
            .replace(",", "%2C")
            .replace(" ", "%20")
        )

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

    def _split_json_path(field: str) -> tuple[str, str]:
        """
        Split dotted JSON path "payload.foo.bar" into:
        (parent_expr, last_key) where parent_expr is the PostgREST expr for the parent JSON,
        using -> (JSON) to keep it as JSON for key ops.
        """
        if "." not in field:
            # treat whole column as parent; last_key is the column itself
            return field, field
        parts = field.split(".")
        parent = parts[:-1]
        last = parts[-1]
        # Build parent expr with JSON operator chain (->), starting from base column
        base = parent[0]
        chain = "".join([f"->{p}" for p in parent[1:]]) if len(parent) > 1 else ""
        return f"{base}{chain}", last

    def _parse_relative_time(spec: object) -> str:
        """
        Convert simple relative tokens to ISO8601 UTC:
        - "now"                    -> now
        - "-24h", "-7d", "-30m"    -> now + negative delta
        - "2025-08-16T12:34:56Z"   -> left as-is
        Returns an ISO8601 string with timezone (Z or +00:00).
        """
        from datetime import datetime, timedelta, timezone

        if spec is None:
            raise ValueError("relative time value is required")

        s = str(spec).strip().lower()
        if s == "now":
            return datetime.now(timezone.utc).isoformat()

        # ISO-ish? keep as-is
        import re as _re

        if _re.match(r"^\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}", s, _re.I):
            return s

        m = _re.match(r"^(-?\d+)([smhdw])$", s)  # seconds/minutes/hours/days/weeks
        if not m:
            raise ValueError(f"unsupported relative time '{spec}' (try -24h, -7d, now)")

        n = int(m.group(1))
        unit = m.group(2)
        mult = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]
        delta = timedelta(seconds=n * mult)
        ts = datetime.now(timezone.utc) + delta
        return ts.isoformat()

    def _op_is_textual(op: str) -> bool:
        """
        Decide whether comparisons should treat JSON paths as text (->>) or JSON (->).
        Textual ops use ->>; numeric/range ops prefer ->.
        """
        op = (op or "").lower()
        return op in {
            "eq",
            "=",
            "neq",
            "ne",
            "!=",
            "like",
            "ilike",
            "match",
            "imatch",
            "contains",
            "icontains",
            "starts_with",
            "istarts_with",
            "ends_with",
            "iends_with",
            "is",
            "is_null",
            "not_null",
            "isnull",
            "notnull",
            "is-null",
            "not-null",
            "in",
            "not_in",
        }

    def _as_filter_token(field: str, op: str, value: object, cast: str | None) -> str:
        """
        Convert a single *leaf* into a PostgREST boolean token fragment usable inside
        or(...), and(...), not.and(...), etc.

        Examples:
        col.eq.value                 -> "col.eq.value"
        col.in.(a,b)                -> "col.in.(a,b)"
        not.in                      -> "col.not.in.(a,b)"
        regex                       -> "col.imatch.^foo.*$"
        JSON key presence           -> "parent.cs.{key}" or "parent.ov.{k1,k2}"
        *any pattern sugar          -> "or(col.like.p1,col.like.p2,...)"
        """
        op = (op or "eq").lower()

        # 1) ANY pattern sugar → build a *single* or(...) token
        if op in (
            "starts_with_any",
            "istarts_with_any",
            "ends_with_any",
            "iends_with_any",
            "contains_any",
            "icontains_any",
        ):
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError(f"{op} expects a non-empty array of strings")
            kind = (
                "starts" if "starts" in op else ("ends" if "ends" in op else "contains")
            )
            oper = "ilike" if op.startswith("i") else "like"
            col = _colexpr(field, op=oper, cast=cast, text_mode=True)
            parts = []
            for v in value:
                if kind == "starts":
                    patt = f"{v}%"
                elif kind == "ends":
                    patt = f"%{v}"
                else:
                    patt = f"%{v}%"
                parts.append(f"{col}.{oper}.{_pg_encode(patt)}")
            return f"or({','.join(parts)})"

        # 2) JSON key presence family
        if op in ("has_key", "exists", "exists_strict"):
            coltxt = _colexpr(field, op=op, cast=cast, text_mode=True)
            return f"{coltxt}.not.is.null"

            # relative time helpers for grouped boolean expressions
        if op == "since":
            start_iso = _parse_relative_time(value)
            col = _colexpr(field, op=op, cast=cast, text_mode=True)
            return f"{col}.gte.{_pg_encode(start_iso)}"

        if op == "between_relative":
            if not isinstance(value, dict):
                raise ValueError(
                    f"{field}.between_relative requires object value with start/end"
                )
            start_iso = _parse_relative_time(value.get("start"))
            end_iso = _parse_relative_time(value.get("end") or "now")
            col = _colexpr(field, op=op, cast=cast, text_mode=True)
            return f"and({col}.gte.{_pg_encode(start_iso)},{col}.lte.{_pg_encode(end_iso)})"

        if op in ("has_keys_any", "exists_any"):
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError("has_keys_any expects a non-empty array of keys")
            parent_expr, _ = _split_json_path(field)
            keys = "{" + ",".join(_pg_encode(k) for k in value) + "}"
            # overlaps of key sets -> parent.ov.{k1,k2}
            return f"{parent_expr}.ov.{keys}"

        if op in ("has_keys_all", "exists_all"):
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError("has_keys_all expects a non-empty array of keys")
            parent_expr, _ = _split_json_path(field)
            keys = "{" + ",".join(_pg_encode(k) for k in value) + "}"
            # contains all keys -> parent.cs.{k1,k2}
            return f"{parent_expr}.cs.{keys}"

        # 3) Regex ops
        if op in ("match", "imatch"):
            col = _colexpr(field, op=op, cast=cast, text_mode=True)
            return f"{col}.{op}.{_pg_encode(value)}"

        # 4) IN / NOT IN
        if op == "in":
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError("IN requires a non-empty list")
            col = _colexpr(field, op=op, cast=cast, text_mode=_op_is_textual(op))
            inner = ",".join(_pg_encode(v) for v in value)
            return f"{col}.in.({inner})"

        if op == "not_in":
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError("NOT IN requires a non-empty list")
            col = _colexpr(field, op=op, cast=cast, text_mode=_op_is_textual(op))
            inner = ",".join(_pg_encode(v) for v in value)
            return f"{col}.not.in.({inner})"

        # 5) Simple scalar/pattern comparisons (eq/neq/gt/gte/lt/lte/like/ilike/is)
        pgop = {
            "eq": "eq",
            "=": "eq",
            "ne": "neq",
            "neq": "neq",
            "!=": "neq",
            "gt": "gt",
            "gte": "gte",
            "lt": "lt",
            "lte": "lte",
            "like": "like",
            "ilike": "ilike",
            "is": "is",
        }.get(op, op)

        col = _colexpr(field, op=op, cast=cast, text_mode=_op_is_textual(op))
        return f"{col}.{pgop}.{_pg_encode(value)}"

    def _as_group_token(node: dict) -> str:
        """
        Recursively serialize a group node into a single boolean token:
        and(a,b), or(a,b), or not.and(...)/not.or(...)
        """
        if _is_group(node):
            op = (node.get("op") or "").lower()
            conds = node.get("conditions") or []
            if not isinstance(conds, list):
                conds = [conds]
            parts = []
            for c in conds:
                if _is_group(c):
                    parts.append(_as_group_token(c))
                else:
                    fld, cop, val, cst = _normalize_leaf(c)
                    parts.append(_as_filter_token(fld, cop, val, cst))
            if op == "and":
                return f"and({','.join(parts)})"
            if op == "or":
                return f"or({','.join(parts)})"
            if op == "not":
                # default to NOT of AND if unspecified; if the single child is an OR,
                # caller should have normalized already in _apply_group.
                return f"not.and({','.join(parts)})"
            raise ValueError(f"unsupported group op '{op}'")
        # leaf
        fld, cop, val, cst = _normalize_leaf(node)
        _assert_allowed_column(fld)
        return _as_filter_token(fld, cop, val, cst)

    def _serialize_node_to_postgrest(node: dict) -> str:
        if _is_group(node):
            op = node["op"].lower()
            parts = [_serialize_node_to_postgrest(c) for c in node["conditions"]]

            if op == "and":
                return f"and({','.join(parts)})"
            if op == "or":
                return f"or({','.join(parts)})"
            # --- NEW: group NOT ---
            if op == "not":
                # If there is exactly one child:
                if len(node["conditions"]) == 1:
                    child = node["conditions"][0]
                    inner = _serialize_node_to_postgrest(child)
                    if _is_group(child):
                        # child already serialized like "and(...)" or "or(...)" → "not.and(...)" / "not.or(...)"
                        return f"not.{inner}"
                    # child is a leaf → wrap as NOT of an AND group for valid PostgREST syntax
                    return f"not.and({inner})"
                # Multiple children → NOT of an AND group
                return f"not.and({','.join(parts)})"

        # leaf
        field, op, value, cast = _normalize_leaf(node)
        col = _colexpr(field, op=op, cast=cast, text_mode=True)

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
            else:
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
                operator = (
                    "not.ilike"
                    if iop in ("icontains", "istarts_with", "iends_with")
                    else "not.like"
                )
                return f"{col}.{operator}.{_pg_encode(patt)}"

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

    # ---- record boolean expression when present ----
    if debug_explain:
        try:
            # If we have a canonical group, capture the exact token
            if isinstance(where, dict) and ("op" in where and "conditions" in where):
                _debug_meta["where_token"] = _serialize_node_to_postgrest(where)
            else:
                # Fallback: summarize flat where keys to avoid heavy duplication
                _debug_meta["where_flat_keys"] = sorted(list((where or {}).keys()))
        except Exception as e:
            _debug_meta["notes"].append(f"where debug skipped: {type(e).__name__}: {e}")

    def _apply_group(qh, node: dict):
        """
        Apply a boolean group node of the form:
        {"op":"and"|"or"|"not", "conditions":[ ... ]}
        - Leaves are delegated to _apply_leaf(...)
        - AND: recursively apply each child (implicit AND in PostgREST)
        - OR: build ONE boolean token "or(a,b,...)" and send via qh.or_(...)
        - NOT: build ONE token "not.and(...)" or "not.or(...)" and send via qh.or_(...)
        """
        # Leaf short-circuit
        if not _is_group(node):
            fld, op, val, cast = _normalize_leaf(node)
            _assert_allowed_column(fld)
            return _apply_leaf(qh, fld, op, val, cast)

        op = (node.get("op") or "").lower()
        conds = node.get("conditions") or []
        if not isinstance(conds, list):
            conds = [conds]

        # ---------- group-level NOT ----------
        if op == "not":
            # If NOT wraps a single subgroup that has an 'op', keep that op so we can emit
            # not.and(...) or not.or(...). Otherwise, treat as NOT(AND(...)).
            children = conds
            negate_as = "and"
            if len(children) == 1 and isinstance(children[0], dict):
                c0 = children[0]
                inner = (c0.get("op") or "").lower() if "op" in c0 else None
                if inner in ("and", "or"):
                    negate_as = inner
                    # replace with inner group's conditions to avoid nested parens bloat
                    children = c0.get("conditions") or []
                    if not isinstance(children, list):
                        children = [children]

            parts: list[str] = []
            for ch in children:
                if _is_group(ch):
                    parts.append(_as_group_token(ch))
                else:
                    fld, cop, val, cst = _normalize_leaf(ch)
                    parts.append(_as_filter_token(fld, cop, val, cst))

            token = f"not.{negate_as}({','.join(parts)})"
            return qh.or_(token)

        # ---------- OR fast-path: eq on same column => IN ----------
        if op == "or":
            same_col = None
            values = []
            all_eq_same_col = True
            for ch in conds:
                if _is_group(ch):
                    all_eq_same_col = False
                    break
                fld, cop, val, _ = _normalize_leaf(ch)  # 4-tuple; ignore cast here
                _assert_allowed_column(fld)
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
                # Only safe to use qh.in_(...) when it's a *real* column (not a JSON path)
                expr = _colexpr(
                    same_col, op="in", cast=None, text_mode=_op_is_textual("in")
                )
                if expr == same_col:
                    return qh.in_(same_col, values)
                # JSON path IN not supported directly: fall through to general OR token
                # to avoid exploding into many filters; token form keeps it to one .or_(...)
                # (col.eq.a,col.eq.b,...) is equivalent.
            # General OR: build one boolean token
            parts: list[str] = []
            for ch in conds:
                if _is_group(ch):
                    parts.append(_as_group_token(ch))
                else:
                    fld, cop, val, cst = _normalize_leaf(ch)
                    parts.append(_as_filter_token(fld, cop, val, cst))
            token = f"or({','.join(parts)})"
            return qh.or_(token)

        # ---------- AND (default): apply children sequentially ----------
        # In PostgREST, multiple filters are ANDed, so we just recurse/apply in order.
        for ch in conds:
            qh = (
                _apply_group(qh, ch)
                if _is_group(ch)
                else _apply_leaf(qh, *_normalize_leaf(ch))
            )
        return qh

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
        # --------------------------
        # Flat filters (unified with _colexpr)
        # --------------------------
        def _apply_pred(col: str, spec: Any):
            nonlocal q
            _assert_allowed_column(col)
            # equality shortcut (scalar)
            if not isinstance(spec, dict):
                expr = _colexpr(col, op="eq", cast=None, text_mode=True)
                q = q.eq(col, spec) if expr == col else q.filter(expr, "eq", str(spec))
                return

            op = (spec.get("op") or "eq").lower()
            val = spec.get("value")
            cast = spec.get("cast")
            expr_text = _colexpr(col, op=op, cast=cast, text_mode=True)
            expr_json = _colexpr(col, op=op, cast=cast, text_mode=False)

            # --- IN / NOT IN ---
            if ("in" in spec) or op == "in":
                values = spec.get("in") if "in" in spec else val
                if not isinstance(values, list):
                    raise ValueError(f"{col}.in must be a list")
                if expr_text != col:
                    # Guardrail: IN on JSON path can’t be expressed cleanly; ask caller to OR
                    raise ValueError(
                        "IN on JSON path not supported directly; use an OR group."
                    )
                q = q.in_(col, values)
                return

            if op == "not_in":
                if not isinstance(val, (list, tuple)) or len(val) == 0:
                    raise ValueError(
                        f"where.{col}.value must be a non-empty list for op 'not_in'"
                    )
                csv = ",".join(map(str, val))
                q = q.filter(expr_text, "not.in", f"({csv})")
                return

            # --- regex (PostgREST ~ / ~*) ---
            if op == "match":
                q = q.filter(expr_text, "match", str(val))
                return
            if op == "imatch":
                q = q.filter(expr_text, "imatch", str(val))
                return

            # --- JSON key existence (pragmatic: key present AND value not null) ---
            if op in ("exists", "has_key"):
                q = q.filter(expr_json, "not.is", "null")
                return

                # --- relative time helpers ---
            if op == "since":
                # value: "7d", "-24h", "now", or an ISO timestamp
                start_iso = _parse_relative_time(val)
                q = q.filter(expr_text, "gte", start_iso)
                return

            if op == "between_relative":
                # value: {"start":"-24h"|"now"|ISO, "end":"now"|ISO|"+0s"}
                if not isinstance(val, dict):
                    raise ValueError(
                        f"{col}.between_relative requires object value with start/end"
                    )
                start_iso = _parse_relative_time(val.get("start"))
                end_iso = _parse_relative_time(val.get("end") or "now")
                q = q.filter(expr_text, "gte", start_iso).filter(
                    expr_text, "lte", end_iso
                )
                return

            # --- between variants ---
            if op == "between":
                if not (isinstance(val, (list, tuple)) and len(val) == 2):
                    raise ValueError(f"{col}.between requires [low, high]")
                lo, hi = val
                if expr_text == col:
                    q = q.gte(col, lo).lte(col, hi)
                else:
                    q = q.filter(expr_text, "gte", str(lo)).filter(
                        expr_text, "lte", str(hi)
                    )
                return

            if op == "between_exclusive":
                if not (isinstance(val, (list, tuple)) and len(val) == 2):
                    raise ValueError(
                        "between_exclusive expects a 2-item [low, high] array"
                    )
                lo, hi = val
                if expr_json == col:
                    q = q.gt(col, lo).lt(col, hi)
                else:
                    q = q.filter(expr_json, "gt", str(lo)).filter(
                        expr_json, "lt", str(hi)
                    )
                return

            if op == "not_between":
                if not (isinstance(val, (list, tuple)) and len(val) == 2):
                    raise ValueError("not_between expects a 2-item [low, high] array")
                lo, hi = val
                # Build one boolean token: not.and(col.gte.lo,col.lte.hi)
                coltok = expr_json
                token = f"not.and({coltok}.gte.{_pg_encode(lo)},{coltok}.lte.{_pg_encode(hi)})"
                if debug_explain:
                    _debug_meta.setdefault("keyset_or", []).append(token)
                q = q.or_(token)
                return

            # --- simple comparisons on real column fast-paths ---
            if op in ("eq", "="):
                q = (
                    q.eq(col, val)
                    if expr_text == col
                    else q.filter(expr_text, "eq", str(val))
                )
                return
            if op in ("ne", "neq", "!="):
                q = (
                    q.neq(col, val)
                    if expr_text == col
                    else q.filter(expr_text, "neq", str(val))
                )
                return
            if op in ("gt", ">"):
                q = (
                    q.gt(col, val)
                    if expr_text == col
                    else q.filter(expr_text, "gt", str(val))
                )
                return
            if op in ("gte", ">="):
                q = (
                    q.gte(col, val)
                    if expr_text == col
                    else q.filter(expr_text, "gte", str(val))
                )
                return
            if op in ("lt", "<"):
                q = (
                    q.lt(col, val)
                    if expr_text == col
                    else q.filter(expr_text, "lt", str(val))
                )
                return
            if op in ("lte", "<="):
                q = (
                    q.lte(col, val)
                    if expr_text == col
                    else q.filter(expr_text, "lte", str(val))
                )
                return

                # --- array containment on ARRAY-typed columns (flat path) ---
            # contains_all  -> col.cs.{a,b}     (array contains ALL of the given values)
            # contains_any  -> col.ov.{a,b}     (array overlaps ANY of the given values)
            # contained_by  -> col.cd.{a,b}     (array is contained BY the given set)
            if op in ("contains_all", "contains_any", "contained_by"):
                if not isinstance(val, (list, tuple)) or not val:
                    raise ValueError(
                        f"where.{col}.{op} requires a non-empty list value"
                    )
                # These operators are for *array* columns (NOT dotted JSON paths)
                if "." in col:
                    raise ValueError(
                        f"{col}.{op} is only supported on array columns (not JSON paths). "
                        "Consider using a view that exposes a real array column."
                    )
                arr_lit = "{" + ",".join(str(v) for v in val) + "}"
                oper = {
                    "contains_all": "cs",
                    "contains_any": "ov",
                    "contained_by": "cd",
                }[op]
                return q.filter(col, oper, arr_lit)

            # --- pattern helpers ---
            if op == "like":
                q = q.filter(expr_text, "like", str(val))
                return
            if op == "ilike":
                q = q.filter(expr_text, "ilike", str(val))
                return
            if op in ("contains", "icontains"):
                patt = f"%{val}%"
                q = q.filter(expr_text, "ilike" if op == "icontains" else "like", patt)
                return
            if op in ("starts_with", "istarts_with"):
                patt = f"{val}%"
                q = q.filter(
                    expr_text, "ilike" if op == "istarts_with" else "like", patt
                )
                return
            if op in ("ends_with", "iends_with"):
                patt = f"%{val}"
                q = q.filter(expr_text, "ilike" if op == "iends_with" else "like", patt)
                return

            # --- null checks / explicit IS ---
            if op in ("is_null", "isnull", "is-null"):
                q = q.filter(expr_text, "is", "null")
                return
            if op in ("not_null", "notnull", "not-null"):
                q = q.filter(expr_text, "not.is", "null")
                return
            if op == "is":
                q = q.filter(
                    expr_text, "is", "null" if val in (None, "null") else str(val)
                )
                return

                # --- JSONB key existence family (flat path) ---
            # --- JSON key present (practical): parent->>key is not null ---
            if op in ("has_key", "exists", "exists_strict"):
                expr_text = _colexpr(
                    col, op=op, cast=cast, text_mode=True
                )  # uses ->> for JSON paths
                q = q.filter(expr_text, "not.is", "null")
                return

            if op in ("has_keys_any", "exists_any"):
                if not isinstance(val, (list, tuple)) or not val:
                    raise ValueError("has_keys_any expects a non-empty array of keys")
                parent_expr, _ = _split_json_path(col)
                keys = "{" + ",".join(_pg_encode(k) for k in val) + "}"
                q = q.or_(f"{parent_expr}.ov.{keys}")
                return

            if op in ("has_keys_all", "exists_all"):
                if not isinstance(val, (list, tuple)) or not val:
                    raise ValueError("has_keys_all expects a non-empty array of keys")
                parent_expr, _ = _split_json_path(col)
                keys = "{" + ",".join(_pg_encode(k) for k in val) + "}"
                q = q.or_(f"{parent_expr}.cs.{keys}")
                return

            # --- pattern ANY sugar (flat path) ---
            if op in (
                "starts_with_any",
                "istarts_with_any",
                "ends_with_any",
                "iends_with_any",
                "contains_any",
                "icontains_any",
            ):
                if not isinstance(val, (list, tuple)) or not val:
                    raise ValueError(f"{op} expects a non-empty array of strings")
                case_ins = op.startswith(("i",)) or "i" in op.split("_")[0]
                kind = (
                    "starts"
                    if "starts" in op
                    else ("ends" if "ends" in op else "contains")
                )
                oper = "ilike" if ("i" in op) else "like"
                col_expr = _colexpr(col, op=oper, cast=cast, text_mode=True)

                patt_list = []
                for v in val:
                    if kind == "starts":
                        patt_list.append(f"{v}%")
                    elif kind == "ends":
                        patt_list.append(f"%{v}")
                    else:
                        patt_list.append(f"%{v}%")

                parts = [f"{col_expr}.{oper}.{_pg_encode(p)}" for p in patt_list]
                q = q.or_(f"or({','.join(parts)})")
                return

            # --- NOT (covers simple + pattern NOTs) ---
            if op == "not":
                inner = val if isinstance(val, dict) else {"op": "eq", "value": val}
                iop = (inner.get("op") or "eq").lower()
                ival = inner.get("value")

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
                    q = q.filter(expr_text, oper, patt)
                    return

                op_str = _map_simple_op(iop)
                crit = (
                    "null" if (op_str == "is" and ival in (None, "null")) else str(ival)
                )
                q = q.filter(expr_text, f"not.{op_str}", crit)
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

    def _cursor_sign(payload: dict, secret: str) -> str:
        msg = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        mac = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(mac).decode("ascii").rstrip("=")

    def _cursor_verify(payload: dict, sig: str, secret: str) -> bool:
        try:
            expected = _cursor_sign(payload, secret)
            # constant-time compare
            return hmac.compare_digest(expected, sig)
        except Exception:
            return False

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

        if debug_explain:
            _debug_meta.setdefault("cursor", {})["or_disjuncts"] = list(disjuncts)

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
        # Optional sealing
        secret = os.getenv("CURSOR_SIGNING_SECRET")
        if secret:
            # If a signature is provided, verify; if not, we still proceed (first page).
            sig = cursor.get("sig")
            # Consider only the directional part for signature (after/before)
            payload_for_sig = {}
            if "after" in cursor and isinstance(cursor["after"], dict):
                payload_for_sig = {"mode": "keyset", "after": cursor["after"]}
            elif "before" in cursor and isinstance(cursor["before"], dict):
                payload_for_sig = {"mode": "keyset", "before": cursor["before"]}

            if payload_for_sig and (
                not sig or not _cursor_verify(payload_for_sig, sig, secret)
            ):
                raise ValueError("invalid or missing cursor signature")

        q = _apply_keyset_cursor(q, normalized_order, cursor)

    # If caller didn’t pass order but gave a tiebreaker, apply it.
    if (
        not applied_any_order
        and isinstance(tiebreaker, dict)
        and tiebreaker.get("field")
    ):
        q = _apply_order(q, tiebreaker)

    # If still no order was applied, fall back to a stable tiebreaker
    if not applied_any_order and not args.get("_force_no_order"):
        q = _apply_order(q, {"field": "id", "desc": False})
        normalized_order = [{"field": "id", "desc": False}]

    if isinstance(limit, int) and limit > 0:
        q = q.limit(limit)
    if isinstance(offset, int) and offset >= 0:
        q = q.range(
            offset, (offset + (limit or 100)) - 1
        )  # supabase uses inclusive ranges

    if debug_explain:
        _debug_meta["limit"] = args.get("limit")
        _debug_meta["offset"] = args.get("offset")
        agg = args.get("aggregate") or {}
        if "count" in agg:
            _debug_meta["count_mode"] = agg.get("count")

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

    # --- attach debug echo if requested ---
    if debug_explain and _debug_meta:
        result.setdefault("meta", {})["debug"] = _debug_meta

    # Keyset: include next_cursor if possible
    if isinstance(rows, list) and rows and normalized_order:
        tail = rows[-1]
        nc = _build_cursor_from_row(tail, normalized_order)
        if nc:
            nxt = {"mode": "keyset", **nc}
            secret = os.getenv("CURSOR_SIGNING_SECRET")
            if secret:
                nxt["sig"] = _cursor_sign(nxt, secret)
            result.setdefault("meta", {})["next_cursor"] = nxt

    if debug_explain:
        result.setdefault("meta", {})["debug"] = _debug_meta

    return result


def _db_write_adapter(args: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
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


def _env_ports(name: str, default: set[int]) -> set[int]:
    raw = os.getenv(name, "")
    vals = {p.strip() for p in raw.split(",") if p.strip()}
    parsed = {int(p) for p in vals if p.isdigit()}
    return parsed or set(default)


def _resolve_addrs(host: str) -> list[str]:
    addrs: list[str] = []
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            infos = socket.getaddrinfo(host, None, family, socket.SOCK_STREAM)
            for _, _, _, _, sockaddr in infos:
                addrs.append(sockaddr[0])
        except socket.gaierror:
            continue
    return addrs


def _is_disallowed_ip_host(host: str) -> bool:
    """
    Block private/loopback/link-local/unspecified/multicast and IMDS.
    Safer default on resolution failure.
    """
    try:
        for ip_str in _resolve_addrs(host):
            ip = ipaddress.ip_address(ip_str)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or ip.is_unspecified
            ):
                return True
            if ip_str == "169.254.169.254":  # cloud instance metadata
                return True
        return False
    except Exception:
        return True


def _host_port_allowed(parsed) -> Tuple[bool, Optional[str]]:
    host = (parsed.hostname or "").lower()
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    allowed_ports = _env_ports("HTTP_FETCH_ALLOWED_PORTS", HTTP_DEFAULT_ALLOWED_PORTS)
    if port not in allowed_ports:
        return False, f"Port {port} not allowed for host {host}"
    return True, None

    # --- decoding helpers (body + charset) ---
    def _decompress(raw: bytes, enc: str | None) -> bytes:
        """Return decompressed bytes (supports gzip, deflate). Unknown -> raw."""
        if not raw or not enc:
            return raw
        enc = enc.lower().strip()
        try:
            if enc == "gzip":
                return gzip.decompress(raw)
            if enc == "deflate":
                # Some servers send deflate as zlib-wrapped, others raw DEFLATE.
                try:
                    return zlib.decompress(raw)
                except zlib.error:
                    return zlib.decompress(raw, -zlib.MAX_WBITS)
            # If you later add brotli:
            # if enc in ("br","brotli") and brotli:
            #     return brotli.decompress(raw)
        except Exception:
            # If decompression fails, fall back to raw so callers still get something
            return raw
        return raw

    def _detect_charset(content_type: str | None) -> str:
        """Extract charset from Content-Type; default utf-8."""
        if not content_type:
            return "utf-8"
        m = re.search(r"charset\s*=\s*([^\s;]+)", content_type, flags=re.I)
        return m.group(1).strip().strip('"').strip("'") if m else "utf-8"


def _json_pointer_get(doc: Any, pointer: str) -> Any:
    """
    Minimal RFC6901-ish JSON Pointer: "/a/0/b"
    Returns None on traversal failure.
    """
    if pointer == "" or pointer == "/":
        return doc
    if not pointer.startswith("/"):
        return None
    cur = doc
    for raw_token in pointer.split("/")[1:]:
        token = raw_token.replace("~1", "/").replace("~0", "~")
        try:
            if isinstance(cur, list):
                idx = int(token)
                cur = cur[idx]
            elif isinstance(cur, dict):
                cur = cur.get(token)
            else:
                return None
        except Exception:
            return None
    return cur


def _url_with_param(base_url: str, key: str, value: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    q = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    q[key] = value
    new_q = urllib.parse.urlencode(q, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=new_q))


def _looks_like_url(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))


def _parse_link_header_next(link_value: str, want_rel: str = "next") -> str | None:
    """
    Parse RFC5988 Link header to find rel="next".
    Example:
      Link: <https://api.example.com/items?page=3>; rel="next", <...>; rel="prev"
    """
    try:
        parts = [p.strip() for p in link_value.split(",")]
        for part in parts:
            if ";" not in part:
                continue
            url_part, *params = [x.strip() for x in part.split(";")]
            if not (url_part.startswith("<") and url_part.endswith(">")):
                continue
            url_str = url_part[1:-1]
            rel_val = None
            for p in params:
                if p.lower().startswith("rel="):
                    rel_val = p.split("=", 1)[1].strip().strip('"').strip("'")
                    break
            if rel_val == want_rel:
                return url_str
    except Exception:
        pass
    return None


def _get_header_case_insensitive(hdict: dict[str, str], name: str) -> str | None:
    ln = name.lower()
    for k, v in hdict.items():
        if k.lower() == ln:
            return v
    return None


def _http_fetch(args: "HttpFetchArgs", meta: "MetaDict") -> Dict[str, Any]:
    """
    Minimal, safe HTTP client for agents.
    Args:
      url: str               -- required
      method: str            -- GET|POST|PUT|PATCH|DELETE (default GET)
      headers: dict          -- optional request headers
      params: dict           -- optional query params (merged into url)
      body: str|bytes|dict|list -- optional; dict/list will be JSON-encoded
      timeout: int|float     -- seconds (default 15)
      allow_redirects: bool  -- currently follows by default (urllib)
      max_bytes: int         -- cap read size (default 2_000_000)
      paginate: {            -- optional pagination
        mode: "cursor" | "header_link" | "page",
        # cursor mode:
        json_pointer: str,         # e.g. "/next"
        cursor_param: str,         # default "cursor" if pointer returns a token
        # header_link mode:
        header: str,               # default "Link"
        rel: str,                  # default "next" (only for Link header)
        # page mode:
        page_param: str,           # default "page"
        start_page: int,           # default 1
        # shared:
        max_pages: int,            # default 10
        items_pointer: str,        # e.g. "/items"
        concat_json: bool,         # default False
        delay_ms: int,             # polite delay between pages (ms)
        delay_jitter_ms: int       # extra random 0..jitter ms
      }
    Env allow/deny:
      HTTP_FETCH_ALLOW_HOSTS="api1.com,api2.com"
      HTTP_FETCH_DENY_HOSTS="bad.com,evil.test"
    Returns:
      {
        "status": int,
        "url": str,
        "final_url": str,
        "headers": { ... },
        "bytes_len": int,
        "elapsed_ms": int,
        "json": {...} | None,
        "text": str | None,
        "truncated": bool,
        "not_modified": bool,
        "pagination": {
          "pages": [{"status", "url", "bytes_len"}, ...],
          "pages_count": int,
          "items": [...],
          "items_count": int
        }                           # when pagination used
      }
    """
    t0 = time.time()

    # --- tiny local helpers used by pagination ---
    def _json_pointer_get(doc: Any, pointer: str) -> Any:
        if pointer in ("", "/"):
            return doc
        if not isinstance(pointer, str) or not pointer.startswith("/"):
            return None
        cur = doc
        for raw_token in pointer.split("/")[1:]:
            token = raw_token.replace("~1", "/").replace("~0", "~")
            try:
                if isinstance(cur, list):
                    cur = cur[int(token)]
                elif isinstance(cur, dict):
                    cur = cur.get(token)
                else:
                    return None
            except Exception:
                return None
        return cur

    def _url_with_param(base_url: str, key: str, value: str) -> str:
        parsed0 = urllib.parse.urlparse(base_url)
        q = dict(urllib.parse.parse_qsl(parsed0.query, keep_blank_values=True))
        q[key] = value
        new_q = urllib.parse.urlencode(q, doseq=True)
        return urllib.parse.urlunparse(parsed0._replace(query=new_q))

    def _looks_like_url(s: Any) -> bool:
        return isinstance(s, str) and (
            s.startswith("http://") or s.startswith("https://")
        )

    def _parse_link_header_next(link_value: str, want_rel: str = "next") -> str | None:
        """
        Parse RFC5988 Link header to find rel="next".
        Example: Link: <https://api.example.com/items?page=3>; rel="next", <...>; rel="prev"
        """
        try:
            parts = [p.strip() for p in link_value.split(",")]
            for part in parts:
                if ";" not in part:
                    continue
                url_part, *params = [x.strip() for x in part.split(";")]
                if not (url_part.startswith("<") and url_part.endswith(">")):
                    continue
                url_str = url_part[1:-1]
                rel_val = None
                for p in params:
                    if p.lower().startswith("rel="):
                        rel_val = p.split("=", 1)[1].strip().strip('"').strip("'")
                        break
                if rel_val == want_rel:
                    return url_str
        except Exception:
            pass
        return None

    def _get_header_case_insensitive(hdict: dict[str, str], name: str) -> str | None:
        ln = name.lower()
        for k, v in hdict.items():
            if k.lower() == ln:
                return v
        return None

    # --- inputs & method ---
    url = args.get("url")
    if not url or not isinstance(url, str):
        raise ValueError("http.fetch requires a string 'url'")

    method = str(args.get("method", "GET")).upper()
    if method not in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"):
        raise ValueError(f"Unsupported HTTP method: {method}")

    # Merge query params
    params = args.get("params") or {}
    if params:
        parsed = urllib.parse.urlparse(url)
        q = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
        q.update(params)
        new_q = urllib.parse.urlencode(q, doseq=True)
        url = urllib.parse.urlunparse(parsed._replace(query=new_q))

    # Scheme & host checks
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http/https URLs are allowed")
    host = parsed.hostname or ""

    # Enforce allowed ports (env-controllable)
    ok_port, why = _host_port_allowed(parsed)
    if not ok_port:
        return {"status": 0, "error": "PortNotAllowed", "message": why}

    # Block private/loopback/link-local/IMDS targets (SSRF hardening)
    if _is_disallowed_ip_host(host):
        return {
            "status": 0,
            "error": "DeniedIP",
            "message": f"Disallowed IP for host: {host}",
        }

    # Simple allow/deny lists via env
    allow_env = os.getenv("HTTP_FETCH_ALLOW_HOSTS", "").strip()
    deny_env = os.getenv("HTTP_FETCH_DENY_HOSTS", "").strip()
    allow_list = {h.strip().lower() for h in allow_env.split(",") if h.strip()}
    deny_list = {h.strip().lower() for h in deny_env.split(",") if h.strip()}

    if deny_list and host.lower() in deny_list:
        return {
            "status": 0,
            "error": "DeniedHost",
            "message": f"Host blocked by denylist: {host}",
        }
    if allow_list and host.lower() not in allow_list:
        return {
            "status": 0,
            "error": "NotAllowedHost",
            "message": f"Host not in allowlist: {host}",
        }

    headers = dict(args.get("headers") or {})
    headers.setdefault("User-Agent", "personal-agent/1.0 (+registry-http.fetch)")
    timeout = float(args.get("timeout", HTTP_DEFAULT_TIMEOUT_S))
    max_bytes = int(args.get("max_bytes", HTTP_MAX_BYTES))

    # Body handling
    body = args.get("body", None)
    data: bytes | None = None
    if body is None:
        data = None
    elif isinstance(body, (dict, list)):
        headers.setdefault("Content-Type", "application/json")
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    elif isinstance(body, str):
        headers.setdefault("Content-Type", "text/plain; charset=utf-8")
        data = body.encode("utf-8")
    elif isinstance(body, (bytes, bytearray)):
        data = bytes(body)
    else:
        raise ValueError("Unsupported 'body' type; expected dict/list/str/bytes")

    # ---- auth helpers (optional) ----
    auth = args.get("auth") or {}
    atype = (auth.get("type") or "").lower()

    # Avoid overwriting caller-supplied Authorization
    def _maybe_set_auth_header(value: str) -> None:
        if "Authorization" not in {k.title(): v for k, v in headers.items()}:
            headers.setdefault("Authorization", value)

    if atype == "bearer" and auth.get("token"):
        _maybe_set_auth_header(f"Bearer {auth['token']}")
    elif atype == "basic" and auth.get("username") and auth.get("password"):
        b64 = base64.b64encode(
            f"{auth['username']}:{auth['password']}".encode("utf-8")
        ).decode("ascii")
        _maybe_set_auth_header(f"Basic {b64}")
    elif atype == "hmac":
        secret_env = auth.get("secret_env")
        if secret_env:
            secret = os.getenv(secret_env, "")
            if secret:
                algo = (auth.get("algo") or "sha256").lower()
                header_name = auth.get("header", "X-Signature")
                payload = b"" if data is None else data
                digest = hmac.new(
                    secret.encode("utf-8"), payload, getattr(hashlib, algo)
                ).digest()
                headers.setdefault(
                    header_name, base64.b64encode(digest).decode("ascii")
                )

    # ---- correlation & idempotency (optional) ----
    _raw_trace = (args or {}).get("trace")
    trace = _raw_trace if isinstance(_raw_trace, dict) else {}
    corr_id = trace.get("correlation_id")
    idem_key = trace.get("idempotency_key")

    if corr_id:
        headers.setdefault("X-Correlation-Id", str(corr_id))
    if idem_key:
        headers.setdefault("Idempotency-Key", str(idem_key))

    # ---- conditional GET / cache validators (optional) ----
    cache = args.get("cache") or {}
    etag = cache.get("etag")
    last_mod = cache.get("last_modified")
    if etag:
        headers.setdefault("If-None-Match", str(etag))
    if last_mod:
        headers.setdefault("If-Modified-Since", str(last_mod))

    # HTTPS context (default)
    ctx = ssl.create_default_context()

    # ---- retry & redirect settings ----
    allow_redirects = bool(
        args.get("allow_redirects", True)
    )  # urllib follows by default
    max_redirects = int(args.get("max_redirects", 5))

    retry_cfg = args.get("retry") or {}
    retry_max = int(retry_cfg.get("max", 0))
    retry_on = set(int(s) for s in (retry_cfg.get("on") or [429, 502, 503, 504]))
    backoff_ms = int(retry_cfg.get("backoff_ms", 200))
    jitter = bool(retry_cfg.get("jitter", True))

    attempt = 0
    current_url = url

    while True:
        attempt += 1
        try:
            # Build a fresh Request each attempt
            req = urllib.request.Request(url=current_url, data=data, method=method)
            for k, v in headers.items():
                lk = k.lower()
                if lk in {"host", "content-length", "transfer-encoding", "connection"}:
                    continue
                req.add_header(k, v)

            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                # Final location (urllib may auto-follow)
                final_url = resp.geturl() or current_url
                if allow_redirects and max_redirects >= 0 and final_url != current_url:
                    parsed_final = urllib.parse.urlparse(final_url)
                    final_host = parsed_final.hostname or ""
                    # Re-validate hostname policy
                    if deny_list and final_host.lower() in deny_list:
                        return {
                            "status": 0,
                            "error": "DeniedHost",
                            "message": f"Host blocked by denylist: {final_host}",
                        }
                    if allow_list and final_host.lower() not in allow_list:
                        return {
                            "status": 0,
                            "error": "NotAllowedHost",
                            "message": f"Host not in allowlist: {final_host}",
                        }
                    # Re-validate port + IP class (SSRF)
                    ok_port2, why2 = _host_port_allowed(parsed_final)
                    if not ok_port2:
                        return {"status": 0, "error": "PortNotAllowed", "message": why2}
                    if _is_disallowed_ip_host(final_host):
                        return {
                            "status": 0,
                            "error": "DeniedIP",
                            "message": f"Disallowed IP for host: {final_host}",
                        }

                # Read (with cap)
                raw = resp.read(max_bytes + 1)
                truncated = len(raw) > max_bytes
                if truncated:
                    raw = raw[:max_bytes]

                # Decompress if needed
                enc = (resp.headers.get("Content-Encoding") or "").lower()
                if enc == "gzip":
                    try:
                        import gzip

                        raw = gzip.decompress(raw)
                    except Exception:
                        pass
                elif enc == "deflate":
                    try:
                        import zlib

                        raw = zlib.decompress(raw)
                    except Exception:
                        pass

                # Content negotiation
                ctype = (resp.headers.get("Content-Type") or "").lower()
                text: str | None = None
                j: Any | None = None
                if "application/json" in ctype or ctype.endswith("+json"):
                    try:
                        text = raw.decode("utf-8", errors="ignore")
                        j = json.loads(text) if text else None
                    except Exception:
                        j = None
                elif ctype.startswith("text/") or "xml" in ctype or "html" in ctype:
                    try:
                        text = raw.decode("utf-8", errors="ignore")
                    except Exception:
                        text = None
                else:
                    # Heuristic: looks like JSON
                    if raw and raw.lstrip()[:1] in (b"{", b"["):
                        try:
                            text = raw.decode("utf-8", errors="ignore")
                            j = json.loads(text)
                        except Exception:
                            pass

                # ---- optional pagination (cursor | header_link | page) ----
                paginated_info = None
                pages_meta: list[dict[str, Any]] = []
                combined_items: list[Any] = []

                paginate = args.get("paginate") or {}
                if paginate:
                    trace_enabled = bool(paginate.get("trace"))
                    page_traces: list[dict[str, Any]] = []
                    mode = (paginate.get("mode") or "cursor").lower()
                    max_pages = int(paginate.get("max_pages", 10))
                    items_ptr = paginate.get("items_pointer")
                    concat_json = bool(paginate.get("concat_json", False))
                    # NEW: polite delay knobs
                    delay_ms = int(paginate.get("delay_ms", 0))
                    delay_jitter_ms = int(paginate.get("delay_jitter_ms", 0))

                    # record page 1
                    pages_meta.append(
                        {"status": resp.status, "url": final_url, "bytes_len": len(raw)}
                    )
                    if trace_enabled:
                        page_traces.append(
                            {
                                "page": 1,
                                "status": resp.status,
                                "url": final_url,
                                "bytes": len(raw),
                            }
                        )

                    # ---- cursor mode ----
                    if mode == "cursor" and isinstance(j, (dict, list)):
                        ptr = paginate.get("json_pointer")
                        cursor_param = paginate.get("cursor_param", "cursor")

                        next_ref = _json_pointer_get(j, ptr) if ptr else None
                        page_idx = 1
                        while next_ref and page_idx < max_pages:
                            page_idx += 1
                            next_url = (
                                str(next_ref)
                                if _looks_like_url(next_ref)
                                else _url_with_param(
                                    final_url, cursor_param, str(next_ref)
                                )
                            )

                            parsed_next = urllib.parse.urlparse(next_url)
                            next_host = parsed_next.hostname or ""
                            if (deny_list and next_host.lower() in deny_list) or (
                                allow_list and next_host.lower() not in allow_list
                            ):
                                break
                            ok_port3, why3 = _host_port_allowed(parsed_next)
                            if not ok_port3 or _is_disallowed_ip_host(next_host):
                                break

                            try:
                                req_n = urllib.request.Request(
                                    url=next_url, data=None, method="GET"
                                )
                                for k2, v2 in headers.items():
                                    if k2.lower() in {
                                        "host",
                                        "content-length",
                                        "transfer-encoding",
                                        "connection",
                                    }:
                                        continue
                                    req_n.add_header(k2, v2)

                                with urllib.request.urlopen(
                                    req_n, timeout=timeout, context=ctx
                                ) as resp_n:
                                    status_n = getattr(
                                        resp_n, "status", resp_n.getcode()
                                    )
                                    raw_n = resp_n.read(max_bytes + 1)
                                    if len(raw_n) > max_bytes:
                                        raw_n = raw_n[:max_bytes]

                                    ctype_n = (
                                        resp_n.headers.get("Content-Type") or ""
                                    ).lower()
                                    j_n = None
                                    if (
                                        "application/json" in ctype_n
                                        or ctype_n.endswith("+json")
                                    ):
                                        try:
                                            text_n = raw_n.decode(
                                                "utf-8", errors="ignore"
                                            )
                                            j_n = json.loads(text_n) if text_n else None
                                        except Exception:
                                            j_n = None

                                    # ---- stop conditions ----
                                    if status_n == 204:
                                        pages_meta.append(
                                            {
                                                "status": status_n,
                                                "url": resp_n.geturl() or next_url,
                                                "bytes_len": 0,
                                            }
                                        )
                                        if trace_enabled:
                                            page_traces.append(
                                                {
                                                    "page": page_idx,
                                                    "status": status_n,
                                                    "url": resp_n.geturl() or next_url,
                                                    "bytes": 0,
                                                }
                                            )
                                        break
                                    if raw_n is None or len(raw_n) == 0:
                                        break
                                    if concat_json and items_ptr:
                                        if isinstance(j_n, (dict, list)):
                                            arr = _json_pointer_get(j_n, items_ptr)
                                            if (
                                                not isinstance(arr, list)
                                                or len(arr) == 0
                                            ):
                                                break

                                    pages_meta.append(
                                        {
                                            "status": status_n,
                                            "url": resp_n.geturl() or next_url,
                                            "bytes_len": len(raw_n),
                                        }
                                    )
                                    if trace_enabled:
                                        page_traces.append(
                                            {
                                                "page": page_idx,
                                                "status": status_n,
                                                "url": resp_n.geturl() or next_url,
                                                "bytes": len(raw_n),
                                            }
                                        )

                                    if (
                                        concat_json
                                        and items_ptr
                                        and isinstance(j_n, (dict, list))
                                    ):
                                        next_items = _json_pointer_get(j_n, items_ptr)
                                        if isinstance(next_items, list):
                                            combined_items.extend(next_items)

                                    # advance cursor
                                    next_ref = (
                                        _json_pointer_get(j_n, ptr)
                                        if (ptr and isinstance(j_n, (dict, list)))
                                        else None
                                    )

                                    # polite delay before next page
                                    if next_ref and (delay_ms or delay_jitter_ms):
                                        _d = delay_ms or 0
                                        if delay_jitter_ms:
                                            import random

                                            _d += random.randint(0, delay_jitter_ms)
                                        time.sleep(_d / 1000.0)
                            except Exception:
                                break

                    # ---- header_link mode ----
                    elif mode == "header_link":
                        header_name = paginate.get("header") or "Link"
                        rel = paginate.get("rel") or "next"

                        page_idx = 1
                        next_url = None
                        link_val = _get_header_case_insensitive(
                            dict(resp.headers.items()), header_name
                        )
                        if link_val and header_name.lower() == "link":
                            next_url = _parse_link_header_next(link_val, rel)
                        elif link_val and _looks_like_url(link_val):
                            next_url = link_val

                        while next_url and page_idx < max_pages:
                            page_idx += 1

                            # SSRF/allow/deny/port checks for the next hop
                            parsed_next = urllib.parse.urlparse(next_url)
                            next_host = parsed_next.hostname or ""
                            if (deny_list and next_host.lower() in deny_list) or (
                                allow_list and next_host.lower() not in allow_list
                            ):
                                break
                            ok_port3, why3 = _host_port_allowed(parsed_next)
                            if not ok_port3 or _is_disallowed_ip_host(next_host):
                                break

                            try:
                                req_n = urllib.request.Request(
                                    url=next_url, data=None, method="GET"
                                )
                                for k2, v2 in headers.items():
                                    if k2.lower() in {
                                        "host",
                                        "content-length",
                                        "transfer-encoding",
                                        "connection",
                                    }:
                                        continue
                                    req_n.add_header(k2, v2)

                                with urllib.request.urlopen(
                                    req_n, timeout=timeout, context=ctx
                                ) as resp_n:
                                    status_n = getattr(
                                        resp_n, "status", resp_n.getcode()
                                    )
                                    raw_n = resp_n.read(max_bytes + 1)
                                    if len(raw_n) > max_bytes:
                                        raw_n = raw_n[:max_bytes]

                                    ctype_n = (
                                        resp_n.headers.get("Content-Type") or ""
                                    ).lower()
                                    j_n = None
                                    if (
                                        "application/json" in ctype_n
                                        or ctype_n.endswith("+json")
                                    ):
                                        try:
                                            text_n = raw_n.decode(
                                                "utf-8", errors="ignore"
                                            )
                                            j_n = json.loads(text_n) if text_n else None
                                        except Exception:
                                            j_n = None

                                    # ---- stop conditions ----
                                    if status_n == 204:
                                        pages_meta.append(
                                            {
                                                "status": status_n,
                                                "url": resp_n.geturl() or next_url,
                                                "bytes_len": 0,
                                            }
                                        )
                                        if trace_enabled:
                                            page_traces.append(
                                                {
                                                    "page": page_idx,
                                                    "status": status_n,
                                                    "url": resp_n.geturl() or next_url,
                                                    "bytes": 0,
                                                }
                                            )
                                        break
                                    if raw_n is None or len(raw_n) == 0:
                                        break
                                    if concat_json and items_ptr:
                                        if isinstance(j_n, (dict, list)):
                                            arr = _json_pointer_get(j_n, items_ptr)
                                            if (
                                                not isinstance(arr, list)
                                                or len(arr) == 0
                                            ):
                                                break

                                    pages_meta.append(
                                        {
                                            "status": status_n,
                                            "url": resp_n.geturl() or next_url,
                                            "bytes_len": len(raw_n),
                                        }
                                    )
                                    if trace_enabled:
                                        page_traces.append(
                                            {
                                                "page": page_idx,
                                                "status": status_n,
                                                "url": resp_n.geturl() or next_url,
                                                "bytes": len(raw_n),
                                            }
                                        )

                                    if (
                                        concat_json
                                        and items_ptr
                                        and isinstance(j_n, (dict, list))
                                    ):
                                        next_items = _json_pointer_get(j_n, items_ptr)
                                        if isinstance(next_items, list):
                                            combined_items.extend(next_items)

                                    # read next link of the new page
                                    next_url = None
                                    link_val = _get_header_case_insensitive(
                                        dict(resp_n.headers.items()), header_name
                                    )
                                    if link_val and header_name.lower() == "link":
                                        next_url = _parse_link_header_next(
                                            link_val, rel
                                        )
                                    elif link_val and _looks_like_url(link_val):
                                        next_url = link_val

                                    # polite delay before next page (only if we have another page URL)
                                    if next_url and (delay_ms or delay_jitter_ms):
                                        _d = delay_ms or 0
                                        if delay_jitter_ms:
                                            import random

                                            _d += random.randint(0, delay_jitter_ms)
                                        time.sleep(_d / 1000.0)
                            except Exception:
                                break

                    # ---- page mode (page=1,2,3,...) ----
                    elif mode == "page":
                        paginate = args.get("paginate") or {}
                        if isinstance(paginate, bool):
                            paginate = {}

                        page_param = paginate.get("page_param", "page")
                        start_page = int(paginate.get("start_page", 1))
                        max_pages = int(paginate.get("max_pages", 10))
                        delay_ms = int(paginate.get("delay_ms", 0))
                        delay_jitter_ms = int(paginate.get("delay_jitter_ms", 0))

                        page_idx = 1
                        cur_page = (
                            start_page + 1
                        )  # assume first call fetched start_page
                        while page_idx < max_pages:
                            page_idx += 1
                            next_url = _url_with_param(
                                final_url, page_param, str(cur_page)
                            )
                            cur_page += 1

                            parsed_next = urllib.parse.urlparse(next_url)
                            next_host = parsed_next.hostname or ""
                            if (deny_list and next_host.lower() in deny_list) or (
                                allow_list and next_host.lower() not in allow_list
                            ):
                                break
                            ok_port3, why3 = _host_port_allowed(parsed_next)
                            if not ok_port3 or _is_disallowed_ip_host(next_host):
                                break

                            try:
                                req_n = urllib.request.Request(
                                    url=next_url, data=None, method="GET"
                                )
                                for k2, v2 in headers.items():
                                    if k2.lower() in {
                                        "host",
                                        "content-length",
                                        "transfer-encoding",
                                        "connection",
                                    }:
                                        continue
                                    req_n.add_header(k2, v2)

                                with urllib.request.urlopen(
                                    req_n, timeout=timeout, context=ctx
                                ) as resp_n:
                                    status_n = getattr(
                                        resp_n, "status", resp_n.getcode()
                                    )
                                    raw_n = resp_n.read(max_bytes + 1)
                                    if len(raw_n) > max_bytes:
                                        raw_n = raw_n[:max_bytes]

                                    ctype_n = (
                                        resp_n.headers.get("Content-Type") or ""
                                    ).lower()
                                    j_n = None
                                    if (
                                        "application/json" in ctype_n
                                        or ctype_n.endswith("+json")
                                    ):
                                        try:
                                            text_n = raw_n.decode(
                                                "utf-8", errors="ignore"
                                            )
                                            j_n = json.loads(text_n) if text_n else None
                                        except Exception:
                                            j_n = None

                                    # ---- stop conditions ----
                                    if status_n == 204:
                                        pages_meta.append(
                                            {
                                                "status": status_n,
                                                "url": resp_n.geturl() or next_url,
                                                "bytes_len": 0,
                                            }
                                        )
                                        if trace_enabled:
                                            page_traces.append(
                                                {
                                                    "page": page_idx,
                                                    "status": status_n,
                                                    "url": resp_n.geturl() or next_url,
                                                    "bytes": 0,
                                                }
                                            )
                                        break
                                    if raw_n is None or len(raw_n) == 0:
                                        break
                                    if concat_json and items_ptr:
                                        arr = (
                                            _json_pointer_get(j_n, items_ptr)
                                            if isinstance(j_n, (dict, list))
                                            else None
                                        )
                                        if not isinstance(arr, list) or len(arr) == 0:
                                            break

                                    pages_meta.append(
                                        {
                                            "status": status_n,
                                            "url": resp_n.geturl() or next_url,
                                            "bytes_len": len(raw_n),
                                        }
                                    )
                                    if trace_enabled:
                                        page_traces.append(
                                            {
                                                "page": page_idx,
                                                "status": status_n,
                                                "url": resp_n.geturl() or next_url,
                                                "bytes": len(raw_n),
                                            }
                                        )

                                    if (
                                        concat_json
                                        and items_ptr
                                        and isinstance(j_n, (dict, list))
                                    ):
                                        next_items = _json_pointer_get(j_n, items_ptr)
                                        if isinstance(next_items, list):
                                            combined_items.extend(next_items)

                                    # polite delay before next page if continuing
                                    if page_idx < max_pages and (
                                        delay_ms or delay_jitter_ms
                                    ):
                                        _d = delay_ms
                                        if delay_jitter_ms:
                                            import random

                                            _d += random.randint(0, delay_jitter_ms)
                                        time.sleep(_d / 1000.0)
                            except Exception:
                                break

                    # Package pagination metadata if we paged beyond the first page or concatenated
                    if len(pages_meta) >= 1:
                        paginated_info = {
                            "pages": pages_meta,
                            "pages_count": len(pages_meta),
                        }
                        if concat_json and items_ptr:
                            paginated_info["items"] = combined_items
                            paginated_info["items_count"] = len(combined_items)
                        if trace_enabled:
                            paginated_info["traces"] = page_traces

                elapsed_ms = int((time.time() - t0) * 1000)
                result_obj = {
                    "status": resp.status,
                    "url": url,
                    "final_url": final_url,
                    "headers": dict(resp.headers.items()),
                    "bytes_len": len(raw),
                    "elapsed_ms": elapsed_ms,
                    "json": j,
                    "text": text,
                    "truncated": truncated,
                    "not_modified": (resp.status == 304),
                }
                if paginated_info is not None:
                    result_obj["pagination"] = paginated_info

                if corr_id or idem_key:
                    result_obj["trace"] = {}
                    if corr_id:
                        result_obj["trace"]["correlation_id"] = str(corr_id)
                    if idem_key:
                        result_obj["trace"]["idempotency_key"] = str(idem_key)

                return result_obj

        except urllib.error.HTTPError as e:
            try:
                body_text = e.read().decode("utf-8", errors="ignore")
            except Exception:
                body_text = None

            if e.code in retry_on and attempt <= retry_max:
                ra = None
                try:
                    ra = e.headers.get("Retry-After")
                except Exception:
                    ra = None
                if ra and str(ra).isdigit():
                    time.sleep(int(ra))
                else:
                    delay = backoff_ms * (2 ** (attempt - 1))
                    if jitter:
                        import random

                        delay += random.randint(0, backoff_ms)
                    time.sleep(delay / 1000.0)
                continue

            elapsed_ms = int((time.time() - t0) * 1000)
            return {
                "status": e.code,
                "url": url,
                "final_url": getattr(e, "url", current_url),
                "headers": dict(getattr(e, "headers", {}) or {}),
                "bytes_len": len(body_text.encode("utf-8")) if body_text else 0,
                "elapsed_ms": elapsed_ms,
                "error": "HTTPError",
                "text": body_text,
                "truncated": False,
            }

        except (urllib.error.URLError, TimeoutError) as e:
            if attempt <= retry_max:
                delay = backoff_ms * (2 ** (attempt - 1))
                if jitter:
                    import random

                    delay += random.randint(0, backoff_ms)
                time.sleep(delay / 1000.0)
                continue

            elapsed_ms = int((time.time() - t0) * 1000)
            return {
                "status": 0,
                "url": url,
                "final_url": current_url,
                "headers": {},
                "bytes_len": 0,
                "elapsed_ms": elapsed_ms,
                "error": type(e).__name__,
                "message": str(e),
                "truncated": False,
            }
