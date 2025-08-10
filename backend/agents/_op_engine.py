# backend/agents/_op_engine.py
from typing import Any, Dict, List
import re, json
from backend.services.supabase_service import supabase

def table_columns(table: str) -> List[str]:
    try:
        sample = supabase.table(table).select("*").limit(1).execute().data or []
        return list(sample[0].keys()) if sample else []
    except Exception:
        return []

def _prune_where(where: Any, columns: List[str]) -> Any:
    if not where or not columns: return where
    cols = set(columns)
    if isinstance(where, str):
        parts = re.split(r"\s+and\s+", where, flags=re.I)
        kept = [p for p in parts if re.match(r"\s*([a-zA-Z0-9_]+)\s*=", p) and p.split("=")[0].strip() in cols]
        return " AND ".join(kept) if kept else None
    if isinstance(where, list):
        pruned = [w for w in (_prune_where(w, columns) for w in where) if w]
        return pruned or None
    if isinstance(where, dict):
        return {k:v for k,v in where.items() if k in cols} or None
    return None

def _apply_where(q, where: Any):
    if not where: return q
    if isinstance(where, str):
        parts = re.split(r"\s+and\s+", where, flags=re.I)
        for cond in parts:
            m = re.match(r"\s*([a-zA-Z0-9_]+)\s*=\s*(.+)\s*$", cond)
            if not m: continue
            key, val = m.group(1), m.group(2).strip().strip("'\"")
            q = q.eq(key, val)
        return q
    if isinstance(where, list):
        for w in where: q = _apply_where(q, w)
        return q
    if isinstance(where, dict):
        for k, v in where.items():
            if isinstance(v, dict):
                for oper, val in v.items():
                    o = str(oper).lower()
                    if   o == ">=": q = q.gte(k, val)
                    elif o == "<=": q = q.lte(k, val)
                    elif o == ">":  q = q.gt(k, val)
                    elif o == "<":  q = q.lt(k, val)
                    elif o == "!=": q = q.neq(k, val)
                    elif o == "like": q = q.like(k, val)
                    else: q = q.eq(k, val)
            else:
                q = q.eq(k, v)
        return q
    return q

def execute_ops(ops: List[Dict[str, Any]]) -> List[Any]:
    out: List[Any] = []
    for step in ops:
        op = str(step.get("op","")).lower()
        table = step.get("table")
        if not op or not table:
            out.append({"error":"missing op/table","step":step}); continue

        cols = table_columns(table)
        where = _prune_where(step.get("where"), cols)

        if op == "select":
            q = supabase.table(table).select("*")
            q = _apply_where(q, where or {})
            for pair in (step.get("order") or []):
                if isinstance(pair, list) and len(pair)==2:
                    q = q.order(pair[0], desc=(str(pair[1]).lower()=="desc"))
            if step.get("limit"): q = q.limit(int(step["limit"]))
            out.append(q.execute().data or []); continue

        if op == "update":
            q = supabase.table(table).update(step.get("set") or {})
            q = _apply_where(q, where or {})
            out.append(q.execute().data or []); continue

        if op == "insert":
            values = step.get("values"); 
            if isinstance(values, dict): values = [values]
            out.append(supabase.table(table).insert(values or []).execute().data or []); continue

        if op == "delete":
            q = supabase.table(table).delete()
            q = _apply_where(q, where or {})
            out.append(q.execute().data or []); continue

        if op == "upsert":
            values = step.get("values"); 
            if isinstance(values, dict): values = [values]
            out.append(supabase.table(table).upsert(values or []).execute().data or []); continue

        out.append({"error": f"unsupported op {op}", "step": step})
    return out
