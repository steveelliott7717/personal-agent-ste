# backend/agents/router_agent.py
from __future__ import annotations
from typing import Tuple, Dict, Any, Optional, List
import json, traceback, logging, re

# Agent handlers
from .finance_agent import handle_finance
from .meals_agent import handle_meals
from .workouts_agent import handle_workouts
from .grooming_agent import handle_grooming

# LLM reasoner + response protocol
from reasoner.policy import reason_with_memory
from utils.agent_protocol import make_response

# Supabase for router memory/aliases
from services.supabase_service import supabase

ROUTER_VERSION = "2025-08-08-llm-router+memory-v1"
print(f"[router] loaded version={ROUTER_VERSION}")
logger = logging.getLogger("router")

# ---------------- Registry ----------------
AGENTS: Dict[str, Dict[str, Any]] = {
    "finance": {
        "handle": handle_finance,
        "desc": "Personal finance: expenses, budgets, hypothetical changes, ledger queries.",
        "capabilities": [
            "List and search transactions",
            "Summarize spending by time window",
            "Propose hypothetical budgets",
            "Insert/update/delete ledger rows",
        ],
        "namespaces": ["expenses"],
    },
    "meals": {
        "handle": handle_meals,
        "desc": "Meals and food logging, simple calorie/meal tracking.",
        "capabilities": ["Log meals", "Show meals for a day", "Summarize calories"],
        "namespaces": ["meals"],
    },
    "workouts": {
        "handle": handle_workouts,
        "desc": "Workout logging and retrieval.",
        "capabilities": ["Log a workout", "Show workouts today", "Summarize volume"],
        "namespaces": ["workouts"],
    },
    "grooming": {
        "handle": handle_grooming,
        "desc": "Personal grooming tracker.",
        "capabilities": ["Log grooming events", "Show recent grooming"],
        "namespaces": ["grooming"],
    },
}
ALLOWED = set(AGENTS.keys())
CLARIFY_DEFAULTS = ["finance", "meals", "workouts", "grooming"]

# --------------- LLM policy ---------------
ROUTER_SYSTEM = """You are the routing coordinator for a personal-agents app.
Decide which agent should handle the user's request, OR answer directly, OR ask the user to clarify.

Return ONLY minified JSON (no prose, no code fences). Schemas:

1) Route to agent:
{"agent":"finance|meals|workouts|grooming","reason":"...","confidence":0.0-1.0,"rewrite":"...optional..."}

2) Answer directly (no agent call):
{"agent":"none","response":"...short answer...","reason":"...","confidence":0.0-1.0}

3) Ask user to choose (you are unsure):
{"agent":"clarify","question":"one short question to decide routing","options":["finance","meals","workouts","grooming"],"reason":"...","confidence":0.0-1.0,"rewrite":"...optional best-guess task phrasing..."}

Guidance:
- If about transactions, budgets, spending, expenses, money -> finance.
- Meals/food/calories -> meals.
- Workouts/running/lifting -> workouts.
- Grooming/hygiene -> grooming.
- If chit-chat or the best reply is a short answer, use agent="none".
- If the domain is ambiguous, prefer agent="clarify" with a crisp question.
- Keep "rewrite" concise and actionable for the chosen agent.
"""

# -------------- Normalization --------------
def _normalize(q: str) -> str:
    q = (q or "").strip().lower()
    q = q.replace("^&", "&")
    q = q.replace("&", " & ")
    q = re.sub(r"[^a-z0-9 &]+", " ", q)
    q = re.sub(r"\s+", " ", q)
    return q

# -------------- Router memory (Supabase) --------------
def _save_router_event(
    user_id: str,
    query_text: str,
    decision: str,
    *,
    rewrite: str | None = None,
    response: str | None = None,
    options: List[str] | None = None,
    reason: str | None = None,
    confidence: float | None = None,
    question: str | None = None,
) -> None:
    try:
        payload = {
            "user_id": user_id,
            "query_text": query_text,
            "decision": decision,          # finance|meals|workouts|grooming|clarify|none|fallback|error
            "rewrite": rewrite,
            "response": response,
            "options": options,
            "reason": reason,
            "confidence": confidence,
            "question": question,
        }
        supabase.table("router_memory").insert(payload).execute()
    except Exception:
        logger.exception("router: failed to save memory")

def _load_router_context(user_id: str, k: int = 6) -> list[dict]:
    """Load recent non-error decisions for prompt context (oldest->newest)."""
    try:
        rows = (
            supabase.table("router_memory")
            .select("*")
            .eq("user_id", user_id)
            .neq("decision", "error")
            .order("created_at", desc=True)
            .limit(k)
            .execute()
            .data
            or []
        )
        return rows[::-1]
    except Exception:
        logger.exception("router: failed to load context")
        return []

def _format_context(rows: list[dict]) -> str:
    # Keep the history concise and LLM-friendly
    compact = []
    for r in rows:
        compact.append({
            "when": r.get("created_at"),
            "query": (r.get("query_text") or "").strip(),
            "decision": r.get("decision"),
            "rewrite": r.get("rewrite"),
            "response": r.get("response"),
            "reason": r.get("reason"),
        })
    return json.dumps(compact, ensure_ascii=False)

# -------------- User aliases (optional) --------------
def _find_alias_route(user_id: str, query: str) -> Optional[str]:
    """
    If you create a `router_alias` table (user_id, phrase, route),
    this will route immediately when the phrase is found.
    Silently no-ops if table doesn't exist.
    """
    try:
        qn = _normalize(query)
        res = (
            supabase.table("router_alias")
            .select("phrase,route")
            .eq("user_id", user_id)
            .execute()
        )
        rows = res.data or []
        for r in rows:
            phrase = _normalize(r.get("phrase") or "")
            route = (r.get("route") or "").strip().lower()
            if phrase and phrase in qn and route in ALLOWED:
                return route
    except Exception:
        # Table may not exist; that's fine.
        pass
    return None

# -------------- Prompt + LLM call --------------
def _build_router_prompt(user_text: str, user_id: str) -> str:
    catalog = {k: {"desc": v["desc"], "capabilities": v["capabilities"]} for k, v in AGENTS.items()}
    history = _format_context(_load_router_context(user_id, k=6))
    return (
        f"{ROUTER_SYSTEM}\n\n"
        f"AGENT_CATALOG:\n{json.dumps(catalog)}\n\n"
        f"USER_HISTORY (oldest->newest):\n{history}\n\n"
        f"USER_REQUEST:\n{user_text}\n\n"
        "Return ONLY the JSON."
    )

def _parse_llm_json(raw: str) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        lines = s.splitlines()
        if lines and lines[0].lower().startswith("json"):
            s = "\n".join(lines[1:])
    start = s.find("{"); end = s.rfind("}")
    candidate = s[start:end+1] if (start != -1 and end != -1 and end > start) else s
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        logger.exception("router: could not parse LLM JSON")
        return None

def _llm_route(user_text: str, user_id: str) -> Optional[Dict[str, Any]]:
    try:
        raw = reason_with_memory(
            agent_name="router",
            query=_build_router_prompt(user_text, user_id),
            namespace="routing",
            k=4
        )
    except Exception:
        logger.exception("router reasoner failed")
        return None

    obj = _parse_llm_json(raw)
    if not obj:
        return None

    agent = str(obj.get("agent", "")).strip().lower()
    reason = obj.get("reason")
    rewrite = obj.get("rewrite")
    response = obj.get("response")
    options = obj.get("options") or []
    question = obj.get("question")
    try:
        conf = float(obj.get("confidence", 0))
    except Exception:
        conf = 0.0

    return {
        "agent": agent,
        "reason": reason,
        "rewrite": rewrite,
        "response": response,
        "options": options,
        "question": question,
        "confidence": conf,
    }

# -------------- Heuristic fallback --------------
def _fallback_route(query: str) -> str:
    q = _normalize(query)
    if any(w in q for w in ["expense", "expenses", "spend", "spent", "budget", "purchase", "purchased", "money", "cost"]):
        return "finance"
    if any(w in q for w in ["meal", "meals", "breakfast", "lunch", "dinner", "calorie", "food", "diet"]):
        return "meals"
    if any(w in q for w in ["workout", "workouts", "gym", "run", "lift", "strength", "cardio", "exercise"]):
        return "workouts"
    if any(w in q for w in ["groom", "grooming", "shower", "skincare", "haircut", "clip", "trim"]):
        return "grooming"
    return "meals"

# -------------- Public entrypoint --------------
def route_request(query: str, user_id: str = "anon") -> Tuple[str, dict | str]:
    print(f"[router] incoming={_normalize(query)!r} user={user_id!r}")
    try:
        # alias shortcut (optional)
        alias_route = _find_alias_route(user_id, query)
        if alias_route:
            print(f"[router] alias matched -> {alias_route}")
            _save_router_event(user_id, query, alias_route, reason="alias")
            handler = AGENTS[alias_route]["handle"]
            return alias_route, handler(query)

        decision = _llm_route(query, user_id)

        # Direct LLM answer
        if decision and decision["agent"] == "none":
            print(f"[router] LLM direct answer (conf={decision.get('confidence')})")
            _save_router_event(
                user_id, query, "none",
                response=decision.get("response"),
                reason=decision.get("reason"),
                confidence=decision.get("confidence"),
            )
            return "router", make_response(
                agent="router",
                intent="answer",
                data={"response": decision.get("response"), "reason": decision.get("reason")}
            )

        # Clarify (or low confidence)
        if decision and (decision["agent"] == "clarify" or decision.get("confidence", 0) < 0.55):
            opts = [o for o in (decision.get("options") or CLARIFY_DEFAULTS) if o in ALLOWED] or CLARIFY_DEFAULTS
            question = decision.get("question") or "Which area should this go to?"
            print(f"[router] LLM clarify (conf={decision.get('confidence')}) -> options={opts}")
            _save_router_event(
                user_id, query, "clarify",
                rewrite=decision.get("rewrite"),
                reason=decision.get("reason"),
                confidence=decision.get("confidence"),
                options=opts,
                question=question,
            )
            return "router", make_response(
                agent="router",
                intent="clarify",
                data={
                    "question": question,
                    "options": opts,
                    "suggested_rewrite": decision.get("rewrite"),
                    "reason": decision.get("reason"),
                }
            )

        # Normal route
        if decision and decision["agent"] in ALLOWED:
            agent = decision["agent"]
            routed_text = decision.get("rewrite") or query
            print(f"[router] LLM chose -> {agent} (conf={decision.get('confidence')} reason={decision.get('reason')})")
            _save_router_event(
                user_id, query, agent,
                rewrite=routed_text,
                reason=decision.get("reason"),
                confidence=decision.get("confidence"),
            )
            handler = AGENTS[agent]["handle"]
            return agent, handler(routed_text)

        # Fallback if LLM gave garbage
        agent = _fallback_route(query)
        print(f"[router] Fallback chose -> {agent}")
        _save_router_event(user_id, query, "fallback", reason=f"heuristic->{agent}")
        handler = AGENTS[agent]["handle"]
        return agent, handler(query)

    except Exception:
        print("[router] ERROR in routing:")
        traceback.print_exc()
        _save_router_event(user_id, query, "error", reason="exception in router")
        # Last resort: a clarify card so the UI doesn’t 500
        return "router", make_response(
            agent="router",
            intent="clarify",
            data={"question": "Something went wrong. Where should I send this?", "options": CLARIFY_DEFAULTS}
        )
