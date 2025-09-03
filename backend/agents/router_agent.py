# backend/agents/router_agent.py
from __future__ import annotations
from typing import Tuple, Dict, Any, Optional, Callable
import json, logging, time, importlib

# ✅ package-qualified imports
from backend.services.supabase_service import supabase
from backend.reasoner.policy import reason_with_memory
from backend.utils.agent_protocol import make_response
from backend.semantics.store import upsert as emb_upsert
import time

from backend.services.supabase_service import supabase


def _log_decision(
    agent_slug: str,
    user_id: str,
    query_text: str,
    was_success: bool,
    latency_ms: int = 0,
    reason: str | None = None,
    confidence: float | None = None,
    error: str | None = None,
    extra: dict | None = None,
) -> None:
    try:
        payload = {
            "agent_slug": agent_slug,
            "user_id": user_id,
            "query_text": query_text,
            "was_success": was_success,
            "latency_ms": latency_ms,
            "extra": {
                "reason": reason,
                "confidence": confidence,
                "error": error,
                **(extra or {}),
            },
        }
        payload["extra"] = {k: v for k, v in payload["extra"].items() if v is not None}
        supabase.table("agent_decisions").insert(payload).execute()
    except Exception:
        logger.exception("[router] failed to log agent_decision")


ROUTER_VERSION = "2025-08-09-supabase-registry-v1"
logger = logging.getLogger("router")
print(f"[router] loaded version={ROUTER_VERSION}")

# ----- Registry cache (auto refresh) -----
_REG: Dict[str, Dict[str, Any]] = {}
_LAST: float = 0.0
_TTL = 60.0  # seconds


def _load_registry(force: bool = False) -> Dict[str, Dict[str, Any]]:
    global _REG, _LAST
    if not force and _REG and (time.time() - _LAST < _TTL):
        return _REG

    rows = []
    try:
        res = supabase.table("agents").select("*").eq("status", "enabled").execute()
        print(
            "[router] fetched agents:",
            [
                (
                    r.get("slug"),
                    r.get("module_path"),
                    r.get("callable_name"),
                    r.get("status"),
                )
                for r in rows
            ],
        )

        rows = getattr(res, "data", None) or []
    except Exception:
        logger.exception("[router] failed to read agents table")
        rows = []

    reg: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        slug = (r.get("slug") or "").strip().lower()
        module_path = r.get("module_path") or f"backend.agents.{slug}_agent"
        callable_name = r.get("callable_name") or f"handle_{slug}"
        if module_path.startswith("agents."):
            module_path = "backend." + module_path
        try:
            mod = importlib.import_module(module_path)
            if callable_name.startswith("class:"):
                cls_name = callable_name.split(":", 1)[1]
                Cls = getattr(mod, cls_name)
                inst = Cls()
                handle: Callable[[str], dict | str] = lambda q, _i=inst: _i.handle(q)
            else:
                handle = getattr(mod, callable_name)
            reg[slug] = {
                "handle": handle,
                "desc": r.get("description") or slug,
                "capabilities": r.get("capabilities") or [],
                "namespaces": r.get("namespaces") or [slug],
            }
            print(f"[router] registered agent: {slug} -> {module_path}.{callable_name}")
        except Exception:
            logger.exception(
                f"[router] failed to import {module_path}.{callable_name} for slug={slug}"
            )

    _REG = reg
    _LAST = time.time()
    print(f"[router] registry loaded: {sorted(_REG.keys())}")
    return _REG


def _catalog() -> Dict[str, Any]:
    reg = _load_registry()
    return {
        k: {"desc": v["desc"], "capabilities": v["capabilities"]}
        for k, v in reg.items()
    }


# ----- LLM prompt -----
ROUTER_SYSTEM = """You are the routing coordinator for a personal-agents app.

You MUST choose an agent from AGENT_CATALOG when a user request is task-like or data-related
and any catalog agent plausibly fits. Only use {"agent":"none", ...} for small-talk, greetings,
or purely informational questions you can answer in one short sentence without calling an agent.

Return ONLY minified JSON in one of these schemas:

1) Route to agent:
{"agent":"<slug>","reason":"short reason","confidence":0.0-1.0,"rewrite":"optional concise task phrasing"}

2) Answer directly (no agent call):
{"agent":"none","response":"short answer","reason":"why no agent is needed","confidence":0.0-1.0}

3) Ask the user to choose (you are unsure which agent fits):
{"agent":"clarify","question":"one short disambiguating question","options":[<slugs>],
 "reason":"why you asked","confidence":0.0-1.0,"rewrite":"optional best-guess task phrasing"}

Hard rules:
- If at least one agent in AGENT_CATALOG plausibly matches the user task, DO NOT choose agent="none".
- Prefer routing even if your confidence is moderate; use "clarify" only when you cannot pick between agents.
- Use "rewrite" to turn vague user text into a crisp, actionable task for the chosen agent.

Guidance (examples):
- "show today's meals" -> route to the agent whose description/capabilities mention meals/meal planning/meal logs.
- "mark lunch done" -> route to that same meals agent; include a rewrite like "mark today's lunch complete".
- "log today's workout" -> route to workouts agent.
- "how much did I spend last week?" -> route to finance agent.
- "hi", "thanks", "tell me a joke" -> agent="none".
"""


def _build_prompt(user_text: str, user_id: str) -> str:
    return (
        f"{ROUTER_SYSTEM}\n\n"
        f"AGENT_CATALOG:\n{json.dumps(_catalog())}\n\n"
        f"USER_REQUEST:\n{user_text}\n\n"
        "Return ONLY the JSON."
    )


def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`")
        lines = s.splitlines()
        if lines and lines[0].lower().startswith("json"):
            s = "\n".join(lines[1:])
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(s[start : end + 1])
    except Exception:
        logger.exception("[router] LLM parse error")
        return None


def _llm_route(user_text: str, user_id: str) -> Optional[Dict[str, Any]]:
    try:
        raw = reason_with_memory(
            agent_name="router",
            query=_build_prompt(user_text, user_id),
            namespace="routing",
            k=4,
        )
        obj = _parse_json(raw)
        if not obj:
            # LLM returned something we couldn't parse
            _log_decision("router", user_id, user_text, False, reason="parse_failed")
            return None

        try:
            obj["confidence"] = float(obj.get("confidence", 0))
        except Exception:
            obj["confidence"] = 0.0

        # Soft log that the LLM produced a routing proposal (final success/fail is logged later)
        _log_decision(
            "router",
            user_id,
            user_text,
            True,
            reason="llm_routed",
            confidence=obj.get("confidence"),
        )
        return obj

    except Exception as e:
        logger.exception("[router] reasoner failed")
        _log_decision(
            "router", user_id, user_text, False, reason="reasoner_failed", error=str(e)
        )
        return None


# ----- public entry -----
def route_request(query: str, user_id: str = "anon") -> Tuple[str, dict | str]:
    reg = _load_registry()
    allowed = list(reg.keys())
    print(f"[router] incoming user={user_id!r} query={query!r} allowed={allowed}")

    try:
        decision = _llm_route(query, user_id)

        # Direct answer
        if decision and decision.get("agent") == "none":
            _log_decision(
                "router",
                user_id,
                query,
                True,
                extra={
                    "reason": decision.get("reason"),
                    "confidence": decision.get("confidence"),
                },
            )
            return "router", make_response(
                agent="router",
                intent="answer",
                data={
                    "response": decision.get("response"),
                    "reason": decision.get("reason"),
                    "confidence": decision.get("confidence"),
                },
            )

        # Clarify (or low confidence)
        if (
            not decision
            or decision.get("agent") == "clarify"
            or decision.get("confidence", 0) < 0.55
        ):
            opts = (
                decision.get("options")
                if decision and decision.get("options")
                else allowed
            )
            q = (
                decision.get("question")
                if decision and decision.get("question")
                else "Which agent should handle this?"
            )
            _log_decision(
                "router",
                user_id,
                query,
                False,
                extra={
                    "reason": (decision or {}).get("reason"),
                    "confidence": (decision or {}).get("confidence"),
                },
            )
            return "router", make_response(
                agent="router",
                intent="clarify",
                data={
                    "question": q,
                    "options": opts,
                    "suggested_rewrite": decision.get("rewrite") if decision else None,
                },
            )

        # Normal route
        agent = str(decision.get("agent") or "").strip().lower()
        if agent in reg:
            text = decision.get("rewrite") or query

            # Log to routing memory for future retrieval
            try:
                doc_id = (
                    f"{user_id}:{int(time.time())}"  # or a short hash if you prefer
                )
                emb_upsert(
                    namespace="routing",
                    doc_id=doc_id,
                    text=text,  # the post-rewrite text you routed on
                    metadata={
                        "reason": decision.get("reason"),
                        "confidence": decision.get("confidence"),
                        "ref": agent,  # keep in metadata too
                    },
                    kind="utterance",
                    ref=agent,
                )
            except Exception:
                logger.exception("[router] failed to log routing utterance")

            # Time the agent call, then log analytics
            _start = time.time()
            result = reg[agent]["handle"](text)
            _latency_ms = int((time.time() - _start) * 1000)

            _log_decision(
                agent_slug=agent,
                user_id=user_id,
                query_text=text,
                was_success=True,  # flip to False if you detect failures in result handling
                latency_ms=_latency_ms,
                extra={
                    "reason": decision.get("reason"),
                    "confidence": decision.get("confidence"),
                },
            )

            return agent, result

        # If the agent isn’t in registry, clarify (no fallback)
        _log_decision(
            "router", user_id, query, False, extra={"reason": "agent_not_in_registry"}
        )
        return "router", make_response(
            agent="router",
            intent="clarify",
            data={
                "question": "I don’t recognize that agent. Choose one:",
                "options": allowed,
            },
        )

    except Exception:
        logger.exception("[router] fatal error")
        _log_decision("router", user_id, query, False, extra={"reason": "fatal_error"})
        return "router", make_response(
            agent="router",
            intent="clarify",
            data={
                "question": "Something went wrong. Which agent should handle this?",
                "options": allowed,
            },
        )
