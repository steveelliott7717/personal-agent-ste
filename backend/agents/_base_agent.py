# backend/agents/_base_agent.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import json, logging, inspect
from services.supabase_service import supabase
from utils.agent_protocol import make_response, AgentResponse
from reasoner.policy import reason_with_memory
from ._op_engine import execute_ops, table_columns  # shared CRUD executor

log = logging.getLogger("agent")


class BaseAgent:
    """
    Reusable agent skeleton.
    Subclasses configure AGENT_META and (optionally) override choose_tags/after_execute.
    """

    AGENT_META: Dict[str, Any] = {
        "slug": "base",
        "title": "Base",
        "description": "",
        "handler_key": "base.handle",
        "namespaces": [],            # e.g., ["meals"]
        "capabilities": [],
        "keywords": [],
        "status": "enabled",
        # Optional hints (overridable via agent_settings):
        "default_tables": [],        # e.g., ["recipe_templates","meal_plan"]
        "instruction_tags": [],      # e.g., ["planning","logging"]
        "fallback_system": "",       # used if agent_instructions/core is absent
        "post_hooks": [],            # ["plugins.meals.attach_daylist_summary:post_summarize"]

        # Optional registry fields; auto-filled if omitted:
        # "module_path": "agents.meals_agent",
        # "callable_name": "class:MealsAgent"  or "handle_meals"
        # "version": "v1"
    }

    # cached settings pulled from Supabase (agent_settings)
    _settings: Dict[str, Any] | None = None

    def __init__(self):
        self._register_if_needed()
        self._settings = self._load_settings()

    # ---------- Supabase helpers ----------
    def _safe_exec(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            return None

    def _load_settings(self) -> Dict[str, Any]:
        """Load agent_settings into a dict(key -> value). Non-fatal if table missing."""
        slug = self.AGENT_META.get("slug", "")
        res = self._safe_exec(
            supabase.table("agent_settings")
            .select("key,value")
            .eq("agent_slug", slug)
            .execute
        )
        items = (getattr(res, "data", None) or []) if res else []
        out: Dict[str, Any] = {}
        for row in items:
            k, v = row.get("key"), row.get("value")
            if isinstance(k, str):
                out[k] = v
        return out

    def _get_setting(self, key: str, default: Any = None) -> Any:
        if self._settings is None:
            self._settings = self._load_settings()
        return self._settings.get(key, default) if isinstance(self._settings, dict) else default

    # ---------- self-registration (UPSERT; safe if tables absent) ----------
    def _register_if_needed(self) -> None:
        slug = self.AGENT_META.get("slug", "")
        if not slug:
            return
        # Derive defaults
        cls_name = self.__class__.__name__
        module_path = self.AGENT_META.get("module_path") or f"agents.{slug}_agent"
        callable_name = self.AGENT_META.get("callable_name") or f"class:{cls_name}"

        row = {
            "slug": slug,
            "title": self.AGENT_META.get("title", slug.title()),
            "description": self.AGENT_META.get("description", ""),
            "module_path": module_path,
            "callable_name": callable_name,
            "namespaces": self.AGENT_META.get("namespaces", [slug]),
            "capabilities": self.AGENT_META.get("capabilities", []),
            "status": self.AGENT_META.get("status", "enabled"),
            "version": self.AGENT_META.get("version", ""),
        }

        # If agents table exists, upsert; otherwise ignore silently
        try:
            # prefer explicit UPSERT (if supported by your client)
            supabase.table("agents").upsert(row, on_conflict="slug").execute()
        except Exception:
            # fallback: try insert once if table exists but row missing
            try:
                exists = supabase.table("agents").select("id").eq("slug", slug).limit(1).execute().data
                if not exists:
                    supabase.table("agents").insert(row).execute()
            except Exception:
                # table may not exist yet—don’t crash
                pass

    # ---------- instruction loading ----------
    def _get_instruction(self, tag: str) -> Optional[str]:
        try:
            res = (supabase.table("agent_instructions")
                   .select("instructions")
                   .eq("agent_name", self.AGENT_META["slug"])
                   .eq("tag", tag)
                   .order("created_at", desc=True)
                   .limit(1)
                   .execute())
            if getattr(res, "data", None):
                return res.data[0]["instructions"]
        except Exception:
            return None
        return None

    def choose_tags(self, user_text: str) -> List[str]:
        """Subclasses can analyze user_text and return instruction tags to include."""
        return []

    def _system_prompt(self, user_text: str) -> str:
        core = self._get_instruction("core") or self.AGENT_META.get("fallback_system", "")
        extras: List[str] = []
        for tag in self.choose_tags(user_text):
            text = self._get_instruction(tag)
            if text:
                extras.append(text)
        return core + (("\n\n" + "\n\n".join(extras)) if extras else "")

    # ---------- schema hint (merge meta + settings override) ----------
    def _schema_hint(self) -> str:
        tables = self._get_setting("default_tables", None) or self.AGENT_META.get("default_tables", []) or []
        hint = {"tables": {}}
        for t in tables:
            hint["tables"][t] = {"columns": table_columns(t)}
        return json.dumps(hint)

    def _build_action_prompt(self, user_text: str) -> str:
        return (
f"""{self._system_prompt(user_text)}

SCHEMA_HINT:
{self._schema_hint()}

USER_REQUEST:
{user_text}

Return ONLY compact JSON with keys:
- thoughts (string, optional short rationale)
- operations (array of objects with: op, table, where?, order?, limit?, set?, values?)
- response_template? (string for UI)

If you need no DB calls, return operations: [].
"""
        )

    # ---------- plan parsing ----------
    def _parse_plan(self, raw: str) -> Dict[str, Any]:
        s = (raw or "").strip()
        if s.startswith("```"):
            s = s.strip("`")
            lines = s.splitlines()
            if lines and lines[0].lower().startswith("json"):
                s = "\n".join(lines[1:])
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start:end+1]
        return json.loads(s)

    # ---------- main entry ----------
    def run(self, user_text: str) -> AgentResponse:
        slug = self.AGENT_META.get("slug", "?")
        try:
            plan_raw = reason_with_memory(
                agent_name=slug,
                query=self._build_action_prompt(user_text),
                namespace=(self.AGENT_META.get("namespaces") or [slug])[0],
                k=8
            )
        except Exception as e:
            return make_response(agent=slug, intent="error", data={"message": f"reasoner error: {e}"})
        try:
            plan = self._parse_plan(plan_raw if isinstance(plan_raw, str) else json.dumps(plan_raw))
        except Exception as e:
            return make_response(agent=slug, intent="error",
                                 data={"message": f"Could not parse plan: {e}", "raw": str(plan_raw)})

        ops = plan.get("operations", []) or []
        try:
            results = execute_ops(ops)
        except Exception as e:
            return make_response(agent=slug, intent="error",
                                 data={"message": f"execution error: {e}", "operations": ops})

        data = {
            "thoughts": plan.get("thoughts"),
            "operations": ops,
            "results": results,
            "response_template": plan.get("response_template"),
        }
        data = self.after_execute(user_text, data)
        return make_response(agent=slug, intent="auto", data=data)

    # ---------- plugin hook (Supabase-aware) ----------
    def after_execute(self, user_text: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run post hooks from AGENT_META['post_hooks'] and/or agent_settings('post_hooks').
        """
        from importlib import import_module

        # Merge hooks: settings override can add/replace
        hooks_meta = list(self.AGENT_META.get("post_hooks", []) or [])
        hooks_cfg = self._get_setting("post_hooks", None)
        if isinstance(hooks_cfg, list):
            hooks = hooks_cfg
        elif isinstance(hooks_cfg, str):
            try:
                hooks = json.loads(hooks_cfg)
                if not isinstance(hooks, list): hooks = hooks_meta
            except Exception:
                hooks = hooks_meta
        else:
            hooks = hooks_meta

        for dotted in hooks:
            try:
                mod, func = dotted.rsplit(":", 1)
                fn = getattr(import_module(mod), func)
                data = fn(agent_slug=self.AGENT_META["slug"], user_text=user_text, data=data) or data
            except Exception:
                log.exception("[%s] post hook failed: %s", self.AGENT_META["slug"], dotted)
        return data

    # ---------- public wrapper ----------
    def handle(self, query: str) -> AgentResponse:
        return self.run(query)
