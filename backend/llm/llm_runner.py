from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Iterable

import requests
from openai import OpenAI

# --- Add: Imports for other clients ---
import anthropic
import google.generativeai as genai

# Repo helpers:
from backend.services.agent_settings import get_settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("llm_runner")
if not log.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[llm_runner] %(message)s")
    handler.setFormatter(fmt)
    log.addHandler(handler)
log.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class LLMRunResult:
    ok: bool
    model: str
    messages: List[Dict[str, Any]]
    response_text: str
    raw_response: Dict[str, Any] | None
    error: str | None
    debug: Dict[str, Any]

    # --- Dict-like conveniences so legacy callers using .get() won't crash ---
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def keys(self) -> Iterable[str]:
        return self.to_dict().keys()

    def items(self) -> Iterable[Tuple[str, Any]]:
        return self.to_dict().items()

    def values(self) -> Iterable[Any]:
        return self.to_dict().values()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _coerce_json(value: Any) -> Any:
    """Parse JSON-looking strings into Python; otherwise return unchanged."""
    if isinstance(value, str):
        s = value.strip()
        if (s.startswith("{") and s.endswith("}")) or (
            s.startswith("[") and s.endswith("]")
        ):
            try:
                return json.loads(s)
            except Exception:
                return value
    return value


def _join_prompt(value: Any) -> Optional[str]:
    """
    Normalize a prompt into a string:
      - list[str] => join with newlines
      - str => stripped string
      - else => None
    """
    if value is None:
        return None
    if isinstance(value, list):
        return "\n".join(str(x).rstrip() for x in value if str(x).strip())
    if isinstance(value, str):
        s = value.strip()
        return s if s else None
    return None


class _SafeParams(dict):
    """dict that leaves unknown placeholders intact when used with format_map."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _apply_params(text: Optional[str], params: Dict[str, Any]) -> Optional[str]:
    if not text:
        return text
    try:
        return text.format_map(_SafeParams({k: str(v) for k, v in params.items()}))
    except Exception:
        # Never fail composition because of formatting — return original if something odd happens
        return text


def _compose_system_text(
    *,
    agent_slug: str,
    system_prompt_override: Optional[str],
    agent_settings: Dict[str, Any],
    params: Dict[str, Any],
) -> tuple[str, str]:
    """
    Build the full system message (and return a prompt_source label):
      - Common rules (from agent_settings.includes, if any)
      - Agent-specific prompt (string or list)
      - Optional primer
      - Params applied to all above
    """
    parts: List[str] = [f"[Agent: {agent_slug}]"]

    # COMMON RULES (from includes)
    includes = agent_settings.get("includes")
    if isinstance(includes, list) and includes:
        parts.append("=== COMMON RULES ===")
        for inc_slug in includes:
            inc_settings = get_settings(inc_slug)
            inc_prompt = _apply_params(
                _join_prompt(inc_settings.get("system_prompt")), params
            )
            if inc_prompt:
                parts.append(inc_prompt)

    # AGENT PROMPT PRIORITY: override -> agent_settings -> get_system_prompt() fallback
    prompt_source = "missing"
    agent_prompt = _apply_params(_join_prompt(system_prompt_override), params)
    if agent_prompt:
        prompt_source = "override"
    else:
        agent_prompt = _apply_params(
            _join_prompt(agent_settings.get("system_prompt")), params
        )
        if agent_prompt:
            prompt_source = "agent_settings.system_prompt"

    if agent_prompt:
        parts.append("\n=== AGENT PROMPT ===")
        parts.append(agent_prompt)

    # PRIMER (optional, also allow list)
    primer = _apply_params(_join_prompt(agent_settings.get("primer")), params)
    if primer:
        parts.append("\n=== PRIMER ===")
        parts.append(primer)

    return "\n".join(parts), prompt_source


def _pick_model(agent_settings: Dict[str, Any], default_model: Optional[str]) -> str:
    # Support both 'model' and 'oai_model'
    return (
        agent_settings.get("model")
        or agent_settings.get("oai_model")
        or default_model
        or os.getenv("OPENAI_MODEL")
        or "gpt-4o-mini"
    )


def _preview(text: str, limit: int = 800) -> str:
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    return t[:limit] + " … [truncated]"


def _get_provider_for_model(model_name: str) -> str:
    """Determines the API provider from the model name string."""
    name = (model_name or "").lower()
    if name.startswith("gpt-"):
        return "openai"
    if name.startswith("claude-"):
        return "anthropic"
    if name.startswith("gemini-"):
        return "google"
    return "openai"  # Default provider


def _remote_log_once(
    *,
    supabase_url: Optional[str],
    supabase_key: Optional[str],
    body: Dict[str, Any],
) -> tuple[int, str]:
    if not supabase_url or not supabase_key:
        return (0, "no_supabase")
    try:
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
        }
        url = supabase_url.rstrip("/") + "/rest/v1/agent_logs"
        r = requests.post(url, headers=headers, data=json.dumps(body))
        return (r.status_code, r.text)
    except Exception as e:
        return (599, str(e))


def _remote_log(
    *,
    supabase_url: Optional[str],
    supabase_key: Optional[str],
    agent_slug: str,
    verb: str,
    role: str,
    step: int,
    run_id: Optional[str],
    ok: Optional[bool],
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Best-effort remote logging into agent_logs:
    - Try with 'payload'
    - If 400 mentions missing 'payload' (schema/cache issue), retry without it
    """
    if not supabase_url or not supabase_key:
        return

    base = {
        "agent_slug": agent_slug,
        "verb": verb,  # NOT NULL in your schema
        "role": role,
        "step": step,
    }
    if run_id:
        base["run_id"] = run_id
    if ok is not None:
        base["ok"] = ok

    body = dict(base)
    if payload is not None:
        body["payload"] = payload

    code, text = _remote_log_once(
        supabase_url=supabase_url, supabase_key=supabase_key, body=body
    )
    if code >= 300:
        # Retry once without payload if the error references 'payload'
        if "payload" in (text or "").lower():
            body2 = dict(base)  # no payload
            code2, text2 = _remote_log_once(
                supabase_url=supabase_url, supabase_key=supabase_key, body=body2
            )
            if code2 >= 300:
                log.info(f"[agent_logs] non-2xx (retry no payload): {code2} {text2}")
        else:
            log.info(f"[agent_logs] non-2xx: {code} {text}")


def _messages_from_examples(examples: Any) -> List[Dict[str, str]]:
    """
    Convert few-shot examples (from Supabase) to messages.
    Supports:
      - list of {role, content}
      - list of {"user": "...", "assistant": "..."}
      - content can be str | dict | list (dict/list are JSON-encoded)
    """
    msgs: List[Dict[str, str]] = []
    if not examples:
        return msgs

    try:
        ex = json.loads(examples) if isinstance(examples, str) else examples

        if isinstance(ex, list):
            for item in ex:
                if not isinstance(item, dict):
                    continue

                def _encode_content(val: Any) -> str:
                    if isinstance(val, (dict, list)):
                        return json.dumps(val, ensure_ascii=False)
                    return str(val)

                if "role" in item and "content" in item:
                    msgs.append(
                        {
                            "role": str(item["role"]),
                            "content": _encode_content(item["content"]),
                        }
                    )
                else:
                    if "user" in item:
                        msgs.append(
                            {"role": "user", "content": _encode_content(item["user"])}
                        )
                    if "assistant" in item:
                        msgs.append(
                            {
                                "role": "assistant",
                                "content": _encode_content(item["assistant"]),
                            }
                        )
    except Exception:
        # swallow — examples are optional
        pass

    return msgs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _call_openai(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> Tuple[str, Dict[str, Any]]:
    """Makes a call to the OpenAI API and returns (response_text, raw_response_dict)."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore
        temperature=float(temperature) if temperature is not None else 0.2,
        max_tokens=max_tokens,
    )
    raw = (
        resp.to_dict()
        if hasattr(resp, "to_dict")
        else json.loads(resp.model_dump_json())
    )
    text_out = (raw.get("choices") or [{}])[0].get("message", {}).get(
        "content", ""
    ) or ""
    return text_out, raw


def _call_anthropic(
    model: str,
    messages: List[Dict[str, str]],
    temperature: Optional[float],
    max_tokens: Optional[int] = 4096,  # Claude requires max_tokens
) -> Tuple[str, Dict[str, Any]]:
    """Makes a call to the Anthropic (Claude) API."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Anthropic separates the system prompt from the message list.
    system_prompt = ""
    if messages and messages[0].get("role") == "system":
        system_prompt = messages[0].get("content", "")
        messages_for_api = messages[1:]
    else:
        messages_for_api = messages

    resp = client.messages.create(
        model=model,
        system=system_prompt,
        messages=messages_for_api,  # type: ignore
        temperature=float(temperature) if temperature is not None else 0.2,
        max_tokens=max_tokens or 4096,
    )

    raw = resp.model_dump()

    # Extract text from the response content block
    text_out = ""
    if resp.content:
        for block in resp.content:
            if hasattr(block, "text"):
                text_out += block.text

    # Normalize usage stats for logging consistency
    if "usage" in raw and raw["usage"]:
        raw["usage"]["total_tokens"] = raw["usage"].get("input_tokens", 0) + raw[
            "usage"
        ].get("output_tokens", 0)
        raw["usage"]["prompt_tokens"] = raw["usage"].get("input_tokens", 0)
        raw["usage"]["completion_tokens"] = raw["usage"].get("output_tokens", 0)

    return text_out, raw


def _call_google(
    model: str,
    messages: List[Dict[str, str]],
    temperature: Optional[float],
    max_tokens: Optional[int],
) -> Tuple[str, Dict[str, Any]]:
    """Placeholder for calling the Google (Gemini) API."""
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    gemini = genai.GenerativeModel(model)

    # Gemini uses 'model' for the assistant role and has no 'system' role.
    # The system prompt is combined with the first user message.
    system_prompt = ""
    if messages and messages[0].get("role") == "system":
        system_prompt = messages[0].get("content", "")
        messages_for_api = messages[1:]
    else:
        messages_for_api = messages

    # Prepend system prompt to the first user message
    if system_prompt and messages_for_api and messages_for_api[0].get("role") == "user":
        messages_for_api[0][
            "content"
        ] = f"{system_prompt}\n\n{messages_for_api[0]['content']}"

    # Map 'assistant' role to 'model' and wrap content in 'parts'
    gemini_messages = []
    for msg in messages_for_api:
        role = "model" if msg.get("role") == "assistant" else msg.get("role", "user")
        gemini_messages.append(
            {"role": role, "parts": [{"text": msg.get("content", "")}]}
        )

    resp = gemini.generate_content(gemini_messages)

    # The raw response object doesn't easily convert to a dict, so we build one.
    raw = {
        "usage": {
            "prompt_tokens": resp.usage_metadata.prompt_token_count,
            "completion_tokens": resp.usage_metadata.candidates_token_count,
            "total_tokens": resp.usage_metadata.total_token_count,
        }
    }

    return resp.text, raw


# ---------------------------------------------------------------------------


def run_llm_agent(
    *,
    agent_slug: str,
    user_text: Optional[str] = None,
    input_payload: Optional[Dict[str, Any]] = None,
    system_prompt: Optional[str] = None,
    default_model: Optional[str] = None,
    temperature: Optional[float] = 0.2,
    max_tokens: Optional[int] = None,
    run_id: Optional[str] = None,
    step: int = 0,
) -> LLMRunResult:
    """
    1) Load agent & common settings from Supabase
    2) Compose system message (common rules + agent prompt + primer), apply params
    3) Add optional few-shot examples
    4) Call OpenAI Chat Completions
    5) Best-effort remote logging to agent_logs
    """

    # 1) Load settings
    agent_settings = {
        k: _coerce_json(v) for k, v in (get_settings(agent_slug) or {}).items()
    }
    model = _pick_model(agent_settings, default_model)

    # Allow settings to override sampling (optional)
    if isinstance(agent_settings.get("temperature"), (int, float)):
        temperature = float(agent_settings["temperature"])
    if isinstance(agent_settings.get("max_tokens"), int):
        max_tokens = agent_settings["max_tokens"]

    # Params for prompt templating (agent <- payload.params)
    params: Dict[str, Any] = {}
    if isinstance(agent_settings.get("params"), dict):
        params.update(agent_settings["params"])
    if input_payload and isinstance(input_payload.get("params"), dict):
        params.update(input_payload["params"])

    # 2) Compose system
    system_text, prompt_source = _compose_system_text(
        agent_slug=agent_slug,
        system_prompt_override=system_prompt,
        agent_settings=agent_settings,
        params=params,
    )

    # 3) Build user content (also parametric)
    if user_text is None and input_payload:
        for k in ("text", "query", "task", "prompt", "input"):
            v = input_payload.get(k)
            if isinstance(v, str) and v.strip():
                user_text = v.strip()
                break
    if not user_text:
        user_text = "Proceed with your task using the rules above."
    user_text = _apply_params(user_text, params)

    # 4) Few-shot examples
    examples = agent_settings.get("examples")
    few_shot_msgs = _messages_from_examples(examples)

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_text}]
    messages.extend(few_shot_msgs)
    messages.append({"role": "user", "content": user_text})

    # Debug breadcrumbs
    log.info(f"{agent_slug} model: {model}")
    log.info(f"{agent_slug} prompt_source: {prompt_source}")
    log.info(f"{agent_slug} system composed (preview):\n{_preview(system_text)}")
    if few_shot_msgs:
        log.info(f"{agent_slug} examples included: {len(few_shot_msgs)}")
    log.info(f"{agent_slug} temperature={temperature} max_tokens={max_tokens}")
    log.info(f"{agent_slug} user_text: {_preview(user_text, 240)}")

    # Remote logs (best-effort)
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")

    _remote_log(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        agent_slug=agent_slug,
        verb="system_prompt",
        role="system",
        step=step,
        run_id=run_id,
        ok=True,
        payload={
            "model": model,
            "system_preview": _preview(system_text, 400),
            "prompt_source": prompt_source,
            "params": params,
        },
    )
    _remote_log(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        agent_slug=agent_slug,
        verb="user_message",
        role="user",
        step=step,
        run_id=run_id,
        ok=True,
        payload={"text_preview": _preview(user_text, 200)},
    )

    # 5) Call the appropriate LLM provider
    raw = None
    text_out = ""
    try:
        provider = _get_provider_for_model(model)
        if provider == "openai":
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            text_out, raw = _call_openai(
                client, model, messages, temperature, max_tokens
            )
        elif provider == "anthropic":
            # You will need to install and configure the anthropic SDK
            text_out, raw = _call_anthropic(model, messages, temperature, max_tokens)
        elif provider == "google":
            # You will need to install and configure the google-generativeai SDK
            text_out, raw = _call_google(model, messages, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown provider '{provider}' for model '{model}'")

        _remote_log(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            agent_slug=agent_slug,
            # Use a generic verb name now
            verb="llm_response_ok",
            role="assistant_raw",
            step=step,
            run_id=run_id,
            ok=True,
            payload={"usage": raw.get("usage", {})},
        )
        _remote_log(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            agent_slug=agent_slug,
            verb="assistant_text",
            role="assistant",
            step=step,
            run_id=run_id,
            ok=True,
            payload={"preview": _preview(text_out, 400)},
        )

        log.info(f"{agent_slug} assistant (preview): {_preview(text_out)}")

        return LLMRunResult(
            ok=True,
            model=model,
            messages=messages,
            response_text=text_out,
            raw_response=raw,
            error=None,
            debug={
                "agent_settings": agent_settings,
                "prompt_source": prompt_source,
                "params": params,
                "rendered_messages_preview": [
                    {"role": m["role"], "content": _preview(m["content"], 240)}
                    for m in messages
                ],
            },
        )
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        log.info(f"{agent_slug} llm_error: {err_msg}")
        _remote_log(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            agent_slug=agent_slug,
            verb="llm_error",
            role="assistant_raw",
            step=step,
            run_id=run_id,
            ok=False,
            payload={"error": err_msg},
        )
        return LLMRunResult(
            ok=False,
            model=model,
            messages=messages,
            response_text="",
            raw_response=raw,
            error=err_msg,
            debug={
                "agent_settings": agent_settings,
                "prompt_source": prompt_source,
                "params": params,
                "rendered_messages_preview": [
                    {"role": m["role"], "content": _preview(m["content"], 240)}
                    for m in messages
                ],
            },
        )
