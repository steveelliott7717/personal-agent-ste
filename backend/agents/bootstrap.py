# backend/agents/bootstrap.py
import pkgutil, importlib, inspect, logging
from typing import Optional, Dict, Any
from backend.utils.agent_registry import register_agent

logger = logging.getLogger("agent-bootstrap")


def bootstrap_register_all():
    import backend.agents as agents_pkg  # package where your agents live

    for m in pkgutil.iter_modules(agents_pkg.__path__, agents_pkg.__name__ + "."):
        try:
            mod = importlib.import_module(m.name)
            meta: Optional[Dict[str, Any]] = getattr(mod, "AGENT_META", None)
            if isinstance(meta, dict) and "slug" in meta and "handler_key" in meta:
                register_agent(meta)
        except Exception:
            logger.exception("bootstrap: failed registering from %s", m.name)
