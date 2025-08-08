class SemanticAgentMixin:
    """Optional mixin for agents that want semantic memory + tools."""
    namespace: str = "generic"

    def build_knowledge(self):
        """
        Return rows to embed:
        [{"doc_id": "...", "text": "...", "metadata": {...}}, ...]
        """
        return []

    def tools(self) -> dict:
        """Return {tool_name: callable} mapping (deterministic helpers)."""
        return {}
