"""B0: No Memory Baseline.

This baseline provides no context from prior events.
Only the current query is passed to the LLM.
"""

from statebench.baselines.base import MemoryStrategy
from statebench.schema.timeline import Event


class NoMemoryStrategy(MemoryStrategy):
    """No memory - only the current query is used."""

    @property
    def name(self) -> str:
        return "no_memory"

    def process_event(self, event: Event) -> None:
        """No-op: we don't store anything."""
        pass

    def build_context(self, query: str) -> str:
        """Return an explicit reminder that this baseline has no context."""
        return "[No memory baseline] Prior conversation is intentionally ignored."

    def reset(self) -> None:
        """No-op: nothing to reset."""
        pass

    def get_system_prompt(self) -> str:
        return (
            "You are an AI assistant. Answer the user's question. "
            "If you don't have enough information, say so."
        )
