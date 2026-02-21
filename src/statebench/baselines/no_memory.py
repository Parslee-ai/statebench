"""B0: No Memory Baseline.

This baseline provides no context from prior events.
Only the current query is passed to the LLM.

v1.0: Returns ContextResult with empty provenance.
"""

from statebench.baselines.base import ContextResult, MemoryStrategy
from statebench.schema.timeline import Event


class NoMemoryStrategy(MemoryStrategy):
    """No memory - only the current query is used.

    v1.0: Returns ContextResult with empty provenance lists.
    """

    @property
    def name(self) -> str:
        return "no_memory"

    def process_event(self, event: Event) -> None:
        """No-op: we don't store anything."""
        pass

    def build_context(self, query: str) -> ContextResult:
        """Return empty context with no provenance.

        v1.0: Returns ContextResult instead of str.
        """
        context = "[No memory baseline] Prior conversation is intentionally ignored."
        return ContextResult(
            context=context,
            facts_included=[],
            facts_excluded=[],
            inclusion_reasons={},
            token_count=len(context) // 4,
        )

    def reset(self) -> None:
        """No-op: nothing to reset."""
        pass

    def get_system_prompt(self) -> str:
        return (
            "You are an AI assistant. Answer the user's question. "
            "If you don't have enough information, say so."
        )
