"""MemgineStrategy: MemoryStrategy adapter for StateBench.

Thin adapter wrapping MemgineEngine behind the MemoryStrategy ABC.
"""

from __future__ import annotations

from statebench.baselines.base import ContextResult, MemoryStrategy
from statebench.memgine.config import MemgineConfig
from statebench.memgine.engine import MemgineEngine
from statebench.schema.timeline import Event, InitialState


class MemgineStrategy(MemoryStrategy):
    """State-aware deterministic memory engine baseline.

    Combines LCM's architectural patterns (immutable store, hierarchical DAG,
    deterministic compaction) with StateBench's semantic layer awareness
    (layer-specific compaction rules, supersession-chain tracking,
    constraint preservation).
    """

    def __init__(
        self,
        token_budget: int = 8000,
        **kwargs: object,
    ) -> None:
        super().__init__(token_budget)
        config = MemgineConfig(token_budget=token_budget)
        self._engine = MemgineEngine(config)

    @property
    def name(self) -> str:
        return "memgine"

    @property
    def expects_initial_state(self) -> bool:
        return True

    def initialize_from_state(self, initial_state: InitialState) -> None:
        """Initialize from timeline's initial state."""
        self._engine.ingest_initial_state(initial_state)

    def process_event(self, event: Event) -> None:
        """Process an event and update internal state."""
        self._engine.ingest_statebench_event(event)

    def build_context(self, query: str) -> ContextResult:
        """Build context with provenance for an LLM query."""
        return self._engine.build_context(query)

    def consolidate(self) -> dict[str, int]:
        """Run a consolidation/dreaming pass. Returns operation counts."""
        return self._engine.consolidate()

    def reset(self) -> None:
        """Reset internal state for a new timeline."""
        self._engine.reset()

    def get_system_prompt(self) -> str:
        return (
            "You are an AI agent. Answer based ONLY on the structured context.\n\n"
            "CRITICAL RULES:\n"
            "1. For YES/NO decisions, CHECK ALL CONSTRAINTS - if ANY blocks, answer NO\n"
            "2. Multiple constraints must ALL be satisfied simultaneously\n"
            "3. NEVER invent details not explicitly stated — but DO perform "
            "arithmetic on stated values (e.g. $120K - $150K = -$30K) and DO "
            "recalculate schedules when dates shift (preserve the same durations)\n"
            "4. If info wasn't provided, say 'not specified' - don't assume\n"
            "5. Items marked [HYPOTHETICAL] are what-if scenarios - not real\n"
            "6. Items marked [DRAFT] are tentative - not finalized\n\n"
            "ANSWER GUIDANCE:\n"
            "7. If Recent Context CONTRADICTS or UPDATES a Current Fact, the "
            "conversation correction takes precedence over the stored fact\n"
            "8. ALWAYS answer using information from Current Facts when relevant data exists. "
            "Only say 'That information is not available.' when Current Facts contains "
            "absolutely no relevant data — do NOT name entities or terms from the question\n\n"
            "⚠️ REPAIR/CORRECTION RULES:\n"
            "9. Lines marked '⚠️ RECALCULATE' under a fact show stale conclusions — "
            "the base value they used has changed. Recalculate using the current value above, "
            "preserving the same durations and proportions from the original\n"
            "10. Show your arithmetic explicitly when recalculating\n"
            "11. Facts with (depends on: X) are derived from X — if X changed, "
            "recalculate this fact too\n\n"
            "CONTEXT FORMAT:\n"
            "- If info is marked '(changed from: ...)' only the NEW value applies\n"
            "- [org] = organizational/policy data, [usr] = user-provided info\n"
            "- 'Known Unknowns' = things NOT in the data — say 'not specified'\n\n"
            "Be accurate, concise, and explicit about what you know vs. don't know."
        )
