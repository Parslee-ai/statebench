"""MemgineStrategy: MemoryStrategy adapter for StateBench.

Thin adapter wrapping MemgineEngine behind the MemoryStrategy ABC.
"""

from __future__ import annotations

from statebench.baselines.base import ContextResult, MemoryStrategy
from statebench.memgine.config import MemgineConfig
from statebench.memgine.engine import MemgineEngine
from statebench.schema.timeline import Event, InitialState


def frontier_capable(model: str | None) -> bool:
    """Whether ``model`` is strong enough to use the verbose ``(changed from: X)``
    supersession annotation without resurrecting the old value.

    Empirically (StateBench supersession track, CAR-local models): a small model
    (qwen3-4b) resurrects values shown in the annotation (SFRR 0.33 -> 0.15 when
    hidden, no accuracy cost), while a stronger model (qwen3-30b) does not resurrect
    (SFRR 0.15 baseline) and is *hurt* by hiding the value (accuracy 1.00 -> 0.89).
    Frontier remote models (Opus/GPT-5.x, the leaderboard path) match the strong
    behavior. Unknown -> True (frontier-safe; never silently regress that path).
    """
    if not model:
        return True
    m = model.lower()
    if any(s in m for s in (
        "gpt-4o", "gpt-5", "gpt-4.", "claude", "opus", "sonnet",
        "gemini-2.", "gemini-3", "o1", "o3",
    )):
        return True
    small_local = m.startswith(("mlx/", "ollama/", "apple/")) or any(
        s in m for s in ("0.6b", "1.7b", "-3b", "-4b", "-7b", "-8b", "-9b", "-13b")
    )
    if small_local:
        # Large local models behave like frontier on this annotation.
        if any(s in m for s in ("30b", "32b", "70b", "72b", "a3b", "120b")):
            return True
        return False
    return True


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
        show_superseded_values: bool | None = None,
        cascade_supersession_to_transcript: bool | None = None,
        model: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(token_budget)
        # Both supersession-safety behaviors key off model capability: weaker
        # models both resurrect annotated old values AND parrot the old value
        # left in the transcript, so hide the annotation and cascade to Layer 3.
        frontier = frontier_capable(model)
        if show_superseded_values is None:
            show_superseded_values = frontier
        if cascade_supersession_to_transcript is None:
            cascade_supersession_to_transcript = not frontier
        config = MemgineConfig(
            token_budget=token_budget,
            show_superseded_values=show_superseded_values,
            cascade_supersession_to_transcript=cascade_supersession_to_transcript,
        )
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
