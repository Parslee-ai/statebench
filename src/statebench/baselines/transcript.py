"""B1: Full Transcript Replay Baseline.

This baseline stores all conversation turns and replays them
as context, truncated to fit the token budget.

v1.0: Returns ContextResult with turn-based provenance.
"""

import tiktoken

from statebench.baselines.base import ContextResult, FactMetadata, MemoryStrategy
from statebench.schema.timeline import ConversationTurn, Event


class TranscriptReplayStrategy(MemoryStrategy):
    """Full transcript replay, truncated to token budget.

    v1.0: Tracks which turns are included/excluded for provenance.
    """

    def __init__(self, token_budget: int = 8000):
        super().__init__(token_budget)
        self.turns: list[ConversationTurn] = []
        self._encoder = tiktoken.get_encoding("cl100k_base")
        self._turn_counter = 0

    @property
    def name(self) -> str:
        return "transcript_replay"

    def process_event(self, event: Event) -> None:
        """Store conversation turns."""
        if isinstance(event, ConversationTurn) and event.speaker == "user":
            self.turns.append(event)
            self._turn_counter += 1

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._encoder.encode(text))

    def _turn_to_fact_metadata(self, turn: ConversationTurn, idx: int) -> FactMetadata:
        """Convert a conversation turn to FactMetadata for provenance."""
        return FactMetadata(
            fact_id=f"T-{idx:04d}",
            key=f"turn_{idx}",
            value=turn.text[:100] + ("..." if len(turn.text) > 100 else ""),
            layer=3,  # Working set layer
            is_valid=True,
            scope="session",
            authority="peer",
            source=turn.speaker,
        )

    def build_context(self, query: str) -> ContextResult:
        """Build context from transcript, newest first, truncated to budget.

        v1.0: Returns ContextResult with turn provenance.
        """
        if not self.turns:
            return ContextResult(
                context="",
                facts_included=[],
                facts_excluded=[],
                inclusion_reasons={},
                token_count=0,
            )

        # Reserve tokens for query and response
        available_budget = self.token_budget - 500  # Reserve for query + overhead

        # Build transcript newest-first (so we keep recent context if truncated)
        lines: list[str] = []
        total_tokens = 0
        included_turns: list[tuple[int, ConversationTurn]] = []
        excluded_turns: list[tuple[int, ConversationTurn]] = []
        truncated = False

        for idx, turn in enumerate(reversed(self.turns)):
            original_idx = len(self.turns) - 1 - idx
            line = f"{turn.speaker.title()}: {turn.text}"
            line_tokens = self._count_tokens(line)

            if total_tokens + line_tokens > available_budget:
                # Mark remaining turns as excluded
                truncated = True
                excluded_turns.append((original_idx, turn))
            else:
                lines.insert(0, line)
                total_tokens += line_tokens
                included_turns.insert(0, (original_idx, turn))

        if truncated:
            lines.insert(0, "[Earlier conversation truncated...]")

        # Build provenance
        facts_included = [
            self._turn_to_fact_metadata(turn, idx)
            for idx, turn in included_turns
        ]
        facts_excluded = [
            self._turn_to_fact_metadata(turn, idx)
            for idx, turn in excluded_turns
        ]

        inclusion_reasons = {
            f"T-{idx:04d}": "within token budget"
            for idx, _ in included_turns
        }
        inclusion_reasons.update({
            f"T-{idx:04d}": "excluded due to token budget"
            for idx, _ in excluded_turns
        })

        context = "Conversation history:\n\n" + "\n\n".join(lines) if lines else ""

        return ContextResult(
            context=context,
            facts_included=facts_included,
            facts_excluded=facts_excluded,
            inclusion_reasons=inclusion_reasons,
            token_count=total_tokens,
        )

    def reset(self) -> None:
        """Clear the transcript."""
        self.turns = []
        self._turn_counter = 0

    def get_system_prompt(self) -> str:
        return (
            "You are an AI assistant. Use the conversation history "
            "to answer the user's question. Be concise and accurate."
        )
