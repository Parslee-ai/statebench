"""B1: Full Transcript Replay Baseline.

This baseline stores all conversation turns and replays them
as context, truncated to fit the token budget.
"""

import tiktoken

from statebench.baselines.base import MemoryStrategy
from statebench.schema.timeline import Event, ConversationTurn


class TranscriptReplayStrategy(MemoryStrategy):
    """Full transcript replay, truncated to token budget."""

    def __init__(self, token_budget: int = 8000):
        super().__init__(token_budget)
        self.turns: list[ConversationTurn] = []
        self._encoder = tiktoken.get_encoding("cl100k_base")

    @property
    def name(self) -> str:
        return "transcript_replay"

    def process_event(self, event: Event) -> None:
        """Store conversation turns."""
        if isinstance(event, ConversationTurn) and event.speaker == "user":
            self.turns.append(event)

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._encoder.encode(text))

    def build_context(self, query: str) -> str:
        """Build context from transcript, newest first, truncated to budget."""
        if not self.turns:
            return ""

        # Reserve tokens for query and response
        available_budget = self.token_budget - 500  # Reserve for query + overhead

        # Build transcript newest-first (so we keep recent context if truncated)
        lines = []
        total_tokens = 0

        for turn in reversed(self.turns):
            line = f"{turn.speaker.title()}: {turn.text}"
            line_tokens = self._count_tokens(line)

            if total_tokens + line_tokens > available_budget:
                # Add truncation marker
                lines.insert(0, "[Earlier conversation truncated...]")
                break

            lines.insert(0, line)
            total_tokens += line_tokens

        return "Conversation history:\n\n" + "\n\n".join(lines)

    def reset(self) -> None:
        """Clear the transcript."""
        self.turns = []

    def get_system_prompt(self) -> str:
        return (
            "You are an AI assistant. Use the conversation history "
            "to answer the user's question. Be concise and accurate."
        )
