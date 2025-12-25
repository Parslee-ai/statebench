"""B2: Rolling Conversation Summary Baseline.

This baseline maintains a rolling summary of the conversation
plus the last K turns as recent context.

v1.0: Returns ContextResult with basic provenance.
"""

import tiktoken
from anthropic import Anthropic
from openai import OpenAI

from statebench.baselines.base import ContextResult, MemoryStrategy, wrap_legacy_context
from statebench.schema.timeline import ConversationTurn, Event


class RollingSummaryStrategy(MemoryStrategy):
    """Rolling summary + last K turns."""

    def __init__(
        self,
        token_budget: int = 8000,
        recent_turns: int = 5,
        provider: str = "openai",
    ):
        """Initialize the strategy.

        Args:
            token_budget: Maximum tokens for context
            recent_turns: Number of recent turns to keep verbatim
            provider: LLM provider for summarization ("openai" or "anthropic")
        """
        super().__init__(token_budget)
        self.recent_turns = recent_turns
        self.provider = provider
        self.turns: list[ConversationTurn] = []
        self.summary: str = ""
        self._encoder = tiktoken.get_encoding("cl100k_base")

        # Initialize client lazily
        self._client: OpenAI | Anthropic | None = None

    def _get_client(self) -> OpenAI | Anthropic:
        """Get or create the LLM client."""
        if self._client is None:
            if self.provider == "openai":
                self._client = OpenAI()
            else:
                self._client = Anthropic()
        return self._client

    @property
    def name(self) -> str:
        return "rolling_summary"

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._encoder.encode(text))

    def _summarize(self, text: str) -> str:
        """Summarize text using LLM."""
        client = self._get_client()

        prompt = (
            "Summarize the following conversation, focusing on:\n"
            "1. Key decisions and commitments made\n"
            "2. Important facts and preferences stated\n"
            "3. Any changes to previous decisions\n\n"
            "Be concise but preserve critical details.\n\n"
            f"Conversation:\n{text}"
        )

        try:
            if self.provider == "openai" and isinstance(client, OpenAI):
                openai_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                )
                return openai_response.choices[0].message.content or ""
            elif isinstance(client, Anthropic):
                anthropic_response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )
                # Extract text from first content block
                content = anthropic_response.content[0]
                if hasattr(content, "text"):
                    return str(content.text)
                return ""
            else:
                return ""
        except Exception:
            fallback_lines = text.splitlines()[-5:]
            return " ; ".join(fallback_lines)

    def process_event(self, event: Event) -> None:
        """Store conversation turns, updating summary as needed."""
        if isinstance(event, ConversationTurn):
            self.turns.append(event)

            # Check if we need to summarize older turns
            if len(self.turns) > self.recent_turns:
                # Summarize older turns
                older_turns = self.turns[:-self.recent_turns]
                older_text = "\n".join(
                    f"{t.speaker.title()}: {t.text}" for t in older_turns
                )

                if self.summary:
                    # Combine with existing summary
                    combined = (
                        f"Previous summary:\n{self.summary}\n\n"
                        f"New conversation:\n{older_text}"
                    )
                    self.summary = self._summarize(combined)
                else:
                    self.summary = self._summarize(older_text)

                # Keep only recent turns
                self.turns = self.turns[-self.recent_turns:]

    def build_context(self, query: str) -> ContextResult:
        """Build context from summary + recent turns.

        v1.0: Returns ContextResult with basic provenance.
        """
        parts = []

        if self.summary:
            parts.append(f"Conversation summary:\n{self.summary}")

        if self.turns:
            recent = "\n".join(
                f"{t.speaker.title()}: {t.text}"
                for t in self.turns[-self.recent_turns:]
            )
            parts.append(f"Recent conversation:\n{recent}")

        context = "\n\n---\n\n".join(parts)
        return wrap_legacy_context(context)

    def reset(self) -> None:
        """Clear turns and summary."""
        self.turns = []
        self.summary = ""

    def get_system_prompt(self) -> str:
        return (
            "You are an AI assistant. Use the conversation summary and "
            "recent context to answer questions accurately."
        )
