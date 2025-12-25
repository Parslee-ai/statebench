"""B4: Fact Extraction Memory Baseline (Mem0-style).

This baseline extracts facts from each conversation turn and stores
them as a list. For queries, it retrieves the top-k most relevant facts.
"""

import numpy as np
import tiktoken
from openai import OpenAI

from statebench.baselines.base import ContextResult, MemoryStrategy, wrap_legacy_context
from statebench.schema.timeline import ConversationTurn, Event


class FactExtractionStrategy(MemoryStrategy):
    """Mem0-style fact extraction and retrieval."""

    def __init__(
        self,
        token_budget: int = 8000,
        top_k: int = 10,
    ):
        """Initialize the strategy.

        Args:
            token_budget: Maximum tokens for context
            top_k: Number of facts to retrieve
        """
        super().__init__(token_budget)
        self.top_k = top_k
        self.facts: list[str] = []
        self.embeddings: list[list[float]] = []
        self._embedding_cache: dict[str, list[float]] = {}
        self._fact_set: set[str] = set()
        self._encoder = tiktoken.get_encoding("cl100k_base")
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        """Get or create the OpenAI client."""
        if self._client is None:
            self._client = OpenAI()
        return self._client

    @property
    def name(self) -> str:
        return "fact_extraction"

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._encoder.encode(text))

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for texts."""
        client = self._get_client()
        embeddings: list[list[float]] = []
        to_fetch = [text for text in texts if text not in self._embedding_cache]
        if to_fetch:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=to_fetch,
            )
            for text, data in zip(to_fetch, response.data):
                self._embedding_cache[text] = data.embedding
        for text in texts:
            embeddings.append(self._embedding_cache[text])
        return embeddings

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a_arr = np.array(a)
        b_arr = np.array(b)
        denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if denom == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / denom)

    def _extract_facts(self, text: str, speaker: str) -> list[str]:
        """Extract facts from a conversation turn."""
        client = self._get_client()

        prompt = (
            f"Extract key facts, decisions, preferences, and commitments from this message.\n"
            f"Return each fact on a separate line. Be concise.\n"
            f"If there are no extractable facts, return 'NONE'.\n\n"
            f"{speaker}: {text}"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )

        content = response.choices[0].message.content or ""
        if content.strip().upper() == "NONE":
            return []

        # Split into lines and clean up
        facts = [
            line.strip().lstrip("- ").lstrip("• ")
            for line in content.split("\n")
            if line.strip() and not line.strip().upper() == "NONE"
        ]

        return facts

    def process_event(self, event: Event) -> None:
        """Extract facts from conversation turns."""
        if isinstance(event, ConversationTurn):
            new_facts = self._extract_facts(event.text, event.speaker)

            if new_facts:
                # Add timestamp context
                ts_str = event.ts.strftime("%Y-%m-%d %H:%M")
                new_facts_with_ts = [f"[{ts_str}] {fact}" for fact in new_facts]

                # Compute embeddings for new facts
                new_embeddings = self._embed(new_facts_with_ts)

                for fact_text, embedding in zip(new_facts_with_ts, new_embeddings):
                    if fact_text in self._fact_set:
                        continue
                    self.facts.append(fact_text)
                    self.embeddings.append(embedding)
                    self._fact_set.add(fact_text)

    def build_context(self, query: str) -> ContextResult:
        """Retrieve relevant facts for the query.

        v1.0: Returns ContextResult with basic provenance.
        """
        if not self.facts:
            return wrap_legacy_context("")

        # Embed the query
        query_embedding = self._embed([query])[0]

        # Compute similarities
        similarities = [
            (i, self._cosine_similarity(query_embedding, emb))
            for i, emb in enumerate(self.embeddings)
        ]

        # Sort by similarity and get top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_facts = [self.facts[i] for i, _ in similarities[:self.top_k]]

        # Build context
        context = "Relevant facts from memory:\n\n" + "\n".join(
            f"- {fact}" for fact in top_facts
        )
        return wrap_legacy_context(context)

    def reset(self) -> None:
        """Clear all stored facts."""
        self.facts = []
        self.embeddings = []

    def get_system_prompt(self) -> str:
        return (
            "You are an AI assistant with access to extracted facts from memory. "
            "Use these facts to answer questions accurately. "
            "If facts conflict, prefer more recent information."
        )
