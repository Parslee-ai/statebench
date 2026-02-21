"""B3: RAG over Transcript Chunks Baseline.

This baseline chunks conversation turns and retrieves the most
relevant chunks via embedding similarity for each query.
"""

import numpy as np
import tiktoken
from openai import OpenAI

from statebench.baselines.base import ContextResult, MemoryStrategy, wrap_legacy_context
from statebench.schema.timeline import ConversationTurn, Event


class RAGTranscriptStrategy(MemoryStrategy):
    """RAG over transcript chunks using embedding similarity."""

    def __init__(
        self,
        token_budget: int = 8000,
        chunk_size: int = 3,  # Number of turns per chunk
        top_k: int = 5,  # Number of chunks to retrieve
    ):
        """Initialize the strategy.

        Args:
            token_budget: Maximum tokens for context
            chunk_size: Number of turns per chunk
            top_k: Number of chunks to retrieve
        """
        super().__init__(token_budget)
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.turns: list[ConversationTurn] = []
        self.chunks: list[str] = []
        self.embeddings: list[list[float]] = []
        self._current_chunk: list[ConversationTurn] = []
        self._encoder = tiktoken.get_encoding("cl100k_base")
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        """Get or create the OpenAI client."""
        if self._client is None:
            self._client = OpenAI()
        return self._client

    @property
    def name(self) -> str:
        return "rag_transcript"

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._encoder.encode(text))

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for texts."""
        client = self._get_client()
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [e.embedding for e in response.data]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

    def _build_chunks(self) -> None:
        """Finalize the current chunk (if any) and compute its embedding."""
        if not self._current_chunk:
            return
        chunk_text = "\n".join(
            f"{t.speaker.title()}: {t.text}" for t in self._current_chunk
        )
        self.chunks.append(chunk_text)
        self.embeddings.append(self._embed([chunk_text])[0])
        self._current_chunk = []

    def process_event(self, event: Event) -> None:
        """Store conversation turns."""
        if isinstance(event, ConversationTurn):
            self.turns.append(event)
            self._current_chunk.append(event)
            if len(self._current_chunk) >= self.chunk_size:
                self._build_chunks()

    def build_context(self, query: str) -> ContextResult:
        """Retrieve relevant chunks for the query.

        v1.0: Returns ContextResult with basic provenance.
        """
        if not self.turns:
            return wrap_legacy_context("")

        # Ensure any remaining partial chunk is materialized
        self._build_chunks()

        if not self.chunks:
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
        top_chunks = []
        total_tokens = 0
        available_budget = self.token_budget - 500
        for idx, _score in similarities:
            chunk = self.chunks[idx]
            chunk_tokens = self._count_tokens(chunk)
            if total_tokens + chunk_tokens > available_budget:
                break
            top_chunks.append(chunk)
            total_tokens += chunk_tokens
            if len(top_chunks) >= self.top_k:
                break

        # Build context
        context = "Relevant conversation context:\n\n" + "\n\n---\n\n".join(top_chunks)
        return wrap_legacy_context(context)

    def reset(self) -> None:
        """Clear all stored data."""
        self.turns = []
        self.chunks = []
        self.embeddings = []
        self._current_chunk = []

    def get_system_prompt(self) -> str:
        return (
            "You are an AI assistant. Use the provided conversation context "
            "to answer questions. Focus on the most relevant information."
        )
