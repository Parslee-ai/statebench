"""Ablation Baselines for StateBench.

These baselines isolate specific components to test causal claims:
1. state_based_no_supersession: State structure without invalidation tracking
2. fact_extraction_with_supersession: Fact extraction + explicit supersession
3. transcript_latest_wins: Transcript with "latest instruction wins" heuristic
"""

from datetime import datetime
from typing import Any

import numpy as np
import tiktoken
from openai import OpenAI

from statebench.baselines.base import ContextResult, MemoryStrategy, wrap_legacy_context
from statebench.schema.state import (
    IdentityRole,
    PersistentFact,
    Source,
    WorkingSetItem,
)
from statebench.schema.timeline import (
    ConversationTurn,
    Event,
    InitialState,
    StateWrite,
    Supersession,
)


class StateBasedNoSupersessionStrategy(MemoryStrategy):
    """State-based context WITHOUT supersession tracking.

    This ablation keeps the structured state layers but does NOT
    mark facts as invalid when superseded. All facts remain visible.
    This tests whether supersession tracking is the key differentiator.
    """

    def __init__(
        self,
        token_budget: int = 8000,
        working_set_size: int = 5,
    ):
        super().__init__(token_budget)
        self.working_set_size = working_set_size
        self.identity: IdentityRole | None = None
        self.facts: dict[str, PersistentFact] = {}
        self.working_set: list[WorkingSetItem] = []
        self.environment: dict[str, str | datetime] = {}
        self._environment_ts: dict[str, datetime] = {}
        self._encoder = tiktoken.get_encoding("cl100k_base")
        self._fact_counter = 0

    @property
    def name(self) -> str:
        return "state_based_no_supersession"

    @property
    def expects_initial_state(self) -> bool:
        return True

    def initialize_from_state(self, initial_state: InitialState) -> None:
        """Initialize state from timeline's initial state."""
        self.identity = initial_state.identity_role
        for fact in initial_state.persistent_facts:
            self.facts[fact.key] = fact.model_copy(deep=True)
        self.working_set = [
            WorkingSetItem(item_type="context", content=item.content, ts=item.ts)
            for item in initial_state.working_set
        ]
        self.environment = dict(initial_state.environment)
        now = datetime.min
        self._environment_ts = {k: now for k in self.environment.keys()}

    def process_event(self, event: Event) -> None:
        """Process events WITHOUT invalidating superseded facts."""
        if isinstance(event, ConversationTurn):
            self.working_set.append(WorkingSetItem(
                item_type="context",
                content=f"{event.speaker.title()}: {event.text}",
                ts=event.ts,
            ))
            if len(self.working_set) > self.working_set_size:
                self.working_set = self.working_set[-self.working_set_size:]

        elif isinstance(event, StateWrite):
            for write in event.writes:
                if write.layer == "persistent_facts":
                    self._fact_counter += 1
                    fact = PersistentFact(
                        id=f"F-{self._fact_counter:04d}",
                        key=write.key,
                        value=write.value,
                        source=Source(type="user", authority="peer"),
                        ts=event.ts,
                        is_valid=True,  # Always valid
                    )
                    self.facts[write.key] = fact
                elif write.layer == "environment":
                    self._update_environment(write.key, write.value, event.ts, write.supersedes)

        elif isinstance(event, Supersession):
            # KEY DIFFERENCE: Do NOT mark old facts as invalid
            for write in event.writes:
                if write.layer == "persistent_facts":
                    # Just add new fact, don't invalidate old one
                    self._fact_counter += 1
                    fact = PersistentFact(
                        id=f"F-{self._fact_counter:04d}",
                        key=write.key,
                        value=write.value,
                        source=Source(type="user", authority="peer"),
                        ts=event.ts,
                        is_valid=True,
                    )
                    self.facts[write.key] = fact
                elif write.layer == "environment":
                    self._update_environment(write.key, write.value, event.ts, write.supersedes)

    def build_context(self, query: str) -> ContextResult:
        """Build context showing ALL facts (no filtering).

        v1.0: Returns ContextResult with basic provenance.
        """
        parts = []

        if self.identity:
            identity_text = (
                f"User: {self.identity.user_name}\nRole: {self.identity.authority}"
            )
            if self.identity.department:
                identity_text += f"\nDepartment: {self.identity.department}"
            parts.append(f"## Identity\n{identity_text}")

        # ALL facts - no filtering for validity
        if self.facts:
            facts_text = "\n".join(
                f"- {f.value} (source: {f.source.type})"
                for f in sorted(self.facts.values(), key=lambda x: x.ts)
            )
            parts.append(f"## Facts\n{facts_text}")

        if self.working_set:
            working_text = "\n".join(item.content for item in self.working_set)
            parts.append(f"## Recent Context\n{working_text}")

        if self.environment:
            sorted_env = sorted(
                self.environment.items(),
                key=lambda item: self._environment_ts.get(item[0], datetime.min),
                reverse=True,
            )[:5]
            env_text = "\n".join(f"- {k}: {v}" for k, v in sorted_env)
            parts.append(f"## Environment\n{env_text}")

        context = "\n\n".join(parts)
        return wrap_legacy_context(context)

    def reset(self) -> None:
        self.identity = None
        self.facts = {}
        self.working_set = []
        self.environment = {}
        self._environment_ts = {}
        self._fact_counter = 0

    def _update_environment(
        self,
        key: str,
        value: str | datetime,
        ts: datetime,
        supersedes: str | None,
    ) -> None:
        if supersedes:
            self.environment.pop(supersedes, None)
            self._environment_ts.pop(supersedes, None)

        prev_ts = self._environment_ts.get(key)
        if prev_ts and prev_ts > ts:
            return
        self.environment[key] = value
        self._environment_ts[key] = ts

        if len(self.environment) > 5:
            oldest_key = min(self._environment_ts.items(), key=lambda item: item[1])[0]
            if oldest_key != key:
                self.environment.pop(oldest_key, None)
                self._environment_ts.pop(oldest_key, None)

    def get_system_prompt(self) -> str:
        return (
            "You are an AI agent. Answer based on the structured context provided."
            "Use the most recent information when facts conflict. Be accurate and concise."
        )


class FactExtractionWithSupersessionStrategy(MemoryStrategy):
    """Fact extraction WITH explicit supersession tracking.

    This ablation adds supersession semantics to fact extraction:
    when a supersession event occurs, the old fact is marked invalid.
    This tests whether supersession helps even without structured state.
    """

    def __init__(
        self,
        token_budget: int = 8000,
        top_k: int = 10,
    ):
        super().__init__(token_budget)
        self.top_k = top_k
        # {text, embedding, valid, superseded_by, source_key, tokens}
        self.facts: list[dict[str, Any]] = []
        self._encoder = tiktoken.get_encoding("cl100k_base")
        self._client: OpenAI | None = None
        self._embedding_cache: dict[str, list[float]] = {}

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    @property
    def name(self) -> str:
        return "fact_extraction_with_supersession"

    def _embed(self, texts: list[str]) -> list[list[float]]:
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
        a_arr = np.array(a)
        b_arr = np.array(b)
        denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if denom == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / denom)

    def _extract_tokens(self, text: str) -> set[str]:
        tokens = set()
        for raw in text.replace("_", " ").split():
            cleaned = "".join(ch for ch in raw.lower() if ch.isalnum())
            if len(cleaned) >= 3:
                tokens.add(cleaned)
        return tokens

    def _extract_facts(self, text: str, speaker: str) -> list[str]:
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
        return [
            line.strip().lstrip("- ").lstrip("• ")
            for line in content.split("\n")
            if line.strip() and not line.strip().upper() == "NONE"
        ]

    def _detect_supersession(self, new_text: str) -> list[int]:
        """Detect which existing facts are superseded by new text.

        Uses lightweight token overlap heuristics instead of additional LLM calls.
        """
        if not self.facts:
            return []

        superseded_indices = []
        new_tokens = self._extract_tokens(new_text)
        cancel_markers = {"cancel", "replace", "update", "supersede", "change"}
        has_cancel = bool(cancel_markers & new_tokens)
        new_embedding = None

        for i, fact in enumerate(self.facts):
            if not fact["valid"]:
                continue
            fact_tokens = fact.get("tokens") or set()
            overlap = new_tokens & fact_tokens
            if not overlap:
                continue
            if has_cancel or any(word in new_text.lower() for word in ["no longer", "instead"]):
                superseded_indices.append(i)
            else:
                # As a fallback, compare embeddings for high similarity but timestamp difference
                if new_embedding is None:
                    new_embedding = self._embed([new_text])[0]
                similarity = self._cosine_similarity(
                    new_embedding,
                    fact["embedding"],
                )
                if similarity > 0.95:
                    superseded_indices.append(i)

        return superseded_indices

    def _add_fact(self, text: str, ts: datetime, source_key: str | None) -> None:
        """Add a fact with embedding and metadata."""
        embedding = self._embed([text])[0]
        self.facts.append({
            "text": text,
            "embedding": embedding,
            "valid": True,
            "superseded_by": None,
            "source_key": source_key,
            "ts": ts,
            "tokens": self._extract_tokens(text),
        })

    def _invalidate_by_source(self, source_key: str, superseded_text: str | None) -> None:
        """Invalidate any fact produced from a given state key."""
        for fact in self.facts:
            if fact["valid"] and fact.get("source_key") == source_key:
                fact["valid"] = False
                fact["superseded_by"] = superseded_text

    def process_event(self, event: Event) -> None:
        """Extract facts with supersession detection."""
        if isinstance(event, ConversationTurn):
            new_facts = self._extract_facts(event.text, event.speaker)

            for fact_text in new_facts:
                ts_str = event.ts.strftime("%Y-%m-%d %H:%M")
                full_text = f"[{ts_str}] {fact_text}"

                # Detect supersession
                superseded = self._detect_supersession(fact_text)
                for idx in superseded:
                    self.facts[idx]["valid"] = False
                    self.facts[idx]["superseded_by"] = full_text

                # Add new fact (no explicit source key)
                self._add_fact(full_text, event.ts, source_key=None)

        elif isinstance(event, StateWrite):
            for write in event.writes:
                if write.layer != "persistent_facts":
                    continue
                superseded_text = None
                if write.supersedes:
                    superseded_text = write.value
                    self._invalidate_by_source(write.supersedes, superseded_text)
                ts_str = event.ts.strftime("%Y-%m-%d %H:%M")
                full_text = f"[{ts_str}] {write.value}"
                self._add_fact(full_text, event.ts, source_key=write.key)

        elif isinstance(event, Supersession):
            # Explicit supersession event
            for write in event.writes:
                if write.layer != "persistent_facts":
                    continue
                superseding_text = f"[{event.ts.strftime('%Y-%m-%d %H:%M')}] {write.value}"
                if write.supersedes:
                    self._invalidate_by_source(write.supersedes, superseding_text)
                self._add_fact(superseding_text, event.ts, source_key=write.key)

    def build_context(self, query: str) -> ContextResult:
        """Retrieve only VALID facts.

        v1.0: Returns ContextResult with basic provenance.
        """
        valid_facts = [f for f in self.facts if f["valid"]]
        if not valid_facts:
            return wrap_legacy_context("")

        query_embedding = self._embed([query])[0]
        similarities = [
            (i, self._cosine_similarity(query_embedding, f["embedding"]))
            for i, f in enumerate(valid_facts)
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_facts = [valid_facts[i]["text"] for i, _ in similarities[:self.top_k]]

        context = (
            "Valid facts from memory (superseded facts filtered out):\n\n"
            + "\n".join(f"- {fact}" for fact in top_facts)
        )
        return wrap_legacy_context(context)

    def reset(self) -> None:
        self.facts = []

    def get_system_prompt(self) -> str:
        return (
            "You are an AI assistant with access to validated facts from memory. "
            "Superseded facts have been removed. Use only the current facts provided."
        )


class TranscriptLatestWinsStrategy(MemoryStrategy):
    """Transcript replay with "latest instruction wins" heuristic.

    This ablation adds explicit instruction to prefer recent statements
    over older ones. Tests whether simple recency heuristics help.
    """

    def __init__(self, token_budget: int = 8000):
        super().__init__(token_budget)
        self.turns: list[ConversationTurn] = []
        self._latest_per_speaker: dict[str, ConversationTurn] = {}
        self._encoder = tiktoken.get_encoding("cl100k_base")

    @property
    def name(self) -> str:
        return "transcript_latest_wins"

    def _count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def process_event(self, event: Event) -> None:
        if isinstance(event, ConversationTurn) and event.speaker == "user":
            self.turns.append(event)
            self._latest_per_speaker[event.speaker] = event

    def build_context(self, query: str) -> ContextResult:
        """Build context with latest-wins heuristic.

        v1.0: Returns ContextResult with basic provenance.
        """
        if not self.turns:
            return wrap_legacy_context("")

        available_budget = self.token_budget - 500

        lines: list[str] = []
        total_tokens = 0

        seen_speakers = set()
        for turn in reversed(self.turns):
            if turn.speaker in seen_speakers:
                continue
            seen_speakers.add(turn.speaker)
            line = f"{turn.speaker.title()}: {turn.text}"
            line_tokens = self._count_tokens(line)

            if total_tokens + line_tokens > available_budget:
                lines.insert(0, "[Earlier conversation truncated...]")
                break

            lines.insert(0, line)
            total_tokens += line_tokens

        # Highlight explicitly what the latest instruction per speaker is
        latest_summary = "\n".join(
            f"- {speaker.title()}: {turn.text}"
            for speaker, turn in self._latest_per_speaker.items()
        )

        context = (
            "Conversation history:\n\n" + "\n\n".join(lines)
            + ("\n\nLatest commitments:\n" + latest_summary if latest_summary else "")
        )
        return wrap_legacy_context(context)

    def reset(self) -> None:
        self.turns = []
        self._latest_per_speaker = {}

    def get_system_prompt(self) -> str:
        return (
            "You are an AI assistant. Use the conversation history to answer questions.\n\n"
            "IMPORTANT: When information conflicts, ALWAYS prefer the LATEST statement. "
            "More recent instructions, decisions, or facts supersede earlier ones. "
            "If a user says 'Actually X' or 'Change to Y' or 'Cancel Z', the new instruction "
            "takes precedence over anything said before.\n\n"
            "Be concise and accurate."
        )
