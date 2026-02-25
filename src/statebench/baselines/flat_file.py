"""Flat-File Baseline: Mimics OpenClaw's current memory system.

This baseline simulates how OpenClaw currently manages agent memory:
- MEMORY.md: Long-term facts stored as prose (no supersession tracking)
- WORKING.md: Current task context
- Daily notes: Conversation history

Key limitations this baseline demonstrates:
- No supersession tracking (old facts persist alongside new ones)
- No structured fact extraction (prose format)
- No query-relevance sorting
- No engine-level filtering
- Truncation when context exceeds budget (oldest dropped first)
"""

from __future__ import annotations

from datetime import datetime

import tiktoken

from statebench.baselines.base import ContextResult, FactMetadata, MemoryStrategy, wrap_legacy_context
from statebench.schema.timeline import (
    ConversationTurn,
    Event,
    InitialState,
    StateWrite,
    Supersession,
)


class FlatFileStrategy(MemoryStrategy):
    """Simulates OpenClaw's flat-file memory system.

    Memory is stored as unstructured text in three "files":
    - identity_text: SOUL.md / USER.md equivalent
    - memory_text: MEMORY.md equivalent (append-only, no supersession)
    - working_text: WORKING.md + daily notes equivalent
    """

    def __init__(self, token_budget: int = 8000, **kwargs: object) -> None:
        super().__init__(token_budget)
        self._identity: str = ""
        self._memory_lines: list[str] = []  # MEMORY.md lines
        self._working_lines: list[str] = []  # WORKING.md / daily notes
        self._encoder = tiktoken.get_encoding("cl100k_base")

    @property
    def name(self) -> str:
        return "flat_file"

    @property
    def expects_initial_state(self) -> bool:
        return True

    def initialize_from_state(self, initial_state: InitialState) -> None:
        """Load initial state as flat text (like reading files at boot)."""
        # Identity → SOUL.md
        ir = initial_state.identity_role
        lines = [f"User: {ir.user_name}", f"Role: {ir.authority}"]
        if ir.department:
            lines.append(f"Department: {ir.department}")
        if ir.organization:
            lines.append(f"Organization: {ir.organization}")
        self._identity = "\n".join(lines)

        # Persistent facts → MEMORY.md (just appended as prose, no structure)
        for fact in initial_state.persistent_facts:
            self._memory_lines.append(f"- {fact.key}: {fact.value}")

        # Working set → WORKING.md
        for item in initial_state.working_set:
            self._working_lines.append(item.content)

    def process_event(self, event: Event) -> None:
        """Process events by appending to flat files."""
        if isinstance(event, ConversationTurn):
            self._working_lines.append(f"{event.speaker.title()}: {event.text}")

        elif isinstance(event, StateWrite):
            for write in event.writes:
                if write.layer == "persistent_facts":
                    # Just append — no supersession tracking!
                    self._memory_lines.append(f"- {write.key}: {write.value}")
                elif write.layer == "environment":
                    self._working_lines.append(f"[env] {write.key}: {write.value}")

        elif isinstance(event, Supersession):
            for write in event.writes:
                if write.layer == "persistent_facts":
                    # Flat file: just append the new value
                    # The old value is still there — this is the core weakness
                    self._memory_lines.append(f"- {write.key}: {write.value}")
                elif write.layer == "environment":
                    self._working_lines.append(f"[env] {write.key}: {write.value}")

    def build_context(self, query: str) -> ContextResult:
        """Assemble context by concatenating files with truncation."""
        parts: list[str] = []

        # Identity always included (SOUL.md)
        if self._identity:
            parts.append(f"## Identity\n{self._identity}")

        # MEMORY.md — all facts, no filtering, no relevance sorting
        if self._memory_lines:
            memory_text = "\n".join(self._memory_lines)
            parts.append(f"## Memory\n{memory_text}")

        # WORKING.md + daily notes — recent conversation
        if self._working_lines:
            working_text = "\n".join(self._working_lines)
            parts.append(f"## Recent Context\n{working_text}")

        context = "\n\n".join(parts)

        # Truncate from the beginning (oldest content) if over budget
        tokens = self._encoder.encode(context)
        if len(tokens) > self.token_budget:
            # Keep identity, truncate memory and working from oldest
            identity_part = f"## Identity\n{self._identity}" if self._identity else ""
            identity_tokens = len(self._encoder.encode(identity_part))
            remaining_budget = self.token_budget - identity_tokens

            # Split remaining budget: 60% memory, 40% working
            memory_budget = int(remaining_budget * 0.6)
            working_budget = remaining_budget - memory_budget

            # Truncate memory (keep latest lines)
            memory_text = self._truncate_to_budget(
                self._memory_lines, memory_budget
            )
            working_text = self._truncate_to_budget(
                self._working_lines, working_budget
            )

            parts = []
            if identity_part:
                parts.append(identity_part)
            if memory_text:
                parts.append(f"## Memory\n{memory_text}")
            if working_text:
                parts.append(f"## Recent Context\n{working_text}")

            context = "\n\n".join(parts)

        token_count = len(self._encoder.encode(context))
        return ContextResult(
            context=context,
            facts_included=[],  # No provenance in flat-file system
            facts_excluded=[],
            inclusion_reasons={},
            token_count=token_count,
        )

    def _truncate_to_budget(self, lines: list[str], budget: int) -> str:
        """Keep the most recent lines that fit within token budget."""
        if not lines:
            return ""
        # Start from end, add lines until budget exhausted
        kept: list[str] = []
        total = 0
        for line in reversed(lines):
            line_tokens = len(self._encoder.encode(line))
            if total + line_tokens > budget:
                break
            kept.append(line)
            total += line_tokens
        kept.reverse()
        return "\n".join(kept)

    def reset(self) -> None:
        """Reset for new timeline."""
        self._identity = ""
        self._memory_lines = []
        self._working_lines = []

    def get_system_prompt(self) -> str:
        return (
            "You are an AI assistant. Answer questions based on the context provided. "
            "Be concise and accurate. If information conflicts, use the most recent version."
        )
