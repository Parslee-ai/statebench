"""Immutable Store: append-only event log.

Source of truth for all raw data. Events are never modified.
In-memory implementation with dict indices.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from statebench.memgine.types import EntryId, EntryKind, Layer, StoreEntry
from statebench.schema.state import AuthorityLevel, Scope


class ImmutableStore:
    """Append-only event log with dict indices.

    Interface is narrow enough for a DB backend later.
    """

    def __init__(self) -> None:
        self._entries: list[StoreEntry] = []
        self._by_fact_id: dict[str, EntryId] = {}
        self._by_layer: dict[Layer, list[EntryId]] = {1: [], 2: [], 3: [], 4: []}

    def append(
        self,
        layer: Layer,
        kind: EntryKind,
        key: str,
        value: str,
        ts: datetime,
        *,
        fact_id: str | None = None,
        source_type: str | None = None,
        source_identity: str | None = None,
        authority: AuthorityLevel = "peer",
        scope: Scope = "global",
        supersedes_fact_id: str | None = None,
        depends_on: tuple[str, ...] = (),
        is_constraint: bool = False,
        constraint_type: str | None = None,
        expires: datetime | None = None,
        urgency: Literal["low", "medium", "high", "critical"] = "medium",
    ) -> StoreEntry:
        """Create and store an immutable entry. Returns the entry."""
        entry_id = len(self._entries)
        entry = StoreEntry(
            id=entry_id,
            layer=layer,
            kind=kind,
            key=key,
            value=value,
            ts=ts,
            fact_id=fact_id,
            source_type=source_type,
            source_identity=source_identity,
            authority=authority,
            scope=scope,
            supersedes_fact_id=supersedes_fact_id,
            depends_on=depends_on,
            is_constraint=is_constraint,
            constraint_type=constraint_type,
            expires=expires,
            urgency=urgency,
        )
        self._entries.append(entry)
        self._by_layer[layer].append(entry_id)
        if fact_id is not None:
            self._by_fact_id[fact_id] = entry_id
        return entry

    def get(self, entry_id: EntryId) -> StoreEntry:
        """Lookup by ID. Raises IndexError if not found."""
        return self._entries[entry_id]

    def get_by_fact_id(self, fact_id: str) -> StoreEntry | None:
        """Layer 2 index: lookup by fact_id."""
        entry_id = self._by_fact_id.get(fact_id)
        if entry_id is None:
            return None
        return self._entries[entry_id]

    def get_range(self, start: EntryId, end: EntryId) -> list[StoreEntry]:
        """Get entries in [start, end) range for summarization spans."""
        return self._entries[start:end]

    def get_by_layer(self, layer: Layer) -> list[StoreEntry]:
        """Get all entries for a layer, in insertion order."""
        return [self._entries[eid] for eid in self._by_layer[layer]]

    def __len__(self) -> int:
        return len(self._entries)

    def total_tokens(self) -> int:
        """Estimate total tokens across all entries."""
        return sum(e.token_estimate() for e in self._entries)

    def layer_tokens(self, layer: Layer) -> int:
        """Estimate total tokens for a layer."""
        return sum(self._entries[eid].token_estimate() for eid in self._by_layer[layer])

    def clear(self) -> None:
        """Reset the store."""
        self._entries.clear()
        self._by_fact_id.clear()
        for layer in self._by_layer:
            self._by_layer[layer].clear()
