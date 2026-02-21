"""Core data types for the Memgine engine.

StoreEntry: Immutable record in the append-only store.
SummaryNode: Hierarchical summary with provenance pointers.
CompactionResult: What a compaction pass produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from statebench.schema.state import AuthorityLevel, Scope

# Semantic type aliases
EntryId = int
NodeId = int
Layer = Literal[1, 2, 3, 4]
EntryKind = Literal[
    "identity",
    "fact",
    "fact_superseded",
    "conversation",
    "working_set",
    "environment",
]


@dataclass(frozen=True)
class StoreEntry:
    """Immutable record in the append-only store.

    Once created, never modified. Mutable state (validity, superseded_by)
    lives in LayerState indices.
    """

    id: EntryId
    layer: Layer
    kind: EntryKind
    key: str
    value: str
    ts: datetime

    # Layer 2 specific
    fact_id: str | None = None
    source_type: str | None = None
    source_identity: str | None = None
    authority: AuthorityLevel = "peer"
    scope: Scope = "global"
    supersedes_fact_id: str | None = None
    depends_on: tuple[str, ...] = ()
    is_constraint: bool = False
    constraint_type: str | None = None

    # Layer 4 specific
    expires: datetime | None = None
    urgency: Literal["low", "medium", "high", "critical"] = "medium"

    def token_estimate(self) -> int:
        """Rough token count (chars / 4)."""
        return max(1, len(self.value) // 4)


@dataclass(frozen=True)
class SummaryNode:
    """A node in the hierarchical summary DAG.

    Leaf nodes summarize spans of StoreEntry IDs.
    Condensed nodes summarize other SummaryNodes.
    """

    id: NodeId
    layer: Layer
    summary_text: str
    # For leaf nodes: the store entries this summarizes
    entry_ids: tuple[EntryId, ...] = ()
    # For condensed nodes: the child summary nodes
    child_node_ids: tuple[NodeId, ...] = ()
    # Layer 2 metadata
    covered_fact_ids: tuple[str, ...] = ()
    all_superseded: bool = False

    @property
    def is_leaf(self) -> bool:
        return len(self.entry_ids) > 0

    def token_estimate(self) -> int:
        return max(1, len(self.summary_text) // 4)


@dataclass
class CompactionResult:
    """Result of a compaction pass."""

    level: int  # 0, 1, or 2
    entries_compacted: int
    nodes_created: int
    tokens_before: int
    tokens_after: int
    entries_discarded: list[EntryId] = field(default_factory=list)
    nodes_created_ids: list[NodeId] = field(default_factory=list)
