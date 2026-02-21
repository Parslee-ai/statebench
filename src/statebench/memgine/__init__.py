"""Memgine: State-Aware Deterministic Memory Engine.

Combines LCM's architectural patterns (immutable store, hierarchical DAG,
deterministic compaction) with StateBench's semantic layer awareness.
"""

from statebench.memgine.compaction import CompactionEngine
from statebench.memgine.config import MemgineConfig
from statebench.memgine.dag import SummaryDAG
from statebench.memgine.engine import MemgineEngine
from statebench.memgine.layers import LayerState
from statebench.memgine.store import ImmutableStore
from statebench.memgine.strategy import MemgineStrategy
from statebench.memgine.types import CompactionResult, StoreEntry, SummaryNode

__all__ = [
    "CompactionEngine",
    "CompactionResult",
    "ImmutableStore",
    "LayerState",
    "MemgineConfig",
    "MemgineEngine",
    "MemgineStrategy",
    "StoreEntry",
    "SummaryDAG",
    "SummaryNode",
]
