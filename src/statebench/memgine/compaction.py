"""Compaction Engine: layer-aware, threshold-based compaction.

Phase 1: Level 2 (deterministic) only. No LLM calls.

Thresholds (from LCM):
- Below soft (70%): Zero-cost continuity — no overhead
- Soft to hard (70-95%): Compaction triggered
- At hard (95%): Blocking compaction

Layer-specific rules:
| Layer | Level 2 (Deterministic)         |
|-------|----------------------------------|
| 1     | Never compacted                  |
| 2 V   | Compress to key=value            |
| 2 S   | Discard entirely                 |
| 2 C   | Never compacted                  |
| 3     | Keep recent 3, discard rest      |
| 4     | Discard expired + truncate       |
"""

from __future__ import annotations

from datetime import datetime

from statebench.memgine.config import MemgineConfig
from statebench.memgine.dag import SummaryDAG
from statebench.memgine.layers import LayerState
from statebench.memgine.store import ImmutableStore
from statebench.memgine.types import CompactionResult, Layer


class CompactionEngine:
    """Layer-aware compaction engine.

    Phase 1: Only Level 2 (deterministic) compaction.
    """

    def __init__(
        self,
        config: MemgineConfig,
        store: ImmutableStore,
        layer_state: LayerState,
        dag: SummaryDAG,
    ) -> None:
        self._config = config
        self._store = store
        self._layers = layer_state
        self._dag = dag
        # Track which entries have been compacted into DAG nodes
        self._compacted_entries: set[int] = set()
        # Track which entries have been discarded (shouldn't appear in context)
        self._discarded_entries: set[int] = set()

    @property
    def compacted_entries(self) -> set[int]:
        return self._compacted_entries

    @property
    def discarded_entries(self) -> set[int]:
        return self._discarded_entries

    def current_usage(self) -> float:
        """Current token usage as a fraction of budget."""
        total = self._effective_tokens()
        return total / max(1, self._config.token_budget)

    def _effective_tokens(self) -> int:
        """Calculate effective token count (raw entries minus compacted/discarded + DAG summaries)."""
        raw = 0
        for entry in self._store.get_range(0, len(self._store)):
            if entry.id not in self._compacted_entries and entry.id not in self._discarded_entries:
                raw += entry.token_estimate()

        dag_tokens = 0
        for layer in (1, 2, 3, 4):
            for node in self._dag.get_by_layer(layer):
                dag_tokens += node.token_estimate()

        return raw + dag_tokens

    def needs_compaction(self) -> bool:
        """Check if compaction is needed (usage above soft threshold)."""
        return self.current_usage() >= self._config.thresholds.soft

    def needs_blocking_compaction(self) -> bool:
        """Check if blocking compaction is needed (usage above hard threshold)."""
        return self.current_usage() >= self._config.thresholds.hard

    def maybe_compact(self, now: datetime | None = None) -> CompactionResult | None:
        """Run compaction if needed. Returns result or None if not needed."""
        if not self.needs_compaction():
            return None
        return self.compact_level2(now=now)

    def compact_level2(self, *, now: datetime | None = None) -> CompactionResult:
        """Level 2: Purely deterministic compaction. No LLM.

        Layer 1: Never compacted
        Layer 2 (valid facts): Compress to "key=value" summary
        Layer 2 (superseded): Discard entirely
        Layer 2 (constraints): Never compacted
        Layer 3: Keep recent N, discard rest
        Layer 4: Discard expired, truncate
        """
        tokens_before = self._effective_tokens()
        entries_compacted = 0
        nodes_created = 0
        entries_discarded: list[int] = []
        nodes_created_ids: list[int] = []

        # --- Layer 2: Superseded facts → discard ---
        for fact_id in list(self._layers.get_superseded_fact_ids()):
            entry = self._store.get_by_fact_id(fact_id)
            if entry and entry.id not in self._discarded_entries:
                self._discarded_entries.add(entry.id)
                entries_discarded.append(entry.id)
                entries_compacted += 1

        # --- Layer 2: Valid non-constraint facts → compress to key=value ---
        valid_non_constraint_ids = [
            fid
            for fid in self._layers.get_valid_fact_ids()
            if fid not in self._layers.constraints
        ]
        entries_to_compact: list[int] = []
        for fid in valid_non_constraint_ids:
            entry = self._store.get_by_fact_id(fid)
            if entry and entry.id not in self._compacted_entries and entry.id not in self._discarded_entries:
                entries_to_compact.append(entry.id)

        if entries_to_compact:
            # Build compressed summary
            lines: list[str] = []
            covered_fact_ids: list[str] = []
            for eid in entries_to_compact:
                entry = self._store.get(eid)
                if entry.fact_id:
                    lines.append(f"[{entry.fact_id}] {entry.key}={entry.value}")
                    covered_fact_ids.append(entry.fact_id)

            if lines:
                summary_text = "\n".join(lines)
                node = self._dag.create_leaf(
                    layer=2,
                    summary_text=summary_text,
                    entry_ids=entries_to_compact,
                    covered_fact_ids=covered_fact_ids,
                )
                for eid in entries_to_compact:
                    self._compacted_entries.add(eid)
                entries_compacted += len(entries_to_compact)
                nodes_created += 1
                nodes_created_ids.append(node.id)

        # --- Layer 3: Keep recent N, discard rest ---
        keep_n = self._config.working_set_keep_recent
        ws_entries = self._layers.working_set_entries
        if len(ws_entries) > keep_n:
            to_discard = ws_entries[:-keep_n]
            for eid in to_discard:
                if eid not in self._discarded_entries:
                    self._discarded_entries.add(eid)
                    entries_discarded.append(eid)
                    entries_compacted += 1

        # --- Layer 4: Discard expired ---
        if now:
            for key, eid in list(self._layers.environment_entries.items()):
                entry = self._store.get(eid)
                if entry.expires and entry.expires <= now:
                    if eid not in self._discarded_entries:
                        self._discarded_entries.add(eid)
                        entries_discarded.append(eid)
                        entries_compacted += 1
                        del self._layers.environment_entries[key]

        # Truncate environment to max
        env_max = self._config.environment_max
        env_entries = list(self._layers.environment_entries.items())
        if len(env_entries) > env_max:
            # Sort by entry id (insertion order) and keep most recent
            sorted_env = sorted(env_entries, key=lambda x: x[1])
            to_remove = sorted_env[: len(sorted_env) - env_max]
            for key, eid in to_remove:
                if eid not in self._discarded_entries:
                    self._discarded_entries.add(eid)
                    entries_discarded.append(eid)
                    entries_compacted += 1
                del self._layers.environment_entries[key]

        tokens_after = self._effective_tokens()
        return CompactionResult(
            level=2,
            entries_compacted=entries_compacted,
            nodes_created=nodes_created,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            entries_discarded=entries_discarded,
            nodes_created_ids=nodes_created_ids,
        )

    def layer_usage(self, layer: Layer) -> int:
        """Current effective token count for a layer."""
        raw = 0
        for entry in self._store.get_by_layer(layer):
            if entry.id not in self._compacted_entries and entry.id not in self._discarded_entries:
                raw += entry.token_estimate()
        dag_tokens = sum(n.token_estimate() for n in self._dag.get_by_layer(layer))
        return raw + dag_tokens

    def clear(self) -> None:
        """Reset compaction state."""
        self._compacted_entries.clear()
        self._discarded_entries.clear()
