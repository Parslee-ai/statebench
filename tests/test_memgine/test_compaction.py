"""Tests for CompactionEngine."""

from datetime import datetime, timedelta

from statebench.memgine.compaction import CompactionEngine
from statebench.memgine.config import CompactionThresholds, MemgineConfig
from statebench.memgine.dag import SummaryDAG
from statebench.memgine.layers import LayerState
from statebench.memgine.store import ImmutableStore


def _make_components(
    budget: int = 100, soft: float = 0.70, hard: float = 0.95
) -> tuple[MemgineConfig, ImmutableStore, LayerState, SummaryDAG, CompactionEngine]:
    config = MemgineConfig(
        token_budget=budget,
        thresholds=CompactionThresholds(soft=soft, hard=hard),
    )
    store = ImmutableStore()
    layers = LayerState()
    dag = SummaryDAG(store)
    compactor = CompactionEngine(config, store, layers, dag)
    return config, store, layers, dag, compactor


def test_zero_cost_continuity_below_threshold() -> None:
    """Below soft threshold, no compaction happens."""
    _, store, layers, _, compactor = _make_components(budget=1000)
    ts = datetime(2024, 1, 1)

    # Add a small amount of data (well below 70%)
    store.append(layer=2, kind="fact", key="k1", value="short", ts=ts, fact_id="F-001")
    layers.register_fact(entry_id=0, fact_id="F-001", fact_key="k1")

    assert not compactor.needs_compaction()
    result = compactor.maybe_compact()
    assert result is None  # No compaction needed


def test_compaction_triggers_at_soft_threshold() -> None:
    """Above soft threshold, compaction runs."""
    # Use tiny budget so a few entries exceed it
    _, store, layers, _, compactor = _make_components(budget=10, soft=0.5)
    ts = datetime(2024, 1, 1)

    # Add enough data to exceed 50% of 10 tokens (~5 tokens)
    for i in range(5):
        entry = store.append(
            layer=2,
            kind="fact",
            key=f"key_{i}",
            value=f"value number {i} is here",
            ts=ts,
            fact_id=f"F-{i:03d}",
        )
        layers.register_fact(entry_id=entry.id, fact_id=f"F-{i:03d}", fact_key=f"key_{i}")

    assert compactor.needs_compaction()
    result = compactor.maybe_compact()
    assert result is not None
    assert result.level == 2
    assert result.entries_compacted > 0


def test_superseded_facts_discarded() -> None:
    """Superseded facts are discarded entirely at Level 2."""
    _, store, layers, _, compactor = _make_components(budget=10, soft=0.1)
    ts = datetime(2024, 1, 1)

    # Create two facts, supersede the first
    entry1 = store.append(
        layer=2, kind="fact", key="budget", value="$50k", ts=ts, fact_id="F-001"
    )
    layers.register_fact(entry_id=entry1.id, fact_id="F-001", fact_key="budget")

    entry2 = store.append(
        layer=2, kind="fact", key="budget", value="$75k", ts=ts, fact_id="F-002"
    )
    layers.register_fact(entry_id=entry2.id, fact_id="F-002", fact_key="budget")

    layers.invalidate_fact("F-001", "F-002")

    result = compactor.compact_level2()
    assert entry1.id in compactor.discarded_entries


def test_constraints_never_compacted() -> None:
    """Constraint facts are preserved verbatim."""
    _, store, layers, dag, compactor = _make_components(budget=10, soft=0.1)
    ts = datetime(2024, 1, 1)

    entry = store.append(
        layer=2,
        kind="fact",
        key="policy",
        value="Budget must not exceed $100k",
        ts=ts,
        fact_id="F-001",
        is_constraint=True,
    )
    layers.register_fact(
        entry_id=entry.id, fact_id="F-001", fact_key="policy", is_constraint=True
    )

    result = compactor.compact_level2()
    # Constraint should NOT be compacted or discarded
    assert entry.id not in compactor.compacted_entries
    assert entry.id not in compactor.discarded_entries


def test_working_set_keep_recent_n() -> None:
    """Working set keeps only recent N entries at Level 2."""
    config, store, layers, _, compactor = _make_components(budget=10, soft=0.1)
    config.working_set_keep_recent = 3
    ts = datetime(2024, 1, 1)

    for i in range(7):
        entry = store.append(
            layer=3,
            kind="conversation",
            key="user",
            value=f"message number {i}",
            ts=ts,
        )
        layers.register_working_set(entry.id)

    result = compactor.compact_level2()
    # First 4 entries (0-3) should be discarded, last 3 (4-6) kept
    assert result.entries_compacted >= 4
    for eid in [0, 1, 2, 3]:
        assert eid in compactor.discarded_entries
    for eid in [4, 5, 6]:
        assert eid not in compactor.discarded_entries


def test_environment_expired_discarded() -> None:
    """Expired environment signals are discarded."""
    _, store, layers, _, compactor = _make_components(budget=10, soft=0.1)
    now = datetime(2024, 1, 10)
    past = datetime(2024, 1, 1)
    expired = datetime(2024, 1, 5)

    entry = store.append(
        layer=4,
        kind="environment",
        key="deadline",
        value="Jan 5 deadline",
        ts=past,
        expires=expired,
    )
    layers.register_environment("deadline", entry.id)

    result = compactor.compact_level2(now=now)
    assert entry.id in compactor.discarded_entries


def test_compaction_creates_dag_nodes() -> None:
    """Compaction creates DAG summary nodes from raw entries."""
    _, store, layers, dag, compactor = _make_components(budget=10, soft=0.1)
    ts = datetime(2024, 1, 1)

    for i in range(5):
        entry = store.append(
            layer=2,
            kind="fact",
            key=f"key_{i}",
            value=f"a somewhat long value for fact number {i}",
            ts=ts,
            fact_id=f"F-{i:03d}",
        )
        layers.register_fact(entry_id=entry.id, fact_id=f"F-{i:03d}", fact_key=f"key_{i}")

    result = compactor.compact_level2()
    assert result.entries_compacted == 5
    assert result.nodes_created == 1
    assert len(result.nodes_created_ids) == 1
    # All entries should be in the compacted set
    for i in range(5):
        assert i in compactor.compacted_entries
    # DAG node should exist
    node = dag.get(result.nodes_created_ids[0])
    assert len(node.covered_fact_ids) == 5
