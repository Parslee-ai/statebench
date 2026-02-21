"""Tests for SummaryDAG."""

from datetime import datetime

from statebench.memgine.dag import SummaryDAG
from statebench.memgine.store import ImmutableStore


def _make_store_with_entries(n: int) -> ImmutableStore:
    store = ImmutableStore()
    ts = datetime(2024, 1, 1)
    for i in range(n):
        store.append(
            layer=3, kind="conversation", key="user", value=f"msg-{i}", ts=ts
        )
    return store


def test_create_leaf_and_expand() -> None:
    store = _make_store_with_entries(5)
    dag = SummaryDAG(store)

    node = dag.create_leaf(
        layer=3,
        summary_text="Messages 0-2 summary",
        entry_ids=[0, 1, 2],
    )

    assert node.is_leaf
    assert node.id == 0

    # Expand returns original entry IDs
    expanded = dag.expand(node.id)
    assert expanded == [0, 1, 2]


def test_create_condensed_and_expand() -> None:
    store = _make_store_with_entries(6)
    dag = SummaryDAG(store)

    leaf1 = dag.create_leaf(layer=3, summary_text="0-2", entry_ids=[0, 1, 2])
    leaf2 = dag.create_leaf(layer=3, summary_text="3-5", entry_ids=[3, 4, 5])

    condensed = dag.create_condensed(
        layer=3,
        summary_text="All messages",
        child_node_ids=[leaf1.id, leaf2.id],
    )

    assert not condensed.is_leaf

    # Expand recursively resolves to all original entries
    expanded = dag.expand(condensed.id)
    assert expanded == [0, 1, 2, 3, 4, 5]


def test_expand_to_entries() -> None:
    store = _make_store_with_entries(3)
    dag = SummaryDAG(store)

    node = dag.create_leaf(layer=3, summary_text="summary", entry_ids=[0, 1, 2])
    entries = dag.expand_to_entries(node.id)

    assert len(entries) == 3
    assert entries[0].value == "msg-0"
    assert entries[2].value == "msg-2"


def test_layer2_metadata() -> None:
    store = ImmutableStore()
    ts = datetime(2024, 1, 1)
    store.append(layer=2, kind="fact", key="budget", value="$50k", ts=ts, fact_id="F-001")
    store.append(layer=2, kind="fact", key="deadline", value="March", ts=ts, fact_id="F-002")

    dag = SummaryDAG(store)
    node = dag.create_leaf(
        layer=2,
        summary_text="[F-001] budget=$50k\n[F-002] deadline=March",
        entry_ids=[0, 1],
        covered_fact_ids=["F-001", "F-002"],
    )

    assert node.covered_fact_ids == ("F-001", "F-002")
    assert node.all_superseded is False


def test_get_by_layer() -> None:
    store = _make_store_with_entries(3)
    dag = SummaryDAG(store)

    dag.create_leaf(layer=3, summary_text="l3", entry_ids=[0, 1])
    dag.create_leaf(layer=2, summary_text="l2", entry_ids=[2])

    assert len(dag.get_by_layer(3)) == 1
    assert len(dag.get_by_layer(2)) == 1
    assert len(dag.get_by_layer(1)) == 0


def test_clear() -> None:
    store = _make_store_with_entries(3)
    dag = SummaryDAG(store)
    dag.create_leaf(layer=3, summary_text="s", entry_ids=[0])

    dag.clear()
    assert len(dag.get_by_layer(3)) == 0
