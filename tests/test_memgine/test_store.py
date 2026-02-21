"""Tests for the ImmutableStore."""

from datetime import datetime

from statebench.memgine.store import ImmutableStore


def test_append_and_get() -> None:
    store = ImmutableStore()
    ts = datetime(2024, 1, 1)
    entry = store.append(layer=2, kind="fact", key="budget", value="$50k", ts=ts, fact_id="F-001")

    assert entry.id == 0
    assert entry.layer == 2
    assert entry.key == "budget"
    assert entry.value == "$50k"
    assert entry.fact_id == "F-001"

    retrieved = store.get(0)
    assert retrieved is entry


def test_get_by_fact_id() -> None:
    store = ImmutableStore()
    ts = datetime(2024, 1, 1)
    store.append(layer=2, kind="fact", key="budget", value="$50k", ts=ts, fact_id="F-001")
    store.append(layer=2, kind="fact", key="deadline", value="March", ts=ts, fact_id="F-002")

    entry = store.get_by_fact_id("F-001")
    assert entry is not None
    assert entry.key == "budget"

    assert store.get_by_fact_id("F-999") is None


def test_get_by_layer() -> None:
    store = ImmutableStore()
    ts = datetime(2024, 1, 1)
    store.append(layer=1, kind="identity", key="id", value="Alice", ts=ts)
    store.append(layer=2, kind="fact", key="budget", value="$50k", ts=ts, fact_id="F-001")
    store.append(layer=3, kind="conversation", key="user", value="Hello", ts=ts)
    store.append(layer=2, kind="fact", key="deadline", value="March", ts=ts, fact_id="F-002")

    layer2 = store.get_by_layer(2)
    assert len(layer2) == 2
    assert layer2[0].key == "budget"
    assert layer2[1].key == "deadline"


def test_get_range() -> None:
    store = ImmutableStore()
    ts = datetime(2024, 1, 1)
    for i in range(5):
        store.append(layer=3, kind="conversation", key="user", value=f"msg-{i}", ts=ts)

    entries = store.get_range(1, 4)
    assert len(entries) == 3
    assert entries[0].value == "msg-1"
    assert entries[2].value == "msg-3"


def test_immutability() -> None:
    store = ImmutableStore()
    ts = datetime(2024, 1, 1)
    entry = store.append(layer=2, kind="fact", key="k", value="v", ts=ts, fact_id="F-001")

    # StoreEntry is frozen, so setting attributes should raise
    try:
        entry.value = "modified"  # type: ignore[misc]
        assert False, "Should have raised"
    except AttributeError:
        pass


def test_clear() -> None:
    store = ImmutableStore()
    ts = datetime(2024, 1, 1)
    store.append(layer=2, kind="fact", key="k", value="v", ts=ts, fact_id="F-001")
    assert len(store) == 1

    store.clear()
    assert len(store) == 0
    assert store.get_by_fact_id("F-001") is None
