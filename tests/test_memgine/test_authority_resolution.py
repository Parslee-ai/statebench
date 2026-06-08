"""Tests for authority-aware same-key conflict resolution (Whose-Facts-Win fix).

A later low-authority note must not silently override a standing high-authority
directive; recency only breaks ties between equal-authority sources.
"""

from datetime import datetime, timedelta

from statebench.memgine.config import MemgineConfig
from statebench.memgine.engine import MemgineEngine
from statebench.schema.state import Source

T0 = datetime(2026, 1, 1, 12, 0)


def _engine(gap: int = 1) -> MemgineEngine:
    return MemgineEngine(MemgineConfig(token_budget=8000, authority_resolution_gap=gap))


def _ingest(eng, fid, key, val, authority, scope="global", ts=T0):
    return eng.ingest_fact(fid, key, val, Source(type="user", authority=authority), ts, scope=scope)


def _valid(eng):
    return set(eng._layers.get_valid_fact_ids())


def test_higher_authority_newcomer_supersedes_lower() -> None:
    eng = _engine()
    _ingest(eng, "f1", "purchase", "approved (manager)", "manager", ts=T0)
    _ingest(eng, "f2", "purchase", "frozen (CEO)", "executive", ts=T0 + timedelta(hours=1))
    assert _valid(eng) == {"f2"}
    assert eng._layers.superseded_by.get("f1") == "f2"


def test_standing_higher_authority_survives_later_lower_note() -> None:
    # The key fix: recency alone would let the later manager note win; authority must not.
    eng = _engine()
    _ingest(eng, "f1", "purchase", "frozen (CEO)", "executive", ts=T0)
    _ingest(eng, "f2", "purchase", "approved (manager)", "manager", ts=T0 + timedelta(hours=1))
    assert _valid(eng) == {"f1"}
    assert eng._layers.superseded_by.get("f2") == "f1"


def test_equal_authority_keeps_both() -> None:
    eng = _engine()
    _ingest(eng, "f1", "x", "A", "peer", ts=T0)
    _ingest(eng, "f2", "x", "B", "peer", ts=T0 + timedelta(hours=1))
    assert _valid(eng) == {"f1", "f2"}


def test_different_scope_not_resolved() -> None:
    eng = _engine()
    _ingest(eng, "f1", "x", "A", "manager", scope="global", ts=T0)
    _ingest(eng, "f2", "x", "B", "executive", scope="task", ts=T0 + timedelta(hours=1))
    assert _valid(eng) == {"f1", "f2"}


def test_same_value_not_a_conflict() -> None:
    eng = _engine()
    _ingest(eng, "f1", "x", "same", "manager", ts=T0)
    _ingest(eng, "f2", "x", "same", "executive", ts=T0 + timedelta(hours=1))
    # identical value -> no authority retirement here (handled as dup elsewhere)
    assert _valid(eng) == {"f1", "f2"}


def test_gap_zero_disables_resolution() -> None:
    eng = _engine(gap=0)
    _ingest(eng, "f1", "purchase", "approved (manager)", "manager", ts=T0)
    _ingest(eng, "f2", "purchase", "frozen (CEO)", "executive", ts=T0 + timedelta(hours=1))
    assert _valid(eng) == {"f1", "f2"}


def test_multiple_standing_losers_point_at_single_winner() -> None:
    # Two equal-authority facts coexist (neither retires the other); a higher
    # authority then arrives and retires BOTH, each pointing directly at the one
    # winner (within-call global resolution, not a chain).
    eng = _engine()
    _ingest(eng, "f0", "k", "peer A", "peer", ts=T0)
    _ingest(eng, "f1", "k", "peer B", "peer", ts=T0 + timedelta(hours=1))
    assert _valid(eng) == {"f0", "f1"}  # equal authority -> both stand
    _ingest(eng, "f2", "k", "policy val", "policy", ts=T0 + timedelta(hours=2))
    assert _valid(eng) == {"f2"}
    assert eng._layers.superseded_by.get("f0") == "f2"
    assert eng._layers.superseded_by.get("f1") == "f2"


def test_constraint_fact_not_silently_retired_by_equal_authority() -> None:
    eng = _engine()
    _ingest(eng, "c1", "budget", "max $100k per policy", "policy", ts=T0)
    # a peer note with a different value does NOT outrank policy -> policy stays
    _ingest(eng, "n1", "budget", "let's do $150k", "peer", ts=T0 + timedelta(hours=1))
    assert "c1" in _valid(eng) and "n1" not in _valid(eng)


def test_explicit_supersession_unaffected() -> None:
    # Explicit supersedes still works and authority resolution doesn't interfere.
    eng = _engine()
    _ingest(eng, "f1", "x", "old", "executive", ts=T0)
    eng.ingest_fact(
        "f2", "x", "new", Source(type="user", authority="peer"),
        T0 + timedelta(hours=1), supersedes="f1",
    )
    assert _valid(eng) == {"f2"}
