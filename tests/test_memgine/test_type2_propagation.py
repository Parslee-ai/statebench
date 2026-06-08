"""Tests for Type II supersession propagation (CUPMem-style transitive cascade).

When a fact's premise is superseded, everything DERIVED from it (transitively) is
flagged recalc-pending — not just the direct dependent, and not hard-deleted.
"""

from datetime import datetime, timedelta

from statebench.memgine.config import MemgineConfig
from statebench.memgine.engine import MemgineEngine
from statebench.memgine.layers import LayerState
from statebench.schema.state import Source

T0 = datetime(2026, 1, 1, 12, 0)


def _engine() -> MemgineEngine:
    return MemgineEngine(MemgineConfig(token_budget=8000))


def _fact(eng, fid, key, val, *, depends_on=None, supersedes=None, ts=T0):
    return eng.ingest_fact(
        fid, key, val, Source(type="user", authority="peer"), ts,
        depends_on=depends_on, supersedes=supersedes,
    )


def test_multi_hop_dependent_chain_all_flagged() -> None:
    eng = _engine()
    _fact(eng, "F", "unit_price", "$10")
    _fact(eng, "D1", "subtotal", "$100", depends_on=["F"])       # derived from F
    _fact(eng, "D2", "total_with_tax", "$110", depends_on=["D1"])  # derived from D1
    # Supersede the base fact F.
    _fact(eng, "G", "unit_price", "$12", supersedes="F", ts=T0 + timedelta(hours=1))

    # Base is dead; both derived facts (1- and 2-hop) are flagged for recalculation.
    assert "F" not in eng._layers.get_valid_fact_ids()
    assert "D1" in eng._needs_review
    assert "D2" in eng._needs_review  # the transitive hop — the actual fix


def test_dependents_stay_valid_not_deleted() -> None:
    # Recalc-pending, not hard-invalidated: still in valid_facts (shown as RECALCULATE).
    eng = _engine()
    _fact(eng, "F", "base", "10")
    _fact(eng, "D1", "derived", "20", depends_on=["F"])
    _fact(eng, "G", "base", "12", supersedes="F", ts=T0 + timedelta(hours=1))
    assert "D1" in eng._layers.get_valid_fact_ids()  # not deleted
    assert "D1" not in eng._layers.superseded_by      # no successor — needs recalc
    assert "D1" in eng._needs_review


def test_independent_fact_not_flagged() -> None:
    # A fact that does NOT depend on the superseded base is untouched (no over-propagation).
    eng = _engine()
    _fact(eng, "F", "base", "10")
    _fact(eng, "D1", "derived", "20", depends_on=["F"])
    _fact(eng, "X", "unrelated", "hello")  # depends on nothing
    _fact(eng, "G", "base", "12", supersedes="F", ts=T0 + timedelta(hours=1))
    assert "X" not in eng._needs_review
    assert "X" in eng._layers.get_valid_fact_ids()


def test_cyclic_dependency_terminates() -> None:
    # Pathological cycle must not infinite-loop.
    layers = LayerState()
    layers.valid_facts = {"A": 1, "B": 2}
    layers.validity = {1: True, 2: True}
    layers.dependents = {"A": {"B"}, "B": {"A"}}
    invalidated: set[str] = set()
    layers._invalidate_recursive("A", "G", invalidated)
    assert invalidated == {"A", "B"}  # terminates, both reached once


def test_deep_chain_no_recursion_error() -> None:
    # The cascade is shared by ALL supersession; a deep chain must not blow the
    # Python recursion limit (it is iterative).
    layers = LayerState()
    n = 3000
    for i in range(n):
        layers.valid_facts[f"f{i}"] = i
        layers.validity[i] = True
    for i in range(n - 1):
        layers.dependents.setdefault(f"f{i}", set()).add(f"f{i + 1}")
    invalidated: set[str] = set()
    layers._invalidate_recursive("f0", "G", invalidated)  # must not raise
    assert len(invalidated) == n  # whole chain reached


def test_flagged_dependent_rendered_recalc_not_live_fact() -> None:
    # A transitively-flagged fact is shown as recalc-pending (RECALCULATE), never as
    # a plain live "- [D]" current-fact line the model could read as current.
    eng = _engine()
    _fact(eng, "F", "base_rate", "10%")
    _fact(eng, "D", "computed_fee", "stale-fee-from-10pct", depends_on=["F"])
    _fact(eng, "G", "base_rate", "12%", supersedes="F", ts=T0 + timedelta(hours=1))
    ctx = eng.build_context("what is the computed fee?")
    plain_live_lines = [
        ln for ln in ctx.context.splitlines() if ln.strip().startswith("- [D]")
    ]
    assert not plain_live_lines  # D is not a live current fact
    assert "D" in eng._needs_review  # it is flagged recalc-pending instead
