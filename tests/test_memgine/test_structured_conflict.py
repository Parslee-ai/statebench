"""Integration tests: the structured-conflict tracks actually EXERCISE the
authority-resolution and Type II propagation engine fixes (which are latent on the
NL-based authority_hierarchy / repair_propagation tracks)."""

from statebench.baselines import get_baseline
from statebench.generator.engine import TimelineGenerator
from statebench.schema.timeline import Query


def _run_to_query(tl):
    s = get_baseline("memgine", token_budget=8000)
    s.reset()
    if getattr(s, "expects_initial_state", False):
        s.initialize_from_state(tl.initial_state)
    q = None
    for e in tl.events:
        if isinstance(e, Query):
            q = e
            break
        s.process_event(e)
    return s, q


def test_authority_conflict_track_fires_authority_resolution() -> None:
    gen = TimelineGenerator(seed=3)
    fired = 0
    tls = list(gen.generate_track("authority_conflict", count=8))
    assert all(t.track == "authority_conflict" for t in tls)
    for tl in tls:
        s, _ = _run_to_query(tl)
        # exactly one valid same-key fact survives (the higher-authority one),
        # via an authority-reasoned correction.
        if any(c.get("reason") == "authority" for c in s._engine._corrections):
            fired += 1
        valid = s._engine._layers.get_valid_fact_ids()
        assert valid == {"AC-HIGH"}, f"{tl.id}: expected only AC-HIGH valid, got {valid}"
    assert fired == len(tls)  # the fix is exercised on every timeline


def test_dependency_chain_track_fires_type2_propagation() -> None:
    gen = TimelineGenerator(seed=3)
    tls = list(gen.generate_track("dependency_chain", count=6))
    assert all(t.track == "dependency_chain" for t in tls)
    for tl in tls:
        s, _ = _run_to_query(tl)
        nr = s._engine._needs_review
        # base superseded -> direct (MID) AND transitive (TOP) dependents flagged.
        assert any(f.startswith("DC-MID") for f in nr), f"{tl.id}: MID not flagged"
        assert any(f.startswith("DC-TOP") for f in nr), f"{tl.id}: TOP not flagged (transitive)"
        # base itself is superseded
        assert "DC-BASE" in s._engine._layers.superseded_by
