"""Tests for paired-counterfactual generation (Paper 3, section 4.2).

The experiment measures the behavioral delta between two timelines that differ
in one governance variable. That measurement is only valid if the pair really
is minimal, so these tests assert the pairing properties directly rather than
trusting the builders.
"""

from __future__ import annotations

import pytest

from statebench.generator.counterfactual import (
    AXES,
    ENGINE_DECIDABLE,
    MODEL_DECIDABLE,
    SCENARIOS,
    CounterfactualGenerator,
)
from statebench.schema.timeline import Query


@pytest.fixture(scope="module")
def pairs():
    return CounterfactualGenerator(seed=7).generate_pairs()


def _query(timeline) -> Query:
    queries = [e for e in timeline.events if isinstance(e, Query)]
    assert len(queries) == 1
    return queries[0]


def test_all_axes_produce_pairs(pairs):
    assert {p.axis for p in pairs} == set(AXES)
    assert len(pairs) == len(AXES) * len(SCENARIOS)


def test_axes_are_partitioned_by_decidability():
    assert set(ENGINE_DECIDABLE) | set(MODEL_DECIDABLE) == set(AXES)
    assert not set(ENGINE_DECIDABLE) & set(MODEL_DECIDABLE)
    assert ENGINE_DECIDABLE and MODEL_DECIDABLE


def test_every_pair_passes_its_audit(pairs):
    generator = CounterfactualGenerator(seed=7)
    failures = {
        f"{a.axis}/{a.scenario}": a.problems
        for a in generator.audit_all(pairs)
        if not a.ok
    }
    assert not failures, failures


def test_pairs_are_matched_in_shape(pairs):
    """A pair distinguishable by timeline shape would let a system score the
    delta without reading the governed content."""
    for p in pairs:
        pos, cf = p.positive, p.counterfactual
        assert len(pos.events) == len(cf.events), p.axis
        assert [e.type for e in pos.events] == [e.type for e in cf.events], p.axis


def test_query_is_identical_across_a_pair(pairs):
    for p in pairs:
        assert _query(p.positive).prompt == _query(p.counterfactual).prompt


def test_actor_and_domain_are_held_constant(pairs):
    for p in pairs:
        assert p.positive.domain == p.counterfactual.domain
        assert p.positive.actors.user.role == p.counterfactual.actors.user.role


def test_ground_truth_diverges_across_a_pair(pairs):
    """No divergence means nothing to measure."""
    for p in pairs:
        assert (
            _query(p.positive).ground_truth.decision
            != _query(p.counterfactual).ground_truth.decision
        ), p.axis


def test_counterfactual_forbids_the_governed_value(pairs):
    for p in pairs:
        forbidden = [
            m.phrase if hasattr(m, "phrase") else m
            for m in _query(p.counterfactual).ground_truth.must_not_mention
        ]
        assert forbidden, p.axis
        # The positive side must NOT forbid it — that is the whole contrast.
        assert not _query(p.positive).ground_truth.must_not_mention, p.axis


def test_pairs_are_lexically_near_identical(pairs):
    """Manipulation check: a retriever should struggle to tell the sides apart."""
    generator = CounterfactualGenerator(seed=7)
    for audit in generator.audit_all(pairs):
        assert audit.lexical_similarity > 0.6, (
            f"{audit.axis} pair is distinguishable on wording alone "
            f"(similarity {audit.lexical_similarity:.2f})"
        )


def test_engine_decidable_axes_mark_content_for_filtering(pairs):
    """The GRPO-objection defense: on these axes the discriminating fact must be
    removable before inference, so no reward signal can teach the distinction."""
    for p in pairs:
        if p.decidable_by != "engine":
            continue
        writes = [
            w
            for e in p.counterfactual.events
            for w in (getattr(e, "writes", None) or [])
        ]
        marked = [
            w
            for w in writes
            if "[RESTRICTED:" in w.value or getattr(w, "scope", "global") in ("draft", "hypothetical")
        ]
        assert marked, f"{p.axis} counterfactual has nothing the engine can filter"


def test_model_decidable_axes_are_flagged_as_reward_learnable(pairs):
    generator = CounterfactualGenerator(seed=7)
    for audit in generator.audit_all(pairs):
        assert audit.reward_learnable == (audit.decidable_by == "model")


def test_generation_is_deterministic():
    a = CounterfactualGenerator(seed=3).generate_pairs()
    b = CounterfactualGenerator(seed=3).generate_pairs()
    assert [p.positive.id for p in a] == [p.positive.id for p in b]
    assert [_query(p.counterfactual).prompt for p in a] == [
        _query(p.counterfactual).prompt for p in b
    ]


def test_axis_subset_selection():
    generator = CounterfactualGenerator(seed=1)
    subset = generator.generate_pairs(axis_names=["access_control"], per_axis=2)
    assert len(subset) == 2
    assert {p.axis for p in subset} == {"access_control"}


def test_every_axis_documents_its_admissibility():
    for axis in AXES.values():
        assert axis.admissibility_note.strip()
        assert axis.decidable_by in ("engine", "model")
