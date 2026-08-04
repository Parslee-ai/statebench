"""Tests for the applicability suite and its metrics (Paper 3, §4.1 and §5.2).

Governance dominates the matrix: once state is governed, the required behavior
no longer varies with how well the experience transfers. That leaves the suite
REJECT-leaning, which makes degenerate strategies dangerous, so most of these
tests ensure a system cannot score well by always refusing or always answering.
"""

from __future__ import annotations

import pytest

from statebench.evaluation.applicability_metrics import (
    ApplicabilityMetrics,
    expected_from_timelines,
    looks_like_refusal,
)
from statebench.evaluation.metrics import QueryResult
from statebench.generator.applicability import (
    EXPERIENCE_STATUSES,
    GOVERNANCE_STATUSES,
    MATRIX,
    ApplicabilityGenerator,
)


@pytest.fixture(scope="module")
def suite():
    return ApplicabilityGenerator(seed=3).generate(per_cell=1)


def _result(tid: str, response: str, correct: bool) -> QueryResult:
    r = QueryResult(
        timeline_id=tid,
        query_idx=0,
        track="applicability",
        domain="sales",
        expected_decision="x",
        must_mention=[],
        must_not_mention=[],
        response=response,
    )
    r.decision_correct = correct
    return r


# --------------------------------------------------------------------------- #
# Matrix structure
# --------------------------------------------------------------------------- #

def test_matrix_is_the_full_factorial():
    assert len(MATRIX) == len(GOVERNANCE_STATUSES) * len(EXPERIENCE_STATUSES)
    assert {(c.governance, c.experience) for c in MATRIX} == {
        (g, e) for g in GOVERNANCE_STATUSES for e in EXPERIENCE_STATUSES
    }


def test_governance_dominates_experience_applicability():
    """The paper's thesis as ground truth: once state is governed, how well the
    experience transfers no longer changes the required behavior."""
    from statebench.generator.applicability import SUBSTITUTING, WITHDRAWING

    for governance in WITHDRAWING + SUBSTITUTING:
        row = {c.expected_applicability for c in MATRIX if c.governance == governance}
        assert len(row) == 1, f"{governance} row varies with experience: {row}"


def test_withdrawing_and_substituting_governance_differ():
    """Withdrawing removes the value (refuse); substituting replaces it (adapt
    to the new one). Collapsing them marked correct supersession answers wrong."""
    from statebench.generator.applicability import SUBSTITUTING, WITHDRAWING

    for c in MATRIX:
        if c.governance in WITHDRAWING:
            assert c.expected_applicability == "REJECT", c
        elif c.governance in SUBSTITUTING:
            assert c.expected_applicability == "ADAPT", c


def test_only_ungoverned_state_can_be_kept_verbatim():
    """KEEP means the past value is still correct, which requires live state."""
    keep = [c for c in MATRIX if c.expected_applicability == "KEEP"]
    assert keep
    assert all(c.governance == "valid" for c in keep)


def test_diagonal_assigns_responsibility_to_the_right_plane():
    by = {(c.governance, c.experience): c for c in MATRIX}
    assert by[("valid", "applicable")].responsible_plane == "retrieval"
    assert by[("valid", "adaptable")].responsible_plane == "reconstruction"
    assert by[("unauthorized", "applicable")].responsible_plane == "state resolution"


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

def test_suite_covers_every_cell(suite):
    assert len(suite) >= len(MATRIX)
    assert all(t.track == "applicability" for t in suite)


def test_valid_row_is_oversampled_for_usable_n(suite):
    """KEEP lives only in the valid row; without oversampling there would be a
    single sample and the use-cell rates would be unmeasurable."""
    expected, _, _ = expected_from_timelines(suite)
    counts = {v: sum(1 for x in expected.values() if x == v) for v in ("KEEP", "ADAPT", "REJECT")}
    assert counts["KEEP"] >= 4 and counts["ADAPT"] >= 4


def test_ground_truth_carries_applicability_labels(suite):
    expected, principles, obsolete = expected_from_timelines(suite)
    assert len(expected) == len(suite)
    assert set(expected.values()) <= {"KEEP", "ADAPT", "REJECT"}
    # ADAPT cells must name the stale detail that has to die in adaptation.
    adapt_keys = [k for k, v in expected.items() if v == "ADAPT"]
    assert adapt_keys and all(obsolete[k] for k in adapt_keys)
    assert all(principles[k] for k in adapt_keys)


def test_governed_cells_forbid_the_governed_value(suite):
    from statebench.schema.timeline import Query

    governed = [t for t in suite if "APPL" in t.id]
    forbidding = [
        t
        for t in governed
        for e in t.events
        if isinstance(e, Query) and e.ground_truth.must_not_mention
    ]
    assert forbidding, "no governed cell forbids its value"


def test_generation_is_deterministic():
    a = ApplicabilityGenerator(seed=11).generate(per_cell=1)
    b = ApplicabilityGenerator(seed=11).generate(per_cell=1)
    assert [t.id for t in a] == [t.id for t in b]


# --------------------------------------------------------------------------- #
# Refusal detection
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "response,expected",
    [
        ("I do not have that information.", True),
        ("I cannot share that figure.", True),
        ("There is no record of an agreed rate.", True),
        ("The discount is 22%.", False),
        ("Use the current value of $60,000.", False),
    ],
)
def test_refusal_detection(response, expected):
    assert looks_like_refusal(response) is expected


# --------------------------------------------------------------------------- #
# Degenerate strategies must be visible
# --------------------------------------------------------------------------- #

def test_blanket_refusal_does_not_beat_its_own_baseline(suite):
    expected, principles, obsolete = expected_from_timelines(suite)
    results = [_result(t.id, "I do not have that information.", False) for t in suite]
    m = ApplicabilityMetrics.from_results(results, expected, principles, obsolete)

    # It scores high on raw classification precisely because the suite is
    # REJECT-heavy — which is why the baseline is reported alongside.
    assert m.always_refuse_baseline == pytest.approx(
        m.applicability_classification_accuracy
    )
    assert not m.beats_blanket_refusal
    assert m.false_refusal_rate == 1.0
    assert "DOES NOT BEAT" in m.summary()


def test_answering_everything_fails_every_reject_cell(suite):
    expected, principles, obsolete = expected_from_timelines(suite)
    results = [_result(t.id, "The value is 22%.", True) for t in suite]
    m = ApplicabilityMetrics.from_results(results, expected, principles, obsolete)
    assert m.correct_rejection_rate == 0.0
    assert m.false_refusal_rate == 0.0  # it never refuses, so never falsely
    # Its whole score comes from use-cells; the REJECT cells are all missed.
    assert m.correct_classification == m.use_cells


def test_correct_and_false_refusal_are_computed_over_disjoint_sets(suite):
    expected, _, _ = expected_from_timelines(suite)
    results = [_result(t.id, "I do not have that information.", False) for t in suite]
    m = ApplicabilityMetrics.from_results(results, expected)
    assert m.reject_cells + m.use_cells == m.classified
    assert m.reject_cells and m.use_cells


def test_a_discriminating_system_beats_the_baseline(suite):
    """Refuse exactly on REJECT cells, answer exactly on the rest."""
    expected, principles, obsolete = expected_from_timelines(suite)
    results = []
    for t in suite:
        want = expected[(t.id, 0)]
        if want == "REJECT":
            results.append(_result(t.id, "I do not have that information.", False))
        else:
            results.append(_result(t.id, "The value is 22%.", True))
    m = ApplicabilityMetrics.from_results(results, expected, principles, obsolete)
    assert m.applicability_classification_accuracy == 1.0
    assert m.correct_rejection_rate == 1.0
    assert m.false_refusal_rate == 0.0
    assert m.beats_blanket_refusal


def test_historical_detail_carryover_is_measured_on_adapt_cells(suite):
    expected, principles, obsolete = expected_from_timelines(suite)
    adapt = [k for k, v in expected.items() if v == "ADAPT"]
    stale = obsolete[adapt[0]][0]
    results = [
        _result(tid, f"Use {stale} as before.", True) if (tid, 0) in adapt else
        _result(tid, "I do not have that information.", False)
        for tid in {k[0] for k in expected}
    ]
    m = ApplicabilityMetrics.from_results(results, expected, principles, obsolete)
    assert m.historical_detail_carryover_rate > 0


def test_no_labelled_results_yields_zeroed_metrics():
    m = ApplicabilityMetrics.from_results([_result("X", "hi", True)], {})
    assert m.classified == 0
    assert m.applicability_classification_accuracy == 0.0


# --------------------------------------------------------------------------- #
# CLI / generator wiring
# --------------------------------------------------------------------------- #

def test_tracks_are_registered_with_the_cli():
    from typing import get_args

    from statebench.cli import AVAILABLE_TRACKS
    from statebench.schema.timeline import Track

    assert set(get_args(Track)) <= set(AVAILABLE_TRACKS)
    assert "applicability" in AVAILABLE_TRACKS


def test_counterfactual_tracks_emit_balanced_pairs():
    from statebench.generator.engine import TimelineGenerator

    timelines = list(TimelineGenerator(seed=5).generate_track("cf_access_control", 3))
    assert len(timelines) == 6  # count is PAIRS, so twice as many timelines
    assert sum(1 for t in timelines if "-CF-" in t.id) == 3
    assert sum(1 for t in timelines if "-POS-" in t.id) == 3


def test_applicability_track_generates_via_the_engine():
    from statebench.generator.engine import TimelineGenerator

    timelines = list(TimelineGenerator(seed=5).generate_track("applicability", 18))
    assert timelines and all(t.track == "applicability" for t in timelines)
