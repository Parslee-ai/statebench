"""Regression tests for the supersession-detection generator.

Three defects shipped in the v1.0 release, all of them invisible in a results
file because each one produces numbers that look like model failure:

1. **Two clocks.** Conversation turns advanced one counter while state writes
   derived their timestamps from a second, and all the writes were appended
   after the whole conversation. 105 of the 1,400 released timelines carry
   non-monotonic timestamps as a result.
2. **Inverted causal order.** The same appending put the write recording the
   *original* value after the turn that corrected it. On a track whose subject
   is detecting a correction, the superseded value was the most recent thing
   anyone had written down.
3. **A value that changed to itself.** ``rate_a`` and ``rate_b`` were drawn
   independently from overlapping pools, so one case in nine produced a change
   from ``$150`` to ``$150`` — with ``$150`` then both required and forbidden.
   No response could pass. Two such queries shipped.

The fidelity auditor (``statebench audit-dataset``) is what surfaced 1 and 3;
these tests pin all three so they cannot come back.
"""

from __future__ import annotations

import pytest

from statebench.generator.engine import TimelineGenerator
from statebench.generator.fidelity import check_dataset
from statebench.generator.templates.detection import (
    DETECTION_TEMPLATES,
    DetectionGroundTruth,
    DetectionTemplate,
    TurnTemplate,
    WriteTemplate,
)
from statebench.schema.timeline import (
    ConversationTurn,
    Query,
    StateWrite,
    Supersession,
)

BY_ID = {t.id: t for t in DETECTION_TEMPLATES}


def _timelines(count: int = 60, seed: int = 2026):
    return list(TimelineGenerator(seed=seed).generate_track(
        "supersession_detection", count=count
    ))


# --------------------------------------------------------------------------- #
# 1. One clock
# --------------------------------------------------------------------------- #

def test_timestamps_are_strictly_increasing():
    for tl in _timelines():
        stamps = [e.ts for e in tl.events]
        assert all(a < b for a, b in zip(stamps, stamps[1:])), (
            f"{tl.id}: {stamps}"
        )


def test_every_template_produces_ordered_events():
    """Not just the ones a random sample happens to hit."""
    gen = TimelineGenerator(seed=17)
    for template in DETECTION_TEMPLATES:
        tl = gen.generate_detection_timeline(template)
        stamps = [e.ts for e in tl.events]
        assert stamps == sorted(stamps), f"{template.id} out of order"


# --------------------------------------------------------------------------- #
# 2. The recorded value precedes the cue that supersedes it
# --------------------------------------------------------------------------- #

def test_state_writes_precede_the_supersession_cue():
    """The defect that made the dead value look like the newest state."""
    for tl in _timelines():
        cue_positions = [
            i
            for i, e in enumerate(tl.events)
            if isinstance(e, ConversationTurn) and e.implicit_supersession
        ]
        if not cue_positions:
            continue
        last_cue = max(cue_positions)
        write_positions = [
            i for i, e in enumerate(tl.events) if isinstance(e, (StateWrite, Supersession))
        ]
        assert all(p < last_cue for p in write_positions), (
            f"{tl.id}: a write is recorded after the correction cue"
        )


def test_the_write_follows_the_turn_that_states_its_value():
    gen = TimelineGenerator(seed=5)
    tl = gen.generate_detection_timeline(BY_ID["DET-EXP-LOC-001"])

    write_idx = next(
        i for i, e in enumerate(tl.events) if isinstance(e, StateWrite)
    )
    stated_value = tl.events[write_idx].writes[0].value
    prior = tl.events[write_idx - 1]
    assert isinstance(prior, ConversationTurn)
    assert stated_value in prior.text


@pytest.mark.parametrize(
    "template_id", ["DET-AUTH-BUD-001", "DET-DFT-PRC-001", "DET-DFT-PLN-002"]
)
def test_annotated_values_still_anchor_to_the_right_turn(template_id):
    """Values like '$85 (DRAFT)' never appear verbatim in the dialogue.

    Matching on the whole string finds nothing, so the write falls back to the
    start of the conversation. Matching on the placeholder finds the turn.
    """
    gen = TimelineGenerator(seed=5)
    tl = gen.generate_detection_timeline(BY_ID[template_id])

    write_idx = next(
        i for i, e in enumerate(tl.events) if isinstance(e, (StateWrite, Supersession))
    )
    cue_idx = next(
        i
        for i, e in enumerate(tl.events)
        if isinstance(e, ConversationTurn) and e.implicit_supersession
    )
    assert write_idx < cue_idx


def test_multi_write_template_interleaves_in_causal_order():
    """DET-REV-DSN-001 establishes A, switches to B, then reverts to A."""
    gen = TimelineGenerator(seed=5)
    tl = gen.generate_detection_timeline(BY_ID["DET-REV-DSN-001"])

    kinds = [
        type(e).__name__ for e in tl.events
    ]
    assert kinds.index("StateWrite") < kinds.index("Supersession")

    supersession = next(e for e in tl.events if isinstance(e, Supersession))
    original = next(e for e in tl.events if isinstance(e, StateWrite))
    assert supersession.writes[0].supersedes == original.writes[0].id


# --------------------------------------------------------------------------- #
# 3. A value never changes to itself
# --------------------------------------------------------------------------- #

def test_paired_variables_never_collide():
    """The pools overlap on purpose; a single case must still show a change."""
    gen = TimelineGenerator(seed=99)
    for template in DETECTION_TEMPLATES:
        for _ in range(60):
            variables = gen._draw_detection_variables(template)
            for name in variables:
                if not name.endswith("_a"):
                    continue
                partner = name[:-2] + "_b"
                if partner in variables:
                    assert variables[name] != variables[partner], (
                        f"{template.id}: {name} and {partner} both "
                        f"{variables[name]!r}"
                    )


def test_the_overlapping_pool_that_shipped_the_defect_is_still_overlapping():
    """Guard the guard: if the pools stop overlapping the test above goes vacuous."""
    rate = BY_ID["DET-TMP-RAT-002"].variables
    assert set(rate["rate_a"]) & set(rate["rate_b"]), (
        "rate_a/rate_b no longer overlap — this test no longer exercises the fix"
    )


def test_required_and_forbidden_phrases_never_coincide():
    for tl in _timelines(count=300):
        gt = next(e for e in tl.events if isinstance(e, Query)).ground_truth
        required = {str(m).lower() for m in gt.must_mention}
        forbidden = {str(m).lower() for m in gt.must_not_mention}
        assert not (required & forbidden), f"{tl.id}: {required & forbidden}"


def test_a_pool_with_no_alternative_fails_loudly():
    impossible = DetectionTemplate(
        id="DET-TEST-IMPOSSIBLE",
        name="impossible",
        description="",
        domain="sales",
        conversation_pattern=[TurnTemplate(role="user", content="The rate is {x_a}.")],
        state_writes=[WriteTemplate(id="F-X", key="x", value="{x_a}")],
        ground_truth=DetectionGroundTruth(
            decision="{x_b}", must_detect=["F-X"],
            must_mention=["{x_b}"], must_not_mention=["{x_a}"],
        ),
        query_template="What is the rate?",
        variables={"x_a": ["$150"], "x_b": ["$150"]},
    )
    gen = TimelineGenerator(seed=1)
    with pytest.raises(ValueError, match="no change to detect"):
        gen.generate_detection_timeline(impossible)


# --------------------------------------------------------------------------- #
# Ground truth the track is named after
# --------------------------------------------------------------------------- #

def test_implicit_supersession_markers_survive_into_the_timeline():
    """They were declared in the templates and dropped by the generator."""
    for tl in _timelines():
        recovered = tl.get_implicit_supersessions()
        assert recovered, f"{tl.id}: no implicit supersession recorded"
        for turn, marker in recovered:
            assert marker.detection_cue
            assert "{" not in marker.detection_cue, "placeholder left unsubstituted"


def test_must_detect_lands_in_the_structured_field():
    """It used to exist only inside a prose reasoning string."""
    for tl in _timelines(count=20):
        gt = next(e for e in tl.events if isinstance(e, Query)).ground_truth
        assert gt.supersession_detection is not None
        assert gt.supersession_detection.must_detect
        for fact_id in gt.supersession_detection.must_detect:
            assert "{" not in fact_id


def test_detected_fact_ids_refer_to_facts_that_exist():
    for tl in _timelines(count=40):
        gt = next(e for e in tl.events if isinstance(e, Query)).ground_truth
        written = {
            w.id
            for e in tl.events
            if isinstance(e, (StateWrite, Supersession))
            for w in e.writes
        }
        for fact_id in gt.supersession_detection.must_detect:
            assert fact_id in written, f"{tl.id}: must_detect names unknown {fact_id}"


# --------------------------------------------------------------------------- #
# Whole-track guarantees
# --------------------------------------------------------------------------- #

def test_track_audits_clean():
    report = check_dataset(_timelines(count=300))
    assert report.issues == [], "\n".join(str(i) for i in report.issues[:5])


def test_generation_stays_reproducible():
    a = _timelines(count=10, seed=4)
    b = _timelines(count=10, seed=4)
    assert [[e.ts for e in t.events] for t in a] == [[e.ts for e in t.events] for t in b]
    assert [t.events[0].text for t in a] == [t.events[0].text for t in b]
