"""Tests for the dataset fidelity checker.

The checker exists because ground truth can be wrong in ways that look exactly
like model failure. Its own false positives would be the same kind of mistake
one layer up, so each check is tested both ways: it must fire on a timeline
that really is broken, and stay quiet on one that only looks unusual.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from statebench.generator.engine import TimelineGenerator
from statebench.generator.fidelity import (
    check_dataset,
    check_timeline,
    format_fidelity_report,
    timeline_corpus,
)
from statebench.schema.state import IdentityRole
from statebench.schema.timeline import (
    Actor,
    Actors,
    ConversationTurn,
    GroundTruth,
    InitialState,
    MentionRequirement,
    Query,
    StateWrite,
    Supersession,
    Timeline,
    Write,
)

T0 = datetime(2026, 3, 1, 9, 0, 0)


def _build(
    *,
    events,
    track: str = "supersession",
    timeline_id: str = "T-TEST",
) -> Timeline:
    return Timeline(
        id=timeline_id,
        domain="sales",
        track=track,  # type: ignore[arg-type]
        actors=Actors(
            user=Actor(id="u1", role="Manager", org="acme"),
            assistant_role="AI_Agent",
        ),
        initial_state=InitialState(
            identity_role=IdentityRole(user_name="Sam", authority="Manager"),
            persistent_facts=[],
            working_set=[],
            environment={"now": T0.isoformat()},
        ),
        events=events,
    )


def _healthy(**gt_overrides) -> Timeline:
    """A timeline nothing should complain about."""
    gt = {
        "decision": "no",
        "must_mention": ["456 Oak Ave"],
        "must_not_mention": [
            MentionRequirement(phrase="123 Main St", kind="superseded")
        ],
    }
    gt.update(gt_overrides)
    return _build(
        events=[
            ConversationTurn(ts=T0, speaker="user", text="Ship to 123 Main St."),
            StateWrite(
                ts=T0 + timedelta(minutes=1),
                writes=[
                    Write(id="F-1", layer="persistent_facts", key="addr", value="123 Main St")
                ],
            ),
            ConversationTurn(
                ts=T0 + timedelta(minutes=2),
                speaker="user",
                text="I moved to 456 Oak Ave.",
            ),
            Supersession(
                ts=T0 + timedelta(minutes=3),
                writes=[
                    Write(
                        id="F-2",
                        layer="persistent_facts",
                        key="addr_v2",
                        value="456 Oak Ave",
                        supersedes="addr",
                    )
                ],
            ),
            Query(ts=T0 + timedelta(minutes=5), prompt="Where do I ship?",
                  ground_truth=GroundTruth(**gt)),
        ]
    )


# --------------------------------------------------------------------------- #
# The quiet case
# --------------------------------------------------------------------------- #

def test_a_well_formed_timeline_produces_nothing():
    assert check_timeline(_healthy()) == []


def test_corpus_includes_every_place_a_fact_can_live():
    corpus = timeline_corpus(_healthy())
    assert "123 Main St" in corpus       # conversation and write
    assert "456 Oak Ave" in corpus       # supersession write
    assert "Sam" in corpus               # identity
    assert T0.isoformat() in corpus      # environment


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

def test_phrase_both_required_and_forbidden():
    """A real v1.0 defect: the value 'changes' to itself, so no answer passes."""
    timeline = _healthy(
        must_mention=["123 Main St"],
        must_not_mention=[MentionRequirement(phrase="123 Main St", kind="superseded")],
    )
    codes = [i.code for i in check_timeline(timeline)]
    assert "phrase_required_and_forbidden" in codes
    assert all(
        i.severity == "error"
        for i in check_timeline(timeline)
        if i.code == "phrase_required_and_forbidden"
    )


def test_out_of_order_events():
    timeline = _healthy()
    timeline.events[1].ts = T0 - timedelta(hours=1)
    codes = [i.code for i in check_timeline(timeline)]
    assert "events_out_of_order" in codes


def test_dangling_supersession():
    timeline = _healthy()
    supersession = next(e for e in timeline.events if isinstance(e, Supersession))
    supersession.writes[0].supersedes = "a_key_that_was_never_written"
    issues = [i for i in check_timeline(timeline) if i.code == "dangling_supersession"]
    assert len(issues) == 1
    assert issues[0].severity == "error"


@pytest.mark.parametrize("reference", ["addr", "F-1"])
def test_supersedes_may_name_either_a_key_or_an_id(reference):
    """Shipped data uses both; flagging one of them would be a false positive."""
    timeline = _healthy()
    next(e for e in timeline.events if isinstance(e, Supersession)).writes[0].supersedes = reference
    assert not [i for i in check_timeline(timeline) if i.code == "dangling_supersession"]


def test_timeline_with_no_query():
    timeline = _healthy()
    timeline.events = [e for e in timeline.events if not isinstance(e, Query)]
    assert "no_query" in [i.code for i in check_timeline(timeline)]


def test_query_before_the_state_it_depends_on():
    timeline = _healthy()
    query = next(e for e in timeline.events if isinstance(e, Query))
    query.ts = T0 + timedelta(seconds=30)
    codes = [i.code for i in check_timeline(timeline)]
    assert "query_precedes_state" in codes


# --------------------------------------------------------------------------- #
# Warnings
# --------------------------------------------------------------------------- #

def test_must_mention_phrase_that_is_a_near_miss():
    """'500 users' required, 'They have 500 active users' planted."""
    timeline = _healthy(must_mention=["456 Oak Avenue, Suite 3"])
    issues = [i for i in check_timeline(timeline) if i.code == "unsupported_must_mention"]
    assert len(issues) == 1
    assert issues[0].severity == "warning"


def test_forbidden_phrase_that_can_never_fire():
    """'MFA reset' forbidden, 'Reset MFA for CEO' planted — no match, ever."""
    timeline = _healthy(
        must_not_mention=[MentionRequirement(phrase="999 Elm Blvd", kind="superseded")]
    )
    issues = [
        i for i in check_timeline(timeline) if i.code == "unreachable_forbidden_phrase"
    ]
    assert len(issues) == 1
    assert issues[0].severity == "warning"


def test_inadmissible_forbidden_phrase_is_reported_once_and_not_also_unreachable():
    timeline = _healthy(must_not_mention=["approved"])
    codes = [i.code for i in check_timeline(timeline)]
    assert codes.count("inadmissible_forbidden_phrase") == 1
    assert "unreachable_forbidden_phrase" not in codes


def test_superseded_tag_with_nothing_invalidating_it():
    timeline = _build(
        events=[
            ConversationTurn(ts=T0, speaker="user", text="Discount is 22%."),
            StateWrite(
                ts=T0 + timedelta(minutes=1),
                writes=[
                    Write(id="F-1", layer="persistent_facts", key="disc", value="22%")
                ],
            ),
            Query(
                ts=T0 + timedelta(minutes=2),
                prompt="What discount applies?",
                ground_truth=GroundTruth(
                    decision="no",
                    must_mention=[],
                    must_not_mention=[
                        MentionRequirement(phrase="22%", kind="superseded")
                    ],
                ),
            ),
        ]
    )
    issues = [
        i
        for i in check_timeline(timeline)
        if i.code == "superseded_phrase_without_invalidation"
    ]
    assert len(issues) == 1
    assert issues[0].severity == "warning"


def test_expiry_counts_as_invalidation():
    """cf_temporal_validity retires facts by clock, not by a supersession event."""
    timeline = _build(
        track="cf_temporal_validity",
        events=[
            ConversationTurn(ts=T0, speaker="user", text="Set the discount."),
            StateWrite(
                ts=T0 + timedelta(minutes=1),
                writes=[
                    Write(
                        id="F-1",
                        layer="persistent_facts",
                        key="disc",
                        value="Discount authority is 22% (valid through 2026-01-31)",
                    )
                ],
            ),
            Query(
                ts=T0 + timedelta(minutes=2),
                prompt="What discount applies?",
                ground_truth=GroundTruth(
                    decision="no",
                    must_mention=[],
                    must_not_mention=[
                        MentionRequirement(phrase="22%", kind="superseded")
                    ],
                ),
            ),
        ],
    )
    codes = [i.code for i in check_timeline(timeline)]
    assert "superseded_phrase_without_invalidation" not in codes


# --------------------------------------------------------------------------- #
# Dataset level
# --------------------------------------------------------------------------- #

def test_report_counts_and_grouping():
    broken = _healthy(
        must_mention=["123 Main St"],
        must_not_mention=[MentionRequirement(phrase="123 Main St", kind="superseded")],
    )
    report = check_dataset([_healthy(), broken])

    assert report.timelines_checked == 2
    assert report.queries_checked == 2
    assert report.affected_timelines() == 1
    assert report.clean is False
    assert report.by_code()["phrase_required_and_forbidden"] == 1


def test_clean_dataset_reports_clean():
    report = check_dataset([_healthy(), _healthy()])
    assert report.clean is True
    assert "No issues found" in format_fidelity_report(report)


def test_report_renders_codes_and_examples():
    report = check_dataset([_healthy(must_mention=["nowhere to be found"])])
    text = format_fidelity_report(report)
    assert "unsupported_must_mention" in text
    assert "Examples" in text


# --------------------------------------------------------------------------- #
# Regression guard on the tracks we control
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "track", ["supersession", "premise_resistance", "premise_maintain"]
)
def test_generated_tracks_have_no_fidelity_errors(track):
    """New tracks must not ship the defects the v1.0 release turned out to have."""
    timelines = list(TimelineGenerator(seed=4242).generate_track(track, count=25))
    report = check_dataset(timelines)
    assert report.errors == [], "\n".join(str(i) for i in report.errors[:5])


def test_no_generated_track_produces_a_fidelity_error():
    """The gate has to be passable, or `audit-dataset --strict` is decoration.

    This caught two real generator defects: non-monotonic timestamps on
    `supersession_detection`, and red-herring insertion on `adversarial`
    stamping chatter later than the supersession that still followed it.
    """
    from statebench.cli import AVAILABLE_TRACKS

    failures: dict[str, list[str]] = {}
    for track in AVAILABLE_TRACKS:
        timelines = list(TimelineGenerator(seed=2026).generate_track(track, count=10))
        errors = check_dataset(timelines).errors
        if errors:
            failures[track] = [str(e) for e in errors[:3]]

    assert not failures, "\n".join(
        f"{track}: {'; '.join(msgs)}" for track, msgs in failures.items()
    )


def test_red_herring_insertion_keeps_events_ordered():
    """Adding a distraction must not reorder the state changes around it."""
    import random

    from statebench.generator.adversarial import TimelinePerturbator

    base = list(TimelineGenerator(seed=3).generate_track("supersession", count=1))[0]
    perturbator = TimelinePerturbator(rng=random.Random(11))

    for _ in range(20):
        variant = perturbator.add_red_herrings(base)
        stamps = [e.ts for e in variant.events]
        assert stamps == sorted(stamps), [str(t) for t in stamps]


def _with_adjacent_filler(timeline):
    """Splice two adjacent filler turns onto the front of a timeline.

    Generated tracks rarely produce adjacent filler pairs, so the shuffle has
    nothing to act on; constructing the case is the only way to exercise it.
    """
    from copy import deepcopy

    variant = deepcopy(timeline)
    t0 = variant.events[0].ts
    variant.events = [
        ConversationTurn(
            ts=t0 - timedelta(minutes=4), speaker="user", text="Thanks for the update."
        ),
        ConversationTurn(
            ts=t0 - timedelta(minutes=2),
            speaker="assistant",
            text="No problem, happy to help.",
        ),
    ] + list(variant.events)
    return variant


def test_temporal_shuffle_reorders_content_but_not_time():
    """The timestamps belong to the slots, not to the turns.

    Letting them travel with the turns made the timeline contradict itself:
    list order said A then B while timestamps said B then A. A system sorting
    by timestamp then saw the *unshuffled* order and was not perturbed at all,
    so the perturbation's strength depended on how the system under test reads
    events — which is exactly what an adversarial control must not do.
    """
    import random

    from statebench.generator.adversarial import TimelinePerturbator

    base = _with_adjacent_filler(
        list(TimelineGenerator(seed=3).generate_track("supersession", count=1))[0]
    )
    shuffled = TimelinePerturbator(rng=random.Random(7)).temporal_shuffle(base)

    # The two filler turns changed places...
    assert [e.text for e in shuffled.events[:2]] == [
        base.events[1].text,
        base.events[0].text,
    ]
    # ...while every timestamp stayed exactly where it was.
    assert [e.ts for e in shuffled.events] == [e.ts for e in base.events]
    stamps = [e.ts for e in shuffled.events]
    assert stamps == sorted(stamps)


def test_temporal_shuffle_never_moves_a_supersession_cue():
    """A cue turn is not filler, whatever politeness markers it contains.

    `_is_filler` matches on "okay"/"got it"/"sure" regardless of substance, so
    a correction phrased "Okay, make that Thursday instead" would otherwise be
    shufflable — and moving the correction moves the answer.
    """
    import random

    from statebench.generator.adversarial import TimelinePerturbator
    from statebench.schema.timeline import ImplicitSupersession

    base = _with_adjacent_filler(
        list(TimelineGenerator(seed=3).generate_track("supersession", count=1))[0]
    )
    base.events[1].implicit_supersession = ImplicitSupersession(
        detection_cue="No problem, happy to help", difficulty="obvious"
    )

    perturbator = TimelinePerturbator(rng=random.Random(7))
    for _ in range(20):
        shuffled = perturbator.temporal_shuffle(base)
        assert [e.text for e in shuffled.events[:2]] == [
            e.text for e in base.events[:2]
        ], "a turn carrying a supersession cue was shuffled"


def test_stacked_perturbations_stay_ordered():
    """Ordering has to survive composition, not just one perturbation at a time.

    The bug that motivated the shared insertion helper only appeared when two
    perturbations composed: add_red_herrings shifted a supersession forward,
    then emphasis_invert inserted ahead of it at `supersession.ts - 1 minute`,
    a time that was now in the past.
    """
    import random

    from statebench.generator.adversarial import TimelinePerturbator

    names = [
        "paraphrase",
        "temporal_shuffle",
        "name_substitute",
        "emphasis_invert",
        "add_red_herrings",
    ]
    bases = list(TimelineGenerator(seed=3).generate_track("supersession", count=3))
    bases += list(
        TimelineGenerator(seed=4).generate_track("supersession_detection", count=3)
    )

    picker = random.Random(0)
    for trial in range(120):
        perturbator = TimelinePerturbator(rng=random.Random(trial))
        timeline = picker.choice(bases)
        applied = []
        for _ in range(picker.randint(2, 5)):
            name = picker.choice(names)
            applied.append(name)
            timeline = getattr(perturbator, name)(timeline)

        stamps = [e.ts for e in timeline.events]
        assert stamps == sorted(stamps), f"trial {trial} after {applied}"


def test_premise_tracks_have_no_warnings_either():
    """The premise templates are literal, so every phrase must be in the text."""
    timelines = []
    for track in ("premise_resistance", "premise_maintain"):
        timelines += list(TimelineGenerator(seed=4242).generate_track(track, count=25))
    report = check_dataset(timelines)
    assert report.issues == [], "\n".join(str(i) for i in report.issues[:5])
