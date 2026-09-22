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


def test_premise_tracks_have_no_warnings_either():
    """The premise templates are literal, so every phrase must be in the text."""
    timelines = []
    for track in ("premise_resistance", "premise_maintain"):
        timelines += list(TimelineGenerator(seed=4242).generate_track(track, count=25))
    report = check_dataset(timelines)
    assert report.issues == [], "\n".join(str(i) for i in report.issues[:5])
