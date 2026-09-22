"""Tests for dependency-distance padding.

Padding is only a valid measurement if it changes exactly one thing: how far
the deciding fact sits from the query. Everything else — ground truth, the
causal order of events, the state layer — must come through untouched. These
tests pin that down, because a padding bug would not fail loudly; it would
quietly produce a curve that means something other than what it claims.
"""

from __future__ import annotations

import random

import pytest

from statebench.generator.distance import (
    DEFAULT_DISTANCE,
    DISTANCE_BY_NAME,
    DISTANCE_PROFILES,
    FILLER_EXCHANGES,
    TIME_SENSITIVE_TRACKS,
    get_profile,
    pad_dataset,
    pad_timeline,
)
from statebench.generator.engine import TimelineGenerator
from statebench.schema.timeline import (
    ConversationTurn,
    MentionRequirement,
    Query,
    StateWrite,
    Supersession,
)


def _timeline(track: str = "supersession", seed: int = 11):
    return next(iter(TimelineGenerator(seed=seed).generate_track(track, count=1)))


def _phrases(timeline) -> list[str]:
    out = []
    for e in timeline.events:
        if isinstance(e, Query):
            for item in list(e.ground_truth.must_mention) + list(
                e.ground_truth.must_not_mention
            ):
                out.append(item.phrase if isinstance(item, MentionRequirement) else item)
    return out


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #

def test_profiles_increase_monotonically():
    """A sweep only makes sense if the axis is ordered."""
    turns = [p.turns for p in DISTANCE_PROFILES]
    assert turns == sorted(turns)
    assert len(set(p.name for p in DISTANCE_PROFILES)) == len(DISTANCE_PROFILES)


def test_default_profile_is_the_unpadded_one():
    """Existing behavior must be the default, or old numbers silently change."""
    default = get_profile(DEFAULT_DISTANCE)
    assert default.exchanges == 0
    assert default.session_gaps == 0


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="Unknown distance profile"):
        get_profile("galaxy_brained")


def test_default_padding_is_a_faithful_copy():
    original = _timeline()
    padded, report = pad_timeline(original, DEFAULT_DISTANCE)
    assert padded.model_dump() == original.model_dump()
    assert report.turns_inserted == 0


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "profile", [p for p in DISTANCE_PROFILES if p.exchanges], ids=lambda p: p.name
)
def test_padding_inserts_only_conversation(profile):
    original = _timeline()
    padded, _ = pad_timeline(original, profile, rng=random.Random(3))

    def counts(tl):
        return (
            sum(isinstance(e, StateWrite) for e in tl.events),
            sum(isinstance(e, Supersession) for e in tl.events),
            sum(isinstance(e, Query) for e in tl.events),
        )

    assert counts(padded) == counts(original)
    added = len(padded.events) - len(original.events)
    assert added == profile.turns


@pytest.mark.parametrize(
    "profile", [p for p in DISTANCE_PROFILES if p.exchanges], ids=lambda p: p.name
)
def test_ground_truth_survives_padding(profile):
    original = _timeline()
    padded, _ = pad_timeline(original, profile, rng=random.Random(3))

    before = [e.ground_truth.model_dump() for e in original.events if isinstance(e, Query)]
    after = [e.ground_truth.model_dump() for e in padded.events if isinstance(e, Query)]
    assert before == after


@pytest.mark.parametrize("track", ["supersession", "premise_resistance", "causality"])
def test_filler_never_matches_a_ground_truth_phrase(track):
    """The guard that keeps padding from manufacturing hits and violations."""
    from statebench.evaluation.rubric import contains_phrase

    original = _timeline(track, seed=5)
    padded, _ = pad_timeline(original, "long_horizon", rng=random.Random(9))

    original_texts = {
        e.text for e in original.events if isinstance(e, ConversationTurn)
    }
    phrases = _phrases(padded)
    assert phrases, "this track should score on something"

    for event in padded.events:
        if not isinstance(event, ConversationTurn) or event.text in original_texts:
            continue
        for phrase in phrases:
            assert not contains_phrase(event.text, phrase), (
                f"filler {event.text!r} matches ground-truth phrase {phrase!r}"
            )


@pytest.mark.parametrize(
    "profile", [p for p in DISTANCE_PROFILES if p.exchanges], ids=lambda p: p.name
)
def test_timestamps_stay_ordered_and_the_query_moves_last(profile):
    original = _timeline()
    padded, _ = pad_timeline(original, profile, rng=random.Random(4))

    stamps = [e.ts for e in padded.events]
    assert stamps == sorted(stamps), "padding must not reorder time"

    query = next(e for e in padded.events if isinstance(e, Query))
    original_query = next(e for e in original.events if isinstance(e, Query))
    assert query.ts > original_query.ts


@pytest.mark.parametrize(
    "profile", [p for p in DISTANCE_PROFILES if p.exchanges], ids=lambda p: p.name
)
def test_padding_goes_between_the_supersession_and_the_query(profile):
    """Distance is measured from the deciding fact, so filler must sit after it."""
    padded, _ = pad_timeline(_timeline(), profile, rng=random.Random(4))

    last_supersession = max(
        i for i, e in enumerate(padded.events) if isinstance(e, Supersession)
    )
    query_idx = min(i for i, e in enumerate(padded.events) if isinstance(e, Query))
    assert last_supersession < query_idx

    original_turn_count = len(
        [e for e in _timeline().events if isinstance(e, ConversationTurn)]
    )
    between = sum(
        isinstance(e, ConversationTurn)
        for e in padded.events[last_supersession:query_idx]
    )
    assert between >= profile.turns - original_turn_count


def test_original_is_not_mutated():
    original = _timeline()
    before = original.model_dump()
    pad_timeline(original, "long_horizon", rng=random.Random(1))
    assert original.model_dump() == before


def test_padding_is_reproducible_for_a_given_seed():
    a, _ = pad_timeline(_timeline(), "cross_session", rng=random.Random(42))
    b, _ = pad_timeline(_timeline(), "cross_session", rng=random.Random(42))
    assert [e.ts for e in a.events] == [e.ts for e in b.events]
    assert [getattr(e, "text", None) for e in a.events] == [
        getattr(e, "text", None) for e in b.events
    ]


# --------------------------------------------------------------------------- #
# Time-sensitive tracks
# --------------------------------------------------------------------------- #

def test_time_sensitive_tracks_keep_their_clock():
    """Advancing days on an expiry track would rewrite its ground truth."""
    assert "environmental_freshness" in TIME_SENSITIVE_TRACKS

    original = _timeline("environmental_freshness", seed=8)
    padded, report = pad_timeline(original, "long_horizon", rng=random.Random(2))

    assert report.clock_preserved_for_time_sensitive is True
    assert report.days_advanced == 0
    assert report.notes

    # Still padded to the same token load — only the clock is held back.
    assert report.turns_inserted == get_profile("long_horizon").turns


def test_ordinary_tracks_do_advance_the_clock():
    _, report = pad_timeline(_timeline(), "cross_session", rng=random.Random(2))
    assert report.days_advanced > 0
    assert report.clock_preserved_for_time_sensitive is False


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def test_report_records_what_happened():
    profile = get_profile("cross_session")
    _, report = pad_timeline(_timeline(), profile, rng=random.Random(6))
    assert report.profile == "cross_session"
    assert report.exchanges_inserted == profile.exchanges
    assert report.session_gaps_inserted == profile.session_gaps


def test_exhausted_filler_is_declared_not_hidden():
    """The longest profile asks for more exchanges than the pool holds."""
    _, report = pad_timeline(_timeline(), "long_horizon", rng=random.Random(6))
    assert get_profile("long_horizon").exchanges > len(FILLER_EXCHANGES)
    assert report.exhausted_filler is True
    assert any("cycles" in n for n in report.notes)


def test_timeline_records_the_distance_it_was_padded_to():
    """A padded dataset must be able to say so; results are read months later."""
    original = _timeline()
    assert original.dependency_distance is None, "generated timelines start unpadded"

    padded, _ = pad_timeline(original, "cross_turn", rng=random.Random(1))
    assert padded.dependency_distance == "cross_turn"

    # And it survives the JSONL round trip the sweep actually uses.
    from statebench.schema.timeline import Timeline

    assert Timeline.model_validate_json(
        padded.model_dump_json()
    ).dependency_distance == "cross_turn"


def test_pad_dataset_pads_every_timeline():
    timelines = list(TimelineGenerator(seed=3).generate_track("supersession", count=4))
    padded, reports = pad_dataset(timelines, "cross_turn", seed=1)
    assert len(padded) == len(reports) == 4
    profile = get_profile("cross_turn")
    for original, result in zip(timelines, padded):
        assert len(result.events) - len(original.events) == profile.turns


def test_padding_a_timeline_with_no_query_fails_loudly():
    timeline = _timeline()
    timeline.events = [e for e in timeline.events if not isinstance(e, Query)]
    with pytest.raises(ValueError, match="no query"):
        pad_timeline(timeline, "cross_turn")


# --------------------------------------------------------------------------- #
# Dataset generation
# --------------------------------------------------------------------------- #

def test_generate_dataset_honours_distance(tmp_path):
    from statebench.generator.engine import generate_dataset
    from statebench.runner.harness import load_timelines

    short = tmp_path / "short.jsonl"
    long_ = tmp_path / "long.jsonl"
    generate_dataset(short, ["supersession"], count_per_track=3, seed=17)
    generate_dataset(
        long_, ["supersession"], count_per_track=3, seed=17, distance="cross_session"
    )

    short_events = [len(t.events) for t in load_timelines(short)]
    long_events = [len(t.events) for t in load_timelines(long_)]
    assert all(b > a for a, b in zip(short_events, long_events))


def test_every_profile_is_reachable_by_name():
    for profile in DISTANCE_PROFILES:
        assert DISTANCE_BY_NAME[profile.name] is profile
