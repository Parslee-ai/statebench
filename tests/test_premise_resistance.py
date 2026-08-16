"""Tests for the premise-resistance track.

The track's whole reason for existing is a scoring interaction: a correct
answer must utter the superseded value in order to reject the premise built on
it, so plain phrase-list matching fails every correct answer. These tests lock
in three things:

* the templates and generated timelines are well-formed and paired;
* a correct rejection scores clean under v2.0 and *would* have scored as
  resurrection under v1.0 — the demonstration itself;
* the guardrail half catches a system that rejects reflexively.

No network, no model calls: the judge runs with ``use_llm_judge=False``.
"""

from __future__ import annotations

import pytest

from statebench.evaluation.judge import ResponseJudge
from statebench.evaluation.metrics import MetricsAggregator, QueryResult
from statebench.evaluation.phrase_quality import infer_kind, is_discriminative
from statebench.evaluation.premise_metrics import (
    compute_premise_metrics,
    format_premise_report,
)
from statebench.generator.engine import TimelineGenerator
from statebench.generator.templates.premise import (
    PREMISE_MAINTAIN_TEMPLATES,
    PREMISE_PAIRS,
    PREMISE_RESISTANCE_TEMPLATES,
    PremiseTemplate,
    get_premise_templates,
)
from statebench.schema.timeline import Query, Supersession

ALL_TEMPLATES = PREMISE_RESISTANCE_TEMPLATES + PREMISE_MAINTAIN_TEMPLATES


@pytest.fixture
def judge() -> ResponseJudge:
    """Deterministic judge — no LLM paraphrase fallback, no network."""
    return ResponseJudge(use_llm_judge=False)


def _query(timeline) -> Query:
    """The single query event of a premise timeline."""
    queries = [e for e in timeline.events if isinstance(e, Query)]
    assert len(queries) == 1, "premise timelines carry exactly one query"
    return queries[0]


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("template", ALL_TEMPLATES, ids=lambda t: t.name)
def test_query_embeds_the_value_it_presupposes(template: PremiseTemplate):
    """The premise must actually be in the query, or the track tests nothing."""
    presupposed = (
        template.stale_value
        if template.presupposes == "stale"
        else template.live_value
    )
    assert presupposed in template.query

    # And the *other* value must not be, or the premise is ambiguous.
    other = (
        template.live_value
        if template.presupposes == "stale"
        else template.stale_value
    )
    assert other not in template.query


@pytest.mark.parametrize("template", PREMISE_RESISTANCE_TEMPLATES, ids=lambda t: t.name)
def test_false_premise_forbids_exactly_the_stale_value(template: PremiseTemplate):
    assert template.forbidden == [template.stale_value]
    assert template.live_value in template.must_mention


@pytest.mark.parametrize("template", PREMISE_MAINTAIN_TEMPLATES, ids=lambda t: t.name)
def test_true_premise_forbids_nothing(template: PremiseTemplate):
    """Nothing is off-limits when the premise is sound; the live value is the answer."""
    assert template.forbidden == []
    assert template.correct_decision == "yes"


@pytest.mark.parametrize("template", ALL_TEMPLATES, ids=lambda t: t.name)
def test_forbidden_phrases_are_admissible(template: PremiseTemplate):
    """A stale value that phrase_quality would drop gives the track no signal."""
    for phrase in template.forbidden:
        assert is_discriminative(phrase), f"{phrase!r} would be skipped at judging"


def test_pairs_differ_only_in_the_presupposed_value():
    """Paired design: same scenario, one variable moved."""
    for false_t, true_t in PREMISE_PAIRS:
        assert false_t.fact_key == true_t.fact_key
        assert false_t.domain == true_t.domain
        assert false_t.stale_value == true_t.stale_value
        assert false_t.live_value == true_t.live_value
        assert false_t.initial_turn == true_t.initial_turn
        assert false_t.supersession_turn == true_t.supersession_turn
        assert false_t.query != true_t.query


def test_template_validation_rejects_incoherent_cases():
    with pytest.raises(ValueError, match="presupposes"):
        PremiseTemplate(
            name="bad", domain="sales", description="",
            fact_key="k", stale_value="a", live_value="b",
            initial_turn="", supersession_turn="", ack_turn="",
            query="", presupposes="neither",
            correct_decision="no", must_mention=[],
        )

    with pytest.raises(ValueError, match="true-premise case"):
        PremiseTemplate(
            name="bad", domain="sales", description="",
            fact_key="k", stale_value="a", live_value="b",
            initial_turn="", supersession_turn="", ack_turn="",
            query="", presupposes="live",
            correct_decision="yes", must_mention=[], forbidden=["a"],
        )


def test_get_premise_templates_selects_halves():
    assert get_premise_templates("stale") == PREMISE_RESISTANCE_TEMPLATES
    assert get_premise_templates("live") == PREMISE_MAINTAIN_TEMPLATES
    with pytest.raises(ValueError):
        get_premise_templates("whatever")


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

def test_generated_timelines_carry_the_supersession_and_the_premise():
    gen = TimelineGenerator(seed=1234)
    timelines = list(gen.generate_track("premise_resistance", count=5))
    assert len(timelines) == 5

    for tl in timelines:
        assert tl.track == "premise_resistance"
        # The value must actually be dead before the query asserts it.
        supersessions = [e for e in tl.events if isinstance(e, Supersession)]
        assert len(supersessions) == 1
        assert supersessions[0].writes[0].supersedes is not None

        gt = _query(tl).ground_truth
        assert gt.decision == "no"
        assert gt.must_not_mention, "false premise must forbid the stale value"
        assert gt.must_mention, "correct answer must name the live value"


def test_maintain_half_generates_without_forbidden_phrases():
    gen = TimelineGenerator(seed=1234)
    timelines = list(gen.generate_track("premise_maintain", count=5))
    for tl in timelines:
        assert tl.track == "premise_maintain"
        gt = _query(tl).ground_truth
        assert gt.decision == "yes"
        assert gt.must_not_mention == []


def test_generation_is_seed_reproducible():
    a = list(TimelineGenerator(seed=7).generate_track("premise_resistance", count=4))
    b = list(TimelineGenerator(seed=7).generate_track("premise_resistance", count=4))
    assert [_query(t).prompt for t in a] == [_query(t).prompt for t in b]


def test_forbidden_phrases_are_tagged_superseded():
    """SFRR must see these as resurrection, not as 'other'."""
    gen = TimelineGenerator(seed=99)
    tl = next(iter(gen.generate_track("premise_resistance", count=1)))
    forbidden = _query(tl).ground_truth.must_not_mention
    assert all(getattr(f, "kind", None) == "superseded" for f in forbidden)
    assert infer_kind("premise_resistance") == "superseded"


# --------------------------------------------------------------------------- #
# The scoring interaction — the point of the track
# --------------------------------------------------------------------------- #

def _judge_response(judge: ResponseJudge, timeline, response: str) -> QueryResult:
    gt = _query(timeline).ground_truth
    return judge.judge(
        response=response,
        ground_truth=gt,
        timeline_id=timeline.id,
        query_idx=0,
        track=timeline.track,
        domain=timeline.domain,
    )


def test_correct_rejection_is_clean_under_v2_and_flagged_under_v1(judge):
    """The central claim: rejecting a premise requires naming the dead value."""
    gen = TimelineGenerator(seed=2026)
    # Pin to the address scenario so the response text is exact.
    timeline = gen.generate_premise_timeline(PREMISE_RESISTANCE_TEMPLATES[0])

    response = (
        "No — you're not shipping to 123 Main St anymore. That address was "
        "superseded when you moved. Your current address is 456 Oak Ave, so "
        "I'll select ground shipping to 456 Oak Ave instead."
    )
    result = _judge_response(judge, timeline, response)

    # v2.0: the stale value appears only under negation, so it is not a violation.
    assert result.must_not_mention_violations == []
    assert result.negated_mentions == ["123 Main St"]
    assert result.resurrected_superseded is False

    # The answer is otherwise correct.
    assert result.decision_correct is True
    assert "456 Oak Ave" in result.must_mention_hits

    # v1.0 counted any occurrence, so this same correct answer was a failure.
    v1_would_flag = bool(result.negated_mentions or result.must_not_mention_violations)
    assert v1_would_flag is True


def test_accepting_the_false_premise_is_a_resurrection(judge):
    gen = TimelineGenerator(seed=2026)
    timeline = gen.generate_premise_timeline(PREMISE_RESISTANCE_TEMPLATES[0])

    response = (
        "Yes, ground shipping to 123 Main St is the right choice for this order."
    )
    result = _judge_response(judge, timeline, response)

    assert result.must_not_mention_violations == ["123 Main St"]
    assert result.resurrected_superseded is True
    assert result.decision_correct is False


def test_partial_leak_after_a_correction_still_counts(judge):
    """One un-negated use defeats negation credit — no free pass for hedging."""
    gen = TimelineGenerator(seed=2026)
    timeline = gen.generate_premise_timeline(PREMISE_RESISTANCE_TEMPLATES[0])

    response = (
        "You're not at 123 Main St anymore. I'll send this order to 123 Main St "
        "with ground shipping."
    )
    result = _judge_response(judge, timeline, response)
    assert result.must_not_mention_violations == ["123 Main St"]
    assert result.resurrected_superseded is True


def test_true_premise_accepted_scores_correct(judge):
    gen = TimelineGenerator(seed=2026)
    timeline = gen.generate_premise_timeline(PREMISE_MAINTAIN_TEMPLATES[0])

    response = "Yes, ground shipping to 456 Oak Ave works for this order."
    result = _judge_response(judge, timeline, response)

    assert result.decision_correct is True
    assert result.must_not_mention_violations == []


def test_reflexive_rejection_fails_the_guardrail(judge):
    """A system that corrects everything must lose points somewhere."""
    gen = TimelineGenerator(seed=2026)
    timeline = gen.generate_premise_timeline(PREMISE_MAINTAIN_TEMPLATES[0])

    response = (
        "No, I can't confirm that — I don't have a reliable shipping address "
        "on file, so I won't select a shipping method."
    )
    result = _judge_response(judge, timeline, response)
    assert result.decision_correct is False


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def _result(track: str, **kwargs) -> QueryResult:
    base = {
        "timeline_id": "T-1",
        "query_idx": 0,
        "track": track,
        "domain": "support",
        "expected_decision": "no" if track == "premise_resistance" else "yes",
        "must_mention": ["456 Oak Ave"],
        "must_not_mention": ["123 Main St"] if track == "premise_resistance" else [],
        "response": "",
    }
    base.update(kwargs)
    return QueryResult(**base)  # type: ignore[arg-type]


def test_premise_metrics_separate_the_two_halves():
    results = [
        # Correct rejection, stale value used only under negation.
        _result(
            "premise_resistance",
            decision_correct=True,
            must_mention_hits=["456 Oak Ave"],
            negated_mentions=["123 Main St"],
        ),
        # Went along with the premise.
        _result(
            "premise_resistance",
            decision_correct=False,
            must_not_mention_violations=["123 Main St"],
            violations_by_kind={"superseded": ["123 Main St"]},
            resurrected_superseded=True,
        ),
        # True premise accepted.
        _result("premise_maintain", decision_correct=True),
        # True premise wrongly rejected.
        _result("premise_maintain", decision_correct=False),
    ]

    m = compute_premise_metrics(results)

    assert m.false_premise_queries == 2
    assert m.true_premise_queries == 2
    assert m.premise_rejection_rate == 0.5
    assert m.correction_rate == 0.5
    assert m.resolved_rate == 0.5
    assert m.resurrection_rate == 0.5
    assert m.false_rejection_rate == 0.5
    assert m.discriminates is False  # 0.5 vs 0.5 — not selective

    # Instrument comparison: v1.0 flags both (one negated, one plain).
    assert m.v1_violation_rate == 1.0
    assert m.v2_violation_rate == 0.5
    # Of the one correct response, v1.0 would have failed it.
    assert m.correct_false_premise_responses == 1
    assert m.v1_false_violation_rate == 1.0


def test_selective_system_discriminates():
    results = [
        _result(
            "premise_resistance",
            decision_correct=True,
            must_mention_hits=["456 Oak Ave"],
            negated_mentions=["123 Main St"],
        ),
        _result("premise_maintain", decision_correct=True),
    ]
    m = compute_premise_metrics(results)
    assert m.premise_rejection_rate == 1.0
    assert m.false_rejection_rate == 0.0
    assert m.discriminates is True


def test_premise_metrics_on_empty_run():
    m = compute_premise_metrics([])
    assert m.false_premise_queries == 0
    assert m.premise_rejection_rate == 0.0
    assert "No premise-track results" in format_premise_report(m)


def test_report_renders():
    m = compute_premise_metrics([
        _result("premise_resistance", decision_correct=True,
                must_mention_hits=["456 Oak Ave"], negated_mentions=["123 Main St"]),
        _result("premise_maintain", decision_correct=True),
    ])
    report = format_premise_report(m)
    assert "Premise Rejection Rate" in report
    assert "v2.0 negation-aware" in report


def test_maintain_track_feeds_fsr_not_sfrr():
    """The ``_maintain`` suffix must route this half to the guardrail metric."""
    agg = MetricsAggregator(baseline="test", model="test")
    agg.add_result(_result("premise_maintain", decision_correct=False))
    agg.add_result(_result("premise_maintain", decision_correct=True))
    agg.add_result(
        _result(
            "premise_resistance",
            decision_correct=True,
            must_mention_hits=["456 Oak Ave"],
            negated_mentions=["123 Main St"],
        )
    )

    benchmark = agg.compute_benchmark_metrics()
    assert benchmark.tracks["premise_maintain"].false_supersession_rate == 0.5
    assert benchmark.tracks["premise_maintain"].sfrr == 0.0
    assert benchmark.overall_false_supersession_rate == 0.5
    # The false-premise half is scored on the should-supersede side, and a
    # correctly-negated mention is not a resurrection.
    assert benchmark.overall_sfrr == 0.0
