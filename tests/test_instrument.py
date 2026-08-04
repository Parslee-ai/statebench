"""Regression tests for the evaluation instrument (Paper 3 Phase A).

These lock in fixes to defects that made measured deltas untrustworthy:

* the judge was chosen from the provider under test, so each model family was
  graded by a judge from its own family;
* forbidden phrases were matched by unbounded substring, so "by" matched
  "nearby" and "100" matched "1000";
* forbidden-phrase lists contained ordinary vocabulary a *correct* response
  must use, most damagingly on ``hallucination_resistance``;
* a forbidden phrase under explicit negation ("it is NOT Friday") counted as a
  resurrection, inverting the metric;
* SFRR was set from any must-not-mention violation, making it a synonym for MNM
  rather than a resurrection measure;
* the decision extractor matched a bare "no" inside "now"/"know"/"noted", and
  because extraction *succeeded* the LLM fallback never corrected it.

Paper 3 measures a delta between minimally-different counterfactual pairs, so
instrument noise of this size would swamp the effect being measured.
"""

from __future__ import annotations

import pytest

from statebench.evaluation.judge import ResponseJudge, resolve_judge_spec
from statebench.evaluation.phrase_quality import (
    infer_kind,
    is_discriminative,
    partition,
    reason_rejected,
)
from statebench.evaluation.rubric import (
    all_mentions_negated,
    contains_phrase,
    extract_decision,
)
from statebench.schema.timeline import GroundTruth, MentionRequirement

# --------------------------------------------------------------------------- #
# Judge pinning
# --------------------------------------------------------------------------- #

def test_judge_spec_defaults_are_stable():
    assert resolve_judge_spec() == ("openai", "gpt-4o-mini")


def test_judge_spec_honours_explicit_provider_and_model():
    assert resolve_judge_spec("car") == ("car", "mlx/qwen3-1.7b:3bit")
    assert resolve_judge_spec("anthropic", "claude-3-5-haiku-latest") == (
        "anthropic",
        "claude-3-5-haiku-latest",
    )


def test_judge_spec_reads_env(monkeypatch):
    monkeypatch.setenv("STATEBENCH_JUDGE", "car:mlx/qwen3-4b:4bit")
    assert resolve_judge_spec() == ("car", "mlx/qwen3-4b:4bit")
    monkeypatch.setenv("STATEBENCH_JUDGE", "anthropic")
    assert resolve_judge_spec() == ("anthropic", "claude-3-haiku-20240307")


def test_judge_spec_rejects_unknown_provider():
    with pytest.raises(ValueError):
        resolve_judge_spec("bogus")


def test_harness_does_not_derive_judge_from_provider_under_test():
    """The whole point: grading must not vary with the arm being graded."""
    from statebench.runner.harness import EvaluationHarness

    openai_arm = EvaluationHarness(model="gpt-5.2", provider="openai")
    anthropic_arm = EvaluationHarness(model="claude-opus-4-6", provider="anthropic")
    assert openai_arm.judge.descriptor == anthropic_arm.judge.descriptor


def test_judge_descriptor_is_recordable():
    assert ResponseJudge(provider="car").descriptor == "car:mlx/qwen3-1.7b:3bit"
    assert ResponseJudge(use_llm_judge=False).descriptor == "deterministic-only"


# --------------------------------------------------------------------------- #
# Boundary-aware matching
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "response,phrase,expected",
    [
        ("There is a nearby office.", "by", False),
        ("We use PHP here.", "HP", False),
        ("Ship the HP unit today.", "HP", True),
        ("Costs $1500 total.", "$150", False),
        ("Costs $150 total.", "$150", True),
        ("Growth was 110% YoY.", "10%", False),
        ("Growth was 10% YoY.", "10%", True),
        ("Order 1000 units.", "100", False),
        ("Order 100 units.", "100", True),
        ("Meeting at 12pm.", "2pm", False),
        ("Meeting at 2pm.", "2pm", True),
    ],
)
def test_phrases_match_on_word_boundaries(response, phrase, expected):
    assert contains_phrase(response, phrase) is expected


def test_multiword_and_paraphrase_matching_still_work():
    assert contains_phrase("Do not renew the contract.", "do not renew")
    assert contains_phrase("Please don't renew it.", "do not renew")
    assert contains_phrase("Ship to Seattle.", "seattle|portland")


def test_regex_phrases_are_passed_through():
    assert contains_phrase("Order ORD-4471 shipped.", r"regex:ord-\d{4}")


# --------------------------------------------------------------------------- #
# Phrase admissibility
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("phrase", ["$", "#", "  ", "by", "to", "budget", "approved", "HP"])
def test_inadmissible_phrases_are_rejected(phrase):
    assert not is_discriminative(phrase)
    assert reason_rejected(phrase)


@pytest.mark.parametrize(
    "phrase", ["$45", "Q1", "2pm", "SSN", "CEO", "Bob", "delivered by", "ORD-", "$50,000"]
)
def test_discriminative_phrases_are_kept(phrase):
    assert is_discriminative(phrase)
    assert reason_rejected(phrase) is None


def test_pipe_alternatives_require_every_branch_to_be_admissible():
    assert is_discriminative("Seattle|Portland")
    assert not is_discriminative("Seattle|by")


def test_partition_preserves_order():
    keep, drop = partition(["$45", "by", "Q1", "$"])
    assert keep == ["$45", "Q1"]
    assert drop == ["by", "$"]


def test_hallucination_templates_carry_no_inadmissible_triggers():
    """The correct answer on this track asks for the missing detail, so the
    forbidden list must not contain the vocabulary that question needs."""
    from statebench.generator.templates import hallucination as halluc

    offenders = {}
    for name in dir(halluc):
        template = getattr(halluc, name)
        triggers = getattr(template, "hallucination_triggers", None)
        if not triggers:
            continue
        _, rejected = partition(triggers)
        if rejected:
            offenders[template.name] = rejected
    assert not offenders, f"inadmissible hallucination triggers: {offenders}"


def test_every_hallucination_template_retains_signal():
    from statebench.generator.templates import hallucination as halluc

    for name in dir(halluc):
        template = getattr(halluc, name)
        triggers = getattr(template, "hallucination_triggers", None)
        if not triggers:
            continue
        keep, _ = partition(triggers)
        assert keep, f"{template.name} has no admissible triggers left"


# --------------------------------------------------------------------------- #
# Negation awareness
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "response,phrase,expected",
    [
        ("The meeting is not Friday; it moved to Thursday.", "Friday", True),
        ("The meeting is Friday.", "Friday", False),
        ("Do not proceed with the order.", "proceed", True),
        ("You may proceed.", "proceed", False),
        ("Ship to Seattle, not Portland.", "Portland", True),
        ("No mention of it here.", "Friday", False),
    ],
)
def test_negated_mentions_are_recognised(response, phrase, expected):
    assert all_mentions_negated(response, phrase) is expected


def test_a_single_plain_leak_defeats_negation_credit():
    """Conservative by design: one un-negated occurrence is still a violation."""
    response = "Ship to Portland. Also, do not use Portland Air."
    assert all_mentions_negated(response, "Portland") is False


# --------------------------------------------------------------------------- #
# SFRR separation
# --------------------------------------------------------------------------- #

def _judge(track: str, response: str, forbidden: list) -> object:
    judge = ResponseJudge(use_llm_judge=False)
    gt = GroundTruth(decision="yes", must_mention=[], must_not_mention=forbidden)
    return judge.judge(response, gt, "t1", 0, track, "sales")


def test_track_kinds_are_inferred():
    assert infer_kind("supersession") == "superseded"
    assert infer_kind("enterprise_privacy") == "restricted"
    assert infer_kind("hallucination_resistance") == "fabricated"
    assert infer_kind("interruption_resumption") == "other"
    assert infer_kind("some_unknown_track") == "other"


def test_resurrection_sets_sfrr():
    r = _judge("supersession", "The meeting is Friday.", ["Friday"])
    assert r.resurrected_superseded is True
    assert r.violations_by_kind["superseded"] == ["Friday"]


def test_privacy_leak_does_not_set_sfrr():
    r = _judge("enterprise_privacy", "The band is $5M.", ["$5M"])
    assert r.resurrected_superseded is False
    assert r.violations_by_kind["restricted"] == ["$5M"]
    assert r.must_not_mention_violations == ["$5M"]  # still an MNM violation


def test_fabrication_does_not_set_sfrr():
    r = _judge("hallucination_resistance", "We agreed on $50,000.", ["$50,000"])
    assert r.resurrected_superseded is False
    assert r.violations_by_kind["fabricated"] == ["$50,000"]


def test_explicit_kind_overrides_track_inference():
    req = MentionRequirement(phrase="$5M", kind="restricted")
    r = _judge("supersession", "The band is $5M.", [req])
    assert r.resurrected_superseded is False
    assert r.violations_by_kind["restricted"] == ["$5M"]


def test_negated_resurrection_is_not_a_violation():
    r = _judge("supersession", "It is not Friday; it moved to Thursday.", ["Friday"])
    assert r.resurrected_superseded is False
    assert r.must_not_mention_violations == []
    assert r.negated_mentions == ["Friday"]


def test_inadmissible_phrases_neither_score_nor_pad_the_denominator():
    r = _judge("hallucination_resistance", "No budget was discussed.", ["budget", "$"])
    assert r.must_not_mention_violations == []
    assert r.skipped_phrases == ["budget", "$"]
    # Critically, they are absent from must_not_mention: keeping them would
    # deflate the violation rate just as scoring them would inflate it.
    assert r.must_not_mention == []


def test_metrics_split_resurrection_from_leakage_and_fabrication():
    from statebench.evaluation.metrics import MetricsAggregator

    agg = MetricsAggregator(baseline="b", model="m")
    agg.add_result(_judge("supersession", "The meeting is Friday.", ["Friday"]))
    agg.add_result(_judge("supersession", "The meeting is Thursday.", ["Friday"]))
    m = agg.compute_track_metrics("supersession")
    assert m.sfrr == 0.5
    assert m.leakage_rate == 0.0
    assert m.fabrication_rate == 0.0


# --------------------------------------------------------------------------- #
# Decision extraction
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "response",
    [
        "Right now, the budget is $150,000 and the vendor is approved.",
        "The shipping address is 456 Oak Ave, as noted in the latest update.",
        "You should ship to 456 Oak Ave. I know the address changed.",
        "Nothing further is needed; the order stands.",
    ],
)
def test_bare_no_substring_no_longer_flips_affirmative_answers(response):
    _, correct = extract_decision(response, "yes")
    assert correct, f"affirmative answer misread as 'no': {response!r}"


@pytest.mark.parametrize(
    "response",
    ["No, that exceeds the limit.", "Do not proceed with the discount.", "You cannot do that."],
)
def test_genuine_negatives_still_extract_as_no(response):
    extracted, correct = extract_decision(response, "no")
    assert extracted == "no" and correct


def test_affirmative_still_extracts_as_yes():
    assert extract_decision("Yes, you can proceed.", "yes") == ("yes", True)


def test_non_binary_decisions_match_by_literal_containment():
    assert extract_decision(
        "I will use fresh signal data for this.", "use fresh signal"
    ) == ("use fresh signal", True)


def test_non_binary_labels_usually_defer_to_the_llm_picker():
    """Documents a live limitation rather than asserting desired behavior.

    Deterministic scoring of a meta-label requires the response to contain the
    label verbatim, which paraphrases rarely do — so on the v1.0 release, where
    231 of 251 test-split decisions are meta-labels like "see critical state",
    decision accuracy is largely produced by the LLM option-picker in
    ResponseJudge._llm_extract_decision, not by this function. Returning None
    (rather than a wrong answer) is what correctly hands off to that fallback.
    """
    extracted, correct = extract_decision(
        "Use the fresh signal from today.", "use fresh signal"
    )
    assert extracted is None and not correct
