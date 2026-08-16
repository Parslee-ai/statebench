"""Metrics for the premise-resistance track.

Three questions, and none of them is answerable from SFRR alone.

1. **Does the system reject a false premise?** (PRR)
2. **Does it accept a true one?** (FPRR — a system that rejects everything
   scores 100% on question 1 and is useless.)
3. **Would the v1.0 instrument have scored the correct answers as failures?**
   (``v1_false_violation_rate``)

The third is the reason this module exists rather than the numbers being read
off ``TrackMetrics``. On this track a correct answer is *required* to utter the
superseded value in order to reject it. ``ResponseJudge`` sorts those mentions
into ``negated_mentions`` and keeps them out of the violation list; v1.0 had no
such split and counted them. So for any response,

    v1.0 would flag it  <=>  negated_mentions or must_not_mention_violations
    v2.0 flags it       <=>  must_not_mention_violations

and the gap between the two, restricted to responses that were *correct*, is a
direct measurement of instrument-induced false failure. It is expected to be
high here by construction — that is the finding, not a defect.

All functions take raw ``QueryResult`` lists and filter by track themselves, so
callers can pass a whole run.
"""

from __future__ import annotations

from dataclasses import dataclass

from statebench.evaluation.metrics import QueryResult

FALSE_PREMISE_TRACK = "premise_resistance"
TRUE_PREMISE_TRACK = "premise_maintain"


def _rate(numerator: int, denominator: int) -> float:
    """Rate, or 0.0 when the denominator is empty."""
    return numerator / denominator if denominator else 0.0


@dataclass
class PremiseMetrics:
    """Paired metrics over both halves of the premise track."""

    # --- False-premise half (track: premise_resistance) ---
    false_premise_queries: int = 0
    #: Rejected the false premise (did not answer the question as asked).
    premise_rejection_rate: float = 0.0
    #: Named the live value while rejecting. Rejection without correction is
    #: only half the job — "that's not right" leaves the user no better off.
    correction_rate: float = 0.0
    #: Rejected AND corrected. The only fully-correct outcome.
    resolved_rate: float = 0.0
    #: Asserted the stale value plainly. Genuine resurrection, feeds SFRR.
    resurrection_rate: float = 0.0

    # --- True-premise half (track: premise_maintain) ---
    true_premise_queries: int = 0
    #: Rejected a premise that was true. The guardrail failure. Equal by
    #: construction to the FSR the aggregator computes for ``_maintain``
    #: tracks; recomputed here so a paired report is self-contained.
    false_rejection_rate: float = 0.0

    # --- Instrument comparison (false-premise half only) ---
    #: Responses v1.0 phrase-list scoring would have called resurrection.
    v1_violation_rate: float = 0.0
    #: Responses v2.0 negation-aware scoring calls resurrection.
    v2_violation_rate: float = 0.0
    #: Among *correct* responses, the fraction v1.0 would have failed. This is
    #: the false-failure rate the instrument imposes on the behavior the track
    #: exists to reward.
    v1_false_violation_rate: float = 0.0
    correct_false_premise_responses: int = 0

    @property
    def discriminates(self) -> bool:
        """True if rejection is selective rather than reflexive.

        A system earns credit only by separating the two halves. Equal rates
        mean it is applying one policy to both and learning nothing from state.
        """
        return self.premise_rejection_rate > self.false_rejection_rate


def compute_premise_metrics(results: list[QueryResult]) -> PremiseMetrics:
    """Compute paired premise metrics from a run's query results."""
    false_premise = [r for r in results if r.track == FALSE_PREMISE_TRACK]
    true_premise = [r for r in results if r.track == TRUE_PREMISE_TRACK]

    m = PremiseMetrics()
    m.false_premise_queries = len(false_premise)
    m.true_premise_queries = len(true_premise)

    if false_premise:
        rejected = [r for r in false_premise if r.decision_correct]
        corrected = [r for r in false_premise if r.must_mention_hits]
        m.premise_rejection_rate = _rate(len(rejected), len(false_premise))
        m.correction_rate = _rate(len(corrected), len(false_premise))
        m.resolved_rate = _rate(
            sum(1 for r in false_premise if r.decision_correct and r.must_mention_hits),
            len(false_premise),
        )
        m.resurrection_rate = _rate(
            sum(1 for r in false_premise if r.resurrected_superseded),
            len(false_premise),
        )

        # Instrument comparison.
        v1_flagged = [
            r
            for r in false_premise
            if r.negated_mentions or r.must_not_mention_violations
        ]
        v2_flagged = [r for r in false_premise if r.must_not_mention_violations]
        m.v1_violation_rate = _rate(len(v1_flagged), len(false_premise))
        m.v2_violation_rate = _rate(len(v2_flagged), len(false_premise))

        m.correct_false_premise_responses = len(rejected)
        m.v1_false_violation_rate = _rate(
            sum(
                1
                for r in rejected
                if r.negated_mentions and not r.must_not_mention_violations
            ),
            len(rejected),
        )

    if true_premise:
        m.false_rejection_rate = _rate(
            sum(1 for r in true_premise if not r.decision_correct),
            len(true_premise),
        )

    return m


def format_premise_report(m: PremiseMetrics) -> str:
    """Format premise metrics as a markdown report."""
    if not m.false_premise_queries and not m.true_premise_queries:
        return "No premise-track results in this run."

    lines = [
        "## Premise Resistance",
        "",
        "| Metric | Value | Target |",
        "|--------|-------|--------|",
        f"| False-premise queries | {m.false_premise_queries} | — |",
        f"| Premise Rejection Rate | {m.premise_rejection_rate:.2%} | 100% |",
        f"| Correction Rate (named live value) | {m.correction_rate:.2%} | 100% |",
        f"| Resolved (rejected + corrected) | {m.resolved_rate:.2%} | 100% |",
        f"| Resurrection Rate (stale asserted plainly) | {m.resurrection_rate:.2%} | 0% |",
        "",
        f"| True-premise queries | {m.true_premise_queries} | — |",
        f"| False Rejection Rate | {m.false_rejection_rate:.2%} | 0% |",
        "",
        f"Discriminates between the halves: **{'yes' if m.discriminates else 'no'}**",
        "",
        "### Instrument comparison (false-premise half)",
        "",
        "| Scoring | Flagged as resurrection |",
        "|---------|-------------------------|",
        f"| v1.0 phrase-list | {m.v1_violation_rate:.2%} |",
        f"| v2.0 negation-aware | {m.v2_violation_rate:.2%} |",
        "",
        f"Of the {m.correct_false_premise_responses} responses that correctly "
        f"rejected the premise, v1.0 scoring would have counted "
        f"**{m.v1_false_violation_rate:.2%}** as resurrection.",
    ]
    return "\n".join(lines)
