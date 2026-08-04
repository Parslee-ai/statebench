"""Applicability metrics for the factorial suite (Paper 3, §5.2).

Scores the disposition of a retrieved experience — KEEP, ADAPT or REJECT —
separately from decision accuracy, because "refuse to use this memory" is a
correct outcome that a single accuracy number cannot distinguish from failing
to answer.

**The degenerate-strategy problem.** Governance dominates the matrix: once
state is governed, the correct disposition no longer depends on how well the
experience transfers. Withdrawing statuses (unauthorized/draft/expired) require
REJECT and substituting ones (superseded/outranked) require ADAPT, which leaves
the suite REJECT-leaning, so a system that refuses everything still scores far
above zero on classification alone. Two guards, both mandatory when reporting:

1. ``always_refuse_baseline`` — the score a trivial refuser would obtain on
   this exact sample. Any headline number must be read against it, not against
   zero.
2. ``correct_rejection_rate`` and ``false_refusal_rate`` are computed over
   disjoint cell sets and are NEVER blended, following the FSR/FAOR discipline
   already used for the maintain tracks. Blanket refusal drives the second to
   1.0 and is immediately visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from statebench.evaluation.metrics import QueryResult
from statebench.evaluation.rubric import contains_phrase

REJECT_DECISION_MARKERS = (
    "acknowledge uncertainty",
    "ask for clarification",
    "do not have",
    "don't have",
    "cannot determine",
    "not available",
    "no record",
    "not specified",
    "unable to",
    "cannot share",
    "can't share",
)


def looks_like_refusal(response: str) -> bool:
    """Whether a response declines to supply the requested value.

    Deliberately lexical and conservative: the point is to detect blanket
    refusal as a *strategy*, which is coarse behavior, not to adjudicate
    borderline hedging.
    """
    lowered = response.lower()
    return any(marker in lowered for marker in REJECT_DECISION_MARKERS)


@dataclass
class ApplicabilityMetrics:
    """Disposition metrics over applicability-suite results."""

    total: int = 0

    # Classification against expected_applicability.
    classified: int = 0
    correct_classification: int = 0
    applicability_classification_accuracy: float = 0.0

    # Disjoint by construction — never averaged together.
    reject_cells: int = 0
    correct_rejections: int = 0
    correct_rejection_rate: float = 0.0
    use_cells: int = 0  # KEEP + ADAPT
    false_refusals: int = 0
    false_refusal_rate: float = 0.0

    # ADAPT-only: did the lesson survive, and did the stale detail die?
    adapt_cells: int = 0
    principle_retained: int = 0
    principle_transfer_rate: float = 0.0
    detail_carryover: int = 0
    historical_detail_carryover_rate: float = 0.0

    # What a system that answers nothing would score here.
    always_refuse_baseline: float = 0.0

    per_cell: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_results(
        cls,
        results: list[QueryResult],
        expected: dict[tuple[str, int], str],
        principles: dict[tuple[str, int], str | None] | None = None,
        obsolete: dict[tuple[str, int], list[str]] | None = None,
    ) -> ApplicabilityMetrics:
        """Aggregate applicability results.

        ``expected`` maps (timeline_id, query_idx) -> KEEP|ADAPT|REJECT, taken
        from ground truth rather than inferred, so a system cannot influence
        what it is graded against.
        """
        m = cls(total=len(results))
        principles = principles or {}
        obsolete = obsolete or {}

        for r in results:
            key = (r.timeline_id, r.query_idx)
            want = expected.get(key)
            if want is None:
                continue
            m.classified += 1
            refused = looks_like_refusal(r.response)
            observed = "REJECT" if refused else "KEEP_OR_ADAPT"

            if want == "REJECT":
                m.reject_cells += 1
                if refused:
                    m.correct_rejections += 1
                    m.correct_classification += 1
            else:
                m.use_cells += 1
                if refused:
                    m.false_refusals += 1
                else:
                    # The response supplied a value; decision correctness
                    # distinguishes KEEP from ADAPT.
                    if r.decision_correct:
                        m.correct_classification += 1

            if want == "ADAPT":
                m.adapt_cells += 1
                principle = principles.get(key)
                if principle and contains_phrase(r.response, principle):
                    m.principle_retained += 1
                stale = obsolete.get(key, [])
                if any(contains_phrase(r.response, p) for p in stale):
                    m.detail_carryover += 1

            m.per_cell.setdefault(want, {"n": 0, "correct": 0})
            m.per_cell[want]["n"] += 1
            m.per_cell[want]["correct"] += float(observed == "REJECT") == (want == "REJECT")

        if m.classified:
            m.applicability_classification_accuracy = (
                m.correct_classification / m.classified
            )
            # A refuser is right on every REJECT cell and wrong everywhere else.
            m.always_refuse_baseline = m.reject_cells / m.classified
        if m.reject_cells:
            m.correct_rejection_rate = m.correct_rejections / m.reject_cells
        if m.use_cells:
            m.false_refusal_rate = m.false_refusals / m.use_cells
        if m.adapt_cells:
            m.principle_transfer_rate = m.principle_retained / m.adapt_cells
            m.historical_detail_carryover_rate = m.detail_carryover / m.adapt_cells
        return m

    @property
    def beats_blanket_refusal(self) -> bool:
        """Whether the system did better than answering nothing at all.

        A negative here invalidates any claim about applicability handling,
        however good the headline accuracy looks.
        """
        return self.applicability_classification_accuracy > self.always_refuse_baseline

    def summary(self) -> str:
        return (
            f"applicability classification : {self.applicability_classification_accuracy:.1%}"
            f"  (blanket-refusal baseline {self.always_refuse_baseline:.1%}"
            f" — {'beats' if self.beats_blanket_refusal else 'DOES NOT BEAT'} it)\n"
            f"correct rejection            : {self.correct_rejection_rate:.1%}"
            f"  [n={self.reject_cells}]\n"
            f"false refusal                : {self.false_refusal_rate:.1%}"
            f"  [n={self.use_cells}]  (never blended with the line above)\n"
            f"principle transfer           : {self.principle_transfer_rate:.1%}"
            f"  [n={self.adapt_cells}]\n"
            f"historical detail carryover  : {self.historical_detail_carryover_rate:.1%}"
            f"  [n={self.adapt_cells}]"
        )


def expected_from_timelines(timelines) -> tuple[dict, dict, dict]:
    """Extract applicability ground truth keyed by (timeline_id, query_idx)."""
    expected, principles, obsolete = {}, {}, {}
    for timeline in timelines:
        idx = 0
        for event in timeline.events:
            gt = getattr(event, "ground_truth", None)
            if gt is None:
                continue
            key = (timeline.id, idx)
            if gt.expected_applicability:
                expected[key] = gt.expected_applicability
                principles[key] = gt.transferable_principle
                obsolete[key] = list(gt.obsolete_details)
            idx += 1
    return expected, principles, obsolete
