"""Extended metrics for StateBench v1.0.

This module provides advanced scoring capabilities:
1. Cost-weighted scoring (severity-based error weighting)
2. Correction latency (turns until correction is reflected)
3. Composite StateBench score (unified benchmark score)
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from statebench.evaluation.detection_metrics import DetectionMetrics
    from statebench.evaluation.metrics import BenchmarkMetrics, QueryResult
    from statebench.schema.timeline import Timeline


# =============================================================================
# 4.1 Cost-Weighted Scoring
# =============================================================================

SEVERITY_WEIGHTS: dict[str, float] = {
    "low": 1.0,
    "medium": 2.0,
    "high": 5.0,
    "critical": 10.0,
}


@dataclass
class CostWeightedMetrics:
    """Cost-weighted error metrics.

    Weights errors by their business severity to provide a more
    realistic assessment of system quality.
    """
    raw_error_count: int = 0
    weighted_error_sum: float = 0.0
    max_possible_weight: float = 0.0
    cost_weighted_score: float = 1.0  # 1 - (weighted_errors / max_weight)

    # Breakdown by severity
    errors_by_severity: dict[str, int] = field(default_factory=dict)
    weight_by_severity: dict[str, float] = field(default_factory=dict)


def compute_cost_weighted_metrics(
    results: list["QueryResult"],
    severities: list[str],
) -> CostWeightedMetrics:
    """Compute cost-weighted error metrics.

    Args:
        results: List of query results
        severities: List of severity levels for each result
                   (must be same length as results)

    Returns:
        CostWeightedMetrics with weighted scores
    """
    if len(results) != len(severities):
        raise ValueError("results and severities must have same length")

    metrics = CostWeightedMetrics()
    metrics.errors_by_severity = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    metrics.weight_by_severity = {"low": 0.0, "medium": 0.0, "high": 0.0, "critical": 0.0}

    for result, severity in zip(results, severities):
        weight = SEVERITY_WEIGHTS.get(severity, 2.0)
        metrics.max_possible_weight += weight

        # Count as error if decision wrong OR must_not_mention violated
        is_error = not result.decision_correct or bool(result.must_not_mention_violations)

        if is_error:
            metrics.raw_error_count += 1
            metrics.weighted_error_sum += weight
            metrics.errors_by_severity[severity] = (
                metrics.errors_by_severity.get(severity, 0) + 1
            )
            metrics.weight_by_severity[severity] = (
                metrics.weight_by_severity.get(severity, 0.0) + weight
            )

    # Compute final score
    if metrics.max_possible_weight > 0:
        metrics.cost_weighted_score = 1.0 - (
            metrics.weighted_error_sum / metrics.max_possible_weight
        )

    return metrics


# =============================================================================
# 4.2 Correction Latency
# =============================================================================

@dataclass
class CorrectionLatencyMetrics:
    """Metrics for how quickly corrections are reflected in responses.

    Measures the "lag" between when a supersession occurs and when
    the system's behavior reflects the new state.
    """
    corrections_tested: int = 0
    avg_latency_turns: float = 0.0
    max_latency_turns: int = 0
    min_latency_turns: int = 0
    immediate_corrections: int = 0  # Latency = 0 (corrected immediately)

    # Distribution
    latency_distribution: dict[int, int] = field(default_factory=dict)


@dataclass
class CorrectionEvent:
    """Represents a correction event in a timeline."""
    turn_index: int  # When the correction occurred
    old_value: str
    new_value: str
    fact_id: str | None = None


def extract_corrections(timeline: "Timeline") -> list[CorrectionEvent]:
    """Extract correction events from a timeline.

    Args:
        timeline: The timeline to analyze

    Returns:
        List of CorrectionEvent objects
    """
    from statebench.schema.timeline import Supersession

    corrections = []
    turn_counter = 0

    for event in timeline.events:
        # Track conversation turns
        if hasattr(event, "speaker"):
            turn_counter += 1

        # Identify supersession events
        if isinstance(event, Supersession):
            for write in event.writes:
                if write.supersedes:
                    corrections.append(CorrectionEvent(
                        turn_index=turn_counter,
                        old_value=write.supersedes,  # The superseded key
                        new_value=write.value,
                        fact_id=getattr(write, "id", None),
                    ))

    return corrections


def compute_correction_latency(
    timelines: list["Timeline"],
    responses_per_timeline: list[list[str]],
) -> CorrectionLatencyMetrics:
    """Compute correction latency metrics.

    For each timeline with a correction, measures how many turns
    after the correction until the response reflects the new value.

    Args:
        timelines: List of timelines with potential corrections
        responses_per_timeline: List of response lists, one per timeline.
                               Each inner list has responses for each query.

    Returns:
        CorrectionLatencyMetrics
    """
    metrics = CorrectionLatencyMetrics()
    all_latencies: list[int] = []

    for timeline, responses in zip(timelines, responses_per_timeline):
        corrections = extract_corrections(timeline)

        for correction in corrections:
            metrics.corrections_tested += 1

            # Find when the new value first appears in responses
            latency = _find_latency(
                correction,
                responses,
                correction.turn_index,
            )

            if latency is not None:
                all_latencies.append(latency)

                # Update distribution
                metrics.latency_distribution[latency] = (
                    metrics.latency_distribution.get(latency, 0) + 1
                )

                if latency == 0:
                    metrics.immediate_corrections += 1

    # Compute aggregates
    if all_latencies:
        metrics.avg_latency_turns = sum(all_latencies) / len(all_latencies)
        metrics.max_latency_turns = max(all_latencies)
        metrics.min_latency_turns = min(all_latencies)

    return metrics


def _find_latency(
    correction: CorrectionEvent,
    responses: list[str],
    correction_turn: int,
) -> int | None:
    """Find how many turns until correction is reflected.

    Returns None if correction is never reflected.
    """
    new_value_lower = correction.new_value.lower()

    # Check each response after the correction turn
    for i, response in enumerate(responses):
        response_lower = response.lower()

        # Check if new value appears in response
        if new_value_lower in response_lower:
            # Latency is the number of turns after correction
            # (Assuming responses are indexed by query number)
            return max(0, i - correction_turn)

    return None  # Never reflected


# =============================================================================
# 4.3 Composite StateBench Score
# =============================================================================

@dataclass
class ProvenanceMetrics:
    """Provenance-related metrics for composite scoring.

    These metrics track how well the system handles fact attribution
    and respects scope/authority boundaries.
    """
    accuracy: float = 0.0  # Overall provenance accuracy
    authority_violation_rate: float = 0.0  # Rate of authority violations
    scope_violation_rate: float = 0.0  # Rate of scope leaks
    superseded_fact_usage_rate: float = 0.0  # Rate of using invalid facts


@dataclass
class StateBenchScore:
    """Composite StateBench v1.0 score.

    Combines multiple metrics into a single score with configurable weights.
    """
    # Component scores (0-100 scale)
    decision_accuracy: float = 0.0
    detection_f1: float = 0.0
    provenance_accuracy: float = 0.0
    sfrr_inverted: float = 0.0  # 100 - SFRR (higher is better)
    must_mention_rate: float = 0.0
    context_efficiency: float = 0.0

    # Composite scores
    overall_score: float = 0.0
    risk_adjusted_score: float = 0.0

    # Breakdown by track
    track_scores: dict[str, float] = field(default_factory=dict)

    # Weights used
    weights: dict[str, float] = field(default_factory=dict)


# Default weights for composite scoring
DEFAULT_WEIGHTS: dict[str, float] = {
    "decision_accuracy": 0.25,
    "detection_f1": 0.20,
    "provenance_accuracy": 0.20,
    "sfrr_inverted": 0.15,
    "must_mention_rate": 0.10,
    "context_efficiency": 0.10,
}


def compute_context_efficiency(
    tokens_used: int,
    token_budget: int,
    accuracy: float,
) -> float:
    """Compute context efficiency score.

    Measures how efficiently the system uses its token budget
    while maintaining accuracy.

    Args:
        tokens_used: Average tokens used per query
        token_budget: Maximum token budget
        accuracy: Decision accuracy (0-1)

    Returns:
        Efficiency score (0-1)
    """
    if token_budget == 0:
        return 0.0

    # Efficiency = accuracy * (1 - tokens_used/budget)
    # This rewards high accuracy with low token usage
    token_efficiency = 1.0 - min(1.0, tokens_used / token_budget)

    # Weight accuracy more heavily (70/30 split)
    return 0.7 * accuracy + 0.3 * token_efficiency


def compute_track_scores(
    metrics: "BenchmarkMetrics",
) -> dict[str, float]:
    """Compute per-track scores.

    Args:
        metrics: Benchmark metrics with track breakdown

    Returns:
        Dict of track name to score (0-100)
    """
    track_scores: dict[str, float] = {}

    for track_name, track_metrics in metrics.tracks.items():
        # Simple composite: accuracy * (1 - sfrr) * must_mention_rate
        base_score = track_metrics.decision_accuracy * 100

        # Penalize for resurrection
        resurrection_penalty = track_metrics.sfrr * 50

        # Bonus for must_mention compliance
        mention_bonus = track_metrics.must_mention_rate * 20

        # Penalty for must_not_mention violations
        violation_penalty = track_metrics.must_not_mention_violation_rate * 30

        track_scores[track_name] = max(
            0.0,
            min(
                100.0,
                base_score - resurrection_penalty + mention_bonus - violation_penalty
            )
        )

    return track_scores


def compute_statebench_score(
    metrics: "BenchmarkMetrics",
    provenance_metrics: ProvenanceMetrics | None = None,
    detection_metrics: "DetectionMetrics | None" = None,
    weights: dict[str, float] | None = None,
) -> StateBenchScore:
    """Compute composite StateBench v1.0 score.

    Args:
        metrics: Core benchmark metrics
        provenance_metrics: Optional provenance metrics (defaults to estimates)
        detection_metrics: Optional detection metrics (defaults to 0)
        weights: Optional custom weights (defaults to DEFAULT_WEIGHTS)

    Returns:
        StateBenchScore with overall and component scores
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    # Use provided or default provenance metrics
    if provenance_metrics is None:
        provenance_metrics = ProvenanceMetrics(
            accuracy=1.0 - metrics.overall_sfrr,  # Estimate from SFRR
            authority_violation_rate=0.0,
            scope_violation_rate=0.0,
        )

    # Get detection F1 or default to 0
    detection_f1 = 0.0
    if detection_metrics is not None:
        detection_f1 = detection_metrics.avg_f1

    # Compute context efficiency
    context_efficiency = compute_context_efficiency(
        tokens_used=int(metrics.avg_tokens_per_query),
        token_budget=metrics.token_budget,
        accuracy=metrics.overall_decision_accuracy,
    )

    # Build component scores (0-100 scale)
    components = {
        "decision_accuracy": metrics.overall_decision_accuracy * 100,
        "detection_f1": detection_f1 * 100,
        "provenance_accuracy": provenance_metrics.accuracy * 100,
        "sfrr_inverted": (1 - metrics.overall_sfrr) * 100,
        "must_mention_rate": metrics.overall_must_mention_rate * 100,
        "context_efficiency": context_efficiency * 100,
    }

    # Compute weighted overall score
    overall = sum(
        components.get(k, 0.0) * weights.get(k, 0.0)
        for k in weights
    )

    # Risk-adjusted score penalizes critical failures
    risk_penalty = (
        provenance_metrics.authority_violation_rate * 50 +
        provenance_metrics.scope_violation_rate * 30
    )
    risk_adjusted = max(0.0, overall - risk_penalty)

    # Compute track scores
    track_scores = compute_track_scores(metrics)

    return StateBenchScore(
        decision_accuracy=components["decision_accuracy"],
        detection_f1=components["detection_f1"],
        provenance_accuracy=components["provenance_accuracy"],
        sfrr_inverted=components["sfrr_inverted"],
        must_mention_rate=components["must_mention_rate"],
        context_efficiency=components["context_efficiency"],
        overall_score=overall,
        risk_adjusted_score=risk_adjusted,
        track_scores=track_scores,
        weights=weights,
    )


# =============================================================================
# Formatting
# =============================================================================

def format_cost_weighted_metrics(metrics: CostWeightedMetrics) -> str:
    """Format cost-weighted metrics as markdown."""
    lines = [
        "## Cost-Weighted Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Raw Error Count | {metrics.raw_error_count} |",
        f"| Weighted Error Sum | {metrics.weighted_error_sum:.2f} |",
        f"| Max Possible Weight | {metrics.max_possible_weight:.2f} |",
        f"| Cost-Weighted Score | {metrics.cost_weighted_score:.2%} |",
        "",
        "### Errors by Severity",
        "",
    ]

    for severity in ["critical", "high", "medium", "low"]:
        count = metrics.errors_by_severity.get(severity, 0)
        weight = metrics.weight_by_severity.get(severity, 0.0)
        lines.append(f"- {severity.title()}: {count} errors ({weight:.1f} weighted)")

    return "\n".join(lines)


def format_correction_latency(metrics: CorrectionLatencyMetrics) -> str:
    """Format correction latency metrics as markdown."""
    lines = [
        "## Correction Latency",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Corrections Tested | {metrics.corrections_tested} |",
        f"| Average Latency | {metrics.avg_latency_turns:.2f} turns |",
        f"| Max Latency | {metrics.max_latency_turns} turns |",
        f"| Immediate Corrections | {metrics.immediate_corrections} |",
    ]

    if metrics.latency_distribution:
        lines.extend([
            "",
            "### Latency Distribution",
            "",
        ])
        for latency in sorted(metrics.latency_distribution.keys()):
            count = metrics.latency_distribution[latency]
            lines.append(f"- {latency} turns: {count} corrections")

    return "\n".join(lines)


def format_statebench_score(score: StateBenchScore) -> str:
    """Format StateBench score as markdown."""
    lines = [
        "## StateBench v1.0 Score",
        "",
        f"### Overall Score: {score.overall_score:.1f}/100",
        f"### Risk-Adjusted Score: {score.risk_adjusted_score:.1f}/100",
        "",
        "### Component Scores",
        "",
        "| Component | Score | Weight |",
        "|-----------|-------|--------|",
    ]

    components = [
        ("Decision Accuracy", score.decision_accuracy, "decision_accuracy"),
        ("Detection F1", score.detection_f1, "detection_f1"),
        ("Provenance Accuracy", score.provenance_accuracy, "provenance_accuracy"),
        ("SFRR (inverted)", score.sfrr_inverted, "sfrr_inverted"),
        ("Must Mention Rate", score.must_mention_rate, "must_mention_rate"),
        ("Context Efficiency", score.context_efficiency, "context_efficiency"),
    ]

    for name, value, key in components:
        weight = score.weights.get(key, 0.0)
        lines.append(f"| {name} | {value:.1f} | {weight:.0%} |")

    if score.track_scores:
        lines.extend([
            "",
            "### Track Scores",
            "",
        ])
        for track, track_score in sorted(score.track_scores.items()):
            lines.append(f"- {track}: {track_score:.1f}/100")

    return "\n".join(lines)
