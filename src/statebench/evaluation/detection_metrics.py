"""Detection metrics for StateBench v1.0.

This module provides metrics for evaluating supersession detection
from natural language, not just handling of explicit supersession events.

Key Metrics:
- Detection Precision: Of detected supersessions, how many were correct?
- Detection Recall: Of actual supersessions, how many were detected?
- Detection F1: Harmonic mean of precision and recall
"""

from dataclasses import dataclass, field


@dataclass
class DetectionResult:
    """Result of detection scoring for a single query."""
    timeline_id: str
    query_idx: int

    # Detection ground truth
    expected_supersessions: list[str]  # Fact IDs that should be detected as superseded

    # Detection inference from response behavior
    detected_supersessions: list[str]  # Fact IDs inferred as detected
    false_supersessions: list[str]  # Incorrectly detected supersessions

    # Computed metrics
    detection_precision: float = 0.0
    detection_recall: float = 0.0
    detection_f1: float = 0.0

    # Detection evidence
    detection_evidence: str = ""  # How we inferred detection


@dataclass
class DetectionMetrics:
    """Aggregated detection metrics."""
    total_cases: int = 0

    # Aggregate metrics (macro-averaged)
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    avg_f1: float = 0.0

    # Micro-averaged (pooled counts)
    total_expected: int = 0
    total_detected: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    micro_precision: float = 0.0
    micro_recall: float = 0.0
    micro_f1: float = 0.0

    # By difficulty
    by_difficulty: dict[str, "DetectionMetrics"] = field(default_factory=dict)

    # By cue type
    by_cue_type: dict[str, "DetectionMetrics"] = field(default_factory=dict)


class DetectionScorer:
    """Scores detection performance from response behavior."""

    def __init__(self) -> None:
        self.results: list[DetectionResult] = []

    def score_detection(
        self,
        timeline_id: str,
        query_idx: int,
        response: str,
        must_mention: list[str],
        must_not_mention: list[str],
        expected_supersessions: list[str],
    ) -> DetectionResult:
        """Score detection based on response behavior.

        Detection is inferred from:
        1. Presence/absence of must_not_mention items (superseded values)
        2. Presence of must_mention items (current values)
        3. Explicit acknowledgment of corrections

        Args:
            timeline_id: ID of the timeline
            query_idx: Index of the query in the timeline
            response: The LLM response text
            must_mention: Phrases that should appear (new values)
            must_not_mention: Phrases that should NOT appear (old values)
            expected_supersessions: Fact IDs that should be detected as superseded
        """
        response_lower = response.lower()

        # Infer detected supersessions from response behavior
        detected: set[str] = set()
        evidence_parts: list[str] = []

        # If must_not_mention items are absent, supersession was likely detected
        for phrase in must_not_mention:
            if phrase.lower() not in response_lower:
                # Successfully avoided the superseded value
                evidence_parts.append(f"Avoided superseded phrase: '{phrase}'")
                # Associate with any expected supersession
                if expected_supersessions:
                    detected.add(expected_supersessions[0])

        # If must_mention items are present, current state is being used
        for phrase in must_mention:
            if phrase.lower() in response_lower:
                evidence_parts.append(f"Used current value: '{phrase}'")
                # This also indicates detection
                if expected_supersessions:
                    detected.add(expected_supersessions[0])

        # Check for explicit correction acknowledgment
        correction_signals = [
            "you mentioned",
            "you said earlier",
            "originally",
            "changed to",
            "updated to",
            "instead of",
            "correction",
            "actually",
        ]
        for signal in correction_signals:
            if signal in response_lower:
                evidence_parts.append(f"Explicit acknowledgment: '{signal}'")
                if expected_supersessions:
                    detected.add(expected_supersessions[0])
                break

        # Compute metrics
        expected_set = set(expected_supersessions)
        detected_set = detected

        true_positives = expected_set & detected_set
        false_positives = detected_set - expected_set

        precision = (
            len(true_positives) / len(detected_set) if detected_set else 1.0
        )
        recall = (
            len(true_positives) / len(expected_set) if expected_set else 1.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        result = DetectionResult(
            timeline_id=timeline_id,
            query_idx=query_idx,
            expected_supersessions=expected_supersessions,
            detected_supersessions=list(detected_set),
            false_supersessions=list(false_positives),
            detection_precision=precision,
            detection_recall=recall,
            detection_f1=f1,
            detection_evidence="; ".join(evidence_parts),
        )

        self.results.append(result)
        return result

    def compute_metrics(self) -> DetectionMetrics:
        """Compute aggregate detection metrics."""
        if not self.results:
            return DetectionMetrics()

        metrics = DetectionMetrics(total_cases=len(self.results))

        # Macro-averaged (average of per-case metrics)
        total_precision = sum(r.detection_precision for r in self.results)
        total_recall = sum(r.detection_recall for r in self.results)
        total_f1 = sum(r.detection_f1 for r in self.results)

        metrics.avg_precision = total_precision / len(self.results)
        metrics.avg_recall = total_recall / len(self.results)
        metrics.avg_f1 = total_f1 / len(self.results)

        # Micro-averaged (pooled counts)
        for result in self.results:
            expected_set = set(result.expected_supersessions)
            detected_set = set(result.detected_supersessions)

            metrics.total_expected += len(expected_set)
            metrics.total_detected += len(detected_set)

            true_positives = expected_set & detected_set
            metrics.true_positives += len(true_positives)
            metrics.false_positives += len(detected_set - expected_set)
            metrics.false_negatives += len(expected_set - detected_set)

        if metrics.total_detected > 0:
            metrics.micro_precision = metrics.true_positives / metrics.total_detected

        if metrics.total_expected > 0:
            metrics.micro_recall = metrics.true_positives / metrics.total_expected

        if (metrics.micro_precision + metrics.micro_recall) > 0:
            metrics.micro_f1 = (
                2 * metrics.micro_precision * metrics.micro_recall
                / (metrics.micro_precision + metrics.micro_recall)
            )

        return metrics


def format_detection_metrics(metrics: DetectionMetrics) -> str:
    """Format detection metrics as a markdown table."""
    lines = [
        "## Detection Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Cases | {metrics.total_cases} |",
        f"| Macro Precision | {metrics.avg_precision:.2%} |",
        f"| Macro Recall | {metrics.avg_recall:.2%} |",
        f"| Macro F1 | {metrics.avg_f1:.2%} |",
        f"| Micro Precision | {metrics.micro_precision:.2%} |",
        f"| Micro Recall | {metrics.micro_recall:.2%} |",
        f"| Micro F1 | {metrics.micro_f1:.2%} |",
        "",
        "### Counts",
        "",
        f"- Expected supersessions: {metrics.total_expected}",
        f"- Detected supersessions: {metrics.total_detected}",
        f"- True positives: {metrics.true_positives}",
        f"- False positives: {metrics.false_positives}",
        f"- False negatives: {metrics.false_negatives}",
    ]

    return "\n".join(lines)
