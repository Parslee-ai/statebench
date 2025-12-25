"""Implicit resurrection and action correctness metrics.

This module detects subtle forms of superseded fact usage:
1. Implicit Resurrection: When the response assumes superseded facts are true
   without explicitly mentioning them
2. Action Correctness: Whether recommended actions align with current state

Example:
- Old fact: "Meeting is at 2pm"
- New fact: "Meeting moved to 4pm"
- Response: "You should leave now to arrive on time"
  -> This is implicit resurrection - the action only makes sense if 2pm is true

These metrics catch cases where must_not_mention passes but the response
is still wrong because it implicitly relies on superseded information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from statebench.evaluation.metrics import QueryResult


@dataclass
class ImplicitResurrectionInstance:
    """An instance where superseded info is implicitly used."""
    timeline_id: str
    query_idx: int
    evidence: str  # What in the response suggests resurrection
    superseded_fact: str  # The fact that should have been forgotten
    detection_method: Literal["action_mismatch", "reasoning_trace", "temporal_error", "value_echo"]


@dataclass
class ActionCorrectnessResult:
    """Result of action correctness analysis for a query."""
    timeline_id: str
    query_idx: int
    recommended_action: str | None
    action_correct: bool
    action_basis: str  # "current_state", "superseded_state", or "unclear"
    explanation: str


@dataclass
class ResurrectionMetrics:
    """Metrics for implicit resurrection detection."""
    total_queries: int = 0

    # Explicit resurrection (from must_not_mention)
    explicit_resurrections: int = 0

    # Implicit resurrection instances
    implicit_resurrections: list[ImplicitResurrectionInstance] = field(default_factory=list)
    implicit_resurrection_count: int = 0

    # Breakdown by detection method
    action_mismatch_count: int = 0
    reasoning_trace_count: int = 0
    temporal_error_count: int = 0
    value_echo_count: int = 0

    # Action correctness
    action_results: list[ActionCorrectnessResult] = field(default_factory=list)
    total_actions: int = 0
    correct_actions: int = 0
    action_accuracy: float = 0.0

    # Combined resurrection rate (explicit + implicit)
    total_resurrection_rate: float = 0.0


# Patterns that suggest action recommendations
ACTION_PATTERNS = [
    "you should",
    "I recommend",
    "I suggest",
    "consider",
    "proceed with",
    "go ahead and",
    "please",
    "don't forget to",
    "make sure to",
    "you'll need to",
    "you can",
    "try to",
]

# Patterns that suggest temporal reasoning
TEMPORAL_PATTERNS = [
    "still",
    "remains",
    "continues to be",
    "currently",
    "as of now",
    "at this time",
    "ongoing",
]


def extract_action(response: str) -> str | None:
    """Extract action recommendation from a response if present."""
    response_lower = response.lower()

    for pattern in ACTION_PATTERNS:
        if pattern in response_lower:
            # Find the sentence containing the action
            start = response_lower.find(pattern)
            # Find sentence boundaries
            sentence_start = response.rfind(".", 0, start)
            if sentence_start == -1:
                sentence_start = 0
            else:
                sentence_start += 1

            sentence_end = response.find(".", start)
            if sentence_end == -1:
                sentence_end = len(response)
            else:
                sentence_end += 1

            return response[sentence_start:sentence_end].strip()

    return None


def detect_value_echo(
    response: str,
    superseded_values: list[str],
    current_values: list[str],
) -> list[str]:
    """Detect if response echoes superseded values in a transformed way.

    For example:
    - Superseded: "$50,000"
    - Response mentions "$50k" or "fifty thousand" -> value echo
    """
    echoes = []
    response_lower = response.lower()

    # Check for numeric transformations
    for value in superseded_values:
        # Skip if current values contain similar patterns
        if any(v in value for v in current_values):
            continue

        # Extract numbers from the value
        import re
        numbers = re.findall(r'\d+', value)

        for num in numbers:
            # Check for abbreviated forms
            if len(num) >= 4:
                # e.g., 50000 -> 50k
                abbreviated = num[:-3] + "k"
                if abbreviated in response_lower:
                    echoes.append(f"Value echo: {value} -> {abbreviated}")

    return echoes


def detect_temporal_confusion(
    response: str,
    supersession_time: str | None,
) -> bool:
    """Detect if response shows temporal confusion about state validity."""
    response_lower = response.lower()

    # Check for patterns that suggest treating old state as current
    confusion_patterns = [
        "still valid",
        "still applies",
        "hasn't changed",
        "remains in effect",
        "as previously",
    ]

    for pattern in confusion_patterns:
        if pattern in response_lower:
            return True

    return False


def analyze_action_correctness(
    response: str,
    current_decision: str,
    superseded_decision: str | None,
) -> tuple[bool, str]:
    """Analyze if an action recommendation aligns with current or superseded state.

    Returns:
        Tuple of (is_correct, basis)
    """
    action = extract_action(response)
    if not action:
        return True, "no_action"  # No action to evaluate

    action_lower = action.lower()

    # Check alignment with decisions
    if current_decision.lower() in ["yes", "approve", "proceed"]:
        if any(word in action_lower for word in ["proceed", "go ahead", "approve", "yes"]):
            return True, "current_state"
        if any(word in action_lower for word in ["cancel", "stop", "don't", "no"]):
            return False, "superseded_state"

    elif current_decision.lower() in ["no", "deny", "cancel"]:
        if any(word in action_lower for word in ["cancel", "stop", "don't", "deny", "no"]):
            return True, "current_state"
        if any(word in action_lower for word in ["proceed", "go ahead", "approve", "yes"]):
            return False, "superseded_state"

    return True, "unclear"


def compute_resurrection_metrics(
    query_results: list[QueryResult],
    supersession_context: dict[str, dict[str, object]],
) -> ResurrectionMetrics:
    """Compute resurrection metrics from query results.

    Args:
        query_results: List of QueryResult objects
        supersession_context: Map of timeline_id -> {
            "superseded_values": list[str],
            "current_values": list[str],
            "superseded_decision": str,
            "current_decision": str,
        }

    Returns:
        ResurrectionMetrics with implicit resurrection analysis
    """
    metrics = ResurrectionMetrics()

    # Filter to supersession-relevant tracks
    relevant_tracks = {"supersession", "commitment_durability", "environmental_freshness"}
    relevant_results = [r for r in query_results if r.track in relevant_tracks]
    metrics.total_queries = len(relevant_results)

    for result in relevant_results:
        timeline_id = result.timeline_id
        context = supersession_context.get(timeline_id, {})

        # Count explicit resurrections (already captured in must_not_mention_violations)
        if result.resurrected_superseded:
            metrics.explicit_resurrections += 1

        # Check for implicit resurrection via value echoes
        superseded_values = context.get("superseded_values")
        current_values = context.get("current_values")
        if superseded_values and current_values:
            superseded_list = list(superseded_values) if isinstance(superseded_values, list) else []
            current_list = list(current_values) if isinstance(current_values, list) else []
            echoes = detect_value_echo(result.response, superseded_list, current_list)
            for echo in echoes:
                instance = ImplicitResurrectionInstance(
                    timeline_id=timeline_id,
                    query_idx=result.query_idx,
                    evidence=echo,
                    superseded_fact=str(superseded_values),
                    detection_method="value_echo",
                )
                metrics.implicit_resurrections.append(instance)
                metrics.implicit_resurrection_count += 1
                metrics.value_echo_count += 1

        # Check for temporal confusion
        if detect_temporal_confusion(result.response, None):
            instance = ImplicitResurrectionInstance(
                timeline_id=timeline_id,
                query_idx=result.query_idx,
                evidence="Temporal confusion patterns detected",
                superseded_fact="Unknown",
                detection_method="temporal_error",
            )
            metrics.implicit_resurrections.append(instance)
            metrics.implicit_resurrection_count += 1
            metrics.temporal_error_count += 1

        # Analyze action correctness
        current_decision = context.get("current_decision")
        superseded_decision = context.get("superseded_decision")
        if current_decision and superseded_decision:
            current_str = str(current_decision)
            superseded_str = str(superseded_decision) if superseded_decision else None
            is_correct, basis = analyze_action_correctness(
                result.response,
                current_str,
                superseded_str,
            )

            action = extract_action(result.response)
            if action:
                metrics.total_actions += 1
                action_result = ActionCorrectnessResult(
                    timeline_id=timeline_id,
                    query_idx=result.query_idx,
                    recommended_action=action,
                    action_correct=is_correct,
                    action_basis=basis,
                    explanation=f"Action based on {basis}",
                )
                metrics.action_results.append(action_result)

                if is_correct:
                    metrics.correct_actions += 1
                else:
                    # Action mismatch = implicit resurrection
                    instance = ImplicitResurrectionInstance(
                        timeline_id=timeline_id,
                        query_idx=result.query_idx,
                        evidence=f"Action '{action[:100]}' aligned with superseded state",
                        superseded_fact=str(superseded_decision) if superseded_decision else "",
                        detection_method="action_mismatch",
                    )
                    metrics.implicit_resurrections.append(instance)
                    metrics.implicit_resurrection_count += 1
                    metrics.action_mismatch_count += 1

    # Compute rates
    if metrics.total_actions > 0:
        metrics.action_accuracy = metrics.correct_actions / metrics.total_actions

    if metrics.total_queries > 0:
        total_resurrections = metrics.explicit_resurrections + metrics.implicit_resurrection_count
        metrics.total_resurrection_rate = total_resurrections / metrics.total_queries

    return metrics


def format_resurrection_report(metrics: ResurrectionMetrics) -> str:
    """Format resurrection metrics as a readable report."""
    lines = [
        "# Resurrection Analysis Report",
        "",
        "## Summary",
        f"- Total queries analyzed: {metrics.total_queries}",
        f"- **Total resurrection rate: {metrics.total_resurrection_rate:.2%}**",
        "",
        "## Explicit Resurrection",
        f"- Count: {metrics.explicit_resurrections}",
        f"- Rate: {metrics.explicit_resurrections / max(metrics.total_queries, 1):.2%}",
        "",
        "## Implicit Resurrection",
        f"- Count: {metrics.implicit_resurrection_count}",
        f"- Rate: {metrics.implicit_resurrection_count / max(metrics.total_queries, 1):.2%}",
        "",
        "### Detection Method Breakdown",
        f"- Action mismatch: {metrics.action_mismatch_count}",
        f"- Reasoning trace: {metrics.reasoning_trace_count}",
        f"- Temporal error: {metrics.temporal_error_count}",
        f"- Value echo: {metrics.value_echo_count}",
        "",
        "## Action Correctness",
        f"- Total actions analyzed: {metrics.total_actions}",
        f"- Correct actions: {metrics.correct_actions}",
        f"- **Action accuracy: {metrics.action_accuracy:.2%}**",
    ]

    if metrics.implicit_resurrections:
        lines.extend([
            "",
            "## Implicit Resurrection Instances (sample)",
        ])
        for instance in metrics.implicit_resurrections[:5]:
            lines.append(
                f"- [{instance.detection_method}] {instance.timeline_id}: "
                f"{instance.evidence[:80]}..."
            )

    return "\n".join(lines)
