"""Evaluation and scoring for StateBench."""

from statebench.evaluation.metrics import (
    QueryResult,
    TrackMetrics,
    BenchmarkMetrics,
    MetricsAggregator,
    format_metrics_table,
)
from statebench.evaluation.rubric import ScoringRubric, contains_phrase, extract_decision
from statebench.evaluation.judge import ResponseJudge, create_judge
from statebench.evaluation.track4_metrics import (
    Track4Metrics,
    LeakageInstance,
    FalseRefusalInstance,
    compute_track4_metrics,
    format_track4_report,
)
from statebench.evaluation.resurrection_metrics import (
    ResurrectionMetrics,
    ImplicitResurrectionInstance,
    ActionCorrectnessResult,
    compute_resurrection_metrics,
    format_resurrection_report,
)

__all__ = [
    "QueryResult",
    "TrackMetrics",
    "BenchmarkMetrics",
    "MetricsAggregator",
    "format_metrics_table",
    "ScoringRubric",
    "contains_phrase",
    "extract_decision",
    "ResponseJudge",
    "create_judge",
    # Track 4 metrics
    "Track4Metrics",
    "LeakageInstance",
    "FalseRefusalInstance",
    "compute_track4_metrics",
    "format_track4_report",
    # Resurrection metrics
    "ResurrectionMetrics",
    "ImplicitResurrectionInstance",
    "ActionCorrectnessResult",
    "compute_resurrection_metrics",
    "format_resurrection_report",
]
