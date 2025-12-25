"""Evaluation and scoring for StateBench."""

# v1.0: Detection metrics
from statebench.evaluation.detection_metrics import (
    DetectionMetrics,
    DetectionResult,
    DetectionScorer,
    format_detection_metrics,
)

# v1.0: Extended metrics
from statebench.evaluation.extended_metrics import (
    DEFAULT_WEIGHTS,
    SEVERITY_WEIGHTS,
    CorrectionEvent,
    CorrectionLatencyMetrics,
    CostWeightedMetrics,
    ProvenanceMetrics,
    StateBenchScore,
    compute_context_efficiency,
    compute_correction_latency,
    compute_cost_weighted_metrics,
    compute_statebench_score,
    compute_track_scores,
    extract_corrections,
    format_correction_latency,
    format_cost_weighted_metrics,
    format_statebench_score,
)
from statebench.evaluation.judge import ResponseJudge, create_judge
from statebench.evaluation.metrics import (
    BenchmarkMetrics,
    MetricsAggregator,
    QueryResult,
    TrackMetrics,
    format_metrics_table,
)
from statebench.evaluation.resurrection_metrics import (
    ActionCorrectnessResult,
    ImplicitResurrectionInstance,
    ResurrectionMetrics,
    compute_resurrection_metrics,
    format_resurrection_report,
)
from statebench.evaluation.rubric import ScoringRubric, contains_phrase, extract_decision
from statebench.evaluation.track4_metrics import (
    FalseRefusalInstance,
    LeakageInstance,
    Track4Metrics,
    compute_track4_metrics,
    format_track4_report,
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
    # v1.0: Detection metrics
    "DetectionMetrics",
    "DetectionResult",
    "DetectionScorer",
    "format_detection_metrics",
    # v1.0: Extended metrics
    "CorrectionEvent",
    "CorrectionLatencyMetrics",
    "CostWeightedMetrics",
    "DEFAULT_WEIGHTS",
    "ProvenanceMetrics",
    "SEVERITY_WEIGHTS",
    "StateBenchScore",
    "compute_context_efficiency",
    "compute_correction_latency",
    "compute_cost_weighted_metrics",
    "compute_statebench_score",
    "compute_track_scores",
    "extract_corrections",
    "format_correction_latency",
    "format_cost_weighted_metrics",
    "format_statebench_score",
]
