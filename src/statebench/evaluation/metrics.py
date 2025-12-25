"""Metrics for StateBench evaluation.

Primary metrics:
- SFRR: Superseded Fact Resurrection Rate
- Decision Accuracy
- Constraint Compliance (must_mention, must_not_mention)
- Source Policy Violations

Secondary metrics:
- Token budget used
- Latency (if measured)
"""

from dataclasses import dataclass, field


@dataclass
class QueryResult:
    """Result of evaluating a single query."""
    timeline_id: str
    query_idx: int
    track: str
    domain: str

    # Ground truth
    expected_decision: str
    must_mention: list[str]
    must_not_mention: list[str]

    # Model response
    response: str
    actual_decision: str | None = None

    # Scores
    decision_correct: bool = False
    must_mention_hits: list[str] = field(default_factory=list)
    must_mention_misses: list[str] = field(default_factory=list)
    must_not_mention_violations: list[str] = field(default_factory=list)
    source_violations: list[str] = field(default_factory=list)

    # Derived
    resurrected_superseded: bool = False  # Did the response mention superseded facts?

    # Metadata
    tokens_used: int = 0
    latency_ms: int = 0


@dataclass
class TrackMetrics:
    """Aggregated metrics for a track."""
    track: str
    total_queries: int = 0

    # Primary metrics
    sfrr: float = 0.0  # Superseded Fact Resurrection Rate
    decision_accuracy: float = 0.0
    must_mention_rate: float = 0.0
    must_not_mention_violation_rate: float = 0.0
    source_violation_rate: float = 0.0

    # Counts
    correct_decisions: int = 0
    total_must_mention: int = 0
    must_mention_hits: int = 0
    total_must_not_mention: int = 0
    must_not_mention_violations: int = 0
    resurrection_count: int = 0

    # Secondary
    avg_tokens: float = 0.0
    avg_latency_ms: float = 0.0


@dataclass
class BenchmarkMetrics:
    """Overall benchmark metrics across all tracks."""
    baseline: str
    model: str

    # Track-level metrics
    tracks: dict[str, TrackMetrics] = field(default_factory=dict)

    # Aggregate metrics
    overall_sfrr: float = 0.0
    overall_decision_accuracy: float = 0.0
    overall_must_mention_rate: float = 0.0
    overall_must_not_mention_violation_rate: float = 0.0

    total_queries: int = 0

    # Token usage stats
    total_tokens: int = 0
    avg_tokens_per_query: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0

    # Latency stats
    avg_latency_ms: float = 0.0

    # Configuration
    token_budget: int = 8000
    seed: int | None = None


class MetricsAggregator:
    """Aggregates query results into track and benchmark metrics."""

    def __init__(self, baseline: str, model: str):
        self.baseline = baseline
        self.model = model
        self.results: list[QueryResult] = []

    def add_result(self, result: QueryResult) -> None:
        """Add a query result."""
        self.results.append(result)

    def compute_track_metrics(self, track: str) -> TrackMetrics:
        """Compute metrics for a single track."""
        track_results = [r for r in self.results if r.track == track]

        if not track_results:
            return TrackMetrics(track=track)

        metrics = TrackMetrics(track=track)
        metrics.total_queries = len(track_results)

        for r in track_results:
            # Decision accuracy
            if r.decision_correct:
                metrics.correct_decisions += 1

            # Must mention
            metrics.total_must_mention += len(r.must_mention)
            metrics.must_mention_hits += len(r.must_mention_hits)

            # Must not mention
            metrics.total_must_not_mention += len(r.must_not_mention)
            metrics.must_not_mention_violations += len(r.must_not_mention_violations)

            # Resurrection
            if r.resurrected_superseded:
                metrics.resurrection_count += 1

        # Compute rates
        metrics.decision_accuracy = metrics.correct_decisions / metrics.total_queries

        if metrics.total_must_mention > 0:
            metrics.must_mention_rate = metrics.must_mention_hits / metrics.total_must_mention

        if metrics.total_must_not_mention > 0:
            metrics.must_not_mention_violation_rate = (
                metrics.must_not_mention_violations / metrics.total_must_not_mention
            )

        # SFRR: percentage of queries that resurrected superseded facts
        metrics.sfrr = metrics.resurrection_count / metrics.total_queries

        # Secondary metrics
        total_tokens = sum(r.tokens_used for r in track_results)
        total_latency = sum(r.latency_ms for r in track_results)
        metrics.avg_tokens = total_tokens / metrics.total_queries
        metrics.avg_latency_ms = total_latency / metrics.total_queries

        return metrics

    def compute_benchmark_metrics(self, token_budget: int = 8000, seed: int | None = None) -> BenchmarkMetrics:
        """Compute overall benchmark metrics."""
        metrics = BenchmarkMetrics(baseline=self.baseline, model=self.model)
        metrics.total_queries = len(self.results)
        metrics.token_budget = token_budget
        metrics.seed = seed

        # Get unique tracks
        tracks = set(r.track for r in self.results)

        for track in tracks:
            track_metrics = self.compute_track_metrics(track)
            metrics.tracks[track] = track_metrics

        # Aggregate across tracks
        if metrics.total_queries > 0:
            total_correct = sum(t.correct_decisions for t in metrics.tracks.values())
            total_resurrections = sum(t.resurrection_count for t in metrics.tracks.values())
            total_must_mention = sum(t.total_must_mention for t in metrics.tracks.values())
            total_mm_hits = sum(t.must_mention_hits for t in metrics.tracks.values())
            total_mnm = sum(t.total_must_not_mention for t in metrics.tracks.values())
            total_mnm_violations = sum(t.must_not_mention_violations for t in metrics.tracks.values())

            metrics.overall_decision_accuracy = total_correct / metrics.total_queries
            metrics.overall_sfrr = total_resurrections / metrics.total_queries

            if total_must_mention > 0:
                metrics.overall_must_mention_rate = total_mm_hits / total_must_mention

            if total_mnm > 0:
                metrics.overall_must_not_mention_violation_rate = total_mnm_violations / total_mnm

            # Token stats
            all_tokens = [r.tokens_used for r in self.results if r.tokens_used > 0]
            if all_tokens:
                metrics.total_tokens = sum(all_tokens)
                metrics.avg_tokens_per_query = metrics.total_tokens / len(all_tokens)
                metrics.min_tokens = min(all_tokens)
                metrics.max_tokens = max(all_tokens)

            # Latency stats
            all_latency = [r.latency_ms for r in self.results if r.latency_ms > 0]
            if all_latency:
                metrics.avg_latency_ms = sum(all_latency) / len(all_latency)

        return metrics


def format_metrics_table(metrics: BenchmarkMetrics) -> str:
    """Format metrics as a markdown table."""
    lines = [
        f"# Benchmark Results: {metrics.baseline} / {metrics.model}",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Queries | {metrics.total_queries} |",
        f"| Decision Accuracy | {metrics.overall_decision_accuracy:.2%} |",
        f"| SFRR (Resurrection Rate) | {metrics.overall_sfrr:.2%} |",
        f"| Must Mention Rate | {metrics.overall_must_mention_rate:.2%} |",
        f"| Must Not Mention Violations | {metrics.overall_must_not_mention_violation_rate:.2%} |",
        "",
        "## By Track",
        "",
        "| Track | Queries | Accuracy | SFRR | MM Rate | MNM Violations |",
        "|-------|---------|----------|------|---------|----------------|",
    ]

    for track, tm in sorted(metrics.tracks.items()):
        lines.append(
            f"| {track} | {tm.total_queries} | {tm.decision_accuracy:.2%} | "
            f"{tm.sfrr:.2%} | {tm.must_mention_rate:.2%} | "
            f"{tm.must_not_mention_violation_rate:.2%} |"
        )

    return "\n".join(lines)
