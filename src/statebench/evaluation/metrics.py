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
    # Forbidden phrases excluded from scoring as inadmissible (see
    # evaluation.phrase_quality). Recorded so audits can see what was dropped;
    # they are NOT counted in must_not_mention, so they neither create
    # violations nor pad the denominator.
    skipped_phrases: list[str] = field(default_factory=list)
    # Forbidden phrases that appeared only under negation ("not Friday"). Not
    # violations — the system was distinguishing dead state from live state —
    # but recorded so the behavior stays auditable.
    negated_mentions: list[str] = field(default_factory=list)
    # Violations bucketed by failure class ("superseded" | "restricted" |
    # "fabricated" | "other"). SFRR reads only the "superseded" bucket.
    violations_by_kind: dict[str, list[str]] = field(default_factory=dict)

    # Derived
    resurrected_superseded: bool = False  # Did the response mention superseded facts?

    # Metadata
    tokens_used: int = 0
    latency_ms: int = 0

    # Compaction metrics
    compaction_triggered: bool = False
    compaction_tokens_before: int = 0
    compaction_tokens_after: int = 0

    # Governance metrics, scored against the engine's audit record rather than
    # an author-written phrase list (see evaluation.governance_metrics).
    governance_bypass: bool = False  # asserted something the engine excluded
    bypassed_fact_ids: list[str] = field(default_factory=list)
    facts_excluded_count: int = 0  # denominator for the conditional bypass rate
    unsupported_values: list[str] = field(default_factory=list)  # URR evidence


@dataclass
class TrackMetrics:
    """Aggregated metrics for a track."""
    track: str
    total_queries: int = 0

    # Primary metrics
    sfrr: float = 0.0  # Superseded Fact Resurrection Rate (should-supersede tracks)
    # False Supersession Rate: inverse failure on should-NOT-supersede ("maintain")
    # tracks — fraction of queries where a still-valid fact was wrongly retired.
    # Never aggregated with SFRR; a fix must keep BOTH low.
    false_supersession_rate: float = 0.0
    decision_accuracy: float = 0.0
    must_mention_rate: float = 0.0
    must_not_mention_violation_rate: float = 0.0
    source_violation_rate: float = 0.0
    # Sibling rates to SFRR, split out so each names one failure. Previously all
    # three were folded into SFRR, which made it impossible to tell a
    # resurrection from a privacy leak from a fabrication.
    leakage_rate: float = 0.0  # restricted data reached the response
    fabrication_rate: float = 0.0  # invented detail appeared in the response

    # Counts
    correct_decisions: int = 0
    total_must_mention: int = 0
    must_mention_hits: int = 0
    total_must_not_mention: int = 0
    must_not_mention_violations: int = 0
    resurrection_count: int = 0
    leakage_count: int = 0
    fabrication_count: int = 0

    # Secondary
    avg_tokens: float = 0.0
    avg_latency_ms: float = 0.0

    # Cost-efficiency (v2.0)
    tokens_per_correct_answer: float = 0.0


@dataclass
class BenchmarkMetrics:
    """Overall benchmark metrics across all tracks."""
    baseline: str
    model: str

    # Track-level metrics
    tracks: dict[str, TrackMetrics] = field(default_factory=dict)

    # Aggregate metrics
    overall_sfrr: float = 0.0  # resurrection only — see TrackMetrics.sfrr
    overall_leakage_rate: float = 0.0
    overall_fabrication_rate: float = 0.0
    overall_false_supersession_rate: float = 0.0  # over "maintain" tracks only
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

    # Cost-efficiency metrics (v2.0)
    tokens_per_correct_answer: float = 0.0  # Lower is better
    token_efficiency: float = 0.0  # correct_answers / total_tokens (higher is better)
    cost_weighted_accuracy: float = 0.0  # accuracy / normalized_token_usage

    # Compaction metrics (populated when fact padding is used)
    compaction_triggered_count: int = 0
    avg_compaction_ratio: float = 0.0  # tokens_after / tokens_before (lower = more compaction)


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

            # Resurrection and its siblings, counted per query (a query with
            # two leaked phrases is one leaking query, matching SFRR's shape).
            if r.resurrected_superseded:
                metrics.resurrection_count += 1
            if r.violations_by_kind.get("restricted"):
                metrics.leakage_count += 1
            if r.violations_by_kind.get("fabricated"):
                metrics.fabrication_count += 1

        # Compute rates
        metrics.decision_accuracy = metrics.correct_decisions / metrics.total_queries

        if metrics.total_must_mention > 0:
            metrics.must_mention_rate = metrics.must_mention_hits / metrics.total_must_mention

        if metrics.total_must_not_mention > 0:
            metrics.must_not_mention_violation_rate = (
                metrics.must_not_mention_violations / metrics.total_must_not_mention
            )

        # On "maintain" (should-NOT-*) tracks the correct behavior is to AFFIRM the
        # still-valid fact (ground-truth decision = "yes"). false_supersession_rate
        # is the BEHAVIORAL failure to do so (decision != affirm) — not a lexical
        # signal, since a correct denial reuses the retirement vocabulary. It is the
        # generic "wrongly retired a valid fact" rate: it is FSR on supersession_*
        # maintain tracks and the False Authority Override Rate (FAOR) on
        # authority_maintain. Never aggregated with the should-supersede metrics.
        if track.endswith("_maintain"):
            metrics.false_supersession_rate = (
                metrics.total_queries - metrics.correct_decisions
            ) / metrics.total_queries
        else:
            # SFRR: percentage of queries that resurrected superseded facts.
            # Restricted-data leaks and fabrications are reported separately
            # rather than folded in here.
            metrics.sfrr = metrics.resurrection_count / metrics.total_queries
            metrics.leakage_rate = metrics.leakage_count / metrics.total_queries
            metrics.fabrication_rate = (
                metrics.fabrication_count / metrics.total_queries
            )

        # Secondary metrics
        total_tokens = sum(r.tokens_used for r in track_results)
        total_latency = sum(r.latency_ms for r in track_results)
        metrics.avg_tokens = total_tokens / metrics.total_queries
        metrics.avg_latency_ms = total_latency / metrics.total_queries

        # Cost-efficiency: tokens per correct answer
        if metrics.correct_decisions > 0 and total_tokens > 0:
            metrics.tokens_per_correct_answer = total_tokens / metrics.correct_decisions

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

        # Aggregate across tracks. "maintain" (should-NOT-supersede) tracks measure
        # a DISTINCT failure mode (false supersession) with inverted decision
        # semantics (decision is always "yes"). They are quarantined from every
        # should-supersede aggregate — SFRR, decision accuracy, must-mention, and
        # must-not-mention — and surface only as FSR. A blended number would let a
        # fix trade one failure for the other invisibly (the Belief-R trap).
        ss = [t for k, t in metrics.tracks.items() if not k.endswith("_maintain")]
        ns = [t for k, t in metrics.tracks.items() if k.endswith("_maintain")]
        ss_q = sum(t.total_queries for t in ss)
        ns_q = sum(t.total_queries for t in ns)

        if ss_q:
            metrics.overall_sfrr = sum(t.resurrection_count for t in ss) / ss_q
            metrics.overall_leakage_rate = sum(t.leakage_count for t in ss) / ss_q
            metrics.overall_fabrication_rate = (
                sum(t.fabrication_count for t in ss) / ss_q
            )
            metrics.overall_decision_accuracy = (
                sum(t.correct_decisions for t in ss) / ss_q
            )
            ss_mm = sum(t.total_must_mention for t in ss)
            if ss_mm:
                metrics.overall_must_mention_rate = (
                    sum(t.must_mention_hits for t in ss) / ss_mm
                )
            ss_mnm = sum(t.total_must_not_mention for t in ss)
            if ss_mnm:
                metrics.overall_must_not_mention_violation_rate = (
                    sum(t.must_not_mention_violations for t in ss) / ss_mnm
                )
        if ns_q:
            # FSR: behavioral failure to affirm a still-valid fact.
            metrics.overall_false_supersession_rate = (
                sum(t.total_queries - t.correct_decisions for t in ns) / ns_q
            )

        if metrics.total_queries > 0:
            # Cost metrics use correct answers across all tracks (not the
            # should-supersede-only accuracy above).
            total_correct = sum(t.correct_decisions for t in metrics.tracks.values())

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

            # Cost-efficiency metrics
            if metrics.total_tokens > 0:
                metrics.token_efficiency = total_correct / max(1, metrics.total_tokens)
                metrics.tokens_per_correct_answer = (
                    metrics.total_tokens / max(1, total_correct)
                )
                # Normalize token usage to [0, 1] range relative to budget
                normalized_usage = metrics.avg_tokens_per_query / max(1, token_budget)
                if normalized_usage > 0:
                    metrics.cost_weighted_accuracy = (
                        metrics.overall_decision_accuracy / normalized_usage
                    )

            # Compaction stats
            compacted = [r for r in self.results if r.compaction_triggered]
            metrics.compaction_triggered_count = len(compacted)
            if compacted:
                ratios = [
                    r.compaction_tokens_after / max(1, r.compaction_tokens_before)
                    for r in compacted
                ]
                metrics.avg_compaction_ratio = sum(ratios) / len(ratios)

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
        f"| Tokens/Correct Answer | {metrics.tokens_per_correct_answer:.0f} |",
        f"| Cost-Weighted Accuracy | {metrics.cost_weighted_accuracy:.2f} |",
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
