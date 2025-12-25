"""Command-line interface for StateBench."""

import json
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from statebench.baselines import BASELINE_REGISTRY
from statebench.calibration import create_audit_template, run_calibration
from statebench.evaluation import format_metrics_table
from statebench.generator.engine import generate_dataset
from statebench.release import RELEASE_CONFIG, generate_release, verify_release
from statebench.runner.harness import EvaluationHarness, load_timelines

# Load environment variables from .env file (after imports to satisfy E402)
load_dotenv()

console = Console()

# All available tracks (v0.1 + v0.2 + v1.0)
AVAILABLE_TRACKS = [
    # v0.1 tracks
    "supersession",
    "commitment_durability",
    "interruption_resumption",
    "scope_permission",
    "environmental_freshness",
    # v0.2 tracks
    "hallucination_resistance",
    "scope_leak",
    "causality",
    "repair_propagation",
    "brutal_realistic",
    # v1.0 tracks
    "supersession_detection",
    "adversarial",
    "enterprise_privacy",
    "authority_hierarchy",
]


@click.group()
@click.version_option()
def main() -> None:
    """StateBench: A benchmark for LLM state correctness."""
    pass


@main.command()
@click.option(
    "--tracks",
    "-t",
    multiple=True,
    default=["supersession"],
    help=f"Tracks to generate. Use 'all' for all tracks. Available: {', '.join(AVAILABLE_TRACKS)}",
)
@click.option(
    "--count",
    "-n",
    default=100,
    help="Number of timelines per track (default: 100)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="data/generated/benchmark.jsonl",
    help="Output path for JSONL",
)
@click.option(
    "--seed",
    "-s",
    type=int,
    default=None,
    help="Random seed for reproducibility",
)
def generate(tracks: tuple[str, ...], count: int, output: str, seed: int | None) -> None:
    """Generate synthetic benchmark dataset."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle "all" track
    track_list = list(tracks)
    if "all" in track_list:
        track_list = AVAILABLE_TRACKS

    console.print(f"[bold]Generating {count} timelines per track...[/bold]")
    console.print(f"Tracks: {', '.join(track_list)}")

    total = generate_dataset(
        output_path=output_path,
        tracks=track_list,
        count_per_track=count,
        seed=seed,
    )

    console.print(f"\n[green]Generated {total} timelines to {output_path}[/green]")


@main.command()
@click.option(
    "--dataset",
    "-d",
    type=click.Path(exists=True),
    required=True,
    help="Path to JSONL dataset",
)
@click.option(
    "--baseline",
    "-b",
    type=click.Choice(list(BASELINE_REGISTRY.keys())),
    required=True,
    help="Baseline strategy to evaluate",
)
@click.option(
    "--model",
    "-m",
    default="gpt-4o",
    help="Model to use for generation (default: gpt-4o)",
)
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["openai", "anthropic", "google"]),
    default="openai",
    help="LLM provider (default: openai)",
)
@click.option(
    "--limit",
    "-l",
    type=int,
    default=None,
    help="Limit number of timelines to process",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output path for results JSON",
)
def evaluate(
    dataset: str,
    baseline: str,
    model: str,
    provider: str,
    limit: int | None,
    output: str | None,
) -> None:
    """Evaluate a baseline on a dataset."""
    console.print(f"[bold]Evaluating {baseline} with {model}...[/bold]")

    harness = EvaluationHarness(model=model, provider=provider)
    metrics = harness.evaluate(Path(dataset), baseline, limit=limit)

    # Print results
    console.print("\n")
    console.print(format_metrics_table(metrics))

    # Save if output specified
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "baseline": metrics.baseline,
                "model": metrics.model,
                "total_queries": metrics.total_queries,
                "overall_sfrr": metrics.overall_sfrr,
                "overall_decision_accuracy": metrics.overall_decision_accuracy,
                "overall_must_mention_rate": metrics.overall_must_mention_rate,
                "overall_must_not_mention_violation_rate": metrics.overall_must_not_mention_violation_rate,
                "tracks": {
                    track: {
                        "total_queries": tm.total_queries,
                        "sfrr": tm.sfrr,
                        "decision_accuracy": tm.decision_accuracy,
                        "must_mention_rate": tm.must_mention_rate,
                        "must_not_mention_violation_rate": tm.must_not_mention_violation_rate,
                    }
                    for track, tm in metrics.tracks.items()
                },
            }, f, indent=2)
        console.print(f"\n[green]Results saved to {output_path}[/green]")


@main.command()
@click.option(
    "--dataset",
    "-d",
    type=click.Path(exists=True),
    required=True,
    help="Path to JSONL dataset",
)
@click.option(
    "--baselines",
    "-b",
    multiple=True,
    default=None,
    help="Baselines to compare (default: all)",
)
@click.option(
    "--model",
    "-m",
    default="gpt-4o",
    help="Model to use for generation",
)
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["openai", "anthropic", "google"]),
    default="openai",
    help="LLM provider",
)
@click.option(
    "--limit",
    "-l",
    type=int,
    default=None,
    help="Limit number of timelines per baseline",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="results/comparison.json",
    help="Output path for results",
)
def compare(
    dataset: str,
    baselines: tuple[str, ...],
    model: str,
    provider: str,
    limit: int | None,
    output: str,
) -> None:
    """Compare multiple baselines on a dataset."""
    if not baselines:
        baselines = tuple(BASELINE_REGISTRY.keys())

    console.print(f"[bold]Comparing {len(baselines)} baselines with {model}...[/bold]")

    harness = EvaluationHarness(model=model, provider=provider)
    results = harness.compare_baselines(
        Path(dataset),
        list(baselines),
        limit=limit,
    )

    # Print comparison table
    table = Table(title="Baseline Comparison")
    table.add_column("Baseline")
    table.add_column("SFRR", justify="right")
    table.add_column("Decision Acc", justify="right")
    table.add_column("MM Rate", justify="right")
    table.add_column("MNM Violations", justify="right")

    for baseline, metrics in results.items():
        table.add_row(
            baseline,
            f"{metrics.overall_sfrr:.1%}",
            f"{metrics.overall_decision_accuracy:.1%}",
            f"{metrics.overall_must_mention_rate:.1%}",
            f"{metrics.overall_must_not_mention_violation_rate:.1%}",
        )

    console.print("\n")
    console.print(table)

    # Save results
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            baseline: {
                "model": metrics.model,
                "total_queries": metrics.total_queries,
                "overall_sfrr": metrics.overall_sfrr,
                "overall_decision_accuracy": metrics.overall_decision_accuracy,
                "overall_must_mention_rate": metrics.overall_must_mention_rate,
                "overall_must_not_mention_violation_rate": metrics.overall_must_not_mention_violation_rate,
            }
            for baseline, metrics in results.items()
        }, f, indent=2)
    console.print(f"\n[green]Results saved to {output_path}[/green]")


@main.command()
@click.option(
    "--dataset",
    "-d",
    type=click.Path(exists=True),
    required=True,
    help="Path to JSONL dataset",
)
@click.option(
    "--limit",
    "-l",
    type=int,
    default=5,
    help="Number of timelines to show",
)
def inspect(dataset: str, limit: int) -> None:
    """Inspect timelines in a dataset."""
    timelines = list(load_timelines(Path(dataset)))

    console.print(f"[bold]Dataset: {dataset}[/bold]")
    console.print(f"Total timelines: {len(timelines)}")

    # Track counts
    tracks: dict[str, int] = {}
    domains: dict[str, int] = {}
    for t in timelines:
        tracks[t.track] = tracks.get(t.track, 0) + 1
        domains[t.domain] = domains.get(t.domain, 0) + 1

    console.print("\nBy track:")
    for track, count in sorted(tracks.items()):
        console.print(f"  {track}: {count}")

    console.print("\nBy domain:")
    for domain, count in sorted(domains.items()):
        console.print(f"  {domain}: {count}")

    # Show sample timelines
    console.print(f"\n[bold]Sample timelines (first {limit}):[/bold]")
    for t in timelines[:limit]:
        console.print(f"\n--- {t.id} ({t.track}, {t.domain}) ---")
        console.print(f"User: {t.initial_state.identity_role.user_name}")

        for event in t.events:
            if hasattr(event, "text") and hasattr(event, "speaker"):
                event_text = getattr(event, "text", "")
                event_speaker = getattr(event, "speaker", "")
                console.print(f"  [{event.type}] {event_speaker}: {event_text[:80]}...")
            elif hasattr(event, "prompt") and hasattr(event, "ground_truth"):
                event_prompt = getattr(event, "prompt", "")
                event_gt = getattr(event, "ground_truth", None)
                console.print(f"  [query] {event_prompt}")
                if event_gt:
                    console.print(f"    Expected: {event_gt.decision}")


@main.command()
def baselines() -> None:
    """List available baselines."""
    table = Table(title="Available Baselines")
    table.add_column("Name")
    table.add_column("Description")

    descriptions = {
        # Core baselines
        "no_memory": "B0: No memory - only current query",
        "transcript_replay": "B1: Full transcript replay (truncated to budget)",
        "rolling_summary": "B2: Rolling conversation summary + recent turns",
        "rag_transcript": "B3: RAG over transcript chunks",
        "fact_extraction": "B4: Mem0-style fact extraction and retrieval",
        "state_based": "State-based context with supersession tracking",
        # Ablation baselines
        "state_based_no_supersession": "Ablation: State structure WITHOUT invalidation tracking",
        "fact_extraction_with_supersession": "Ablation: Fact extraction WITH supersession",
        "transcript_latest_wins": "Ablation: Transcript with 'latest wins' heuristic",
    }

    for name in BASELINE_REGISTRY:
        table.add_row(name, descriptions.get(name, ""))

    console.print(table)


@main.command()
@click.option(
    "--version",
    "-v",
    type=click.Choice(list(RELEASE_CONFIG.keys())),
    default="v0.1",
    help="Release version to generate",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="data/releases",
    help="Output directory for release",
)
def release(version: str, output: str) -> None:
    """Generate a canonical benchmark release with train/dev/test splits."""
    output_dir = Path(output) / version
    console.print(f"[bold]Generating release {version}...[/bold]")

    manifest = generate_release(version, output_dir)

    console.print(f"\n[green]Release {version} created in {output_dir}[/green]")
    console.print(f"Total timelines: {manifest['total_timelines']}")

    table = Table(title="Split Summary")
    table.add_column("Split")
    table.add_column("Count", justify="right")
    table.add_column("SHA256", justify="left")

    for split_name, split_info in manifest["splits"].items():
        table.add_row(
            split_name,
            str(split_info["count"]),
            split_info["sha256"][:16] + "...",
        )

    console.print(table)
    console.print(f"\nManifest: {output_dir / 'manifest.json'}")


@main.command()
@click.argument("release_dir", type=click.Path(exists=True))
def verify(release_dir: str) -> None:
    """Verify a benchmark release against its manifest."""
    release_path = Path(release_dir)
    console.print(f"[bold]Verifying release in {release_path}...[/bold]")

    valid, errors = verify_release(release_path)

    if valid:
        console.print("\n[green]✓ Release verified successfully[/green]")
    else:
        console.print("\n[red]✗ Verification failed:[/red]")
        for error in errors:
            console.print(f"  - {error}")


@main.command()
@click.option(
    "--audit-set",
    "-a",
    type=click.Path(exists=True),
    required=True,
    help="Path to human-annotated audit set JSONL",
)
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["openai", "anthropic"]),
    default="openai",
    help="LLM provider for judge",
)
@click.option(
    "--no-llm",
    is_flag=True,
    help="Disable LLM judge (deterministic only)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output path for detailed results JSON",
)
def calibrate(audit_set: str, provider: str, no_llm: bool, output: str | None) -> None:
    """Run judge calibration against human annotations."""
    console.print(f"[bold]Running calibration against {audit_set}...[/bold]")

    result = run_calibration(
        Path(audit_set),
        use_llm_judge=not no_llm,
        provider=provider,
    )

    # Print summary
    table = Table(title="Calibration Results")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Total Items", str(result.total_items))
    table.add_row("Decision Agreement", f"{result.decision_agreement:.1%}")
    table.add_row("Decision Kappa", f"{result.decision_kappa:.3f}")
    table.add_row("Must Mention Precision", f"{result.must_mention_precision:.1%}")
    table.add_row("Must Mention Recall", f"{result.must_mention_recall:.1%}")
    table.add_row("MNM Precision", f"{result.must_not_mention_precision:.1%}")
    table.add_row("MNM Recall", f"{result.must_not_mention_recall:.1%}")

    console.print(table)

    # Kappa interpretation
    kappa = result.decision_kappa
    if kappa > 0.8:
        console.print("\n[green]✓ Kappa > 0.8: Almost perfect agreement[/green]")
    elif kappa > 0.6:
        console.print("\n[yellow]⚠ Kappa 0.6-0.8: Substantial agreement[/yellow]")
    else:
        console.print("\n[red]✗ Kappa < 0.6: Moderate or lower agreement[/red]")

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.model_dump(), f, indent=2)
        console.print(f"\nDetailed results saved to {output_path}")


@main.command("budget-sweep")
@click.option(
    "--dataset",
    "-d",
    type=click.Path(exists=True),
    required=True,
    help="Path to JSONL dataset",
)
@click.option(
    "--baseline",
    "-b",
    type=click.Choice(list(BASELINE_REGISTRY.keys())),
    required=True,
    help="Baseline to evaluate",
)
@click.option(
    "--budgets",
    "-B",
    default="1000,2000,4000,8000,16000",
    help="Comma-separated list of token budgets (default: 1000,2000,4000,8000,16000)",
)
@click.option(
    "--model",
    "-m",
    default="gpt-4o",
    help="Model to use",
)
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["openai", "anthropic", "google"]),
    default="openai",
    help="LLM provider",
)
@click.option(
    "--limit",
    "-l",
    type=int,
    default=None,
    help="Limit number of timelines",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="results/budget_sweep.json",
    help="Output path for results",
)
def budget_sweep(
    dataset: str,
    baseline: str,
    budgets: str,
    model: str,
    provider: str,
    limit: int | None,
    output: str,
) -> None:
    """Run evaluation at multiple token budgets to show budget-performance curves."""
    budget_list = [int(b.strip()) for b in budgets.split(",")]

    console.print(f"[bold]Running budget sweep for {baseline} with {model}...[/bold]")
    console.print(f"Budgets: {budget_list}")

    results = {}

    for budget in budget_list:
        console.print(f"\n[bold]Evaluating at budget={budget}...[/bold]")
        harness = EvaluationHarness(model=model, provider=provider, token_budget=budget)
        metrics = harness.evaluate(Path(dataset), baseline, limit=limit)

        results[budget] = {
            "token_budget": budget,
            "decision_accuracy": metrics.overall_decision_accuracy,
            "sfrr": metrics.overall_sfrr,
            "must_mention_rate": metrics.overall_must_mention_rate,
            "must_not_mention_violation_rate": metrics.overall_must_not_mention_violation_rate,
            "avg_tokens_per_query": metrics.avg_tokens_per_query,
        }

    # Print summary table
    table = Table(title=f"Budget Sweep: {baseline}")
    table.add_column("Budget")
    table.add_column("Decision Acc", justify="right")
    table.add_column("SFRR", justify="right")
    table.add_column("MM Rate", justify="right")
    table.add_column("Avg Tokens", justify="right")

    for budget, r in sorted(results.items()):
        table.add_row(
            str(budget),
            f"{r['decision_accuracy']:.1%}",
            f"{r['sfrr']:.1%}",
            f"{r['must_mention_rate']:.1%}",
            f"{r['avg_tokens_per_query']:.0f}",
        )

    console.print("\n")
    console.print(table)

    # Save results
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "baseline": baseline,
            "model": model,
            "budgets": results,
        }, f, indent=2)
    console.print(f"\n[green]Results saved to {output_path}[/green]")


@main.command("variance-report")
@click.option(
    "--dataset",
    "-d",
    type=click.Path(exists=True),
    required=True,
    help="Path to JSONL dataset",
)
@click.option(
    "--baseline",
    "-b",
    type=click.Choice(list(BASELINE_REGISTRY.keys())),
    required=True,
    help="Baseline to evaluate",
)
@click.option(
    "--seeds",
    "-s",
    default="42,123,456,789,1000",
    help="Comma-separated list of random seeds (default: 42,123,456,789,1000)",
)
@click.option(
    "--model",
    "-m",
    default="gpt-4o",
    help="Model to use",
)
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["openai", "anthropic", "google"]),
    default="openai",
    help="LLM provider",
)
@click.option(
    "--limit",
    "-l",
    type=int,
    default=None,
    help="Limit number of timelines",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="results/variance_report.json",
    help="Output path for results",
)
def variance_report(
    dataset: str,
    baseline: str,
    seeds: str,
    model: str,
    provider: str,
    limit: int | None,
    output: str,
) -> None:
    """Run evaluation across multiple seeds and report mean/std variance."""
    import numpy as np

    seed_list = [int(s.strip()) for s in seeds.split(",")]

    console.print(f"[bold]Running variance report for {baseline} with {model}...[/bold]")
    console.print(f"Seeds: {seed_list}")

    # Note: Currently seeds affect dataset generation, not model sampling
    # For true variance, we'd need temperature>0 or regenerate datasets

    all_decision_acc = []
    all_sfrr = []
    all_mm_rate = []

    for seed in seed_list:
        console.print(f"\n[bold]Evaluating with seed={seed}...[/bold]")
        # For now, run same dataset multiple times (measures model variance at temp=0)
        harness = EvaluationHarness(model=model, provider=provider)
        metrics = harness.evaluate(Path(dataset), baseline, limit=limit)

        all_decision_acc.append(metrics.overall_decision_accuracy)
        all_sfrr.append(metrics.overall_sfrr)
        all_mm_rate.append(metrics.overall_must_mention_rate)

    # Calculate stats
    decision_mean = float(np.mean(all_decision_acc))
    decision_std = float(np.std(all_decision_acc))
    decision_min = float(np.min(all_decision_acc))
    decision_max = float(np.max(all_decision_acc))

    sfrr_mean = float(np.mean(all_sfrr))
    sfrr_std = float(np.std(all_sfrr))
    sfrr_min = float(np.min(all_sfrr))
    sfrr_max = float(np.max(all_sfrr))

    mm_mean = float(np.mean(all_mm_rate))
    mm_std = float(np.std(all_mm_rate))
    mm_min = float(np.min(all_mm_rate))
    mm_max = float(np.max(all_mm_rate))
    results = {
        "baseline": baseline,
        "model": model,
        "n_runs": len(seed_list),
        "decision_accuracy": {
            "mean": decision_mean,
            "std": decision_std,
            "min": decision_min,
            "max": decision_max,
            "values": all_decision_acc,
        },
        "sfrr": {
            "mean": sfrr_mean,
            "std": sfrr_std,
            "min": sfrr_min,
            "max": sfrr_max,
            "values": all_sfrr,
        },
        "must_mention_rate": {
            "mean": mm_mean,
            "std": mm_std,
            "min": mm_min,
            "max": mm_max,
            "values": all_mm_rate,
        },
    }

    # Print summary
    table = Table(title=f"Variance Report: {baseline} ({len(seed_list)} runs)")
    table.add_column("Metric")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    table.add_row(
        "Decision Accuracy",
        f"{decision_mean:.1%}",
        f"±{decision_std:.1%}",
        f"{decision_min:.1%}",
        f"{decision_max:.1%}",
    )
    table.add_row(
        "SFRR",
        f"{sfrr_mean:.1%}",
        f"±{sfrr_std:.1%}",
        f"{sfrr_min:.1%}",
        f"{sfrr_max:.1%}",
    )
    table.add_row(
        "Must Mention Rate",
        f"{mm_mean:.1%}",
        f"±{mm_std:.1%}",
        f"{mm_min:.1%}",
        f"{mm_max:.1%}",
    )

    console.print("\n")
    console.print(table)

    # Stability assessment
    if decision_std < 0.02:
        console.print("\n[green]✓ Results are stable (std < 2%)[/green]")
    else:
        console.print("\n[yellow]⚠ Results show variance (std >= 2%)[/yellow]")

    # Save results
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[green]Results saved to {output_path}[/green]")


@main.command("create-audit-set")
@click.option(
    "--release-dir",
    "-r",
    type=click.Path(exists=True),
    required=True,
    help="Path to release directory",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="data/calibration/audit_template.jsonl",
    help="Output path for audit template",
)
@click.option(
    "--sample-size",
    "-n",
    type=int,
    default=50,
    help="Number of items to sample",
)
@click.option(
    "--seed",
    "-s",
    type=int,
    default=42,
    help="Random seed for sampling",
)
def create_audit_set(release_dir: str, output: str, sample_size: int, seed: int) -> None:
    """Create an audit set template for human annotation."""
    console.print(f"[bold]Creating audit template from {release_dir}...[/bold]")

    count = create_audit_template(
        Path(release_dir),
        Path(output),
        sample_size=sample_size,
        seed=seed,
    )

    console.print(f"\n[green]Created {count} items in {output}[/green]")
    console.print("\nNext steps:")
    console.print("1. Run model on each item to get responses")
    console.print("2. Have annotator fill in human_labels")
    console.print("3. Run: statebench calibrate --audit-set {output}")


@main.command("leaderboard")
@click.option(
    "--baseline",
    "-b",
    type=click.Choice(list(BASELINE_REGISTRY.keys())),
    required=True,
    help="Baseline/strategy being submitted",
)
@click.option(
    "--model",
    "-m",
    default="gpt-4o",
    help="Model used for evaluation",
)
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["openai", "anthropic", "google"]),
    default="openai",
    help="LLM provider",
)
@click.option(
    "--release",
    "-r",
    default="v0.1",
    help="Benchmark release version",
)
@click.option(
    "--split",
    "-s",
    type=click.Choice(["test", "dev"]),
    default="test",
    help="Split to evaluate on (use 'test' for official submissions)",
)
@click.option(
    "--seeds",
    default="42,123,456",
    help="Seeds for variance estimation",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="submissions/submission.json",
    help="Output path for submission file",
)
@click.option(
    "--submitter",
    required=True,
    help="Name/organization of submitter",
)
@click.option(
    "--submission-notes",
    default="",
    help="Optional notes about the submission",
)
@click.option(
    "--limit",
    "-l",
    type=int,
    default=None,
    help="Limit timelines (for development only, invalidates official submission)",
)
def leaderboard(
    baseline: str,
    model: str,
    provider: str,
    release: str,
    split: str,
    seeds: str,
    output: str,
    submitter: str,
    submission_notes: str,
    limit: int | None,
) -> None:
    """Generate an official leaderboard submission.

    This command runs the standardized evaluation protocol and generates
    a submission file that can be used for leaderboard inclusion.

    Protocol:
    1. Uses the official test split (or dev for development)
    2. Runs with multiple seeds for variance estimation
    3. Computes all standard metrics
    4. Generates cryptographic hash for verification
    5. Outputs structured submission file
    """
    import hashlib
    from datetime import datetime

    console.print("[bold]StateBench Leaderboard Submission Protocol[/bold]")
    console.print(f"Release: {release}, Split: {split}")
    console.print(f"Baseline: {baseline}, Model: {model}")
    console.print()

    # Verify release exists
    release_dir = Path(f"data/releases/{release}")
    if not release_dir.exists():
        console.print(f"[red]Release {release} not found at {release_dir}[/red]")
        console.print("Run: statebench release --version {release} first")
        return

    # Verify the release
    from statebench.release import verify_release
    is_valid, errors = verify_release(release_dir)
    if not is_valid:
        console.print("[red]Release verification failed:[/red]")
        for error in errors:
            console.print(f"  - {error}")
        return

    console.print("[green]✓ Release verified[/green]")

    # Load manifest for hash verification
    with open(release_dir / "manifest.json") as f:
        manifest = json.load(f)

    dataset_path = release_dir / f"{split}.jsonl"
    dataset_hash = manifest["splits"][split]["sha256"]

    console.print(f"Dataset: {dataset_path}")
    console.print(f"Hash: {dataset_hash[:16]}...")
    console.print()

    # Warn about limit
    is_official = limit is None
    if not is_official:
        console.print(f"[yellow]⚠ Using limit={limit} - this is NOT an official submission[/yellow]")
        console.print()

    # Run evaluation with multiple seeds
    seed_list = [int(s.strip()) for s in seeds.split(",")]
    console.print(f"Running evaluation with {len(seed_list)} seeds: {seed_list}")

    all_results = []
    for seed in seed_list:
        console.print(f"\n[bold]Seed {seed}...[/bold]")
        # Note: seed is recorded for documentation but LLM calls use provider defaults
        harness = EvaluationHarness(model=model, provider=provider)
        metrics = harness.evaluate(dataset_path, baseline, limit=limit)
        all_results.append({
            "seed": seed,
            "decision_accuracy": metrics.overall_decision_accuracy,
            "sfrr": metrics.overall_sfrr,
            "must_mention_rate": metrics.overall_must_mention_rate,
            "must_not_mention_violation_rate": metrics.overall_must_not_mention_violation_rate,
            "total_queries": metrics.total_queries,
            "avg_tokens": metrics.avg_tokens_per_query,
        })

    # Aggregate results
    def compute_stats(key: str) -> dict[str, float]:
        values = [r[key] for r in all_results]
        import statistics
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    aggregated = {
        "decision_accuracy": compute_stats("decision_accuracy"),
        "sfrr": compute_stats("sfrr"),
        "must_mention_rate": compute_stats("must_mention_rate"),
        "must_not_mention_violation_rate": compute_stats("must_not_mention_violation_rate"),
    }

    # Build submission
    submission = {
        "protocol_version": "1.0",
        "is_official": is_official,
        "submission_timestamp": datetime.utcnow().isoformat() + "Z",
        "submitter": submitter,
        "notes": submission_notes,

        "benchmark": {
            "name": "StateBench",
            "release": release,
            "split": split,
            "dataset_sha256": dataset_hash,
            "total_queries": all_results[0]["total_queries"],
            "limit_applied": limit,
        },

        "system": {
            "baseline": baseline,
            "model": model,
            "provider": provider,
        },

        "evaluation": {
            "seeds": seed_list,
            "num_runs": len(seed_list),
        },

        "results": aggregated,

        "per_seed_results": all_results,
    }

    # Compute submission hash
    submission_content = json.dumps(submission, sort_keys=True)
    submission_hash = hashlib.sha256(submission_content.encode()).hexdigest()
    submission["submission_hash"] = submission_hash

    # Save submission
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(submission, f, indent=2)

    # Print summary
    console.print("\n" + "=" * 60)
    console.print("[bold green]Submission Generated Successfully[/bold green]")
    console.print("=" * 60)

    table = Table(title="Leaderboard Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")

    table.add_row(
        "Decision Accuracy",
        f"{aggregated['decision_accuracy']['mean']:.2%}",
        f"±{aggregated['decision_accuracy']['std']:.2%}",
    )
    table.add_row(
        "SFRR ↓",
        f"{aggregated['sfrr']['mean']:.2%}",
        f"±{aggregated['sfrr']['std']:.2%}",
    )
    table.add_row(
        "Must Mention Rate",
        f"{aggregated['must_mention_rate']['mean']:.2%}",
        f"±{aggregated['must_mention_rate']['std']:.2%}",
    )

    console.print("\n")
    console.print(table)

    console.print(f"\n[bold]Submission file:[/bold] {output_path}")
    console.print(f"[bold]Submission hash:[/bold] {submission_hash[:16]}...")

    console.print("\n[bold]To submit to leaderboard:[/bold]")
    console.print(f"1. Review the submission file: {output_path}")
    console.print("2. Open a PR to github.com/parslee-ai/statebench-leaderboard")
    console.print("3. Add your submission file to submissions/")
    console.print("4. Include your submission hash in the PR description")


# =============================================================================
# v1.0: Split Management Commands
# =============================================================================

@main.command("create-splits")
@click.option(
    "--dataset",
    "-d",
    type=click.Path(exists=True),
    required=True,
    help="Path to JSONL dataset to split",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="data/splits",
    help="Output directory for splits",
)
@click.option(
    "--seed",
    "-s",
    type=int,
    default=42,
    help="Random seed for reproducibility",
)
@click.option(
    "--canaries",
    "-c",
    type=int,
    default=10,
    help="Number of canary items to add to hidden split",
)
@click.option(
    "--train-ratio",
    type=float,
    default=0.60,
    help="Training split ratio (default: 0.60)",
)
@click.option(
    "--dev-ratio",
    type=float,
    default=0.15,
    help="Development split ratio (default: 0.15)",
)
@click.option(
    "--test-ratio",
    type=float,
    default=0.15,
    help="Test split ratio (default: 0.15)",
)
@click.option(
    "--hidden-ratio",
    type=float,
    default=0.10,
    help="Hidden split ratio (default: 0.10)",
)
def create_splits(
    dataset: str,
    output_dir: str,
    seed: int,
    canaries: int,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    hidden_ratio: float,
) -> None:
    """Create train/dev/test/hidden splits with canaries."""
    from statebench.splits import SplitConfig, SplitManager

    console.print("[bold]Creating dataset splits...[/bold]")

    # Load dataset
    timelines = list(load_timelines(Path(dataset)))
    console.print(f"Loaded {len(timelines)} timelines from {dataset}")

    # Create split manager
    config = SplitConfig(
        train=train_ratio,
        dev=dev_ratio,
        test=test_ratio,
        hidden=hidden_ratio,
    )
    manager = SplitManager(version="1.0", config=config)

    # Create splits
    splits = manager.create_splits(timelines, seed=seed)

    # Add canaries to hidden split
    if canaries > 0:
        splits["hidden"], canary_list = manager.add_canaries(
            splits["hidden"],
            n_canaries=canaries,
            seed=seed,
        )
        console.print(f"Added {len(canary_list)} canary items to hidden split")
    else:
        canary_list = None

    # Save splits
    output_path = Path(output_dir)
    metadata = manager.save_splits(splits, output_path, seed, canary_list)

    # Print summary
    table = Table(title="Split Summary")
    table.add_column("Split", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Ratio", justify="right")

    total = sum(metadata.counts.values())
    for split_name in ["train", "dev", "test", "hidden"]:
        count = metadata.counts.get(split_name, 0)
        ratio = count / total if total > 0 else 0
        table.add_row(split_name, str(count), f"{ratio:.1%}")

    console.print(table)
    console.print(f"\n[green]Splits saved to {output_path}[/green]")
    console.print(f"Seed: {seed}")
    if canary_list:
        console.print(f"Canaries: {len(canary_list)} (registry in .canary_registry.json)")


@main.command("check-contamination")
@click.option(
    "--responses",
    "-r",
    type=click.Path(exists=True),
    required=True,
    help="Path to file with model responses (one per line or JSONL)",
)
@click.option(
    "--canary-registry",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to .canary_registry.json",
)
def check_contamination(responses: str, canary_registry: str) -> None:
    """Check model responses for canary contamination."""
    from statebench.splits import check_contamination as do_check
    from statebench.splits import format_contamination_report

    console.print("[bold]Checking for canary contamination...[/bold]")

    # Load responses
    response_list: list[str] = []
    with open(responses) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try JSON first
            try:
                data = json.loads(line)
                if isinstance(data, dict) and "response" in data:
                    response_list.append(data["response"])
                elif isinstance(data, str):
                    response_list.append(data)
                else:
                    response_list.append(line)
            except json.JSONDecodeError:
                response_list.append(line)

    console.print(f"Loaded {len(response_list)} responses")

    # Check contamination
    contamination = do_check(response_list, Path(canary_registry))

    # Report
    report = format_contamination_report(contamination)
    console.print("\n" + report)

    if contamination:
        raise SystemExit(1)  # Fail if contaminated


@main.command("split-stats")
@click.option(
    "--split-dir",
    "-d",
    type=click.Path(exists=True),
    required=True,
    help="Path to splits directory",
)
def split_stats(split_dir: str) -> None:
    """Show statistics for a split directory."""
    from statebench.splits import SplitManager

    split_path = Path(split_dir)
    manager = SplitManager()

    # Load metadata
    metadata_path = split_path / "metadata.json"
    if not metadata_path.exists():
        console.print("[red]No metadata.json found[/red]")
        raise SystemExit(1)

    with open(metadata_path) as f:
        metadata = json.load(f)

    console.print(f"[bold]Split Statistics: {split_dir}[/bold]")
    console.print(f"Version: {metadata.get('version', 'unknown')}")
    console.print(f"Created: {metadata.get('created_at', 'unknown')}")
    console.print(f"Seed: {metadata.get('seed', 'unknown')}")

    table = Table(title="Split Counts")
    table.add_column("Split", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Hash", style="dim")

    counts = metadata.get("counts", {})
    for split_name in ["train", "dev", "test", "hidden"]:
        count = counts.get(split_name, 0)

        # Compute hash if file exists
        split_file = split_path / f"{split_name}.jsonl"
        if split_file.exists():
            timelines = manager.load_split(split_file)
            hash_val = manager.compute_split_hash(timelines)
        else:
            hash_val = "file not found"

        table.add_row(split_name, str(count), hash_val)

    console.print(table)

    if metadata.get("canary_count", 0) > 0:
        console.print(f"\nCanaries: {metadata['canary_count']}")


# =============================================================================
# HuggingFace Hub Integration Commands
# =============================================================================


@main.command("hf-prepare")
@click.option(
    "--release",
    "-r",
    default="v1.0",
    help="Release version to prepare (default: v1.0)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="data/hf-staging",
    help="Output directory for staged files",
)
def hf_prepare(release: str, output: str) -> None:
    """Prepare a release for HuggingFace Hub (local preview).

    Creates Parquet files and a sample JSON for inspection before pushing.
    """
    from statebench.huggingface import prepare_hf_staging

    release_dir = Path(f"data/releases/{release}")
    if not release_dir.exists():
        console.print(f"[red]Release {release} not found at {release_dir}[/red]")
        console.print(f"Run: statebench release --version {release} first")
        return

    console.print(f"[bold]Preparing {release} for HuggingFace Hub...[/bold]")

    output_path = Path(output)
    counts = prepare_hf_staging(release_dir, output_path)

    table = Table(title="Prepared Splits")
    table.add_column("Split", style="cyan")
    table.add_column("Count", justify="right")

    for split_name, count in counts.items():
        table.add_row(split_name, str(count))

    console.print(table)
    console.print(f"\n[green]Staged files saved to {output_path}[/green]")
    console.print("  - Parquet files for each split")
    console.print("  - sample.json for inspection")
    console.print("\nTo push to HuggingFace Hub:")
    console.print(f"  statebench hf-push --release {release}")


@main.command("hf-push")
@click.option(
    "--release",
    "-r",
    default="v1.0",
    help="Release version to push (default: v1.0)",
)
@click.option(
    "--repo",
    default="parslee/statebench",
    help="HuggingFace repository ID (default: parslee/statebench)",
)
@click.option(
    "--private",
    is_flag=True,
    help="Make the repository private",
)
@click.option(
    "--token",
    envvar="HF_TOKEN",
    help="HuggingFace API token (or set HF_TOKEN env var)",
)
def hf_push(release: str, repo: str, private: bool, token: str | None) -> None:
    """Push a release to HuggingFace Hub.

    Requires authentication via --token or HF_TOKEN environment variable.
    You can also run `huggingface-cli login` to cache credentials.
    """
    from statebench.huggingface import push_to_hub

    release_dir = Path(f"data/releases/{release}")
    if not release_dir.exists():
        console.print(f"[red]Release {release} not found at {release_dir}[/red]")
        console.print(f"Run: statebench release --version {release} first")
        return

    console.print(f"[bold]Pushing {release} to {repo}...[/bold]")

    try:
        url = push_to_hub(release_dir, repo_id=repo, private=private, token=token)
        console.print("\n[green]Successfully pushed to HuggingFace Hub![/green]")
        console.print(f"URL: {url}")
        console.print("\nTo load the dataset:")
        console.print("  from datasets import load_dataset")
        console.print(f'  ds = load_dataset("{repo}")')
    except Exception as e:
        console.print(f"\n[red]Failed to push: {e}[/red]")
        console.print("\nTroubleshooting:")
        console.print("  1. Run: huggingface-cli login")
        console.print("  2. Or set HF_TOKEN environment variable")
        console.print("  3. Ensure you have write access to the repository")


if __name__ == "__main__":
    main()
