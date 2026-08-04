"""Re-derive the v1.1 leaderboard under corrected scoring.

Runs (baseline x split x run) units on the published model configuration and
writes one JSON per unit. **Resumable**: a unit whose output file already exists
is skipped, so an interrupted sweep resumes by re-invoking with the same
arguments rather than restarting. Long sweeps here get killed by the harness
around the ten-minute mark, and a non-resumable driver loses everything.

Only the corrected scoring is applied. The legacy-vs-corrected comparison is
already measured in docs/PAPER3_SPEC.md §10a-iv; repeating it would double the
judge calls for a number we have.

Usage::

    uv run python experiments/with_car_secret.py -- \\
        python experiments/rederive_v11.py --baselines memgine --runs 3
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from statebench.evaluation.metrics import MetricsAggregator
from statebench.runner.harness import EvaluationHarness, load_timelines

BASELINES = [
    "no_memory",
    "transcript_replay",
    "transcript_latest_wins",
    "rolling_summary",
    "rag_transcript",
    "fact_extraction",
    "fact_extraction_with_supersession",
    "state_based",
    "state_based_no_supersession",
    "memgine",
]


def unit_path(out_dir: Path, baseline: str, split: str, run: int) -> Path:
    return out_dir / f"{baseline}__{split}__run{run}.json"


def run_unit(
    args: argparse.Namespace, baseline: str, split: str, run: int, out: Path
) -> dict | None:
    dataset = Path(args.release) / f"{split}.jsonl"
    timelines = list(load_timelines(dataset))
    if args.limit:
        timelines = timelines[: args.limit]

    harness = EvaluationHarness(
        model=args.model,
        provider=args.provider,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
    )
    strategy = harness._make_strategy(baseline)
    agg = MetricsAggregator(baseline=baseline, model=args.model)

    blocked = 0
    for timeline in timelines:
        try:
            for r in harness.run_timeline(timeline, strategy):
                agg.add_result(r)
        except Exception as e:  # provider refusal / transient: record and move on
            blocked += 1
            if blocked > args.max_blocked:
                raise RuntimeError(f"too many failed timelines ({blocked}): {e}") from e

    m = agg.compute_benchmark_metrics()
    payload = {
        "baseline": baseline,
        "split": split,
        "run": run,
        "model": args.model,
        "judge": harness.judge.descriptor,
        "n_timelines": len(timelines),
        "n_queries": m.total_queries,
        "blocked_timelines": blocked,
        "overall": {
            "decision_accuracy": m.overall_decision_accuracy,
            "sfrr": m.overall_sfrr,
            "leakage_rate": m.overall_leakage_rate,
            "fabrication_rate": m.overall_fabrication_rate,
            "must_mention_rate": m.overall_must_mention_rate,
            "mnm_violation_rate": m.overall_must_not_mention_violation_rate,
            "false_supersession_rate": m.overall_false_supersession_rate,
        },
        "tracks": {k: asdict(v) for k, v in m.tracks.items()},
    }
    out.write_text(json.dumps(payload, indent=1))
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", default="data/releases/v1.0")
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--baselines", nargs="+", default=BASELINES)
    ap.add_argument("--model", default="gpt-5.2-2025-12-11")
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--judge-provider", default="openai")
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=0, help="0 = full split")
    ap.add_argument("--max-blocked", type=int, default=15)
    ap.add_argument("--out-dir", default="experiments/results/v11_rederived")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    units = [
        (b, s, r)
        for b in args.baselines
        for s in args.splits
        for r in range(1, args.runs + 1)
    ]
    todo = [u for u in units if not unit_path(out_dir, *u).exists()]
    print(f"{len(units)} units, {len(units) - len(todo)} already done, {len(todo)} to run", flush=True)

    for baseline, split, run in todo:
        path = unit_path(out_dir, baseline, split, run)
        print(f"-> {baseline} / {split} / run{run}", flush=True)
        payload = run_unit(args, baseline, split, run, path)
        if payload:
            o = payload["overall"]
            print(
                f"   n={payload['n_queries']} acc={o['decision_accuracy']:.1%} "
                f"sfrr={o['sfrr']:.1%} leak={o['leakage_rate']:.1%} "
                f"mm={o['must_mention_rate']:.1%}",
                flush=True,
            )

    print("all requested units complete", flush=True)


if __name__ == "__main__":
    main()
