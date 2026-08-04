"""Run the governance x applicability factorial (Paper 3, §4.1 and §5.2).

Where the counterfactual pilot asks "does the system respect governance?", this
asks the orthogonal question: given a retrieved experience, does it correctly
KEEP, ADAPT or REJECT it — and does governance correctly dominate that choice?

Reporting discipline. The suite is REJECT-heavy by construction (once state is
superseded/expired/unauthorized/outranked/draft the correct disposition is
REJECT regardless of how well the experience transfers), so raw classification
accuracy is not interpretable alone. Every run prints:

* ``always_refuse_baseline`` — what a system that declines everything scores on
  this exact sample. The headline must be read against this, not against zero.
* ``correct_rejection`` and ``false_refusal`` over disjoint cell sets, never
  blended — the FSR/FAOR discipline. Blanket refusal drives false refusal to
  100% and is immediately visible.

Usage::

    uv run python experiments/applicability_suite.py --model parslee/reasoning
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from statebench.evaluation.applicability_metrics import (
    ApplicabilityMetrics,
    expected_from_timelines,
)
from statebench.evaluation.governance_metrics import GovernanceMetrics
from statebench.evaluation.metrics import MetricsAggregator
from statebench.generator.applicability import MATRIX, ApplicabilityGenerator
from statebench.runner.harness import EvaluationHarness

DEFAULT_BASELINES = [
    "memgine",
    "reconstructive_rag_prompted",
    "reconstructive_rag_engine_filtered",
    "reconstructive_rag_validated",
]


def run(args: argparse.Namespace) -> dict:
    timelines = ApplicabilityGenerator(seed=args.seed).generate(per_cell=args.per_cell)
    expected, principles, obsolete = expected_from_timelines(timelines)
    counts = {
        v: sum(1 for x in expected.values() if x == v) for v in ("KEEP", "ADAPT", "REJECT")
    }
    print(
        f"{len(timelines)} timelines over {len(MATRIX)} cells "
        f"(KEEP {counts['KEEP']} / ADAPT {counts['ADAPT']} / REJECT {counts['REJECT']})\n"
    )

    report: dict = {"model": args.model, "cells": len(MATRIX), "baselines": {}}

    for baseline in args.baselines:
        harness = EvaluationHarness(
            model=args.model,
            provider=args.provider,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
        )
        strategy = harness._make_strategy(baseline)
        aggregator = MetricsAggregator(baseline=baseline, model=args.model)
        results = []
        for i, timeline in enumerate(timelines, 1):
            for r in harness.run_timeline(timeline, strategy):
                aggregator.add_result(r)
                results.append(r)
            if args.verbose and i % 10 == 0:
                print(f"  {baseline}: {i}/{len(timelines)}")

        appl = ApplicabilityMetrics.from_results(results, expected, principles, obsolete)
        gov = GovernanceMetrics.from_results(results)

        print(f"\n=== {baseline} (judge {harness.judge.descriptor}) ===")
        print(appl.summary())
        print(
            f"GBR (queries w/ exclusions)  : {gov.conditional_bypass_rate:.1%}"
            f"  [n={gov.queries_with_exclusions}]\n"
            f"URR                          : {gov.unsupported_reconstruction_rate:.1%}"
        )

        report["baselines"][baseline] = {
            "judge": harness.judge.descriptor,
            "applicability_classification_accuracy": appl.applicability_classification_accuracy,
            "always_refuse_baseline": appl.always_refuse_baseline,
            "beats_blanket_refusal": appl.beats_blanket_refusal,
            "correct_rejection_rate": appl.correct_rejection_rate,
            "reject_cells": appl.reject_cells,
            "false_refusal_rate": appl.false_refusal_rate,
            "use_cells": appl.use_cells,
            "principle_transfer_rate": appl.principle_transfer_rate,
            "historical_detail_carryover_rate": appl.historical_detail_carryover_rate,
            "adapt_cells": appl.adapt_cells,
            "conditional_bypass_rate": gov.conditional_bypass_rate,
            "unsupported_reconstruction_rate": gov.unsupported_reconstruction_rate,
        }

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="parslee/reasoning")
    ap.add_argument("--provider", default="car")
    ap.add_argument("--judge-provider", default="car")
    ap.add_argument("--judge-model", default="parslee/reasoning")
    ap.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    ap.add_argument("--per-cell", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="experiments/results/applicability_suite.json")
    args = ap.parse_args()

    report = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
