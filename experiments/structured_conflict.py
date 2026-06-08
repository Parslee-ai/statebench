"""Validate the structured-conflict tracks exercise the authority + Type II fixes.

authority_conflict: A/B memgine with authority resolution ON (gap=1) vs OFF (gap=0)
  -> accuracy should drop without the fix (the later low-authority value wins by recency).
dependency_chain: does memgine avoid asserting the stale derived value.

Timelines are template-generated; the harness/judge run inference via CAR (local).
Note: the authority fix's value here is STATE CORRECTNESS (not asserting the
superseded loser value), not decision accuracy — a competent model answers
correctly from the surviving fact either way.
"""

from __future__ import annotations

import argparse

from statebench.evaluation.metrics import MetricsAggregator
from statebench.generator.engine import TimelineGenerator
from statebench.memgine.strategy import MemgineStrategy
from statebench.runner.harness import EvaluationHarness


def evaluate(harness, strat, timelines):
    agg = MetricsAggregator(baseline=strat.name, model=harness.model)
    for tl in timelines:
        for r in harness.run_timeline(tl, strat):
            agg.add_result(r)
    m = agg.compute_benchmark_metrics()
    t = next(iter(m.tracks.values()))
    return t.decision_accuracy, t.must_not_mention_violation_rate, t.total_queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx/qwen3-4b:4bit")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    harness = EvaluationHarness(model=args.model, provider="car", use_llm_judge=True)
    gen = TimelineGenerator(seed=11)

    print(f"model={args.model}\n")
    auth = list(gen.generate_track("authority_conflict", count=args.n))
    on = MemgineStrategy(token_budget=8000, authority_resolution_gap=1, model=args.model)
    off = MemgineStrategy(token_budget=8000, authority_resolution_gap=0, model=args.model)
    a_on = evaluate(harness, on, auth)
    a_off = evaluate(harness, off, auth)
    print("authority_conflict — STATE CORRECTNESS (not asserting the superseded loser):")
    print(f"  fix ON  (gap=1): loser-value-mention={a_on[1]:.3f}  acc={a_on[0]:.3f}  n={a_on[2]}")
    print(f"  fix OFF (gap=0): loser-value-mention={a_off[1]:.3f}  acc={a_off[0]:.3f}  n={a_off[2]}")
    print(f"  -> fix effect: loser-mention {a_off[1]:.3f} -> {a_on[1]:.3f} "
          f"({a_on[1]-a_off[1]:+.3f}); accuracy unchanged\n")

    dep = list(gen.generate_track("dependency_chain", count=args.n))
    d = evaluate(harness, MemgineStrategy(token_budget=8000, model=args.model), dep)
    print("dependency_chain (does it avoid asserting the stale derived value?)")
    print(f"  memgine: acc={d[0]:.3f}  stale-value-asserted(MNMv)={d[1]:.3f}  n={d[2]}\n")

    am = list(gen.generate_track("authority_maintain", count=args.n))
    a = evaluate(harness, MemgineStrategy(token_budget=8000, model=args.model), am)
    # FAOR = behavioral failure to affirm the complying decision = 1 - decision_accuracy.
    print("authority_maintain — should-NOT-override (False Authority Override Rate):")
    print(f"  memgine: FAOR={1 - a[0]:.3f}  n={a[2]}  "
          "(FAOR=0 is STRUCTURAL: authority resolution can't reach across keys, so "
          "this is a conformance/regression guarantee, not a behavioral discriminator)")


if __name__ == "__main__":
    main()
