"""Belief-R guardrail end-to-end check: report (SFRR, FSR) as a pair.

SFRR on should-supersede, FSR on should-NOT-supersede ("maintain"). To keep the
two sides comparable (Belief-R warns imbalance hides a biased fix), both are
generated from the SAME matched template population at equal count. A memory fix
is only safe if BOTH stay low. All inference via CAR (local).
"""

from __future__ import annotations

import argparse

from statebench.baselines import get_baseline
from statebench.evaluation.metrics import MetricsAggregator
from statebench.generator.engine import TimelineGenerator
from statebench.generator.templates.supersession import SUPERSESSION_TEMPLATES
from statebench.runner.harness import EvaluationHarness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx/qwen3-4b:4bit")
    ap.add_argument("--baseline", default="memgine")
    ap.add_argument("--n", type=int, default=16, help="timelines per side")
    args = ap.parse_args()

    gen = TimelineGenerator(seed=7)
    templates = [t for t in SUPERSESSION_TEMPLATES if t.maintain_event_template]
    ss, ns = [], []
    for _ in range(args.n):
        t = gen.rng.choice(templates)  # matched template population
        ss.append(gen.generate_supersession_timeline(t))
        ns.append(gen.generate_maintain_timeline(t))

    harness = EvaluationHarness(model=args.model, provider="car", use_llm_judge=True)
    agg = MetricsAggregator(baseline=args.baseline, model=args.model)
    strat = get_baseline(args.baseline, token_budget=8000, model=args.model)
    for tl in ss + ns:
        for r in harness.run_timeline(tl, strat):
            agg.add_result(r)

    m = agg.compute_benchmark_metrics()
    print(f"baseline={args.baseline} model={args.model} (n={args.n}/side, matched templates)")
    for track, tm in m.tracks.items():
        print(f"  {track:24s} n={tm.total_queries:3d} acc={tm.decision_accuracy:.3f} "
              f"SFRR={tm.sfrr:.3f} FSR={tm.false_supersession_rate:.3f}")
    print(f"\n  SFRR={m.overall_sfrr:.3f}  FSR={m.overall_false_supersession_rate:.3f}")
    print("  (guardrail: a supersession fix must keep BOTH low — never trade one for the other)")


if __name__ == "__main__":
    main()
