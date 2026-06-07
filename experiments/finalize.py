"""Finalize the supersession-fix decision + distill front-end, in ONE process.

Runs three experiments sequentially over a single CAR connection (no concurrent
clients -> no daemon channel contention):

  A. Regression + confirm @ qwen3-4b: memgine vs memgine_safe_supersession on
     supersession (does the fix help?), repair_propagation + causality (does
     hiding old values HURT tracks that need old->new deltas?).
  B. Capability @ qwen3-30b-a3b: same A/B on supersession only -- does a stronger
     model resurrect less, shrinking the fix's benefit (=> adaptive) or not (=> default off)?
  C. Distill front-end @ qwen3-4b: memgine vs memgine_distill on supersession_detection
     (implicit) -- does CAR distillation give conversational state real supersession?
"""

from __future__ import annotations

import json

from statebench.baselines import get_baseline
from statebench.runner.harness import EvaluationHarness, load_timelines


def aggregate(results):
    n = len(results)
    if n == 0:
        return {"n": 0}
    with_mnm = [r for r in results if r.must_not_mention]
    return {
        "n": n,
        "decision_accuracy": sum(r.decision_correct for r in results) / n,
        "sfrr": sum(r.resurrected_superseded for r in results) / n,
        "mnm_violation_rate": (
            sum(1 for r in with_mnm if r.must_not_mention_violations) / len(with_mnm)
        )
        if with_mnm
        else 0.0,
        "avg_tokens": sum(r.tokens_used for r in results) / n,
    }


def run(dataset, model, baselines, tracks, label, out):
    timelines = [t for t in load_timelines(dataset) if t.track in tracks]
    harness = EvaluationHarness(model=model, provider="car", use_llm_judge=True)
    print(f"\n=== {label}  (model={model}, {len(timelines)} timelines) ===")
    block = {}
    for baseline in baselines:
        block[baseline] = {}
        for track in tracks:
            tls = [t for t in timelines if t.track == track]
            strat = get_baseline(baseline, token_budget=8000)
            qres = []
            for tl in tls:
                qres.extend(harness.run_timeline(tl, strat))
            m = aggregate(qres)
            block[baseline][track] = m
            print(
                f"  {baseline:26s} {track:24s} n={m['n']:3d} "
                f"acc={m.get('decision_accuracy',0):.3f} SFRR={m.get('sfrr',0):.3f} "
                f"MNMv={m.get('mnm_violation_rate',0):.3f}"
            )
    out[label] = {"model": model, "tracks": tracks, "results": block}


def main():
    DATA = "data/releases/v1.0/dev.jsonl"
    out = {}

    # A. regression + supersession confirm @ 4b
    run(
        DATA, "mlx/qwen3-4b:4bit",
        ["memgine", "memgine_safe_supersession"],
        ["supersession", "repair_propagation", "causality"],
        "A_regression_4b", out,
    )
    # B. capability @ 30b (supersession only)
    run(
        DATA, "mlx/qwen3-30b-a3b:4bit",
        ["memgine", "memgine_safe_supersession"],
        ["supersession"],
        "B_capability_30b", out,
    )
    # C. distill front-end @ 4b (implicit)
    run(
        DATA, "mlx/qwen3-4b:4bit",
        ["memgine", "memgine_distill"],
        ["supersession_detection"],
        "C_distill_4b", out,
    )

    with open("experiments/results/finalize.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved -> experiments/results/finalize.json")


if __name__ == "__main__":
    main()
