"""Controlled isolation of the supersession-cascade's marginal effect.

Runs three Memgine variants in ONE process at larger n so the cascade's effect
is separable from qwen sampling noise:
  base       : show_superseded_values=True,  cascade=False   (default frontier)
  annotation : show_superseded_values=False, cascade=False   (annotation fix only)
  cascade    : show_superseded_values=False, cascade=True     (annotation + transcript cascade)

All inference via CAR (local qwen3-4b).
"""

from __future__ import annotations

import json

from statebench.memgine.strategy import MemgineStrategy
from statebench.runner.harness import EvaluationHarness, load_timelines

DATA = "data/releases/v1.0/full.jsonl"
MODEL = "mlx/qwen3-4b:4bit"
TRACK = "supersession"

VARIANTS = {
    "base": dict(show_superseded_values=True, cascade_supersession_to_transcript=False),
    "annotation": dict(show_superseded_values=False, cascade_supersession_to_transcript=False),
    "cascade": dict(show_superseded_values=False, cascade_supersession_to_transcript=True),
}


def agg(results):
    n = len(results)
    wm = [r for r in results if r.must_not_mention]
    return {
        "n": n,
        "acc": sum(r.decision_correct for r in results) / n,
        "sfrr": sum(r.resurrected_superseded for r in results) / n,
        "mnmv": (sum(1 for r in wm if r.must_not_mention_violations) / len(wm)) if wm else 0.0,
    }


def main():
    timelines = [t for t in load_timelines(DATA) if t.track == TRACK]
    harness = EvaluationHarness(model=MODEL, provider="car", use_llm_judge=True)
    out = {}
    print(f"{TRACK} n={len(timelines)} @ {MODEL}")
    for name, flags in VARIANTS.items():
        strat = MemgineStrategy(token_budget=8000, **flags)
        qres = []
        for tl in timelines:
            qres.extend(harness.run_timeline(tl, strat))  # handles reset + init
        m = agg(qres)
        out[name] = m
        print(f"  {name:11s} acc={m['acc']:.3f} SFRR={m['sfrr']:.3f} MNMv={m['mnmv']:.3f}")
    with open("experiments/results/cascade_ab.json", "w") as f:
        json.dump(out, f, indent=2)
    b, a, c = out["base"], out["annotation"], out["cascade"]
    print(f"\nSFRR: base {b['sfrr']:.3f} -> annotation {a['sfrr']:.3f} -> +cascade {c['sfrr']:.3f}")
    print(f"acc : base {b['acc']:.3f} -> annotation {a['acc']:.3f} -> +cascade {c['acc']:.3f}")


if __name__ == "__main__":
    main()
