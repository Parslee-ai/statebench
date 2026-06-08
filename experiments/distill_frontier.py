"""Does a FRONTIER extractor crack implicit/conversational supersession?

The earlier memgine_distill negative result (acc 0.93->0.60, SFRR 0.47->0.87) was
extraction-quality-bound: local qwen3-4b/8b produced inconsistent keys and missed
semantic/authority-based supersession. This re-runs the implicit-track A/B with a
gpt-5.4 EXTRACTOR (only extraction is paid; generation + judge stay local qwen3-4b,
matching the earlier comparison so the delta isolates extraction quality).

  memgine          : no distill (raw transcript in working set)
  memgine_distill  : gpt-5.4 distills facts + reasoning-based supersession detection
"""

from __future__ import annotations

import json

from statebench.baselines import get_baseline
from statebench.runner.harness import EvaluationHarness, load_timelines

DATA = "data/releases/v1.0/dev.jsonl"
TRACK = "supersession_detection"
GEN_MODEL = "mlx/qwen3-4b:4bit"
EXTRACT_MODEL = "openai/gpt-5.4:latest"


def agg(rs):
    n = len(rs)
    wm = [r for r in rs if r.must_not_mention]
    return {
        "n": n,
        "acc": sum(r.decision_correct for r in rs) / n if n else 0,
        "sfrr": sum(r.resurrected_superseded for r in rs) / n if n else 0,
        "mnmv": (sum(1 for r in wm if r.must_not_mention_violations) / len(wm)) if wm else 0.0,
    }


def main():
    timelines = [t for t in load_timelines(DATA) if t.track == TRACK]
    harness = EvaluationHarness(model=GEN_MODEL, provider="car", use_llm_judge=True)
    print(f"{TRACK} n={len(timelines)}  gen={GEN_MODEL}  extractor={EXTRACT_MODEL}")
    out = {}
    arms = [
        ("memgine", dict()),
        ("memgine_distill", dict(extract_model=EXTRACT_MODEL)),
    ]
    for name, kw in arms:
        strat = get_baseline(name, token_budget=8000, **kw)
        qres = []
        for tl in timelines:
            qres.extend(harness.run_timeline(tl, strat))
        m = agg(qres)
        out[name] = m
        print(f"  {name:18s} acc={m['acc']:.3f} SFRR={m['sfrr']:.3f} MNMv={m['mnmv']:.3f}")
    with open("experiments/results/distill_frontier.json", "w") as f:
        json.dump(out, f, indent=2)
    b, d = out["memgine"], out["memgine_distill"]
    print(f"\ndistill vs baseline: dAcc={d['acc']-b['acc']:+.3f} dSFRR={d['sfrr']-b['sfrr']:+.3f}")
    print("(earlier local-extractor distill: acc 0.93->0.60, SFRR 0.47->0.87)")


if __name__ == "__main__":
    main()
