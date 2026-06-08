"""De-noised frontier confirmation: does safe-mode supersession rendering beat
frontier-mode on TRUE frontier models? n=180, two different-provider frontier
models, one sequential CAR connection.

Tests the assumption behind the adaptive default (frontier_capable -> show old
value). The n=27 dev signal said gpt-5.4 does BETTER with safe mode (hide old
value + cascade) on both accuracy and SFRR -- the opposite of the qwen3-30b-based
assumption. This confirms (or refutes) at larger n across two frontier models.

memgine          = frontier rendering (show_superseded_values=True, cascade=False)
memgine_safe_... = safe rendering      (show=False, cascade=True)
Judge runs on a local model; only the generations are paid.
"""

from __future__ import annotations

import json

from statebench.baselines import get_baseline
from statebench.runner.harness import EvaluationHarness, load_timelines

DATA = "data/releases/v1.0/full.jsonl"
TRACK = "supersession"
MODELS = ["openai/gpt-5.4:latest", "openai/gpt-4.1-mini:latest"]
BASELINES = ["memgine", "memgine_safe_supersession"]


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
    out = {}
    print(f"{TRACK} n={len(timelines)}")
    for model in MODELS:
        harness = EvaluationHarness(model=model, provider="car", use_llm_judge=True)
        out[model] = {}
        print(f"\n=== {model} ===")
        for baseline in BASELINES:
            strat = get_baseline(baseline, token_budget=8000)
            qres = []
            for tl in timelines:
                qres.extend(harness.run_timeline(tl, strat))
            m = agg(qres)
            out[model][baseline] = m
            print(f"  {baseline:26s} acc={m['acc']:.3f} SFRR={m['sfrr']:.3f} MNMv={m['mnmv']:.3f}")
        b = out[model]["memgine"]
        s = out[model]["memgine_safe_supersession"]
        print(f"  -> safe vs frontier: dAcc={s['acc']-b['acc']:+.3f} dSFRR={s['sfrr']-b['sfrr']:+.3f}")

    with open("experiments/results/frontier_confirm_big.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved -> experiments/results/frontier_confirm_big.json")


if __name__ == "__main__":
    main()
