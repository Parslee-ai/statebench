"""Net-benefit + regression check of the supersession fixes across ALL tracks.

Compares frontier-mode memgine (show_superseded_values=True, cascade=False) vs
the supersession-safe mode (both fixes) on every track, to confirm the fixes help
the supersession-bearing tracks without regressing the others.

All inference via CAR (local qwen3-4b).
"""

from __future__ import annotations

import json
from collections import defaultdict

from statebench.baselines import get_baseline
from statebench.runner.harness import EvaluationHarness, load_timelines


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
    data = "data/releases/v1.0/dev.jsonl"
    model = "mlx/qwen3-4b:4bit"
    timelines = list(load_timelines(data))
    by_track = defaultdict(list)
    for t in timelines:
        by_track[t.track].append(t)

    harness = EvaluationHarness(model=model, provider="car", use_llm_judge=True)
    arms = {"frontier": "memgine", "safe": "memgine_safe_supersession"}

    out = {}
    for arm, baseline in arms.items():
        out[arm] = {}
        for track, tls in by_track.items():
            strat = get_baseline(baseline, token_budget=8000)
            qres = []
            for tl in tls:
                qres.extend(harness.run_timeline(tl, strat))
            out[arm][track] = agg(qres)

    with open("experiments/results/cross_track.json", "w") as f:
        json.dump(out, f, indent=2)

    # Per-track table with deltas; flag accuracy regressions.
    print(f"{'track':24s} {'acc_f':>6} {'acc_s':>6} {'dAcc':>6} | {'sfrr_f':>6} {'sfrr_s':>6} {'dSFRR':>6}")
    regressions = []
    for track in sorted(by_track):
        f, s = out["frontier"][track], out["safe"][track]
        dacc = s["acc"] - f["acc"]
        dsfrr = s["sfrr"] - f["sfrr"]
        flag = "  <-- acc regress" if dacc < -0.05 else ""
        if dacc < -0.05:
            regressions.append((track, dacc))
        print(
            f"{track:24s} {f['acc']:>6.3f} {s['acc']:>6.3f} {dacc:>+6.3f} | "
            f"{f['sfrr']:>6.3f} {s['sfrr']:>6.3f} {dsfrr:>+6.3f}{flag}"
        )
    # Overall (query-weighted)
    def overall(arm):
        allrs = [(out[arm][t]["n"], out[arm][t]) for t in by_track]
        N = sum(n for n, _ in allrs)
        return (
            sum(n * m["acc"] for n, m in allrs) / N,
            sum(n * m["sfrr"] for n, m in allrs) / N,
        )
    fa, fs = overall("frontier")
    sa, ss = overall("safe")
    print(f"\nOVERALL  acc {fa:.3f} -> {sa:.3f} ({sa-fa:+.3f}) | SFRR {fs:.3f} -> {ss:.3f} ({ss-fs:+.3f})")
    print(f"accuracy regressions (>5pp): {regressions or 'none'}")


if __name__ == "__main__":
    main()
