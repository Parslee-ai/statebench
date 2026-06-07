"""Task-2 diagnostic: is there a retrieval-drop problem for a second hop to fix?

Memgine includes ALL valid facts and only *compresses* (not drops) them under
budget pressure, query-blind, at query time (engine.build_context -> maybe_compact).
A compressed fact loses its relevance ranking, dependency notes, and
recalculation inlining -- it becomes a bare `key=value` line in a summary blob.

MRAgent-style "second hop" (query-conditioned re-expansion of the dependency
closure of the top-relevant facts) is only worth building if compression
actually (a) triggers on multi-hop timelines and (b) hurts accuracy.

This script establishes that, by running baseline memgine on the multi-hop
tracks across budget/padding regimes and reporting accuracy + how often
compaction fired + how many facts ended up compacted.

All inference through CAR (provider="car").
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from statebench.baselines import get_baseline
from statebench.runner.harness import EvaluationHarness, load_timelines

MULTIHOP_TRACKS = ["causality", "repair_propagation", "commitment_durability"]

# (token_budget, pad_facts) regimes: loose -> tight / padded to force compaction.
REGIMES = [
    (8000, 0),
    (8000, 40),
    (1500, 0),
    (1500, 40),
]


def run_regime(dataset, tracks, model, budget, pad, use_llm_judge):
    timelines = [t for t in load_timelines(dataset) if t.track in tracks]
    harness = EvaluationHarness(
        model=model, provider="car", use_llm_judge=use_llm_judge,
        token_budget=budget, pad_facts=pad,
    )
    strat = get_baseline("memgine", token_budget=budget)
    per_track = defaultdict(list)
    for tl in timelines:
        for r in harness.run_timeline(tl, strat):
            per_track[tl.track].append(r)
    summary = {}
    allr = []
    for track, rs in per_track.items():
        allr.extend(rs)
        n = len(rs)
        summary[track] = {
            "n": n,
            "decision_accuracy": sum(r.decision_correct for r in rs) / n,
            "compaction_rate": sum(r.compaction_triggered for r in rs) / n,
        }
    n = len(allr)
    overall = {
        "n": n,
        "decision_accuracy": sum(r.decision_correct for r in allr) / n if n else 0,
        "compaction_rate": sum(r.compaction_triggered for r in allr) / n if n else 0,
    }
    return overall, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/releases/v1.0/dev.jsonl")
    ap.add_argument("--model", default="mlx/qwen3-4b:4bit")
    ap.add_argument("--tracks", nargs="+", default=MULTIHOP_TRACKS)
    ap.add_argument("--no-llm-judge", action="store_true")
    ap.add_argument("--out", default="experiments/results/task2_diagnostic.json")
    args = ap.parse_args()

    out = {"model": args.model, "tracks": args.tracks, "regimes": []}
    print(f"{'budget':>7} {'pad':>4} {'n':>4} {'acc':>6} {'compact%':>9}")
    for budget, pad in REGIMES:
        overall, per_track = run_regime(
            args.dataset, args.tracks, args.model, budget, pad,
            not args.no_llm_judge,
        )
        out["regimes"].append(
            {"budget": budget, "pad": pad, "overall": overall, "per_track": per_track}
        )
        print(
            f"{budget:>7} {pad:>4} {overall['n']:>4} "
            f"{overall['decision_accuracy']:>6.3f} {overall['compaction_rate']:>9.3f}"
        )

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {args.out}")
    # Verdict hint
    loose = out["regimes"][0]["overall"]
    tight = max(out["regimes"][1:], key=lambda r: r["overall"]["compaction_rate"])
    print(
        f"\nloose(8000/0) acc={loose['decision_accuracy']:.3f} compact={loose['compaction_rate']:.3f} | "
        f"most-compacted({tight['budget']}/{tight['pad']}) "
        f"acc={tight['overall']['decision_accuracy']:.3f} compact={tight['overall']['compaction_rate']:.3f}"
    )
    print("If accuracy drops where compaction rises -> a second hop is worth building.")


if __name__ == "__main__":
    main()
