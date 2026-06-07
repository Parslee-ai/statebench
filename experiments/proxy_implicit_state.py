"""Task-1 proxy: does Memgine's edge survive when state is *implicit* in text?

Memgine derives its supersession/constraint/dependency intelligence entirely
from structured StateWrite/Supersession events (engine.ingest_statebench_event).
Conversational benchmarks like LoCoMo have no such events -- state lives only in
the dialogue text. This experiment proxies that gap with data we already have:

  * supersession            -- EXPLICIT detection_mode: structured Supersession
                               events are present (Memgine's home turf).
  * supersession_detection  -- IMPLICIT detection_mode: the supersession exists
                               only in conversation text, no structured event.

If Memgine's advantage over text-dumping baselines (transcript_replay) collapses
on the implicit track, that is direct evidence Memgine needs an NL->fact
distillation front-end to be competitive on conversational memory.

All generation + judging run through the local CAR daemon (provider="car").
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from statebench.baselines import get_baseline
from statebench.runner.harness import EvaluationHarness, load_timelines

BASELINES = ["memgine", "state_based", "transcript_replay"]
TRACKS = ["supersession", "supersession_detection"]  # explicit, implicit


def aggregate(results):
    """Return dict of metrics over a list[QueryResult]."""
    n = len(results)
    if n == 0:
        return {"n": 0}
    correct = sum(1 for r in results if r.decision_correct)
    resurrected = sum(1 for r in results if r.resurrected_superseded)
    with_mnm = [r for r in results if r.must_not_mention]
    mnm_viol = sum(1 for r in with_mnm if r.must_not_mention_violations)
    return {
        "n": n,
        "decision_accuracy": correct / n,
        "sfrr": resurrected / n,  # superseded-fact resurrection rate
        "mnm_violation_rate": (mnm_viol / len(with_mnm)) if with_mnm else 0.0,
        "avg_tokens": sum(r.tokens_used for r in results) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/releases/v1.0/dev.jsonl")
    ap.add_argument("--model", default="mlx/qwen3-4b:4bit")
    ap.add_argument("--provider", default="car")
    ap.add_argument("--baselines", nargs="+", default=BASELINES)
    ap.add_argument("--tracks", nargs="+", default=TRACKS)
    ap.add_argument("--no-llm-judge", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="cap timelines per track")
    ap.add_argument("--out", default="experiments/results/proxy_implicit_state.json")
    args = ap.parse_args()

    timelines = list(load_timelines(args.dataset))
    by_track = defaultdict(list)
    for t in timelines:
        if t.track in args.tracks:
            by_track[t.track].append(t)
    if args.limit:
        for k in by_track:
            by_track[k] = by_track[k][: args.limit]

    harness = EvaluationHarness(
        model=args.model, provider=args.provider, use_llm_judge=not args.no_llm_judge
    )

    # results[baseline][track] = metrics
    out = {"model": args.model, "provider": args.provider, "results": {}}
    for baseline in args.baselines:
        out["results"][baseline] = {}
        for track in args.tracks:
            tls = by_track.get(track, [])
            strat = get_baseline(baseline, token_budget=8000)
            qres = []
            for tl in tls:
                qres.extend(harness.run_timeline(tl, strat))
            m = aggregate(qres)
            out["results"][baseline][track] = m
            print(
                f"{baseline:18s} {track:24s} "
                f"n={m['n']:3d} acc={m.get('decision_accuracy',0):.3f} "
                f"SFRR={m.get('sfrr',0):.3f} MNMv={m.get('mnm_violation_rate',0):.3f} "
                f"tok={m.get('avg_tokens',0):.0f}"
            )

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # Markdown summary + explicit->implicit delta (the headline).
    print("\n### SFRR (superseded-fact resurrection, lower=better)\n")
    print("| baseline | explicit | implicit | implicit-explicit |")
    print("|---|---|---|---|")
    for b in args.baselines:
        e = out["results"][b].get("supersession", {}).get("sfrr")
        i = out["results"][b].get("supersession_detection", {}).get("sfrr")
        if e is not None and i is not None:
            print(f"| {b} | {e:.3f} | {i:.3f} | {i-e:+.3f} |")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
