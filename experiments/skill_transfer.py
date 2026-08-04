"""RQ2/RQ3: does frontier-distilled skill transfer lift a weak model?

Distillation draws from the **train** split and evaluation from **dev**, so the
episode sets are disjoint by construction rather than by sampling discipline.
A scrub pass then rejects any skill containing an evaluation-set literal, because
templates reuse entity pools across splits and disjoint episodes are not
sufficient to guarantee disjoint *content*.

Arms, each isolating one deflationary explanation:

  weak_alone          floor
  weak_plus_skills    the effect under test
  weak_plus_hand      hand-written procedure -- is this just prompt engineering?
  weak_plus_offfamily skills from the wrong family -- over-generalization (RQ4)
  frontier_alone      ceiling

Reported: gap closure = (skills - alone) / (frontier - alone), with the raw gap
alongside because the ratio is unstable when the denominator is small.

Usage::

    uv run python experiments/with_car_secret.py -- \\
        python experiments/skill_transfer.py --track supersession
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from statebench.baselines.skill_augmented import (
    SkillAugmentedStrategy,
    make_handwritten_store,
)
from statebench.evaluation.metrics import MetricsAggregator
from statebench.runner.harness import EvaluationHarness, load_timelines
from statebench.skills import SkillStore, distill_skills
from statebench.skills.distill import evaluation_literals


def _first_query_answer(timeline) -> str:
    for event in timeline.events:
        if getattr(event, "type", "") == "query":
            gt = event.ground_truth
            mm = [
                (m if isinstance(m, str) else m.phrase) for m in gt.must_mention
            ]
            return f"{gt.decision}" + (f" (must convey: {'; '.join(mm[:2])})" if mm else "")
    return ""


def build_skills(args, eval_timelines) -> tuple[SkillStore, list[dict]]:
    """Distill from the train split, scrubbed against the evaluation set."""
    from statebench.runner.completion import complete

    train = [
        t for t in load_timelines(Path(args.release) / "train.jsonl")
        if t.track == args.track
    ][: args.distill_episodes]
    answers = [_first_query_answer(t) for t in train]
    forbidden = evaluation_literals(eval_timelines)

    print(
        f"distilling from {len(train)} train episodes with {args.distill_model}; "
        f"{len(forbidden)} evaluation literals on the deny-list"
    )
    skills, rejected = distill_skills(
        timelines=train,
        answers=answers,
        task_family=args.track,
        complete=lambda p, model, max_tokens: complete(
            p, provider=args.distill_provider, model=model, max_tokens=max_tokens
        ),
        model=args.distill_model,
        forbidden_literals=forbidden,
        episodes_per_skill=args.episodes_per_skill,
        max_skills=args.max_skills,
    )
    store = SkillStore()
    for s in skills:
        store.add(s)
    print(f"  accepted {len(skills)}, rejected {len(rejected)}")
    for r in rejected:
        print(f"    batch {r['batch']}: {r['reason']} {r.get('literals', '')}")
    for s in skills:
        print(f"  {s.skill_id}: {s.principle[:100]}")
    return store, rejected


def run_arm(args, name, timelines, model, provider, store, family) -> dict:
    harness = EvaluationHarness(
        model=model,
        provider=provider,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
    )
    if store is None:
        strategy = harness._make_strategy("memgine")
    else:
        strategy = SkillAugmentedStrategy(
            token_budget=8000, model=model, skills=store, task_family=family
        )
    agg = MetricsAggregator(baseline=name, model=model)
    fired = 0
    total = 0
    for tl in timelines:
        for r in harness.run_timeline(tl, strategy):
            agg.add_result(r)
            total += 1
            if getattr(strategy, "last_retrieved", None):
                fired += 1
    m = agg.compute_benchmark_metrics()
    out = {
        "arm": name,
        "model": model,
        "n_queries": m.total_queries,
        "decision_accuracy": m.overall_decision_accuracy,
        "sfrr": m.overall_sfrr,
        "must_mention_rate": m.overall_must_mention_rate,
        "skill_firing_rate": fired / total if total else 0.0,
    }
    print(
        f"  {name:24s} acc={out['decision_accuracy']:6.1%} "
        f"sfrr={out['sfrr']:5.1%} fired={out['skill_firing_rate']:5.1%} n={out['n_queries']}"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", default="data/releases/v1.0")
    ap.add_argument("--track", default="supersession")
    ap.add_argument("--offfamily-track", default="scope_permission")
    ap.add_argument("--weak-model", default="mlx/qwen3-4b:4bit")
    ap.add_argument("--weak-provider", default="car")
    ap.add_argument("--frontier-model", default="gpt-5.6-sol")
    ap.add_argument("--frontier-provider", default="openai")
    ap.add_argument("--distill-model", default="gpt-5.6-sol")
    ap.add_argument("--distill-provider", default="openai")
    ap.add_argument("--judge-provider", default="openai")
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--distill-episodes", type=int, default=12)
    ap.add_argument("--episodes-per-skill", type=int, default=3)
    ap.add_argument("--max-skills", type=int, default=4)
    ap.add_argument("--limit", type=int, default=20, help="evaluation timelines")
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--out", default="experiments/results/skill_transfer.json")
    args = ap.parse_args()

    dev = [
        t for t in load_timelines(Path(args.release) / "dev.jsonl")
        if t.track == args.track
    ][: args.limit]
    print(f"track={args.track}  eval on {len(dev)} dev timelines (train/dev disjoint)\n")

    store, rejected = build_skills(args, dev)
    if len(store) == 0:
        raise SystemExit("no skills survived scrubbing — cannot run the transfer arms")

    off_store, _ = (SkillStore(), [])
    for s in store.all():
        clone = type(s)(**{**s.to_dict(), "task_family": args.offfamily_track})
        off_store.add(clone)

    print()
    weak = (args.weak_model, args.weak_provider)
    front = (args.frontier_model, args.frontier_provider)
    all_arms = {
        "weak_alone": (weak, None, None),
        "weak_plus_skills": (weak, store, args.track),
        "weak_plus_hand": (weak, make_handwritten_store(), None),
        "weak_plus_offfamily": (weak, off_store, args.offfamily_track),
        "frontier_alone": (front, None, None),
    }
    chosen = args.arms or list(all_arms)
    results = {}
    for name in chosen:
        if name not in all_arms:
            continue
        (mdl, prov), st_, fam = all_arms[name]
        results[name] = run_arm(args, name, dev, mdl, prov, st_, fam)

    report = {
        "track": args.track,
        "weak_model": args.weak_model,
        "frontier_model": args.frontier_model,
        "distill_model": args.distill_model,
        "n_skills": len(store),
        "n_rejected": len(rejected),
        "skills": [s.to_dict() for s in store.all()],
        "arms": results,
    }

    if {"weak_alone", "weak_plus_skills", "frontier_alone"} <= results.keys():
        lo = results["weak_alone"]["decision_accuracy"]
        hi = results["frontier_alone"]["decision_accuracy"]
        sk = results["weak_plus_skills"]["decision_accuracy"]
        gap = hi - lo
        report["gap"] = gap
        report["raw_gain"] = sk - lo
        report["gap_closure"] = (sk - lo) / gap if abs(gap) > 1e-9 else None
        print(
            f"\n  weak→frontier gap {100*gap:+.1f}pp | skills gain {100*(sk-lo):+.1f}pp"
            + (f" | closure {100*report['gap_closure']:.0f}%" if report["gap_closure"] is not None else "")
        )
        if "weak_plus_hand" in results:
            hand = results["weak_plus_hand"]["decision_accuracy"]
            print(f"  hand-written control gain {100*(hand-lo):+.1f}pp"
                  f"  → distillation adds {100*(sk-hand):+.1f}pp over prompt engineering")
        if "weak_plus_offfamily" in results:
            off = results["weak_plus_offfamily"]["decision_accuracy"]
            print(f"  off-family delta {100*(off-lo):+.1f}pp (negative = artifacts harm)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
