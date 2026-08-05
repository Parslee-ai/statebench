"""Is the skills-vs-description gap real, or artifact-generation noise?

Two prior passes disagreed. Both reported the description-only arm at +- 0.0%
across three evaluation runs, which looked like precision and was in fact the
diagnostic: each pass evaluated **one** generated artifact three times. The
artifact is the thing under test and generating it is stochastic, so the
reported error bars described weak-model sampling and said nothing about the
variable that actually moved between passes.

    pass A:  skills 53.3%   description-only 53.3%   -> episodes add +0.0pp
    pass B:  skills 55.6%   description-only 46.7%   -> episodes add +8.9pp

This experiment points the compute at the right variance: N independent
generations per condition, each evaluated once, compared as distributions. Same
total cost, different question.

A condition's spread here is the spread a practitioner actually faces, since
they get whichever artifact their one generation call happened to produce.

Usage::

    uv run python experiments/with_car_secret.py -- \\
        python experiments/artifact_variance.py --samples 5
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from statebench.baselines.skill_augmented import SkillAugmentedStrategy
from statebench.evaluation.governance_metrics import GovernanceMetrics
from statebench.evaluation.metrics import MetricsAggregator
from statebench.runner.harness import EvaluationHarness, load_timelines
from statebench.skills import SkillStore, distill_skills
from statebench.skills.controls import make_description_only_store
from statebench.skills.distill import evaluation_literals


def _answer(timeline) -> str:
    for event in timeline.events:
        if getattr(event, "type", "") == "query":
            gt = event.ground_truth
            mm = [(m if isinstance(m, str) else m.phrase) for m in gt.must_mention]
            return f"{gt.decision}" + (f" (must convey: {'; '.join(mm[:2])})" if mm else "")
    return ""


def evaluate(args, store, family, timelines) -> dict:
    """One evaluation pass of one artifact set."""
    harness = EvaluationHarness(
        model=args.weak_model,
        provider=args.weak_provider,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
    )
    strategy = (
        harness._make_strategy("memgine")
        if store is None
        else SkillAugmentedStrategy(skills=store, task_family=family)
    )
    agg = MetricsAggregator(baseline="x", model=args.weak_model)
    results = []
    for tl in timelines:
        for r in harness.run_timeline(tl, strategy):
            agg.add_result(r)
            results.append(r)
    m = agg.compute_benchmark_metrics()
    gov = GovernanceMetrics.from_results(results)
    return {
        "decision_accuracy": m.overall_decision_accuracy,
        "sfrr": m.overall_sfrr,
        # Separates the safety question the prior runs could not: is the
        # governance cost attached to procedures in general, or to distilled
        # ones specifically?
        "gbr": gov.conditional_bypass_rate,
    }


def summarize(name: str, samples: list[dict]) -> dict:
    out: dict = {"condition": name, "n_samples": len(samples)}
    for key in ("decision_accuracy", "sfrr", "gbr"):
        vals = [s[key] for s in samples]
        out[key] = st.mean(vals)
        out[key + "_std"] = st.pstdev(vals) if len(vals) > 1 else 0.0
        out[key + "_min"] = min(vals)
        out[key + "_max"] = max(vals)
    print(
        f"  {name:22s} acc={out['decision_accuracy']:6.1%} +-{out['decision_accuracy_std']:5.1%}"
        f"  [{out['decision_accuracy_min']:.1%}-{out['decision_accuracy_max']:.1%}]"
        f"  gbr={out['gbr']:5.1%}  n={len(samples)} generations"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", default="data/releases/v1.0")
    ap.add_argument("--track", default="repair_propagation")
    ap.add_argument("--weak-model", default="mlx/qwen3-4b:4bit")
    ap.add_argument("--weak-provider", default="car")
    ap.add_argument("--distill-model", default="gpt-5.6-sol")
    ap.add_argument("--distill-provider", default="openai")
    ap.add_argument("--judge-provider", default="openai")
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--samples", type=int, default=5, help="independent generations")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--distill-episodes", type=int, default=12)
    ap.add_argument("--out", default="experiments/results/artifact_variance.json")
    args = ap.parse_args()

    from statebench.runner.completion import complete

    def _c(prompt, model, max_tokens):
        return complete(
            prompt, provider=args.distill_provider, model=model, max_tokens=max_tokens
        )

    dev = [
        t for t in load_timelines(Path(args.release) / "dev.jsonl")
        if t.track == args.track
    ][: args.limit]
    train = [
        t for t in load_timelines(Path(args.release) / "train.jsonl")
        if t.track == args.track
    ][: args.distill_episodes]
    forbidden = evaluation_literals(dev)

    print(
        f"track={args.track}  {len(dev)} eval timelines  "
        f"{args.samples} independent generations per condition\n"
    )

    report: dict = {"track": args.track, "samples": args.samples, "conditions": {}}

    # Floor: no artifact, so generation variance does not apply. Repeated to
    # characterise weak-model variance for comparison against artifact variance.
    print("floor (no artifact, repeated for weak-model variance):")
    floor = [evaluate(args, None, None, dev) for _ in range(args.samples)]
    report["conditions"]["weak_alone"] = summarize("weak_alone", floor)

    print("\nindependent artifact generations:")
    distilled: list[dict] = []
    desc: list[dict] = []
    artifacts: dict[str, list] = {"distilled": [], "description_only": []}

    for i in range(args.samples):
        # Vary the episode window so each generation sees a different sample of
        # worked examples, which is what a practitioner distilling twice would get.
        offset = (i * 3) % max(1, len(train))
        rotated = train[offset:] + train[:offset]
        skills, rejected = distill_skills(
            timelines=rotated,
            answers=[_answer(t) for t in rotated],
            task_family=args.track,
            complete=_c,
            model=args.distill_model,
            forbidden_literals=forbidden,
            episodes_per_skill=3,
            max_skills=4,
        )
        store = SkillStore()
        for s in skills:
            store.add(s)
        artifacts["distilled"].append(
            {"n": len(skills), "rejected": len(rejected),
             "principles": [s.principle[:90] for s in skills]}
        )
        distilled.append(evaluate(args, store, args.track, dev) if len(store) else
                         {"decision_accuracy": 0.0, "sfrr": 0.0, "gbr": 0.0})

        d_store = make_description_only_store(args.track, _c, args.distill_model)
        artifacts["description_only"].append(
            {"n": len(d_store),
             "principles": [s.principle[:90] for s in d_store.all()]}
        )
        desc.append(evaluate(args, d_store, args.track, dev) if len(d_store) else
                    {"decision_accuracy": 0.0, "sfrr": 0.0, "gbr": 0.0})
        print(f"  generation {i+1}/{args.samples} done", flush=True)

    print()
    report["conditions"]["distilled"] = summarize("distilled (episodes)", distilled)
    report["conditions"]["description_only"] = summarize("description_only", desc)
    report["artifacts"] = artifacts

    d = report["conditions"]["distilled"]
    o = report["conditions"]["description_only"]
    f = report["conditions"]["weak_alone"]
    gap = d["decision_accuracy"] - o["decision_accuracy"]
    pooled = ((d["decision_accuracy_std"] ** 2 + o["decision_accuracy_std"] ** 2) / 2) ** 0.5
    report["episodes_gap"] = gap
    report["pooled_generation_std"] = pooled
    report["gap_over_noise"] = gap / pooled if pooled > 1e-9 else None

    print(
        f"\n  episodes vs description-only: {100*gap:+.1f}pp"
        f"   pooled generation std {100*pooled:.1f}pp"
        + (f"   ratio {gap/pooled:.2f}x" if pooled > 1e-9 else "")
    )
    print(
        f"  artifact-generation std ({100*pooled:.1f}pp) vs "
        f"weak-model std ({100*f['decision_accuracy_std']:.1f}pp)"
    )
    if pooled > 1e-9 and abs(gap) < pooled:
        print("  -> the gap is SMALLER than generation noise; not resolvable at this n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
