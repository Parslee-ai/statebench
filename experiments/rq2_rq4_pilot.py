"""RQ2/RQ4 pilot: does reconstruction bypass governance, and does it invent state?

This is the decisive experiment for Paper 3 and deliberately runs FIRST. If
reconstruction does not bypass governance and does not synthesize unsupported
state, the paper's central claim is false and the applicability suite need not
be built.

Design. Each counterfactual axis contributes matched pairs that differ in one
governance variable. Three quantities are reported per baseline:

* **Pair accuracy** — fraction of pairs with BOTH sides correct. This is the
  discrimination measure: always-answer and always-refuse both score zero.
  (Δ is also printed, but zero Δ is the TARGET — the sides need different
  answers, so getting both right gives 100%/100% and Δ=0.)
* **GBR** — asserted something state resolution had excluded. Scored against
  the engine's own audit record, so a leak is information the model was never
  shown.
* **URR** — asserted a concrete value traceable to neither an admitted fact nor
  a declared binding.

Read pair accuracy separately for engine-decidable and model-decidable axes.
Only the engine-decidable ones support the strong claim, because there the
discriminating fact never enters the model's context and no amount of
task-success training could supply it.

Usage::

    uv run python experiments/rq2_rq4_pilot.py --dry-run          # plumbing only
    uv run python experiments/rq2_rq4_pilot.py --model mlx/qwen3-4b:4bit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from statebench.baselines.reconstructive import Reconstructor
from statebench.evaluation.governance_metrics import (
    GovernanceMetrics,
    counterfactual_sensitivity,
    format_sensitivity_table,
)
from statebench.evaluation.metrics import MetricsAggregator
from statebench.generator.counterfactual import AXES, CounterfactualGenerator
from statebench.runner.harness import EvaluationHarness

DEFAULT_BASELINES = [
    "memgine",                              # Γ alone
    "reconstructive_rag_prompted",          # C alone, governance by instruction
    "reconstructive_rag_engine_filtered",   # Γ ∘ C
    "reconstructive_rag_validated",         # Γ ∘ C ∘ V
]


def _stub_response(prompt: str) -> str:
    """Canned answer for --dry-run.

    Deliberately leaky: it echoes any dollar figure it can see so the plumbing
    for GBR/URR is exercised rather than trivially returning zero.
    """
    import re

    amounts = re.findall(r"\$\s?\d[\d,]*", prompt)
    if amounts:
        return f"Yes, the figure is {amounts[0]} based on the current record."
    return "I do not have that information in the current state."


def _stub_reconstruction(prompt: str) -> str:
    """Canned typed verdicts for --dry-run.

    The reconstruction stage makes its own LLM call inside build_context, so
    stubbing only the generation path leaves it live. This returns a plausible
    ADAPT bound to the first admitted fact, which exercises the parser, the
    validator and the bindings path rather than short-circuiting them.
    """
    import json
    import re

    experiences = re.findall(r"\[(E-[^\]]+)\]", prompt)
    facts = re.findall(r"\[(F-[^\]]+)\]", prompt)
    payload = []
    for i, exp in enumerate(dict.fromkeys(experiences)):
        if i == 0 and facts:
            payload.append(
                {
                    "experience_id": exp,
                    "verdict": "ADAPT",
                    "guidance": "reuse the prior approach with the current value",
                    "bindings": {facts[0]: "stub"},
                    "rationale": "dry-run stub",
                }
            )
        else:
            payload.append(
                {
                    "experience_id": exp,
                    "verdict": "REJECT",
                    "rationale": "dry-run stub",
                }
            )
    return json.dumps(payload)


def run(args: argparse.Namespace) -> dict:
    generator = CounterfactualGenerator(seed=args.seed)
    pairs = generator.generate_pairs(axis_names=args.axes or None, per_axis=args.per_axis)

    audits = generator.audit_all(pairs)
    inadmissible = [f"{a.axis}/{a.scenario}: {a.problems}" for a in audits if not a.ok]
    if inadmissible:
        raise SystemExit("inadmissible pairs — fix before measuring:\n" + "\n".join(inadmissible))

    axis_of: dict[str, str] = {}
    for pair in pairs:
        axis_of[pair.positive.id] = pair.axis
        axis_of[pair.counterfactual.id] = pair.axis
    decidable_by = {name: axis.decidable_by for name, axis in AXES.items()}

    timelines = [tl for pair in pairs for tl in pair.as_timelines()]
    print(
        f"{len(pairs)} pairs / {len(timelines)} timelines across {len(AXES)} axes "
        f"(mean lexical similarity "
        f"{sum(a.lexical_similarity for a in audits) / len(audits):.3f})\n"
    )

    report: dict = {
        "model": args.model,
        "provider": args.provider,
        "dry_run": args.dry_run,
        "n_pairs": len(pairs),
        "baselines": {},
    }

    for baseline in args.baselines:
        harness = EvaluationHarness(
            model=args.model,
            provider=args.provider,
            use_llm_judge=not (args.dry_run or args.no_judge),
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
        )
        if args.dry_run:
            harness._generate_response = lambda sp, up: (_stub_response(up), 0, 0)  # type: ignore[method-assign]
            # The reconstruction stage calls the model independently of the
            # harness; stub it too or --dry-run still hits the network.
            Reconstructor._complete = lambda self, prompt: _stub_reconstruction(prompt)  # type: ignore[method-assign]

        strategy = harness._make_strategy(baseline)
        aggregator = MetricsAggregator(baseline=baseline, model=args.model)
        results = []
        for i, timeline in enumerate(timelines, 1):
            for r in harness.run_timeline(timeline, strategy):
                aggregator.add_result(r)
                results.append(r)
            if args.verbose and i % 10 == 0:
                print(f"  {baseline}: {i}/{len(timelines)} timelines")

        sensitivity = counterfactual_sensitivity(results, axis_of, decidable_by)
        governance = GovernanceMetrics.from_results(results)

        engine_axes = [s for s in sensitivity.values() if s.decidable_by == "engine"]
        model_axes = [s for s in sensitivity.values() if s.decidable_by == "model"]
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731

        print(f"\n=== {baseline} (judge {harness.judge.descriptor}) ===")
        print(format_sensitivity_table(sensitivity))
        print(
            f"\n  mean PAIR acc engine-decidable : "
            f"{mean([s.pair_accuracy for s in engine_axes]):.1%}"
            f"\n  mean PAIR acc model-decidable  : "
            f"{mean([s.pair_accuracy for s in model_axes]):.1%}"
            f"\n  GBR (all queries)       : {governance.governance_bypass_rate:.1%}"
            f"\n  GBR (queries w/ exclusions): {governance.conditional_bypass_rate:.1%}"
            f"  [n={governance.queries_with_exclusions}]"
            f"\n  URR                     : {governance.unsupported_reconstruction_rate:.1%}"
        )

        report["baselines"][baseline] = {
            "judge": harness.judge.descriptor,
            "sensitivity": {
                name: {
                    "positive": s.positive_accuracy,
                    "counterfactual": s.counterfactual_accuracy,
                    "delta": s.delta,
                    "pair_accuracy": s.pair_accuracy,
                    "decidable_by": s.decidable_by,
                    "n_pairs": s.n_pairs,
                }
                for name, s in sensitivity.items()
            },
            "mean_delta_engine": mean([s.delta for s in engine_axes]),
            "mean_pair_engine": mean([s.pair_accuracy for s in engine_axes]),
            "mean_delta_model": mean([s.delta for s in model_axes]),
            "mean_pair_model": mean([s.pair_accuracy for s in model_axes]),
            "governance_bypass_rate": governance.governance_bypass_rate,
            "conditional_bypass_rate": governance.conditional_bypass_rate,
            "queries_with_exclusions": governance.queries_with_exclusions,
            "unsupported_reconstruction_rate": governance.unsupported_reconstruction_rate,
        }

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mlx/qwen3-4b:4bit")
    ap.add_argument("--provider", default="car")
    ap.add_argument("--judge-provider", default="car")
    ap.add_argument("--judge-model", default="mlx/qwen3-4b:4bit")
    ap.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    ap.add_argument("--per-axis", type=int, default=2, help="scenarios per axis")
    ap.add_argument(
        "--axes",
        nargs="+",
        default=None,
        help=(
            "restrict to named axes. RQ2's decisive slice is the engine-decidable "
            "set (access_control scope_binding actor_isolation), where the "
            "discriminating fact never reaches the model."
        ),
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true", help="stub the model; check plumbing")
    ap.add_argument(
        "--no-judge",
        action="store_true",
        help=(
            "skip LLM judging. GBR and URR are deterministic, so RQ2/RQ4 need no "
            "judge; this drops ~a third of the calls and removes weak-judge noise. "
            "Delta becomes unreliable and should be ignored in this mode."
        ),
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="experiments/results/rq2_rq4_pilot.json")
    args = ap.parse_args()

    report = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
