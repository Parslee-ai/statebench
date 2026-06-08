"""FKE/HP decomposition (TRACK, 2601.15495): is multi-hop failure reasoning-bound
or retrieval-bound? Measure them separately.

For each multi-hop query:
  FKE (Full Knowledge Entailment proxy): fraction of required must_mention
       evidence phrases actually present in the context Memgine assembled — i.e.
       "was the needed information surfaced?" (retrieval/surfacing axis).
  HP  (Holistic Pass): the decision was correct (answer axis).

The split of FAILURES (HP=0):
  FKE high  -> reasoning-bound (evidence was there, model reasoned wrong)
  FKE low   -> retrieval-bound (evidence wasn't surfaced)

Run at a loose budget (Memgine fits everything) and a tight budget (compaction may
drop evidence) to test TRACK's claim that it is BOTH-bound and regime-dependent.
All generation via CAR (local).
"""

from __future__ import annotations

import argparse

from statebench.baselines import get_baseline
from statebench.evaluation.rubric import contains_phrase
from statebench.runner.harness import EvaluationHarness, load_timelines

MULTIHOP = ["causality", "repair_propagation", "commitment_durability"]


def _fact_values(tl) -> list[str]:
    """All fact values asserted in a timeline (initial state + writes)."""
    vals = [f.value for f in tl.initial_state.persistent_facts]
    from statebench.schema.timeline import StateWrite, Supersession
    for ev in tl.events:
        if isinstance(ev, (StateWrite, Supersession)):
            vals += [w.value for w in ev.writes if getattr(w, "layer", None) == "persistent_facts"]
    return vals


def _fke(context: str, must_mention, fact_values) -> float:
    """Fraction of required INPUT-EVIDENCE phrases present in the assembled context.

    Only must_mention phrases that actually appear in some asserted fact value count
    as evidence (a phrase like "over budget" is a derived conclusion, not evidence,
    so it is excluded — it should not be in context). If a query needs no surfaced
    evidence, FKE = 1 (nothing to retrieve).
    """
    phrases = [p if isinstance(p, str) else getattr(p, "phrase", "") for p in must_mention]
    blob = " ".join(fact_values).lower()
    evidence = [p for p in phrases if p and p.lower() in blob]
    if not evidence:
        return 1.0
    hits = sum(1 for p in evidence if contains_phrase(context, p))
    return hits / len(evidence)


def run(dataset, model, budget, tracks, pad_facts=0):
    from statebench.runner.harness import generate_filler_facts
    timelines = [t for t in load_timelines(dataset) if t.track in tracks]
    harness = EvaluationHarness(model=model, provider="car", use_llm_judge=True, token_budget=budget)
    strat = get_baseline("memgine", token_budget=budget, model=model)
    rows = []
    from statebench.schema.timeline import Query
    for tl in timelines:
        strat.reset()
        if getattr(strat, "expects_initial_state", False):
            strat.initialize_from_state(tl.initial_state)
        # Distractor facts to force compaction to actually bind (a small budget
        # alone is non-binding — timelines are tiny).
        if pad_facts:
            seed = hash(tl.id) & 0x7FFFFFFF
            for filler in generate_filler_facts(pad_facts, seed=seed):
                strat.process_event(filler)
        for ev in tl.events:
            if isinstance(ev, Query):
                prompt, ctx = strat.format_prompt_with_provenance(ev.prompt)
                resp, _, _ = harness._generate_response(strat.get_system_prompt(), prompt)
                jr = harness.judge.judge(
                    response=resp, ground_truth=ev.ground_truth, timeline_id=tl.id,
                    query_idx=0, track=tl.track, domain=tl.domain,
                )
                ev_vals = _fact_values(tl)
                phrases = [p if isinstance(p, str) else getattr(p, "phrase", "")
                           for p in ev.ground_truth.must_mention]
                blob = " ".join(ev_vals).lower()
                evidence_empty = not any(p and p.lower() in blob for p in phrases)
                rows.append((
                    _fke(ctx.context, ev.ground_truth.must_mention, ev_vals),
                    bool(jr.decision_correct),
                    bool(ctx.compaction_triggered),
                    evidence_empty,
                ))
            else:
                strat.process_event(ev)
    return rows


def report(label, rows):
    n = len(rows)
    if not n:
        print(f"{label}: no queries"); return
    hp = sum(1 for _, c, _, _ in rows if c) / n
    fke = sum(f for f, *_ in rows) / n
    compaction = sum(1 for _, _, cp, _ in rows if cp) / n
    empty_ev = sum(1 for *_, e in rows if e)
    fails = [(f, c) for f, c, _, _ in rows if not c]
    fke_high_fail = sum(1 for f, _ in fails if f >= 0.8)
    fke_low_fail = sum(1 for f, _ in fails if f < 0.8)
    print(f"\n{label}: n={n}  HP(acc)={hp:.3f}  mean_FKE={fke:.3f}  "
          f"compaction_fired={compaction:.2f}  (empty-evidence queries: {empty_ev}/{n})")
    if fails:
        print(f"  failures={len(fails)}: reasoning-bound (FKE>=.8)={fke_high_fail}  "
              f"retrieval-bound (FKE<.8)={fke_low_fail}")
    else:
        print("  no failures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/releases/v1.0/dev.jsonl")
    ap.add_argument("--model", default="mlx/qwen3-4b:4bit")
    ap.add_argument("--tracks", nargs="+", default=MULTIHOP)
    args = ap.parse_args()
    print(f"FKE/HP decomposition  model={args.model}  tracks={args.tracks}")
    report("loose 8000, no padding — Memgine fits everything",
           run(args.dataset, args.model, 8000, args.tracks))
    report("tight 1200 + 60 distractor facts — forces compaction to BIND",
           run(args.dataset, args.model, 1200, args.tracks, pad_facts=60))


if __name__ == "__main__":
    main()
