"""How much did the v2.0 scoring corrections move the numbers, and did any
baseline ordering flip? (Paper 3 §7 prerequisite.)

Method. Each response is generated ONCE and then scored twice — under the
legacy semantics that produced the published v1.1 tables, and under the
corrected instrument. Comparing two fresh runs would confound the scoring change
with model sampling and with the model change (the published tables are GPT-5.2
/ Opus 4.6; only the Parslee gateway is reachable here). Scoring one set of
responses both ways isolates the instrument exactly and halves the inference.

The legacy scorers are reimplemented locally rather than kept behind a flag in
``statebench``: production should carry one scoring semantics, and the legacy one
is the buggy one. This file is the only place the old behavior survives.

What changed, and what each change should do to the numbers:

* unbounded substring -> word boundaries: fewer spurious violations (SFRR down)
* no admissibility gate -> non-discriminative phrases skipped: fewer spurious
  violations, and a smaller MNM denominator
* negation blind -> negation aware: fewer spurious violations
* SFRR = any violation -> resurrection only: SFRR down, new leakage/fabrication
  rates carry what was removed
* bare "no" substring -> boundary matched: decision accuracy up on binary items

Usage::

    uv run python experiments/instrument_delta.py --baselines memgine state_based
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from statebench.evaluation.judge import ResponseJudge
from statebench.evaluation.rubric import normalize_text
from statebench.runner.harness import EvaluationHarness, load_timelines
from statebench.schema.timeline import Query

DEFAULT_BASELINES = [
    "no_memory",
    "transcript_replay",
    "transcript_latest_wins",
    "rolling_summary",
    "rag_transcript",
    "fact_extraction",
    "fact_extraction_with_supersession",
    "state_based",
    "state_based_no_supersession",
    "memgine",
]


# --------------------------------------------------------------------------- #
# Legacy scorers — the pre-v2.0 behavior, preserved only for this comparison
# --------------------------------------------------------------------------- #

def legacy_contains_phrase(response: str, phrase: str) -> bool:
    """Plain substring containment: "by" matched "nearby", "100" matched "1000"."""
    r, p = normalize_text(response), normalize_text(phrase)
    if p.startswith("regex:"):
        return bool(re.search(p[6:], r))
    if "|" in p:
        return any(alt.strip() in r for alt in p.split("|"))
    if p in r:
        return True
    for pattern, replacement in [
        (r"do not (\w+)", r"don't \1"),
        (r"don't (\w+)", r"do not \1"),
        (r"cannot (\w+)", r"can't \1"),
        (r"can't (\w+)", r"cannot \1"),
        (r"should not (\w+)", r"shouldn't \1"),
        (r"shouldn't (\w+)", r"should not \1"),
    ]:
        try:
            para = re.sub(pattern, replacement, p)
            if para != p and para in r:
                return True
        except re.error:
            continue
    return False


def legacy_extract_decision(response: str, expected: str) -> tuple[str | None, bool]:
    """Pre-v2.0: signals matched as bare substrings, so "now"/"know" set has_no."""
    r, e = normalize_text(response), normalize_text(expected)
    if e in ("yes", "no"):
        yes_signals = ["yes", "go ahead", "proceed", "approved", "can do", "will do"]
        no_signals = [
            "no", "don't", "do not", "cannot", "should not", "shouldn't",
            "stop", "hold off", "not possible", "not permitted", "exceeds",
        ]
        has_yes = any(s in r for s in yes_signals)
        has_no = any(s in r for s in no_signals)
        if has_no and not has_yes:
            return "no", e == "no"
        if has_yes and not has_no:
            return "yes", e == "yes"
        if has_no and has_yes:
            explicit_no = bool(re.search(r"\bno\b", r))
            explicit_yes = bool(re.search(r"\byes\b", r))
            if explicit_no and not explicit_yes:
                return "no", e == "no"
            if explicit_yes and not explicit_no:
                return "yes", e == "yes"
            first_no = min((r.find(s) for s in no_signals if s in r), default=999)
            first_yes = min((r.find(s) for s in yes_signals if s in r), default=999)
            return ("no", e == "no") if first_no < first_yes else ("yes", e == "yes")
        refusal = [
            "not specified", "not provided", "no information", "cannot determine",
            "insufficient", "unclear", "not available", "not mentioned", "unknown",
        ]
        if len(r.split()) >= 5 and not any(s in r for s in refusal):
            return "yes", e == "yes"
        return None, False
    return (e, True) if e in r else (None, False)


@dataclass
class Tally:
    queries: int = 0
    correct: int = 0
    mm_total: int = 0
    mm_hits: int = 0
    mnm_total: int = 0
    mnm_violations: int = 0
    resurrections: int = 0

    def rates(self) -> dict[str, float]:
        return {
            "decision_accuracy": self.correct / max(1, self.queries),
            "sfrr": self.resurrections / max(1, self.queries),
            "must_mention_rate": self.mm_hits / max(1, self.mm_total),
            "mnm_violation_rate": self.mnm_violations / max(1, self.mnm_total),
        }


@dataclass
class Captured:
    """One response plus the ground truth needed to score it either way."""

    track: str
    response: str
    decision: str
    must_mention: list = field(default_factory=list)
    must_not_mention: list = field(default_factory=list)


def _phrase(item) -> str:
    return item if isinstance(item, str) else item.phrase


def score_legacy(captured: list[Captured], judge: ResponseJudge) -> Tally:
    t = Tally()
    for c in captured:
        t.queries += 1
        _, ok = legacy_extract_decision(c.response, c.decision)
        if not ok:
            options = (
                ["yes", "no"] if c.decision in ("yes", "no") else [c.decision, "other"]
            )
            guess = judge._llm_extract_decision(c.response, options)
            ok = bool(guess) and guess.lower() == c.decision.lower()
        t.correct += ok

        for m in c.must_mention:
            p = _phrase(m)
            t.mm_total += 1
            t.mm_hits += legacy_contains_phrase(c.response, p) or judge._llm_check_paraphrase(
                c.response, p
            )
        violations = 0
        for m in c.must_not_mention:
            p = _phrase(m)
            t.mnm_total += 1
            if legacy_contains_phrase(c.response, p):
                t.mnm_violations += 1
                violations += 1
        # Legacy SFRR: ANY forbidden phrase, whatever failure class it detects.
        t.resurrections += violations > 0
    return t


def score_current(captured: list[Captured], judge: ResponseJudge) -> Tally:
    from statebench.schema.timeline import GroundTruth

    t = Tally()
    for c in captured:
        gt = GroundTruth(
            decision=c.decision,
            must_mention=c.must_mention,
            must_not_mention=c.must_not_mention,
        )
        r = judge.judge(c.response, gt, "t", 0, c.track, "sales")
        t.queries += 1
        t.correct += r.decision_correct
        t.mm_total += len(r.must_mention)
        t.mm_hits += len(r.must_mention_hits)
        t.mnm_total += len(r.must_not_mention)
        t.mnm_violations += len(r.must_not_mention_violations)
        t.resurrections += r.resurrected_superseded
    return t


def run(args: argparse.Namespace) -> dict:
    timelines = list(load_timelines(Path(args.dataset)))
    if args.limit:
        # Stratify by track so a subsample keeps every failure mode represented.
        by_track: dict[str, list] = {}
        for tl in timelines:
            by_track.setdefault(tl.track, []).append(tl)
        per = max(1, args.limit // max(1, len(by_track)))
        timelines = [tl for group in by_track.values() for tl in group[:per]]

    print(f"{len(timelines)} timelines across {len({t.track for t in timelines})} tracks")
    report: dict = {"dataset": args.dataset, "model": args.model, "baselines": {}}

    for baseline in args.baselines:
        harness = EvaluationHarness(
            model=args.model,
            provider=args.provider,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
        )
        strategy = harness._make_strategy(baseline)
        captured: list[Captured] = []
        blocked: list[str] = []
        for tl in timelines:
            strategy.reset()
            if getattr(strategy, "expects_initial_state", False):
                init = getattr(strategy, "initialize_from_state", None)
                if callable(init):
                    init(tl.initial_state)
            for event in tl.events:
                if isinstance(event, Query):
                    prompt, _ = strategy.format_prompt_with_provenance(event.prompt)
                    try:
                        text, _, _ = harness._generate_response(
                            strategy.get_system_prompt(), prompt
                        )
                    except RuntimeError as e:
                        # The hosted gateway's safety layer intermittently
                        # refuses StateBench's adversarial prompt-injection
                        # cases — the very content the benchmark uses to test
                        # injection resistance. A refusal is not a model
                        # response, so scoring it either way would be
                        # meaningless; drop the query from BOTH tallies to keep
                        # the paired comparison exact, and report the count.
                        blocked.append(f"{tl.id}:{e}"[:120])
                        continue
                    captured.append(
                        Captured(
                            track=tl.track,
                            response=text,
                            decision=event.ground_truth.decision,
                            must_mention=list(event.ground_truth.must_mention),
                            must_not_mention=list(event.ground_truth.must_not_mention),
                        )
                    )
                else:
                    strategy.process_event(event)

        legacy = score_legacy(captured, harness.judge).rates()
        current = score_current(captured, harness.judge).rates()
        report["baselines"][baseline] = {
            "n_queries": len(captured),
            "n_blocked_by_provider": len(blocked),
            "legacy": legacy,
            "current": current,
            "delta": {k: current[k] - legacy[k] for k in legacy},
        }
        note = f", {len(blocked)} blocked by provider" if blocked else ""
        print(f"\n=== {baseline} (n={len(captured)}{note}) ===")
        for k in legacy:
            # Deltas on sfrr and mnm_violation_rate are EXACT: both scorings
            # are deterministic over identical responses. decision_accuracy and
            # must_mention_rate additionally invoke the LLM judge in each pass,
            # so their deltas carry judge sampling noise (~1 query = 1/n).
            exact = " " if k in ("sfrr", "mnm_violation_rate") else "~"
            print(
                f"  {k:24s} legacy {legacy[k]:6.1%}  ->  current {current[k]:6.1%}"
                f"   ({100 * (current[k] - legacy[k]):+5.1f}pp){exact}"
            )

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="data/releases/v1.0/dev.jsonl")
    ap.add_argument("--model", default="parslee/reasoning")
    ap.add_argument("--provider", default="car")
    ap.add_argument("--judge-provider", default="car")
    ap.add_argument("--judge-model", default="parslee/reasoning")
    ap.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    ap.add_argument("--limit", type=int, default=26, help="stratified timeline subsample")
    ap.add_argument("--out", default="experiments/results/instrument_delta.json")
    args = ap.parse_args()

    report = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
