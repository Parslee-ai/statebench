"""Isolate the length sensitivity of phrase-list scoring. No model calls.

The instrument-delta result shows SFRR falling 8.8-17.6pp under corrected scoring
while must-mention barely moves. Two explanations are compatible with that:

  (a) response length -- longer answers contain more strings by chance;
  (b) degenerate phrases -- forbidden lists contain entries a correct answer may
      legitimately contain.

They are not independent: a degenerate phrase only fires *because* text is long
enough to contain it. This experiment separates them by construction. We take the
real ``must_not_mention`` lists from the release and score **decoy text that
contains no superseded value at all** -- generic business prose. Any violation is
by definition a false positive.

Sweeping decoy length gives the false-positive rate as a function of length, under
both scorings, with zero inference and perfect ground truth: the true violation
count is zero everywhere.

If legacy FPR climbs with length and corrected FPR stays flat, the mechanism is
established directly rather than inferred from the delta.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from statebench.evaluation.phrase_quality import is_discriminative
from statebench.evaluation.rubric import all_mentions_negated, contains_phrase

# Generic business prose. Deliberately contains no entity, amount, or date that
# appears in any StateBench scenario -- so a correct scorer must report zero
# violations at every length.
DECOY_SENTENCES = [
    "I have reviewed the details you shared earlier in this thread.",
    "Based on the current record, here is what I can confirm.",
    "Please let me know if you would like me to follow up with the team.",
    "The relevant information appears to be documented in the system.",
    "I want to make sure I am giving you an accurate answer.",
    "There are a few considerations worth keeping in mind here.",
    "I can help you work through the next steps whenever you are ready.",
    "This should align with what was previously communicated.",
    "Let me summarize the situation as I understand it.",
    "If anything looks incorrect, I am happy to revisit it.",
    "The account shows no outstanding items requiring attention.",
    "I would recommend confirming with the responsible party first.",
    "Everything appears consistent with the standard process.",
    "Please treat this as a preliminary response pending review.",
    "I appreciate your patience while I looked into this.",
    "Here is a short overview of the relevant considerations.",
    "That aligns with the guidance currently on file.",
    "I do not want to speculate beyond what is documented.",
    "You may want to keep a record of this exchange.",
    "I will make a note of your request for future reference.",
]


def load_queries(release: Path) -> list[tuple[str, list[str]]]:
    """Collect (track, full must_not_mention list) per query.

    Per QUERY, not per phrase: real scoring checks every forbidden phrase for a
    query and counts a violation if ANY hits. Sampling one phrase at a time
    understates the false-positive rate by roughly the list length.
    """
    out: list[tuple[str, list[str]]] = []
    for path in sorted(release.glob("*.jsonl")):
        for line in path.open():
            if not line.strip():
                continue
            rec = json.loads(line)
            for event in rec["events"]:
                if event.get("type") != "query":
                    continue
                phrases = [
                    (m if isinstance(m, str) else m.get("phrase", ""))
                    for m in event["ground_truth"].get("must_not_mention", [])
                ]
                phrases = [p for p in phrases if p]
                if phrases:
                    out.append((rec["track"], phrases))
    return out


def legacy_contains(response: str, phrase: str) -> bool:
    """Pre-v2.0 matching: plain substring containment."""
    r, p = response.lower().strip(), phrase.lower().strip()
    if p.startswith("regex:"):
        return bool(re.search(p[6:], r))
    if "|" in p:
        return any(alt.strip() in r for alt in p.split("|"))
    return p in r


def corrected_contains(response: str, phrase: str) -> bool:
    """Post-v2.0: admissibility gate, boundary matching, negation awareness."""
    if not is_discriminative(phrase):
        return False
    if not contains_phrase(response, phrase):
        return False
    return not all_mentions_negated(response, phrase)


def make_decoy(n_sentences: int, rng: random.Random) -> str:
    return " ".join(rng.choice(DECOY_SENTENCES) for _ in range(n_sentences))


def score_query(response: str, phrases: list[str], scorer) -> bool:
    """A query violates if ANY of its forbidden phrases matches."""
    return any(scorer(response, p) for p in phrases)



# --------------------------------------------------------------------------- #
# The decisive test: score the CORRECT answer
# --------------------------------------------------------------------------- #

def correct_answer_probe(release: Path, limit: int = 400) -> dict:
    """Construct a plausibly-correct response per query and score it.

    This is checklist item #1 from the paper applied mechanically: build the
    answer a correct system would give, and see whether the scorer accepts it.

    The constructed answer does three things a real correct answer does, and
    that generic decoy prose does not:
      1. states the required (must_mention) content;
      2. explicitly negates the superseded value -- the discrimination the
         metric exists to reward;
      3. uses the question's own topic vocabulary.
    """
    import json as _json
    rows = []
    for path in sorted(release.glob("*.jsonl")):
        for line in path.open():
            if not line.strip():
                continue
            rec = _json.loads(line)
            for event in rec["events"]:
                if event.get("type") != "query":
                    continue
                gt = event["ground_truth"]
                mm = [(m if isinstance(m, str) else m.get("phrase", ""))
                      for m in gt.get("must_mention", [])]
                mnm = [(m if isinstance(m, str) else m.get("phrase", ""))
                       for m in gt.get("must_not_mention", [])]
                mm = [p for p in mm if p]
                mnm = [p for p in mnm if p]
                if mnm:
                    rows.append((rec["track"], event.get("prompt", ""), mm, mnm))
    rows = rows[:limit]

    out = {"n": len(rows), "by_track": {}, "overall": {}}
    agg = {"plain": [0, 0], "negating": [0, 0]}
    per_track: dict[str, dict] = {}

    for track, prompt, mm, mnm in rows:
        required = "; ".join(mm[:3]) if mm else "the current value on file"
        # (a) states the required content and echoes the question's vocabulary
        plain = f"Based on the current state: {required}. {prompt}".strip()
        # (b) additionally performs the discrimination the metric rewards --
        #     naming the dead value in order to rule it out
        dead = mnm[0]
        negating = (
            f"Based on the current state: {required}. "
            f"Note that it is not {dead} — that value was superseded."
        )
        for label, text in (("plain", plain), ("negating", negating)):
            leg = score_query(text, mnm, legacy_contains)
            cor = score_query(text, mnm, corrected_contains)
            agg[label][0] += leg
            agg[label][1] += cor
            t = per_track.setdefault(track, {"n": 0, "plain": [0, 0], "negating": [0, 0]})
            t[label][0] += leg
            t[label][1] += cor
        per_track[track]["n"] += 1

    n = len(rows)
    for label in ("plain", "negating"):
        out["overall"][label] = {
            "legacy_false_violation_rate": agg[label][0] / n,
            "corrected_false_violation_rate": agg[label][1] / n,
        }
    for track, t in per_track.items():
        out["by_track"][track] = {
            "n": t["n"],
            "plain_legacy": t["plain"][0] / t["n"],
            "plain_corrected": t["plain"][1] / t["n"],
            "negating_legacy": t["negating"][0] / t["n"],
            "negating_corrected": t["negating"][1] / t["n"],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", default="data/releases/v1.0")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="experiments/results/length_bias.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    queries = load_queries(Path(args.release))
    avg_list = sum(len(p) for _, p in queries) / len(queries)
    print(
        f"{len(queries)} queries, mean {avg_list:.1f} forbidden phrases each. "
        f"Decoy text contains no scenario value, so EVERY violation is a false positive.\n"
    )

    lengths = [1, 2, 4, 8, 16, 32]
    report: dict = {"lengths": lengths, "overall": {}, "by_track": {}}

    print(f"{'words':>6s}  {'legacy FPR':>11s}  {'corrected FPR':>14s}")
    for n in lengths:
        leg = cor = 0
        words = 0
        for _ in range(args.trials):
            decoy = make_decoy(n, rng)
            words += len(decoy.split())
            _, phrases = rng.choice(queries)
            leg += score_query(decoy, phrases, legacy_contains)
            cor += score_query(decoy, phrases, corrected_contains)
        avg_words = words / args.trials
        lf, cf = leg / args.trials, cor / args.trials
        report["overall"][n] = {
            "avg_words": avg_words,
            "legacy_fpr": lf,
            "corrected_fpr": cf,
        }
        print(f"{avg_words:6.0f}  {lf:11.1%}  {cf:14.1%}")

    by_track: dict[str, list[list[str]]] = {}
    for track, phrases in queries:
        by_track.setdefault(track, []).append(phrases)

    print(f"\nfalse-positive rate at ~{report['overall'][lengths[-1]]['avg_words']:.0f} words, by track:")
    print(f"{'track':30s} {'legacy':>8s} {'corrected':>10s}  {'phrases':>7s} {'inadmissible':>13s}")
    for track, plists in sorted(by_track.items()):
        leg = cor = 0
        for _ in range(args.trials):
            decoy = make_decoy(lengths[-1], rng)
            phrases = rng.choice(plists)
            leg += score_query(decoy, phrases, legacy_contains)
            cor += score_query(decoy, phrases, corrected_contains)
        uniq = {p for pl in plists for p in pl}
        bad = sum(1 for p in uniq if not is_discriminative(p))
        report["by_track"][track] = {
            "legacy_fpr": leg / args.trials,
            "corrected_fpr": cor / args.trials,
            "n_phrases": len(uniq),
            "n_inadmissible": bad,
        }
        print(
            f"{track:30s} {leg/args.trials:8.1%} {cor/args.trials:10.1%}  "
            f"{len(uniq):7d} {bad:13d}"
        )

    print("\n=== scoring a CORRECT answer (checklist item #1, mechanized) ===")
    ca = correct_answer_probe(Path(args.release))
    report["correct_answer"] = ca
    print(f"n={ca['n']} constructed correct answers")
    for label in ("plain", "negating"):
        o = ca["overall"][label]
        desc = ("states required content + question vocabulary" if label == "plain"
                else "  ... and negates the superseded value")
        print(f"  {desc:52s} legacy {o['legacy_false_violation_rate']:6.1%}"
              f"   corrected {o['corrected_false_violation_rate']:6.1%}")
    print(f"\n  {'track':30s} {'plain leg':>10s} {'negating leg':>13s} {'negating corr':>14s}")
    for t, v in sorted(ca["by_track"].items(), key=lambda kv: -kv[1]["negating_legacy"]):
        print(f"  {t:30s} {v['plain_legacy']:10.1%} {v['negating_legacy']:13.1%} {v['negating_corrected']:14.1%}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

