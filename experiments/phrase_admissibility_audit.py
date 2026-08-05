"""Count forbidden phrases in the release that a correct answer could contain.

This figure appears in the measurement-validity paper (§4.3) and was previously
quoted with no script behind it, which is how it went wrong: the first count
summed ``train + dev + test + full + hidden``, but ``full.jsonl`` *is* the union
of the three splits plus 143 held-out timelines, so 1,257 timelines were counted
twice. That inflated both the numerator and the denominator without changing the
percentage much, which is exactly why it survived a sanity check.

Deduplicating by timeline id gives 1,410 unique timelines. Run this to regenerate
the number rather than quoting it::

    uv run python experiments/phrase_admissibility_audit.py

``--show`` prints example inadmissible phrases per track, which is the useful
output when deciding whether the gate itself is calibrated.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from statebench.evaluation.phrase_quality import is_discriminative
from statebench.runner.harness import load_timelines

# Ordered so the union is built from the widest file first; the splits then add
# nothing, which is the property being asserted.
FILES = ("full", "hidden", "train", "dev", "test")


def audit(release: Path) -> dict:
    seen: set[str] = set()
    total = bad = 0
    per_track: collections.Counter = collections.Counter()
    per_track_total: collections.Counter = collections.Counter()
    examples: dict[str, list[str]] = collections.defaultdict(list)

    for name in FILES:
        path = release / f"{name}.jsonl"
        if not path.exists():
            continue
        for timeline in load_timelines(path):
            if timeline.id in seen:
                continue
            seen.add(timeline.id)
            for event in timeline.events:
                if getattr(event, "type", "") != "query":
                    continue
                for phrase in event.ground_truth.must_not_mention:
                    text = phrase if isinstance(phrase, str) else phrase.phrase
                    total += 1
                    per_track_total[timeline.track] += 1
                    if not is_discriminative(text):
                        bad += 1
                        per_track[timeline.track] += 1
                        if len(examples[timeline.track]) < 8:
                            examples[timeline.track].append(text)

    return {
        "timelines": len(seen),
        "forbidden_phrases": total,
        "inadmissible": bad,
        "rate": bad / total if total else 0.0,
        "by_track": {
            track: {
                "inadmissible": per_track[track],
                "total": per_track_total[track],
                "rate": per_track[track] / per_track_total[track],
            }
            for track in sorted(per_track_total, key=lambda t: -per_track[t])
        },
        "examples": dict(examples),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", default="data/releases/v1.0")
    ap.add_argument("--show", action="store_true", help="print example phrases")
    ap.add_argument("--out", default="experiments/results/phrase_admissibility.json")
    args = ap.parse_args()

    report = audit(Path(args.release))
    print(
        f"{report['timelines']} unique timelines, "
        f"{report['forbidden_phrases']} forbidden phrases, "
        f"{report['inadmissible']} inadmissible ({report['rate']:.1%})\n"
    )
    for track, d in report["by_track"].items():
        if d["inadmissible"]:
            print(f"  {track:30s} {d['inadmissible']:4d} of {d['total']:4d}  {d['rate']:5.1%}")
            if args.show:
                for phrase in report["examples"].get(track, []):
                    print(f"       {phrase!r}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
