"""Does the generator actually plant what the ground truth claims it planted?

``paper-measurement-validity`` audited the *scorer* and found six defects. It
never audited the *generator*. That is a gap with the same shape as the one it
documented: nothing at run time checks that a timeline's ground truth is
satisfiable from the timeline's own text, so a mis-specified template produces
numbers that look exactly like model failures.

Two of the checks here catch defects that are invisible in a results file:

* A ``must_mention`` phrase that appears nowhere in the timeline asks the model
  to produce wording it was never shown. Must-mention scoring has an LLM
  paraphrase fallback, so this is survivable — but it makes the result depend on
  the judge rather than on the system under test, and it depresses the track's
  must-mention rate for every baseline equally, so the ranking looks normal and
  nothing appears wrong.
* A ``must_not_mention`` phrase that appears nowhere in the timeline is worse,
  because must-not-mention scoring is deterministic only — there is no fallback.
  Such a phrase can never be violated by anything, yet it still sits in the
  must-not-mention denominator, so it silently deflates the violation rate. Add
  enough of them and a system looks cleaner than it is.

Neither is hypothetical. In practice both show up as *near misses* rather than
absent facts: a timeline that says "Reset MFA for CEO" with ``"MFA reset"`` on
the forbidden list, or one that says "They have 500 active users" with
``"500 users"`` required. The fact was planted; the phrase does not match it.
That is the same species of defect ``paper-measurement-validity`` found in the
forbidden-phrase lists, one layer earlier in the pipeline.

Severity
--------
``error`` marks a defect that is wrong under any reading: a phrase that is both
required and forbidden, a supersession pointing at nothing, a query that
precedes the fact it depends on.

``warning`` marks something that may be deliberate. The commonest is a
``must_mention`` phrase the model is expected to *infer* rather than quote. A
warning is a prompt to look, not a verdict.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from statebench.evaluation.phrase_quality import is_discriminative, reason_rejected
from statebench.evaluation.rubric import contains_phrase
from statebench.schema.timeline import (
    ConversationTurn,
    MentionRequirement,
    Query,
    StateWrite,
    Supersession,
    Timeline,
)

Severity = Literal["error", "warning"]

#: Phrasings that mark a fact as time-limited rather than event-superseded.
EXPIRY_MARKERS = (
    "valid through",
    "valid until",
    "valid til",
    "expires",
    "expired",
    "expiry",
    "expiration",
    "through 20",
    "until 20",
    "as of 20",
    "no longer valid",
)


@dataclass(frozen=True)
class FidelityIssue:
    """One defect found in one timeline."""

    timeline_id: str
    track: str
    code: str
    severity: Severity
    detail: str
    query_idx: int | None = None

    def __str__(self) -> str:
        where = f"{self.timeline_id}" + (
            f" q{self.query_idx}" if self.query_idx is not None else ""
        )
        return f"[{self.severity}] {self.code} ({where}): {self.detail}"


@dataclass
class FidelityReport:
    """Aggregated findings over a dataset."""

    timelines_checked: int = 0
    queries_checked: int = 0
    issues: list[FidelityIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[FidelityIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[FidelityIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def clean(self) -> bool:
        return not self.errors

    def by_code(self) -> dict[str, int]:
        return dict(Counter(i.code for i in self.issues))

    def affected_timelines(self) -> int:
        return len({i.timeline_id for i in self.issues})


def _phrase_of(item: str | MentionRequirement) -> str:
    return item.phrase if isinstance(item, MentionRequirement) else item


def _kind_of(item: str | MentionRequirement) -> str | None:
    return item.kind if isinstance(item, MentionRequirement) else None


def timeline_corpus(timeline: Timeline) -> str:
    """Everything a memory system could possibly have seen in this timeline.

    Conversation text, every written value and key, the initial facts and the
    environment. If a phrase is not in here, no system can quote it.
    """
    parts: list[str] = []

    for fact in timeline.initial_state.persistent_facts:
        parts.append(f"{fact.key} {fact.value}")
    for item in timeline.initial_state.working_set:
        parts.append(f"{item.item_type} {item.content}")
    for key, value in timeline.initial_state.environment.items():
        parts.append(f"{key} {value}")

    identity = timeline.initial_state.identity_role
    parts.append(
        f"{identity.user_name} {identity.authority} "
        f"{identity.department} {identity.organization}"
    )

    for event in timeline.events:
        if isinstance(event, ConversationTurn):
            parts.append(event.text)
        elif isinstance(event, (StateWrite, Supersession)):
            for write in event.writes:
                parts.append(f"{write.key} {write.value}")

    return "\n".join(parts)


def check_timeline(timeline: Timeline) -> list[FidelityIssue]:
    """Every fidelity issue in one timeline."""
    issues: list[FidelityIssue] = []

    def add(code: str, severity: Severity, detail: str, query_idx: int | None = None):
        issues.append(
            FidelityIssue(
                timeline_id=timeline.id,
                track=timeline.track,
                code=code,
                severity=severity,
                detail=detail,
                query_idx=query_idx,
            )
        )

    corpus = timeline_corpus(timeline)

    # Not every track kills a fact with a Supersession event. The temporal
    # counterfactuals encode a validity window in the value text and let the
    # environment clock retire it, which is a real invalidation and must not be
    # reported as a missing one.
    lowered = corpus.lower()
    has_expiry = any(marker in lowered for marker in EXPIRY_MARKERS)

    # --- Structural checks over the event sequence ---------------------------

    # ``Write.supersedes`` is documented as a fact ID, but the supersession
    # generators populate it with a fact *key* and others with an ID. Both are
    # in shipped data, so a reference is dangling only if it names neither.
    written: set[str] = set()
    for fact in timeline.initial_state.persistent_facts:
        written.update({fact.key, fact.id})

    has_supersession = False
    last_state_change_ts = None

    for event in timeline.events:
        if isinstance(event, StateWrite):
            for write in event.writes:
                written.update({write.key, write.id})
            last_state_change_ts = event.ts
        elif isinstance(event, Supersession):
            for write in event.writes:
                if write.supersedes and write.supersedes not in written:
                    add(
                        "dangling_supersession",
                        "error",
                        f"write {write.id!r} supersedes {write.supersedes!r}, "
                        "which matches no earlier fact key or id",
                    )
                if write.supersedes:
                    has_supersession = True
                written.update({write.key, write.id})
            last_state_change_ts = event.ts

    stamps = [e.ts for e in timeline.events]
    if stamps != sorted(stamps):
        add("events_out_of_order", "error", "event timestamps are not monotonic")

    # --- Per-query ground-truth checks --------------------------------------

    queries = [(i, e) for i, e in enumerate(timeline.events) if isinstance(e, Query)]
    if not queries:
        add("no_query", "error", "timeline has no query, so it scores nothing")

    for query_idx, (event_idx, query) in enumerate(queries):
        gt = query.ground_truth

        if last_state_change_ts is not None and query.ts < last_state_change_ts:
            add(
                "query_precedes_state",
                "error",
                "query is timestamped before the last state change it should "
                "depend on",
                query_idx,
            )

        must = [_phrase_of(m) for m in gt.must_mention]
        forbidden_items = list(gt.must_not_mention)
        forbidden = [_phrase_of(m) for m in forbidden_items]

        if not gt.decision and not must and not forbidden:
            add("empty_ground_truth", "error", "query scores nothing", query_idx)

        overlap = {m.lower() for m in must} & {f.lower() for f in forbidden}
        for phrase in sorted(overlap):
            add(
                "phrase_required_and_forbidden",
                "error",
                f"{phrase!r} is in both must_mention and must_not_mention, so no "
                "response can pass",
                query_idx,
            )

        for phrase in must:
            if not contains_phrase(corpus, phrase):
                add(
                    "unsupported_must_mention",
                    "warning",
                    f"{phrase!r} appears nowhere in the timeline, so a correct "
                    "answer can only match it by paraphrase — the score depends "
                    "on the judge rather than on the system under test",
                    query_idx,
                )

        for item in forbidden_items:
            phrase = _phrase_of(item)
            if not is_discriminative(phrase):
                add(
                    "inadmissible_forbidden_phrase",
                    "warning",
                    f"{reason_rejected(phrase)}; the judge will skip it",
                    query_idx,
                )
                continue
            if not contains_phrase(corpus, phrase):
                add(
                    "unreachable_forbidden_phrase",
                    "warning",
                    f"{phrase!r} appears nowhere in the timeline. Must-not-mention "
                    "scoring is deterministic, so nothing can ever violate it, yet "
                    "it still counts in the denominator and deflates the rate",
                    query_idx,
                )
            if (
                _kind_of(item) == "superseded"
                and not has_supersession
                and not has_expiry
            ):
                add(
                    "superseded_phrase_without_invalidation",
                    "warning",
                    f"{phrase!r} is tagged 'superseded', but nothing in the "
                    "timeline invalidates it — no supersession event and no "
                    "expiry marker — so SFRR may count a value that is still live",
                    query_idx,
                )

    return issues


def check_dataset(timelines: Iterable[Timeline]) -> FidelityReport:
    """Run every check over a dataset."""
    report = FidelityReport()
    for timeline in timelines:
        report.timelines_checked += 1
        report.queries_checked += sum(
            1 for e in timeline.events if isinstance(e, Query)
        )
        report.issues.extend(check_timeline(timeline))
    return report


def format_fidelity_report(report: FidelityReport, max_examples: int = 5) -> str:
    """Markdown summary of a fidelity run."""
    lines = [
        "# Dataset Fidelity Report",
        "",
        f"- Timelines checked: **{report.timelines_checked}**",
        f"- Queries checked: **{report.queries_checked}**",
        f"- Errors: **{len(report.errors)}**",
        f"- Warnings: **{len(report.warnings)}**",
        f"- Timelines with at least one issue: **{report.affected_timelines()}**",
        "",
    ]

    if not report.issues:
        lines.append("No issues found.")
        return "\n".join(lines)

    lines += ["| Code | Severity | Count |", "|------|----------|-------|"]
    severity_of = {i.code: i.severity for i in report.issues}
    for code, count in sorted(report.by_code().items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{code}` | {severity_of[code]} | {count} |")

    lines += ["", "## Examples", ""]
    seen: Counter[str] = Counter()
    for issue in report.issues:
        if seen[issue.code] >= max_examples:
            continue
        seen[issue.code] += 1
        lines.append(f"- {issue}")

    return "\n".join(lines)
