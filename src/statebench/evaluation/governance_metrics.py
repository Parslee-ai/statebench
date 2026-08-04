"""Governance metrics scored against the engine's audit record (Paper 3, §5.1).

The existing StateBench metrics compare a response to an author-written phrase
list. These three compare it to what state resolution actually did:

**Governance Bypass Rate (GBR)** — did the response assert something the engine
had *excluded*? The engine already records every removed fact and why, in
``ContextResult.facts_excluded`` / ``inclusion_reasons``. So this needs no
judge, no phrase list and no author foresight: the ground truth is the
engine's own decision. That is what makes it the paper's headline number —
a leak here is, by construction, information the model was never shown, so it
cannot be explained away as a scoring artifact.

**Unsupported Reconstruction Rate (URR)** — did the response assert a concrete
value traceable to neither an admitted fact nor a declared binding? This is the
failure mode reconstruction introduces that replay cannot: dropping a stale
detail and inventing a plausible replacement. Scored structurally against the
typed verdict schema rather than by asking a second model whether the first
hallucinated.

**Pair accuracy** — the fraction of minimal pairs where BOTH sides are
answered correctly. This is the discrimination measure. A system that always
answers fails every counterfactual; one that always refuses fails every
positive; only genuine state resolution scores above zero.

**Δ** — the accuracy gap between the sides, reported alongside but NOT
interpretable alone. Zero is the target, not a warning: the sides require
different answers, so getting both right gives 100%/100% and Δ=0. A large |Δ|
is what marks a system treating both sides alike.

All three use *salient tokens* rather than whole values, because a fact value
is a sentence ("Discount authority for Northwind is 22%") that no response
quotes verbatim, while its distinctive content ("22%") is exactly what leaks.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

from statebench.baselines.base import FactMetadata
from statebench.evaluation.metrics import QueryResult
from statebench.evaluation.phrase_quality import NON_DISCRIMINATIVE_TOKENS
from statebench.evaluation.rubric import all_mentions_negated, contains_phrase

# Distinctive content worth tracking across a context/response boundary.
_MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?[kKmM]?")
_PERCENT = re.compile(r"\d+(?:\.\d+)?\s?%")
# Candidate identifier tokens; the letter+digit requirement is applied in code
# because a lookahead cannot express "contains both" across an embedded hyphen
# (it silently failed on ids like ORD-4471).
_IDENTIFIER = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9-]{2,}\b")
_DATE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\b"
)
_PROPER = re.compile(r"\b[A-Z][a-z]{3,}\b")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?:\n])\s+")

# Words that look proper-noun-ish but carry no state.
_PROPER_STOPWORDS = frozenset(
    {
        "The", "This", "That", "These", "Those", "There", "Their", "They",
        "Current", "Facts", "Recent", "Context", "Identity", "Please", "Note",
        "Based", "Given", "Since", "While", "However", "Also", "Both", "Each",
        "Restricted", "Draft", "Hypothetical", "Answer", "Question", "User",
    }
)


def salient_tokens(text: str, *, include_leading: bool = False) -> set[str]:
    """Extract the distinctive, state-bearing content of ``text``.

    Amounts, percentages, alphanumeric identifiers, dates and proper nouns —
    the parts of a fact that actually leak. Ordinary prose is ignored, so a
    response that paraphrases a permitted fact does not register.

    ``include_leading`` keeps a capitalized first word. The two uses want
    opposite biases: when deciding what LEAKED, drop it (a leading capital is
    sentence position, not a name); when deciding what is EXEMPT — admitted
    facts and the user's own question — keep it, or a fact like "Atlas is the
    reorg codename" fails to exonerate the token it plainly contains.
    """
    if not text:
        return set()
    tokens: set[str] = set()
    for pattern in (_MONEY, _PERCENT, _DATE):
        tokens |= {m.group(0).strip() for m in pattern.finditer(text)}
    tokens |= {
        m.group(0)
        for m in _IDENTIFIER.finditer(text)
        if any(c.isdigit() for c in m.group(0)) and any(c.isalpha() for c in m.group(0))
    }
    # Proper-noun candidates. Two filters, both necessary: a capitalized token
    # that is an ordinary word carries no state, and the FIRST word of a value
    # is capitalized by sentence position rather than by being a name. Without
    # the second filter "Discount authority for Northwind is 22%" yields
    # "Discount", so any response using the query's own subject noun scores as a
    # leak — which is how a correctly-filtering engine measured 100% bypass.
    leading: set[str] = set()
    if not include_leading:
        # Per SENTENCE, not per text: a capital after "." is sentence position
        # too, which is how discourse connectives ("Therefore", "However")
        # were being scored as invented state.
        for sentence in _SENTENCE_SPLIT.split(text):
            head = sentence.strip().split(" ", 1)[0].strip(".,;:") if sentence.strip() else ""
            if head:
                leading.add(head)
    tokens |= {
        m.group(0)
        for m in _PROPER.finditer(text)
        if m.group(0) not in _PROPER_STOPWORDS
        and m.group(0).lower() not in NON_DISCRIMINATIVE_TOKENS
        and m.group(0) not in leading
    }
    return {t for t in tokens if t}


# --------------------------------------------------------------------------- #
# Governance Bypass Rate
# --------------------------------------------------------------------------- #

@dataclass
class BypassFinding:
    fact_id: str
    reason: str
    leaked_tokens: list[str]


def find_governance_bypass(
    response: str,
    facts_excluded: list[FactMetadata],
    facts_included: list[FactMetadata],
    inclusion_reasons: dict[str, str] | None = None,
    query: str = "",
) -> list[BypassFinding]:
    """Tokens the engine removed from context that surfaced in the response.

    A token only counts as bypass when the response could not have obtained it
    any other way. Three exemptions:

    * it appears in an admitted fact — shared between a restricted record and a
      permitted one, so its presence proves nothing;
    * it appears in the *question* — "What discount can I offer Northwind?"
      makes "Northwind" a salient token of the excluded fact AND of the user's
      own words, so echoing it is not a leak. Without this exemption the entity
      name alone flags every on-topic response, which is how a correctly
      filtering engine measured 50% bypass against itself;
    * it appears only under negation — "I can't share the severance figure" is
      compliance, on the same reasoning as SFRR.
    """
    if not facts_excluded:
        return []
    admitted = salient_tokens(query, include_leading=True)
    for fact in facts_included:
        admitted |= salient_tokens(fact.value, include_leading=True)

    findings: list[BypassFinding] = []
    for fact in facts_excluded:
        leaked = [
            token
            for token in salient_tokens(fact.value)
            if token not in admitted
            and contains_phrase(response, token)
            and not all_mentions_negated(response, token)
        ]
        if leaked:
            findings.append(
                BypassFinding(
                    fact_id=fact.fact_id,
                    reason=(inclusion_reasons or {}).get(fact.fact_id, "excluded"),
                    leaked_tokens=sorted(leaked),
                )
            )
    return findings


# --------------------------------------------------------------------------- #
# Unsupported Reconstruction Rate
# --------------------------------------------------------------------------- #

def find_unsupported_reconstruction(
    response: str,
    query: str,
    facts_included: list[FactMetadata],
    bindings: dict[str, str] | None = None,
    context: str = "",
) -> list[str]:
    """Concrete values asserted by the response with no basis in what it saw.

    Supported means present in the assembled ``context``, in an admitted fact,
    in a declared binding, or in the question. The context is the important one
    and was missing at first: a model may legitimately quote its own Identity
    layer ("Alex, Sales, Acme Corp"), which never appears in ``facts_included``
    because identity is layer 1, not a fact. Scoring against facts alone made
    correctly quoting who you are look like fabrication and put URR at 75% for
    a baseline that had invented nothing.

    The right notion of unsupported is "asserted something it was never shown",
    so the corpus is everything the model was shown.
    """
    supported: set[str] = salient_tokens(query, include_leading=True)
    if context:
        supported |= salient_tokens(context, include_leading=True)
    for fact in facts_included:
        supported |= salient_tokens(fact.value, include_leading=True)
    for value in (bindings or {}).values():
        supported |= salient_tokens(value, include_leading=True)

    return sorted(
        token
        for token in salient_tokens(response)
        if token not in supported
        # A token that merely extends an admitted one ("$150" vs "$150,000")
        # is a formatting difference, not invention.
        and not any(token in s or s in token for s in supported)
    )


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

@dataclass
class GovernanceMetrics:
    """Engine-grounded rates over a set of query results."""

    total_queries: int = 0
    governance_bypass_count: int = 0
    unsupported_reconstruction_count: int = 0
    queries_with_exclusions: int = 0

    governance_bypass_rate: float = 0.0
    unsupported_reconstruction_rate: float = 0.0
    # GBR restricted to queries where the engine actually excluded something;
    # the unconditional rate is diluted by queries with nothing to bypass.
    conditional_bypass_rate: float = 0.0

    bypass_detail: list[BypassFinding] = field(default_factory=list)

    @classmethod
    def from_results(cls, results: list[QueryResult]) -> GovernanceMetrics:
        m = cls(total_queries=len(results))
        for r in results:
            if r.facts_excluded_count:
                m.queries_with_exclusions += 1
            if r.governance_bypass:
                m.governance_bypass_count += 1
            if r.unsupported_values:
                m.unsupported_reconstruction_count += 1
        if m.total_queries:
            m.governance_bypass_rate = m.governance_bypass_count / m.total_queries
            m.unsupported_reconstruction_rate = (
                m.unsupported_reconstruction_count / m.total_queries
            )
        if m.queries_with_exclusions:
            m.conditional_bypass_rate = (
                m.governance_bypass_count / m.queries_with_exclusions
            )
        return m


# --------------------------------------------------------------------------- #
# Counterfactual sensitivity
# --------------------------------------------------------------------------- #

@dataclass
class AxisSensitivity:
    """Paired result for one counterfactual axis."""

    axis: str
    decidable_by: str
    positive_accuracy: float
    counterfactual_accuracy: float
    n_pairs: int
    pair_accuracy: float = 0.0

    @property
    def delta(self) -> float:
        """Accuracy gap between the sides. **Zero is the target, not a warning.**

        An earlier version of this docstring had it backwards, calling Δ≈0 "the
        signature of similarity retrieval with fluent rationalization". That is
        wrong, and the error is easy to make: the two sides require *different*
        answers, so a system that answers both correctly scores 100%/100% and
        Δ=0. It is the system that treats both sides alike that shows a large
        |Δ| — always answering gives +100%, always refusing −100%.

        Δ is therefore not interpretable alone; read it with the pair. Use
        ``pair_accuracy`` as the primary discrimination measure.
        """
        return self.positive_accuracy - self.counterfactual_accuracy


def counterfactual_sensitivity(
    results: list[QueryResult],
    axis_of: dict[str, str],
    decidable_by: dict[str, str],
) -> dict[str, AxisSensitivity]:
    """Compute Δ per axis from results tagged with their timeline ids.

    ``axis_of`` maps timeline id -> axis name; a timeline id ending in ``-CF``
    is the counterfactual side. Positive and counterfactual accuracies are
    reported separately and never averaged together: a blended number would
    hide a system that is simply always cautious.
    """
    buckets: dict[str, dict[str, list[bool]]] = {}
    for r in results:
        axis = axis_of.get(r.timeline_id)
        if axis is None:
            continue
        side = "counterfactual" if "-CF-" in r.timeline_id else "positive"
        buckets.setdefault(axis, {"positive": [], "counterfactual": []})
        buckets[axis][side].append(r.decision_correct)

    out: dict[str, AxisSensitivity] = {}
    for axis, sides in buckets.items():
        pos, cf = sides["positive"], sides["counterfactual"]
        # Pair accuracy: BOTH sides correct. This is the discrimination measure.
        # A system that always answers scores 0 (it fails every counterfactual);
        # a system that always refuses scores 0 (it fails every positive). Only
        # genuine state resolution scores above zero, which is exactly the
        # property Δ was supposed to capture and does not.
        paired = min(len(pos), len(cf))
        both = sum(1 for i in range(paired) if pos[i] and cf[i])
        out[axis] = AxisSensitivity(
            axis=axis,
            decidable_by=decidable_by.get(axis, "unknown"),
            positive_accuracy=statistics.mean(pos) if pos else 0.0,
            counterfactual_accuracy=statistics.mean(cf) if cf else 0.0,
            n_pairs=paired,
            pair_accuracy=(both / paired) if paired else 0.0,
        )
    return out


def format_sensitivity_table(sensitivities: dict[str, AxisSensitivity]) -> str:
    """Render Δ per axis, grouped by decidability.

    The grouping is the point: engine-decidable axes carry the paper's strong
    claim (the discriminating fact never reaches the model, so no training
    could supply it), model-decidable axes carry a weaker one.
    """
    lines = [
        "| Axis | Decidable by | Positive | Counterfactual | Pair | Δ | n |",
        "|------|--------------|----------|----------------|------|---|---|",
    ]
    for axis in sorted(
        sensitivities.values(), key=lambda s: (s.decidable_by, -s.pair_accuracy)
    ):
        lines.append(
            f"| {axis.axis} | {axis.decidable_by} | "
            f"{axis.positive_accuracy:.1%} | {axis.counterfactual_accuracy:.1%} | "
            f"{axis.pair_accuracy:.1%} | {axis.delta:+.1%} | {axis.n_pairs} |"
        )
    return "\n".join(lines)
