"""Query complexity classifier for adaptive context assembly.

Classifies queries into complexity tiers so that context assembly can
adapt: simple lookups need minimal context, constraint checks need all
constraints, and repair/recalculation queries need dependency chains.

Inspired by Honcho's tiered reasoning approach — but deterministic
(keyword heuristics, no LLM calls).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QueryComplexity(Enum):
    """Query complexity tiers."""

    SIMPLE_LOOKUP = "simple_lookup"
    """Direct fact retrieval: 'what is X?', 'who is Y?'"""

    CONSTRAINT_CHECK = "constraint_check"
    """Decision requiring constraint evaluation: 'can we?', 'should we?', 'is X allowed?'"""

    REPAIR = "repair"
    """Recalculation after corrections: 'what is X now?', 'recalculate', dependency chains."""

    TEMPORAL = "temporal"
    """Time-sensitive queries: 'when?', 'deadline', 'schedule', 'what changed?'"""

    AGGREGATION = "aggregation"
    """Multi-fact synthesis: 'summarize', 'how many', 'total', 'list all'."""


@dataclass(frozen=True)
class QueryClassification:
    """Result of query classification."""

    complexity: QueryComplexity
    keywords_matched: tuple[str, ...]
    suggested_layer_weights: dict[int, float]
    """Suggested layer budget overrides (layer -> fraction)."""


# Keyword patterns per complexity tier (ordered by specificity — first match wins
# for constraint/repair/temporal/aggregation; fallback is SIMPLE_LOOKUP).

_CONSTRAINT_PATTERNS: list[str] = [
    "can we", "should we", "is it allowed", "is this allowed",
    "do we have approval", "does this comply", "within budget",
    "approved", "permitted", "authorize", "feasible",
    "proceed with", "go ahead", "move forward",
    "violate", "breach", "exceed", "comply",
    "eligible", "qualified", "meets the requirement",
    "policy allows", "allowed to",
]

_REPAIR_PATTERNS: list[str] = [
    "now that", "given the change", "after the update",
    "recalculate", "revised", "updated",
    "what is the new", "what's the new",
    "how does this affect", "impact of the change",
    "corrected", "adjusted", "changed from",
    "cascade", "propagate", "downstream",
]

_TEMPORAL_PATTERNS: list[str] = [
    "when is", "when does", "when will",
    "deadline", "due date", "schedule",
    "how long", "timeline", "by when",
    "expired", "still valid", "current status",
    "latest", "most recent", "what changed",
]

_AGGREGATION_PATTERNS: list[str] = [
    "how many", "how much", "total",
    "summarize", "summary", "list all",
    "overview", "across all", "combined",
    "count", "aggregate", "everything about",
]

# Default layer weights per complexity tier
_LAYER_WEIGHTS: dict[QueryComplexity, dict[int, float]] = {
    QueryComplexity.SIMPLE_LOOKUP: {
        1: 0.05, 2: 0.60, 3: 0.20, 4: 0.15,
    },
    QueryComplexity.CONSTRAINT_CHECK: {
        1: 0.05, 2: 0.65, 3: 0.20, 4: 0.10,
    },
    QueryComplexity.REPAIR: {
        1: 0.05, 2: 0.60, 3: 0.25, 4: 0.10,
    },
    QueryComplexity.TEMPORAL: {
        1: 0.05, 2: 0.35, 3: 0.25, 4: 0.35,
    },
    QueryComplexity.AGGREGATION: {
        1: 0.05, 2: 0.55, 3: 0.25, 4: 0.15,
    },
}


def classify_query(query: str) -> QueryClassification:
    """Classify a query by complexity using keyword heuristics.

    Returns the most specific matching tier. If no patterns match,
    defaults to SIMPLE_LOOKUP.
    """
    q = query.lower()

    # Check each tier in specificity order (repair > constraint > temporal > aggregation > simple)
    for patterns, complexity in [
        (_REPAIR_PATTERNS, QueryComplexity.REPAIR),
        (_CONSTRAINT_PATTERNS, QueryComplexity.CONSTRAINT_CHECK),
        (_TEMPORAL_PATTERNS, QueryComplexity.TEMPORAL),
        (_AGGREGATION_PATTERNS, QueryComplexity.AGGREGATION),
    ]:
        matched = tuple(p for p in patterns if p in q)
        if matched:
            return QueryClassification(
                complexity=complexity,
                keywords_matched=matched,
                suggested_layer_weights=_LAYER_WEIGHTS[complexity],
            )

    return QueryClassification(
        complexity=QueryComplexity.SIMPLE_LOOKUP,
        keywords_matched=(),
        suggested_layer_weights=_LAYER_WEIGHTS[QueryComplexity.SIMPLE_LOOKUP],
    )
