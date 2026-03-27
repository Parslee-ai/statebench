"""Confidence scoring for facts.

Replaces binary valid/invalid with graduated confidence levels.
Inspired by Honcho's natural language certainty qualifiers.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from statebench.schema.state import AuthorityLevel

Confidence = Literal["definite", "likely", "uncertain", "stale"]


def infer_confidence(
    *,
    authority: AuthorityLevel,
    source_type: str,
    scope: str,
    is_constraint: bool,
    ts: datetime,
    now: datetime | None = None,
    corroborating_count: int = 0,
    constraint_type: str | None = None,
) -> Confidence:
    """Infer confidence level from fact metadata.

    Rules (checked in this order):
    1. Expired TTL → stale
    2. Hypothetical/draft scope → uncertain
    3. Policy/executive source + constraint → definite
    4. Policy/system source → definite
    5. High authority (policy/executive/manager) → likely
    6. Multiple corroborating facts → likely
    7. Unverified source → uncertain
    8. Default → likely
    """
    # TTL-based staleness check (uses type-aware TTL, not fixed 90 days)
    if now is not None and is_expired(
        ts=ts, now=now, source_type=source_type,
        is_constraint=is_constraint, constraint_type=constraint_type,
        scope=scope,
    ):
        return "stale"

    # Hypothetical/draft always uncertain
    if scope in ("hypothetical", "draft"):
        return "uncertain"

    # Policy/executive constraints are definite
    if is_constraint and authority in ("policy", "executive"):
        return "definite"

    # Policy and system sources
    if source_type in ("policy", "system"):
        return "definite"

    # High authority
    if authority in ("policy", "executive", "manager"):
        return "likely"

    # Corroborated by multiple sources
    if corroborating_count >= 2:
        return "likely"

    # Single unverified source
    if authority == "unverified":
        return "uncertain"

    # Default
    return "likely"


CONFIDENCE_DISPLAY: dict[Confidence, str] = {
    "definite": "CONFIRMED",
    "likely": "",  # Default — no annotation needed
    "uncertain": "UNVERIFIED",
    "stale": "STALE",
}


def confidence_annotation(confidence: Confidence) -> str:
    """Get display annotation for a confidence level.

    Returns empty string for 'likely' (the default, no noise needed).
    """
    label = CONFIDENCE_DISPLAY.get(confidence, "")
    return f" [{label}]" if label else ""


# ── Temporal Lifecycle (TTL) ──────────────────────────────────────────

# Default TTL by fact type/source (days)
_TTL_BY_SOURCE: dict[str, int] = {
    "policy": 365,
    "system": 180,
    "external": 90,
    "user": 90,
}

_TTL_BY_CONSTRAINT_TYPE: dict[str, int] = {
    "policy": 365,
    "deadline": 30,
    "budget": 90,
    "capacity": 30,
}


def infer_ttl_days(
    *,
    source_type: str,
    is_constraint: bool,
    constraint_type: str | None = None,
    scope: str = "global",
) -> int:
    """Infer TTL in days based on fact metadata.

    Returns the number of days a fact should be considered fresh.
    After TTL expires, the fact moves to 'stale' confidence.
    """
    # Hypothetical/draft facts expire fast
    if scope in ("hypothetical", "draft"):
        return 7

    # Session-scoped facts expire with the session
    if scope == "session":
        return 1

    # Constraint-specific TTL
    if is_constraint and constraint_type:
        return _TTL_BY_CONSTRAINT_TYPE.get(constraint_type, 90)

    # Source-based TTL
    return _TTL_BY_SOURCE.get(source_type, 90)


def is_expired(
    *,
    ts: datetime,
    now: datetime,
    source_type: str,
    is_constraint: bool = False,
    constraint_type: str | None = None,
    scope: str = "global",
) -> bool:
    """Check if a fact has exceeded its TTL."""
    ttl = infer_ttl_days(
        source_type=source_type,
        is_constraint=is_constraint,
        constraint_type=constraint_type,
        scope=scope,
    )
    age = now - ts
    return age > timedelta(days=ttl)
