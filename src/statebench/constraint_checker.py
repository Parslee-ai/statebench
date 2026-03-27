"""Deterministic constraint pre-computation.

Attempts to evaluate constraints against known facts before the LLM
sees them. When a constraint can be resolved deterministically (e.g.,
numeric comparison), we annotate it with a pass/fail indicator so the
LLM doesn't have to do the math.

Inspired by Honcho's tiered reasoning — push cheap deterministic work
out of the LLM and into the engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ConstraintCheckResult:
    """Result of pre-evaluating a constraint against facts."""

    constraint_fact_id: str
    constraint_text: str
    status: str  # "pass", "fail", "unknown" (can't evaluate deterministically)
    reason: str  # Human-readable explanation
    relevant_fact_ids: tuple[str, ...]  # Facts used in evaluation


def _extract_dollar_amount(text: str) -> float | None:
    """Extract a dollar amount from text (e.g., '$150,000' -> 150000.0)."""
    # Try M suffix first, then K, then plain (most specific first)
    # Word boundary after K/M prevents matching "maximum", "keep", etc.
    m = re.search(r"\$\s*([\d,]+(?:\.\d+)?)\s*[Mm]\b", text)
    if m:
        return float(m.group(1).replace(",", "")) * 1_000_000

    m = re.search(r"\$\s*([\d,]+(?:\.\d+)?)\s*[Kk]\b", text)
    if m:
        return float(m.group(1).replace(",", "")) * 1000

    # Plain dollar amount — match the full number including commas
    m = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1).replace(",", ""))

    return None


def _extract_percentage(text: str) -> float | None:
    """Extract a percentage from text."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    return float(m.group(1)) if m else None


def _extract_integer(text: str) -> int | None:
    """Extract a count/integer from text when it's the primary number."""
    m = re.search(r"\b(\d+)\s+(people|engineers|staff|team members|seats|units|items)\b", text)
    return int(m.group(1)) if m else None


def _is_budget_constraint(text: str) -> bool:
    """Check if a constraint is about budget/cost limits."""
    lower = text.lower()
    return any(kw in lower for kw in [
        "budget", "limit", "cap", "maximum", "cannot exceed",
        "spending", "allocation", "ceiling",
    ])


def _is_approval_constraint(text: str) -> bool:
    """Check if a constraint requires approval."""
    lower = text.lower()
    return any(kw in lower for kw in [
        "approval required", "needs approval", "must be approved",
        "requires sign-off", "authorization required",
    ])


@dataclass(frozen=True)
class FactValue:
    """A fact's ID and text value, for constraint checking."""

    fact_id: str
    key: str
    value: str


def check_constraints(
    constraints: Sequence[FactValue],
    facts: Sequence[FactValue],
) -> list[ConstraintCheckResult]:
    """Pre-evaluate constraints against known facts.

    Returns a ConstraintCheckResult for each constraint. Only evaluates
    constraints that can be resolved deterministically (numeric comparisons,
    boolean checks). Others get status='unknown'.
    """
    results: list[ConstraintCheckResult] = []

    for constraint in constraints:
        result = _check_single(constraint, facts)
        results.append(result)

    return results


def _check_single(
    constraint: FactValue,
    facts: Sequence[FactValue],
) -> ConstraintCheckResult:
    """Check a single constraint against facts."""

    # Budget/cost constraints: compare dollar amounts
    if _is_budget_constraint(constraint.value):
        limit = _extract_dollar_amount(constraint.value)
        if limit is not None:
            # Find facts with dollar amounts that are semantically related
            for fact in facts:
                amount = _extract_dollar_amount(fact.value)
                if amount is not None and _values_related(constraint, fact):
                    if amount <= limit:
                        return ConstraintCheckResult(
                            constraint_fact_id=constraint.fact_id,
                            constraint_text=constraint.value,
                            status="pass",
                            reason=f"${amount:,.0f} <= ${limit:,.0f} limit",
                            relevant_fact_ids=(fact.fact_id,),
                        )
                    else:
                        return ConstraintCheckResult(
                            constraint_fact_id=constraint.fact_id,
                            constraint_text=constraint.value,
                            status="fail",
                            reason=f"${amount:,.0f} exceeds ${limit:,.0f} limit",
                            relevant_fact_ids=(fact.fact_id,),
                        )

    # Approval constraints: check if approval exists in facts
    if _is_approval_constraint(constraint.value):
        for fact in facts:
            lower = fact.value.lower()
            if any(kw in lower for kw in ["approved", "signed off", "authorized"]):
                if _values_related(constraint, fact):
                    return ConstraintCheckResult(
                        constraint_fact_id=constraint.fact_id,
                        constraint_text=constraint.value,
                        status="pass",
                        reason=f"Approval found: {fact.value[:60]}",
                        relevant_fact_ids=(fact.fact_id,),
                    )
            if any(kw in lower for kw in ["denied", "rejected", "not approved"]):
                if _values_related(constraint, fact):
                    return ConstraintCheckResult(
                        constraint_fact_id=constraint.fact_id,
                        constraint_text=constraint.value,
                        status="fail",
                        reason=f"Approval denied: {fact.value[:60]}",
                        relevant_fact_ids=(fact.fact_id,),
                    )

    # Can't evaluate deterministically
    return ConstraintCheckResult(
        constraint_fact_id=constraint.fact_id,
        constraint_text=constraint.value,
        status="unknown",
        reason="Cannot evaluate deterministically",
        relevant_fact_ids=(),
    )


def _values_related(constraint: FactValue, fact: FactValue) -> bool:
    """Check if a constraint and fact are semantically related (keyword overlap)."""
    c_tokens = _stem_tokens(_extract_keywords(f"{constraint.key} {constraint.value}"))
    f_tokens = _stem_tokens(_extract_keywords(f"{fact.key} {fact.value}"))
    overlap = c_tokens & f_tokens
    # Budget constraints need only 1 overlap since amounts provide signal
    if _is_budget_constraint(constraint.value) and _extract_dollar_amount(fact.value):
        return len(overlap) >= 1
    return len(overlap) >= 2


def _stem_tokens(tokens: set[str]) -> set[str]:
    """Naive stemming: strip trailing 's', 'ed', 'ing' for better overlap."""
    stemmed = set()
    for t in tokens:
        stemmed.add(t)
        if t.endswith("s") and len(t) > 4:
            stemmed.add(t[:-1])
        if t.endswith("ed") and len(t) > 5:
            stemmed.add(t[:-2])
        if t.endswith("ing") and len(t) > 6:
            stemmed.add(t[:-3])
    return stemmed


def _extract_keywords(text: str) -> set[str]:
    """Extract lowercase keyword tokens."""
    stop = {"the", "and", "for", "are", "but", "not", "you", "all", "can",
            "this", "that", "with", "from", "will", "has", "had", "was"}
    tokens = set()
    for raw in text.replace("_", " ").split():
        cleaned = "".join(ch for ch in raw.lower() if ch.isalnum())
        if len(cleaned) >= 3 and cleaned not in stop:
            tokens.add(cleaned)
    return tokens


def format_constraint_checklist(results: list[ConstraintCheckResult]) -> str:
    """Format constraint check results as a checklist for the LLM.

    Returns a string like:
        CONSTRAINT CHECKLIST:
        [PASS] Budget: $75,000 <= $200,000 limit (F-001)
        [FAIL] Vendor approval: Not approved (F-003)
        [????] Timeline: Cannot evaluate deterministically (F-005)
    """
    if not results:
        return ""

    lines = ["CONSTRAINT CHECKLIST:"]
    for r in results:
        if r.status == "pass":
            icon = "PASS"
        elif r.status == "fail":
            icon = "FAIL"
        else:
            icon = "????"
        refs = ", ".join(r.relevant_fact_ids) if r.relevant_fact_ids else "—"
        lines.append(f"  [{icon}] {r.reason} (refs: {refs})")

    return "\n".join(lines)
