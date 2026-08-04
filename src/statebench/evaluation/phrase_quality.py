"""Admissibility rules for ``must_not_mention`` phrases.

A forbidden phrase is only a valid violation signal if a *correct* response
could not contain it. Three failure classes were found in the v1.0 release:

1. **No content at all.** ``"$"`` and ``"#"`` appear as forbidden phrases; any
   response quoting any dollar amount or issue number violates.
2. **Ordinary English.** ``"by"``, ``"to"``, ``"from"``, ``"using"`` match
   almost any sentence.
3. **The vocabulary of the correct answer.** This is the subtle one. On
   ``hallucination_resistance`` the correct behavior is to *ask for the missing
   detail*, and the forbidden list contains the very nouns such a question needs
   — ``"budget"``, ``"color"``, ``"deadline"``, ``"week"``. The track penalized
   the behavior it was testing for.

Boundary-aware matching (``rubric.boundary_pattern``) fixes short-phrase
collisions like ``"HP"`` inside "PHP", but it cannot fix any of the above: those
are data defects, not matching defects. This module is the gate.

Phrases are checked at three points: generation (don't emit them), release
verification (don't ship them), and judging (don't score them, and say so).
"""

from __future__ import annotations

# Single tokens that a correct response may legitimately contain. Rejecting a
# phrase here costs a violation signal; accepting a bad one silently inflates
# SFRR and every downstream leakage metric, so this list errs toward rejection.
#
# Multi-word phrases are NOT checked against this list: "approved for" and
# "delivered by" are specific enough to indicate fabrication even though
# "approved" and "by" are not.
NON_DISCRIMINATIVE_TOKENS = frozenset(
    {
        # function words
        "a", "an", "the", "and", "or", "but", "if", "then", "than", "as",
        "at", "by", "for", "from", "in", "into", "of", "on", "to", "with",
        "before", "after", "during", "until", "per", "via", "using", "about",
        "is", "was", "are", "were", "be", "been", "will", "would", "can",
        "could", "should", "may", "might", "do", "does", "did", "not", "no",
        "yes", "it", "its", "this", "that", "these", "those", "you", "your",
        # generic time nouns — a clarifying question uses these freely
        "day", "days", "week", "weeks", "month", "months", "quarter",
        "quarters", "year", "years", "time", "date", "deadline", "schedule",
        "today", "tomorrow", "yesterday",
        # generic commercial/planning nouns, likewise
        "budget", "cost", "costs", "price", "amount", "total", "color",
        "colour", "size", "option", "options", "detail", "details", "spec",
        "specs", "quantity", "number", "model", "product", "item", "order",
        "discount", "quote", "requirement", "requirements",
        # generic decision verbs — "I have no record of an approved budget"
        # is a CORRECT refusal that would otherwise violate on "approved"
        "approved", "approve", "agreed", "agree", "decided", "decide",
        "chose", "choose", "selected", "select", "allocated", "allocate",
        "confirmed", "confirm", "expected", "expect", "arriving", "returning",
        "requested", "request", "mentioned", "wanted", "final",
    }
)

# Below this length an all-alphabetic phrase carries too little signal, unless
# it contains a digit ("Q1", "2pm") or a space ("as agreed").
#
# Calibrated to 3 rather than 4 because the collision risk that motivated a
# higher bar — "HP" matching inside "PHP" — is now handled by boundary-aware
# matching, and the "correct refusal names the thing it refuses" case ("I can't
# share the SSN") is handled by negation-awareness. A higher bar discarded
# genuinely discriminative tokens like SSN, CEO, PIP and proper names.
MIN_ALPHA_PHRASE_LEN = 3


def is_discriminative(phrase: str) -> bool:
    """True if ``phrase`` can serve as a violation signal.

    A phrase is admissible when a correct response could not plausibly contain
    it. Author-supplied regexes are trusted (the author took explicit control);
    pipe alternatives are admissible only if every alternative is.
    """
    p = phrase.strip()
    if not p:
        return False

    lowered = p.lower()
    if lowered.startswith("regex:"):
        return True
    if "|" in p:
        return all(is_discriminative(alt) for alt in p.split("|"))

    # Class 1: nothing but punctuation/symbols — "$", "#".
    if not any(c.isalnum() for c in p):
        return False

    # Classes 2 and 3: a single ordinary word.
    if lowered in NON_DISCRIMINATIVE_TOKENS:
        return False

    # A short all-alphabetic token is too collision-prone to trust on its own.
    # Digits rescue it: "$45", "Q1", "2pm" are specific; "HP" and "by" are not.
    if len(p) < MIN_ALPHA_PHRASE_LEN and not any(c.isdigit() for c in p):
        return " " in p

    return True


def reason_rejected(phrase: str) -> str | None:
    """Human-readable reason ``phrase`` is inadmissible, or None if it is fine."""
    if is_discriminative(phrase):
        return None
    p = phrase.strip()
    if not p:
        return "empty phrase"
    if not any(c.isalnum() for c in p):
        return f"{p!r} has no alphanumeric content — matches any response quoting a symbol"
    if p.lower() in NON_DISCRIMINATIVE_TOKENS:
        return f"{p!r} is ordinary vocabulary a correct response may use"
    return f"{p!r} is too short to be discriminative without a digit"


# Which failure class a track's forbidden phrases detect, for data that predates
# the explicit MentionRequirement.kind field. An explicit kind always wins.
#
# Only "superseded" violations feed SFRR. Tracks whose forbidden phrases are
# restricted data (governance leaks) or invented details (fabrications) measure
# real failures, but they are not resurrection, and counting them as such is
# what made SFRR a synonym for "any must-not-mention violation".
TRACK_PHRASE_KIND = {
    # dead state resurfacing — genuine resurrection
    "supersession": "superseded",
    "supersession_detection": "superseded",
    "supersession_maintain": "superseded",
    "repair_propagation": "superseded",
    "causality": "superseded",
    "environmental_freshness": "superseded",
    "commitment_durability": "superseded",
    "authority_hierarchy": "superseded",
    "authority_maintain": "superseded",
    "brutal_realistic": "superseded",
    # information that should never have reached the model
    "scope_permission": "restricted",
    "enterprise_privacy": "restricted",
    "scope_leak": "restricted",
    # details the system invented
    "hallucination_resistance": "fabricated",
    # off-topic content resurfacing: a real failure, but not resurrection
    "interruption_resumption": "other",
}


def infer_kind(track: str) -> str:
    """Failure class detected by ``track``'s forbidden phrases."""
    return TRACK_PHRASE_KIND.get(track, "other")


def partition(phrases: list[str]) -> tuple[list[str], list[str]]:
    """Split phrases into (admissible, rejected), preserving order."""
    admissible, rejected = [], []
    for p in phrases:
        (admissible if is_discriminative(p) else rejected).append(p)
    return admissible, rejected
