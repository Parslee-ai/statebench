"""Scoring rubrics for StateBench evaluation.

Defines how to score model responses against ground truth constraints.
"""

import re


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip()


def boundary_pattern(phrase: str) -> str:
    """Escape ``phrase`` and wrap it in word boundaries suited to its edges.

    Plain substring containment made short phrases fire inside longer words:
    ``"by"`` matched "nearby", ``"HP"`` matched "PHP", ``"100"`` matched "1000".
    A plain ``\\b`` on both sides is wrong too, because many phrases begin or end
    with a non-word character (``"$45"``, ``"10%"``) where ``\\b`` inverts its
    meaning. So each edge gets a boundary only when that edge is a word
    character: ``"$45"`` still won't match inside "$450", and ``"10%"`` won't
    match inside "110%".
    """
    lead = r"(?<!\w)" if phrase[:1].isalnum() or phrase[:1] == "_" else ""
    trail = r"(?!\w)" if phrase[-1:].isalnum() or phrase[-1:] == "_" else ""
    return lead + re.escape(phrase) + trail


def _contains_token(response_lower: str, phrase_lower: str) -> bool:
    """Boundary-aware containment of a single normalized phrase."""
    if not phrase_lower:
        return False
    return bool(re.search(boundary_pattern(phrase_lower), response_lower))


def contains_phrase(response: str, phrase: str) -> bool:
    """Check if response contains a phrase (case-insensitive, boundary-aware).

    Handles:
    - Exact matches, respecting word boundaries (see :func:`boundary_pattern`)
    - Basic paraphrase patterns (e.g., "not renew" matches "do not renew")
    - Regex patterns (if phrase starts with 'regex:')
    """
    response_lower = normalize_text(response)
    phrase_lower = normalize_text(phrase)

    # Handle regex patterns
    if phrase_lower.startswith("regex:"):
        pattern = phrase_lower[6:]
        return bool(re.search(pattern, response_lower))

    # Handle pipe-separated alternatives (e.g., "do not renew|renegotiate")
    if "|" in phrase_lower:
        alternatives = phrase_lower.split("|")
        return any(_contains_token(response_lower, alt.strip()) for alt in alternatives)

    # Direct containment
    if _contains_token(response_lower, phrase_lower):
        return True

    # Common paraphrase patterns
    paraphrase_patterns = [
        # "not X" variations
        (r"do not (\w+)", r"don't \1"),
        (r"don't (\w+)", r"do not \1"),
        (r"cannot (\w+)", r"can't \1"),
        (r"can't (\w+)", r"cannot \1"),
        # "should not X" variations
        (r"should not (\w+)", r"shouldn't \1"),
        (r"shouldn't (\w+)", r"should not \1"),
    ]

    for pattern, replacement in paraphrase_patterns:
        try:
            paraphrased = re.sub(pattern, replacement, phrase_lower)
            if paraphrased != phrase_lower and _contains_token(
                response_lower, paraphrased
            ):
                return True
        except re.error:
            continue

    return False


# Cues that invert the meaning of a following phrase. A superseded value stated
# under negation ("the meeting is NOT Friday, it moved to Thursday") is the
# system correctly distinguishing dead state from live state — the opposite of
# resurrection — yet plain containment scores it as a violation. The same holds
# for action phrases: "do not proceed" contains "proceed".
NEGATION_CUES = (
    "not ", "n't ", "no longer", "never", "cannot", "can not",
    "instead of", "rather than", "no more", "avoid ", "without ",
    "isn't", "wasn't", "aren't", "weren't", "don't", "doesn't", "didn't",
    "superseded", "outdated", "previously", "used to be", "was changed",
    "no longer valid", "not valid", "invalid",
)

# How far back to look for a negation cue governing a match.
_NEGATION_WINDOW = 40


def all_mentions_negated(response: str, phrase: str) -> bool:
    """True if ``phrase`` occurs and EVERY occurrence sits under a negation cue.

    Used only for must-not-mention scoring. Conservative by construction: a
    single un-negated occurrence makes this False, so a response that leaks the
    value plainly anywhere still counts as a violation.
    """
    response_lower = normalize_text(response)
    phrase_lower = normalize_text(phrase)
    if phrase_lower.startswith("regex:"):
        return False  # author took explicit control of matching

    alternatives = (
        [a.strip() for a in phrase_lower.split("|")]
        if "|" in phrase_lower
        else [phrase_lower]
    )
    found_any = False
    for alt in alternatives:
        if not alt:
            continue
        for match in re.finditer(boundary_pattern(alt), response_lower):
            found_any = True
            window = response_lower[max(0, match.start() - _NEGATION_WINDOW) : match.start()]
            if not any(cue in window for cue in NEGATION_CUES):
                return False  # an un-negated occurrence — genuine leakage
    return found_any


def extract_decision(response: str, expected: str) -> tuple[str | None, bool]:
    """Extract decision from response and check if it matches expected.

    Args:
        response: Model response
        expected: Expected decision (e.g., "yes", "no", specific value)

    Returns:
        Tuple of (extracted_decision, is_correct)
    """
    response_lower = normalize_text(response)
    expected_lower = normalize_text(expected)

    # Binary decisions
    if expected_lower in ("yes", "no"):
        # Look for clear yes/no signals. Matched on word boundaries: plain
        # substring search made bare "no" fire inside "now", "know", "noted"
        # and "nothing", so a correct factual answer with no yes-signal ("Right
        # now, the budget is $150,000") was extracted as "no" — and because
        # extraction succeeded, the LLM fallback never ran to correct it.
        yes_signals = ["yes", "go ahead", "proceed", "approved", "can do", "will do"]
        no_signals = ["no", "don't", "do not", "cannot", "should not", "shouldn't",
                      "stop", "hold off", "not possible", "not permitted", "exceeds"]

        def _has(signals: list[str]) -> bool:
            return any(_contains_token(response_lower, s) for s in signals)

        has_yes = _has(yes_signals)
        has_no = _has(no_signals)

        if has_no and not has_yes:
            return "no", expected_lower == "no"
        elif has_yes and not has_no:
            return "yes", expected_lower == "yes"
        elif has_no and has_yes:
            # Conflicting signals — prefer explicit standalone "no"/"yes" words
            # over contextual matches like "approved" in explanation text
            explicit_no = bool(re.search(r'\bno\b', response_lower))
            explicit_yes = bool(re.search(r'\byes\b', response_lower))
            if explicit_no and not explicit_yes:
                return "no", expected_lower == "no"
            if explicit_yes and not explicit_no:
                return "yes", expected_lower == "yes"
            # Both explicit or neither — fall back to first position
            def _first(signals: list[str]) -> int:
                positions = []
                for s in signals:
                    m = re.search(boundary_pattern(s), response_lower)
                    if m:
                        positions.append(m.start())
                return min(positions, default=999)

            first_no = _first(no_signals)
            first_yes = _first(yes_signals)
            if first_no < first_yes:
                return "no", expected_lower == "no"
            else:
                return "yes", expected_lower == "yes"

        # No explicit yes/no signals found. A substantive factual response
        # (providing requested information without hedging) is an implicit
        # affirmative. Refusal/uncertainty signals indicate non-answers.
        refusal_signals = [
            "not specified", "not provided", "no information",
            "cannot determine", "insufficient", "unclear",
            "not available", "not mentioned", "unknown",
        ]
        has_refusal = any(signal in response_lower for signal in refusal_signals)
        is_substantive = len(response_lower.split()) >= 5

        if is_substantive and not has_refusal:
            return "yes", expected_lower == "yes"

        return None, False

    # Non-binary decisions: check if expected value appears in response
    if expected_lower in response_lower:
        return expected_lower, True

    return None, False


class ScoringRubric:
    """Rubric for scoring a response against ground truth."""

    def __init__(
        self,
        decision: str,
        must_mention: list[str],
        must_not_mention: list[str],
    ):
        self.decision = decision
        self.must_mention = must_mention
        self.must_not_mention = must_not_mention

    def score(self, response: str) -> dict[str, object]:
        """Score a response against the rubric.

        Returns:
            Dictionary with scoring details
        """
        # Decision
        extracted_decision, decision_correct = extract_decision(response, self.decision)

        # Must mention
        must_mention_hits = []
        must_mention_misses = []
        for phrase in self.must_mention:
            if contains_phrase(response, phrase):
                must_mention_hits.append(phrase)
            else:
                must_mention_misses.append(phrase)

        # Must not mention
        must_not_mention_violations = []
        for phrase in self.must_not_mention:
            if contains_phrase(response, phrase):
                must_not_mention_violations.append(phrase)

        # Resurrection: any must_not_mention violation counts
        resurrected_superseded = len(must_not_mention_violations) > 0

        return {
            "decision_correct": decision_correct,
            "extracted_decision": extracted_decision,
            "must_mention_hits": must_mention_hits,
            "must_mention_misses": must_mention_misses,
            "must_not_mention_violations": must_not_mention_violations,
            "resurrected_superseded": resurrected_superseded,
            "must_mention_rate": len(must_mention_hits) / len(self.must_mention) if self.must_mention else 1.0,
            "must_not_mention_clean": len(must_not_mention_violations) == 0,
        }
