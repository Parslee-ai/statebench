"""Scoring rubrics for StateBench evaluation.

Defines how to score model responses against ground truth constraints.
"""

import re


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip()


def contains_phrase(response: str, phrase: str) -> bool:
    """Check if response contains a phrase (case-insensitive).

    Handles:
    - Exact matches
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
        return any(alt.strip() in response_lower for alt in alternatives)

    # Direct containment
    if phrase_lower in response_lower:
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
            if paraphrased != phrase_lower and paraphrased in response_lower:
                return True
        except re.error:
            continue

    return False


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
        # Look for clear yes/no signals
        yes_signals = ["yes", "go ahead", "proceed", "approved", "can do", "will do"]
        no_signals = ["no", "don't", "do not", "cannot", "should not", "shouldn't", "stop", "hold off"]

        has_yes = any(signal in response_lower for signal in yes_signals)
        has_no = any(signal in response_lower for signal in no_signals)

        if has_no and not has_yes:
            return "no", expected_lower == "no"
        elif has_yes and not has_no:
            return "yes", expected_lower == "yes"
        elif has_no and has_yes:
            # Conflicting signals - check which comes first or is stronger
            # "No" followed by explanation usually means no
            first_no = min((response_lower.find(s) for s in no_signals if s in response_lower), default=999)
            first_yes = min((response_lower.find(s) for s in yes_signals if s in response_lower), default=999)
            if first_no < first_yes:
                return "no", expected_lower == "no"
            else:
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
