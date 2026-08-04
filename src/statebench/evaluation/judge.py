"""LLM-as-judge for StateBench evaluation.

Uses a combination of:
1. Deterministic checks (string matching for hard constraints)
2. LLM judge (for paraphrase detection and decision classification)

The dual approach ensures high precision on clear violations while
handling paraphrases gracefully.
"""

import os
import time

from anthropic import Anthropic, APIStatusError
from openai import OpenAI

from statebench.evaluation.metrics import QueryResult
from statebench.evaluation.phrase_quality import infer_kind, is_discriminative
from statebench.evaluation.rubric import (
    all_mentions_negated,
    contains_phrase,
    extract_decision,
)
from statebench.schema.timeline import GroundTruth, MentionRequirement

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0  # seconds

# ---------------------------------------------------------------------------
# Judge selection
#
# The judge is chosen GLOBALLY and is never derived from the provider of the
# system under test. Deriving it (the previous behavior) meant OpenAI arms were
# graded by gpt-4o-mini and Anthropic arms by claude-3-haiku, so every
# cross-model comparison mixed two different graders — each from the same family
# as the model it was grading. The scientific invariant is "one judge across all
# arms being compared", not any particular model, so any provider may be pinned
# as long as it is pinned for the whole comparison.
#
# Resolution order: explicit argument > $STATEBENCH_JUDGE ("provider" or
# "provider:model") > DEFAULT_JUDGE_PROVIDER.
# ---------------------------------------------------------------------------
DEFAULT_JUDGE_PROVIDER = "openai"
DEFAULT_JUDGE_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-haiku-20240307",
    # A small fast local model is plenty for yes/no paraphrase + option pick.
    "car": "mlx/qwen3-1.7b:3bit",
}


def resolve_judge_spec(
    provider: str | None = None, model: str | None = None
) -> tuple[str, str]:
    """Resolve the (provider, model) pair that will grade every arm.

    Never consults the provider of the system under test — see module note.
    """
    if provider is None:
        env = os.environ.get("STATEBENCH_JUDGE", "").strip()
        if env:
            env_provider, _, env_model = env.partition(":")
            provider = env_provider or DEFAULT_JUDGE_PROVIDER
            model = model or (env_model or None)
        else:
            provider = DEFAULT_JUDGE_PROVIDER
    if provider not in DEFAULT_JUDGE_MODELS:
        raise ValueError(
            f"Unknown judge provider {provider!r}; "
            f"expected one of {sorted(DEFAULT_JUDGE_MODELS)}"
        )
    return provider, model or DEFAULT_JUDGE_MODELS[provider]


def _get_phrase(item: str | MentionRequirement) -> str:
    """Extract phrase string from either a string or MentionRequirement."""
    if isinstance(item, str):
        return item
    return item.phrase


def _get_kind(item: str | MentionRequirement, track: str) -> str:
    """Failure class a forbidden phrase detects.

    An explicit ``MentionRequirement.kind`` wins; bare strings and untagged
    requirements (all pre-existing release data) fall back to the track's class.
    """
    if isinstance(item, MentionRequirement) and item.kind != "other":
        return item.kind
    return infer_kind(track)


class ResponseJudge:
    """Judges model responses against ground truth."""

    def __init__(
        self,
        use_llm_judge: bool = True,
        provider: str | None = None,
        model: str | None = None,
        car_model: str | None = None,
    ):
        """Initialize the judge.

        Args:
            use_llm_judge: Whether to use LLM for paraphrase detection
            provider: Judge provider ("openai", "anthropic", or "car"). Resolved
                globally — never from the provider of the system under test.
            model: Judge model id. Defaults per provider.
            car_model: Deprecated alias for ``model`` when provider is "car".
        """
        self.use_llm_judge = use_llm_judge
        self.provider, self.model = resolve_judge_spec(provider, model or car_model)
        self._openai_client: OpenAI | None = None
        self._anthropic_client: Anthropic | None = None

    @property
    def descriptor(self) -> str:
        """Stable identifier recorded alongside results, e.g. ``openai:gpt-4o-mini``.

        Published numbers are only comparable across arms graded by the same
        descriptor, so it belongs in every results file.
        """
        if not self.use_llm_judge:
            return "deterministic-only"
        return f"{self.provider}:{self.model}"

    def _get_openai_client(self) -> OpenAI:
        """Get or create the OpenAI client."""
        if self._openai_client is None:
            self._openai_client = OpenAI()
        return self._openai_client

    def _get_anthropic_client(self) -> Anthropic:
        """Get or create the Anthropic client."""
        if self._anthropic_client is None:
            self._anthropic_client = Anthropic()
        return self._anthropic_client

    def _complete(self, prompt: str, max_tokens: int) -> str:
        """Run one judge completion through the pinned provider/model."""
        if self.provider == "car":
            from statebench.runner.car_provider import car_complete

            return car_complete(prompt, model=self.model, max_tokens=max_tokens)

        if self.provider == "openai":
            result = self._get_openai_client().chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return result.choices[0].message.content or ""

        client = self._get_anthropic_client()
        for attempt in range(MAX_RETRIES):
            try:
                result = client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except APIStatusError as e:
                if e.status_code == 529 and attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY * (2**attempt))
                    continue
                raise
        if result.content:
            first_block = result.content[0]
            if hasattr(first_block, "text"):
                return getattr(first_block, "text", "")
        return ""

    def _llm_check_paraphrase(self, response: str, target_phrase: str) -> bool:
        """Use LLM to check if response contains a paraphrase of target."""
        if not self.use_llm_judge:
            return False

        prompt = f"""Does the following response contain or convey the same meaning as the target phrase?

Target phrase: "{target_phrase}"

Response: "{response}"

Answer with just YES or NO."""

        return "YES" in self._complete(prompt, max_tokens=10).upper()

    def _llm_extract_decision(self, response: str, options: list[str]) -> str | None:
        """Use LLM to extract decision from response."""
        if not self.use_llm_judge:
            return None

        options_str = ", ".join(f'"{o}"' for o in options)
        # For meta-decision labels, add clarifying context so the judge
        # understands what behavioral pattern each option describes
        hints = ""
        for o in options:
            if o == "use fresh signal":
                hints += (
                    '\nNote: "use fresh signal" means the response is based on '
                    "the most recent/updated information rather than stale data."
                )
            elif o == "recall primary task":
                hints += (
                    '\nNote: "recall primary task" means the response correctly '
                    "recalls and provides details about the primary task that was "
                    "being discussed before an interruption occurred."
                )
            elif o == "see critical state":
                hints += (
                    '\nNote: "see critical state" means the response demonstrates '
                    "awareness of the current critical state of affairs, incorporating "
                    "the latest updates and corrections rather than outdated information."
                )
            elif o == "use only permitted information":
                hints += (
                    '\nNote: "use only permitted information" means the response '
                    "answers the question using only non-restricted data. If the "
                    "response provides relevant information without mentioning "
                    "restricted or confidential data, choose this option."
                )
            elif o == "use only global/committed information":
                hints += (
                    '\nNote: "use only global/committed information" means the '
                    "response answers using only confirmed/committed facts and "
                    "does NOT reference hypothetical, draft, or exploratory "
                    "information. If the response sticks to known facts without "
                    "mentioning speculative or unconfirmed data, choose this option."
                )
        prompt = f"""What decision does this response indicate? Choose from: {options_str}
{hints}
Response: "{response}"

Answer with just one of the options, nothing else."""

        # Use enough tokens to reproduce the longest option
        max_option_tokens = max(len(o.split()) * 2 for o in options)
        judge_max_tokens = max(30, max_option_tokens)

        answer = self._complete(prompt, max_tokens=judge_max_tokens)

        # Extract the decision from the answer. Also check if the answer
        # is a truncated prefix of an option (LLM may run out of tokens
        # when reproducing long decision strings).
        answer_lower = answer.lower().strip().strip('"').strip("'")
        for option in options:
            option_lower = option.lower()
            if option_lower in answer_lower:
                return option
        # Truncation fallback: if the answer is a substantial prefix of an option
        if len(answer_lower) >= 15:
            for option in options:
                if option.lower().startswith(answer_lower):
                    return option

        return None

    def judge(
        self,
        response: str,
        ground_truth: GroundTruth,
        timeline_id: str,
        query_idx: int,
        track: str,
        domain: str,
    ) -> QueryResult:
        """Judge a response against ground truth.

        Args:
            response: The model's response
            ground_truth: Ground truth constraints
            timeline_id: ID of the timeline
            query_idx: Index of the query in the timeline
            track: Benchmark track
            domain: Business domain

        Returns:
            QueryResult with scoring details
        """
        # Convert MentionRequirement to strings for QueryResult
        must_mention_strs = [_get_phrase(m) for m in ground_truth.must_mention]
        # Only admissible forbidden phrases enter must_not_mention, so they set
        # the MNM denominator too — keeping a skipped phrase in the denominator
        # would deflate the violation rate just as scoring it would inflate it.
        must_not_mention_strs = [
            p
            for p in (_get_phrase(m) for m in ground_truth.must_not_mention)
            if is_discriminative(p)
        ]

        result = QueryResult(
            timeline_id=timeline_id,
            query_idx=query_idx,
            track=track,
            domain=domain,
            expected_decision=ground_truth.decision,
            must_mention=must_mention_strs,
            must_not_mention=must_not_mention_strs,
            response=response,
        )

        # Step 1: Deterministic decision check
        extracted, decision_correct = extract_decision(response, ground_truth.decision)
        result.actual_decision = extracted
        result.decision_correct = decision_correct

        # If deterministic check is inconclusive, try LLM
        if not decision_correct and extracted is None and self.use_llm_judge:
            options = ["yes", "no"] if ground_truth.decision in ("yes", "no") else [ground_truth.decision, "other"]
            llm_decision = self._llm_extract_decision(response, options)
            if llm_decision:
                result.actual_decision = llm_decision
                result.decision_correct = llm_decision.lower() == ground_truth.decision.lower()

        # Step 2: Must mention checks
        for item in ground_truth.must_mention:
            phrase = _get_phrase(item)
            # Deterministic check first
            if contains_phrase(response, phrase):
                result.must_mention_hits.append(phrase)
            # Try LLM paraphrase check if enabled
            elif self.use_llm_judge and self._llm_check_paraphrase(response, phrase):
                result.must_mention_hits.append(phrase)
            else:
                result.must_mention_misses.append(phrase)

        # Step 3: Must not mention checks (stricter - only deterministic).
        # Inadmissible phrases are skipped rather than scored: a phrase a
        # correct response could contain (see phrase_quality) is not evidence of
        # a violation. They are recorded, not silently dropped, so an audit can
        # see exactly what was excluded from scoring.
        for item in ground_truth.must_not_mention:
            phrase = _get_phrase(item)
            if not is_discriminative(phrase):
                result.skipped_phrases.append(phrase)
                continue
            kind = _get_kind(item, track)
            if contains_phrase(response, phrase):
                # A forbidden phrase used only under negation ("the meeting is
                # NOT Friday") demonstrates the system distinguishing dead from
                # live state. Counting it as resurrection inverts the metric.
                if all_mentions_negated(response, phrase):
                    result.negated_mentions.append(phrase)
                    continue
                result.must_not_mention_violations.append(phrase)
                result.violations_by_kind.setdefault(kind, []).append(phrase)

        # Step 4: Resurrection check. SFRR counts ONLY superseded-state
        # violations. Restricted-data leaks and fabrications are separate
        # failures with their own rates — see TrackMetrics.
        result.resurrected_superseded = bool(result.violations_by_kind.get("superseded"))

        return result


def create_judge(
    use_llm: bool = True,
    provider: str | None = None,
    model: str | None = None,
) -> ResponseJudge:
    """Create a response judge.

    Args:
        use_llm: Whether to use LLM for paraphrase detection
        provider: Judge provider. Resolved globally (see ``resolve_judge_spec``);
            pass None to take $STATEBENCH_JUDGE or the default. Callers must NOT
            pass the provider of the system under test.
        model: Judge model id; defaults per provider.

    Returns:
        Configured ResponseJudge
    """
    return ResponseJudge(use_llm_judge=use_llm, provider=provider, model=model)
