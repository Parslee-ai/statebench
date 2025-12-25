"""LLM-as-judge for StateBench evaluation.

Uses a combination of:
1. Deterministic checks (string matching for hard constraints)
2. LLM judge (for paraphrase detection and decision classification)

The dual approach ensures high precision on clear violations while
handling paraphrases gracefully.
"""

from anthropic import Anthropic
from openai import OpenAI

from statebench.evaluation.rubric import ScoringRubric, contains_phrase, extract_decision
from statebench.evaluation.metrics import QueryResult
from statebench.schema.timeline import Query, GroundTruth


class ResponseJudge:
    """Judges model responses against ground truth."""

    def __init__(
        self,
        use_llm_judge: bool = True,
        provider: str = "openai",
    ):
        """Initialize the judge.

        Args:
            use_llm_judge: Whether to use LLM for paraphrase detection
            provider: LLM provider ("openai" or "anthropic")
        """
        self.use_llm_judge = use_llm_judge
        self.provider = provider
        self._client = None

    def _get_client(self):
        """Get or create the LLM client."""
        if self._client is None:
            if self.provider == "openai":
                self._client = OpenAI()
            else:
                self._client = Anthropic()
        return self._client

    def _llm_check_paraphrase(self, response: str, target_phrase: str) -> bool:
        """Use LLM to check if response contains a paraphrase of target."""
        if not self.use_llm_judge:
            return False

        client = self._get_client()

        prompt = f"""Does the following response contain or convey the same meaning as the target phrase?

Target phrase: "{target_phrase}"

Response: "{response}"

Answer with just YES or NO."""

        if self.provider == "openai":
            result = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
            )
            answer = result.choices[0].message.content or ""
        else:
            result = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = result.content[0].text

        return "YES" in answer.upper()

    def _llm_extract_decision(self, response: str, options: list[str]) -> str | None:
        """Use LLM to extract decision from response."""
        if not self.use_llm_judge:
            return None

        client = self._get_client()

        options_str = ", ".join(f'"{o}"' for o in options)
        prompt = f"""What decision does this response indicate? Choose from: {options_str}

Response: "{response}"

Answer with just one of the options, nothing else."""

        if self.provider == "openai":
            result = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
            )
            answer = result.choices[0].message.content or ""
        else:
            result = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=20,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = result.content[0].text

        # Extract the decision from the answer
        answer_lower = answer.lower().strip().strip('"').strip("'")
        for option in options:
            if option.lower() in answer_lower:
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
        result = QueryResult(
            timeline_id=timeline_id,
            query_idx=query_idx,
            track=track,
            domain=domain,
            expected_decision=ground_truth.decision,
            must_mention=ground_truth.must_mention,
            must_not_mention=ground_truth.must_not_mention,
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
        for phrase in ground_truth.must_mention:
            # Deterministic check first
            if contains_phrase(response, phrase):
                result.must_mention_hits.append(phrase)
            # Try LLM paraphrase check if enabled
            elif self.use_llm_judge and self._llm_check_paraphrase(response, phrase):
                result.must_mention_hits.append(phrase)
            else:
                result.must_mention_misses.append(phrase)

        # Step 3: Must not mention checks (stricter - only deterministic)
        for phrase in ground_truth.must_not_mention:
            if contains_phrase(response, phrase):
                result.must_not_mention_violations.append(phrase)

        # Step 4: Resurrection check
        result.resurrected_superseded = len(result.must_not_mention_violations) > 0

        return result


def create_judge(use_llm: bool = True, provider: str = "openai") -> ResponseJudge:
    """Create a response judge.

    Args:
        use_llm: Whether to use LLM for paraphrase detection
        provider: LLM provider

    Returns:
        Configured ResponseJudge
    """
    return ResponseJudge(use_llm_judge=use_llm, provider=provider)
