"""Evaluation harness for StateBench.

Runs timelines through memory strategies and models,
collecting results for evaluation.
"""

import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from anthropic import Anthropic, APIStatusError
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Google Gemini (lazy import to avoid issues if not installed)
try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

from statebench.baselines import MemoryStrategy, get_baseline
from statebench.evaluation import (
    BenchmarkMetrics,
    MetricsAggregator,
    QueryResult,
    create_judge,
)
from statebench.schema.timeline import ConversationTurn, Query, StateWrite, Supersession, Timeline

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0  # seconds

console = Console()


def load_timelines(path: Path) -> Iterator[Timeline]:
    """Load timelines from a JSONL file."""
    with open(path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                yield Timeline.model_validate(data)


class EvaluationHarness:
    """Runs evaluation of baselines against timelines."""

    def __init__(
        self,
        model: str = "gpt-4o",
        provider: str = "openai",
        use_llm_judge: bool = True,
        token_budget: int = 8000,
    ):
        """Initialize the harness.

        Args:
            model: Model to use for generating responses
            provider: LLM provider ("openai", "anthropic", or "google")
            use_llm_judge: Whether to use LLM for judging
            token_budget: Token budget for context
        """
        self.model = model
        self.provider = provider
        self.token_budget = token_budget
        self._client: Any = None
        # Use OpenAI for judging by default (most reliable)
        judge_provider = "openai" if provider == "google" else provider
        self.judge = create_judge(use_llm=use_llm_judge, provider=judge_provider)

    def _get_client(self) -> Any:
        """Get or create the LLM client."""
        if self._client is None:
            if self.provider == "openai":
                self._client = OpenAI()
            elif self.provider == "anthropic":
                self._client = Anthropic()
            elif self.provider == "google":
                if not HAS_GEMINI:
                    raise ImportError("google-genai package not installed. Run: pip install google-genai")
                self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
        return self._client

    def _generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int, int]:
        """Generate a response from the model.

        Returns:
            Tuple of (response, tokens_used, latency_ms)
        """
        client = self._get_client()
        start_time = time.time()

        completion_budget = max(200, min(1500, self.token_budget // 2))

        if self.provider == "openai":
            # GPT-5.x models use max_completion_tokens, older models use max_tokens
            token_param = "max_completion_tokens" if self.model.startswith("gpt-5") else "max_tokens"
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **{token_param: completion_budget},
            )
            text = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0

        elif self.provider == "anthropic":
            # Retry with exponential backoff for overloaded errors
            anthropic_response = None
            for attempt in range(MAX_RETRIES):
                try:
                    anthropic_response = client.messages.create(
                        model=self.model,
                        max_tokens=completion_budget,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                    )
                    break
                except APIStatusError as e:
                    if e.status_code == 529 and attempt < MAX_RETRIES - 1:
                        delay = RETRY_BASE_DELAY * (2 ** attempt)
                        time.sleep(delay)
                        continue
                    raise
            if anthropic_response is None:
                raise RuntimeError("Failed to get response after retries")
            text = ""
            if anthropic_response.content:
                first_block = anthropic_response.content[0]
                if hasattr(first_block, "text"):
                    text = getattr(first_block, "text", "")
            tokens = anthropic_response.usage.input_tokens + anthropic_response.usage.output_tokens

        elif self.provider == "google":
            # Combine system prompt and user prompt for Gemini
            full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
            response = client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=completion_budget,
                ),
            )
            text = response.text or ""
            # Gemini token counting
            tokens = (
                response.usage_metadata.prompt_token_count +
                response.usage_metadata.candidates_token_count
                if response.usage_metadata else 0
            )

        latency_ms = int((time.time() - start_time) * 1000)
        return text, tokens, latency_ms

    def run_timeline(
        self,
        timeline: Timeline,
        strategy: MemoryStrategy,
    ) -> list[QueryResult]:
        """Run a single timeline through a strategy.

        Args:
            timeline: The timeline to run
            strategy: The memory strategy to use

        Returns:
            List of QueryResults for each query in the timeline
        """
        strategy.reset()

        # Initialize strategies that explicitly request the initial snapshot
        if getattr(strategy, "expects_initial_state", False):
            init_fn = getattr(strategy, "initialize_from_state", None)
            if callable(init_fn):
                init_fn(timeline.initial_state)

        results = []
        query_idx = 0

        for event in timeline.events:
            if isinstance(event, Query):
                # Build context and generate response
                prompt = strategy.format_prompt(event.prompt)
                system_prompt = strategy.get_system_prompt()

                response, tokens, latency = self._generate_response(system_prompt, prompt)

                # Judge the response
                result = self.judge.judge(
                    response=response,
                    ground_truth=event.ground_truth,
                    timeline_id=timeline.id,
                    query_idx=query_idx,
                    track=timeline.track,
                    domain=timeline.domain,
                )
                result.tokens_used = tokens
                result.latency_ms = latency

                results.append(result)
                query_idx += 1

            elif isinstance(event, (ConversationTurn, StateWrite, Supersession)):
                # Process the event through the strategy
                strategy.process_event(event)

        return results

    def evaluate(
        self,
        dataset_path: Path,
        baseline_name: str,
        limit: int | None = None,
    ) -> BenchmarkMetrics:
        """Evaluate a baseline on a dataset.

        Args:
            dataset_path: Path to JSONL dataset
            baseline_name: Name of the baseline to use
            limit: Maximum number of timelines to process

        Returns:
            BenchmarkMetrics with results
        """
        strategy = get_baseline(baseline_name, token_budget=self.token_budget)
        aggregator = MetricsAggregator(baseline=baseline_name, model=self.model)

        timelines = list(load_timelines(dataset_path))
        if limit:
            timelines = timelines[:limit]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Evaluating {baseline_name} on {len(timelines)} timelines...",
                total=len(timelines),
            )

            for timeline in timelines:
                results = self.run_timeline(timeline, strategy)
                for result in results:
                    aggregator.add_result(result)
                progress.advance(task)

        return aggregator.compute_benchmark_metrics()

    def compare_baselines(
        self,
        dataset_path: Path,
        baselines: list[str],
        limit: int | None = None,
    ) -> dict[str, BenchmarkMetrics]:
        """Compare multiple baselines on a dataset.

        Args:
            dataset_path: Path to JSONL dataset
            baselines: List of baseline names to compare
            limit: Maximum number of timelines per baseline

        Returns:
            Dictionary mapping baseline name to BenchmarkMetrics
        """
        results = {}
        for baseline in baselines:
            console.print(f"\n[bold]Evaluating baseline: {baseline}[/bold]")
            results[baseline] = self.evaluate(dataset_path, baseline, limit=limit)

        return results


def run_evaluation(
    dataset_path: str,
    baseline: str,
    model: str = "gpt-4o",
    provider: str = "openai",
    limit: int | None = None,
) -> BenchmarkMetrics:
    """Convenience function to run evaluation.

    Args:
        dataset_path: Path to JSONL dataset
        baseline: Name of the baseline to use
        model: Model to use
        provider: LLM provider
        limit: Maximum timelines to process

    Returns:
        BenchmarkMetrics
    """
    harness = EvaluationHarness(model=model, provider=provider)
    return harness.evaluate(Path(dataset_path), baseline, limit=limit)
