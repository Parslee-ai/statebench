"""StateBench custom task for Lighteval.

This module provides a Lighteval-compatible task for evaluating LLMs on StateBench,
a benchmark for measuring state correctness in multi-turn conversations.

Usage:
    lighteval accelerate \
        "model_name=meta-llama/Llama-2-7b-hf" \
        "statebench|0|0" \
        --custom-tasks lighteval_tasks/statebench_task.py

The task handles multi-turn state by:
1. Loading timelines from parslee/statebench
2. Pre-computing context using transcript_replay baseline
3. Scoring model responses against ground truth
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from aenum import extend_enum
from lighteval.metrics import Metrics
from lighteval.metrics.utils.metric_utils import (
    MetricCategory,
    SampleLevelMetricGrouping,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

# =============================================================================
# Multi-turn Context Pre-computation
# =============================================================================

# Cache for pre-computed contexts (timeline_id -> list of query contexts)
_context_cache: dict[str, list[dict[str, Any]]] = {}


def _get_baseline():
    """Lazy-load the transcript_replay baseline."""
    from statebench.baselines import get_baseline
    return get_baseline("transcript_replay", token_budget=8000)


def _precompute_timeline_queries(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Pre-compute context for each query in a timeline.

    StateBench timelines contain multiple events (conversation turns, state writes)
    followed by query events. We process events sequentially to build context
    at each query point.
    """
    from statebench.huggingface import hf_row_to_timeline
    from statebench.schema.timeline import Query

    timeline = hf_row_to_timeline(dict(row))
    baseline = _get_baseline()
    baseline.reset()

    queries = []
    query_idx = 0

    for event in timeline.events:
        if isinstance(event, Query):
            # Build context at this point in conversation
            context_result = baseline.build_context(event.prompt)

            queries.append({
                "timeline_id": timeline.id,
                "query_idx": query_idx,
                "track": timeline.track,
                "domain": timeline.domain,
                "difficulty": timeline.difficulty,
                "context": context_result.context,
                "query": event.prompt,
                "ground_truth": event.ground_truth,
            })
            query_idx += 1
        else:
            # Process non-query events to update state
            baseline.process_event(event)

    return queries


def _get_query_for_row(row: dict[str, Any], query_idx: int = 0) -> dict[str, Any]:
    """Get pre-computed query context for a timeline row.

    Since Lighteval processes one doc at a time, we cache computed contexts.
    """
    timeline_id = row["id"]

    if timeline_id not in _context_cache:
        _context_cache[timeline_id] = _precompute_timeline_queries(row)

    queries = _context_cache[timeline_id]
    if query_idx < len(queries):
        return queries[query_idx]

    # Fallback to first query if index out of range
    return queries[0] if queries else None


# =============================================================================
# Prompt Function
# =============================================================================

def statebench_prompt(line: dict[str, Any], task_name: str) -> Doc:
    """Convert a StateBench timeline to a Lighteval Doc.

    This function:
    1. Pre-computes context using transcript_replay baseline
    2. Formats the prompt with context + query
    3. Packages ground truth for scoring
    """
    # Get pre-computed query context
    query_data = _get_query_for_row(line)

    if query_data is None:
        # Handle edge case of no queries
        return Doc(
            task_name=task_name,
            query="No query available",
            choices=["N/A"],
            gold_index=0,
        )

    # Build the prompt
    context = query_data["context"]
    query = query_data["query"]
    ground_truth = query_data["ground_truth"]

    prompt = f"""Here is the conversation history and relevant context:

{context}

---

Based on the above context, please answer the following question:

{query}

Answer:"""

    # Package ground truth in instruction field for metric access
    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[ground_truth.decision],  # Expected answer
        gold_index=0,
        instruction=json.dumps({
            "timeline_id": query_data["timeline_id"],
            "track": query_data["track"],
            "expected_decision": ground_truth.decision,
            "decision_type": ground_truth.decision_type,
            "must_mention": [
                m if isinstance(m, str) else m.phrase
                for m in ground_truth.must_mention
            ],
            "must_not_mention": [
                m if isinstance(m, str) else m.phrase
                for m in ground_truth.must_not_mention
            ],
        }),
    )


# =============================================================================
# Custom Metrics
# =============================================================================

def statebench_metrics(doc: Doc, model_response) -> dict[str, float]:
    """Compute StateBench metrics for a single sample.

    Returns:
        dict with keys: decision_accuracy, must_mention_rate,
                       must_not_mention_violations, sfrr_proxy
    """
    response = model_response.result[0] if model_response.result else ""
    response_lower = response.lower()

    # Parse ground truth from instruction
    try:
        gt = json.loads(doc.instruction)
    except (json.JSONDecodeError, TypeError):
        return {
            "decision_accuracy": 0.0,
            "must_mention_rate": 0.0,
            "must_not_mention_violations": 0.0,
        }

    # Decision accuracy (fuzzy match)
    expected = gt["expected_decision"].lower()
    decision_correct = _check_decision_match(response_lower, expected, gt["decision_type"])

    # Must mention rate
    must_mention = gt.get("must_mention", [])
    if must_mention:
        mentions_found = sum(1 for phrase in must_mention if phrase.lower() in response_lower)
        must_mention_rate = mentions_found / len(must_mention)
    else:
        must_mention_rate = 1.0  # No requirements = perfect score

    # Must not mention violations
    must_not_mention = gt.get("must_not_mention", [])
    if must_not_mention:
        violations = sum(1 for phrase in must_not_mention if phrase.lower() in response_lower)
        violation_rate = violations / len(must_not_mention)
    else:
        violation_rate = 0.0  # No forbidden phrases = no violations

    return {
        "decision_accuracy": float(decision_correct),
        "must_mention_rate": must_mention_rate,
        "must_not_mention_violations": violation_rate,
    }


def _check_decision_match(response: str, expected: str, decision_type: str) -> bool:
    """Check if response contains the expected decision."""
    if decision_type == "binary":
        # For yes/no questions
        if expected in ["yes", "true", "correct"]:
            return any(w in response for w in ["yes", "true", "correct", "affirmative"])
        elif expected in ["no", "false", "incorrect"]:
            return any(w in response for w in ["no", "false", "incorrect", "negative"])

    # Default: check if expected appears in response
    return expected in response


# Register metrics with Lighteval
statebench_metric_group = SampleLevelMetricGrouping(
    metric_name=["decision_accuracy", "must_mention_rate", "must_not_mention_violations"],
    higher_is_better={
        "decision_accuracy": True,
        "must_mention_rate": True,
        "must_not_mention_violations": False,
    },
    category=MetricCategory.GENERATIVE,
    sample_level_fn=statebench_metrics,
    corpus_level_fn={
        "decision_accuracy": np.mean,
        "must_mention_rate": np.mean,
        "must_not_mention_violations": np.mean,
    },
)

extend_enum(Metrics, "STATEBENCH", statebench_metric_group)


# =============================================================================
# Task Configurations
# =============================================================================

# All StateBench tracks
TRACKS = [
    "supersession",
    "commitment_durability",
    "interruption_resumption",
    "scope_permission",
    "environmental_freshness",
    "hallucination_resistance",
    "scope_leak",
    "causality",
    "repair_propagation",
    "brutal_realistic",
    "supersession_detection",
    "authority_hierarchy",
    "enterprise_privacy",
]


# Main task (all tracks)
statebench_task = LightevalTaskConfig(
    name="statebench",
    prompt_function=statebench_prompt,
    hf_repo="parslee/statebench",
    hf_subset="default",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[Metrics.STATEBENCH],
    generation_size=512,
    stop_sequence=["\n\n", "Question:", "User:"],
)


# Per-track tasks
class StateBenchTrackTask(LightevalTaskConfig):
    """Task configuration for a specific StateBench track."""

    def __init__(self, track: str):
        super().__init__(
            name=f"statebench:{track}",
            prompt_function=statebench_prompt,
            hf_repo="parslee/statebench",
            hf_subset="default",
            hf_avail_splits=["train", "validation", "test"],
            evaluation_splits=["test"],
            few_shots_split=None,
            few_shots_select=None,
            metrics=[Metrics.STATEBENCH],
            generation_size=512,
            stop_sequence=["\n\n", "Question:", "User:"],
            # Filter to specific track
            hf_filter=lambda x: x["track"] == track,
        )


# Create per-track tasks
TRACK_TASKS = [StateBenchTrackTask(track) for track in TRACKS]


# Export tasks table for Lighteval discovery
TASKS_TABLE = [statebench_task] + TRACK_TASKS


if __name__ == "__main__":
    print(f"StateBench Lighteval tasks loaded: {len(TASKS_TABLE)} tasks")
    print("Main task: statebench")
    print(f"Track tasks: {[f'statebench:{t}' for t in TRACKS]}")
