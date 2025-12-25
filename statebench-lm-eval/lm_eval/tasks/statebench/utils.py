"""StateBench utility functions for lm-evaluation-harness.

This module provides the custom processing functions required by the StateBench
task configuration. It handles:

1. Flattening multi-turn timelines into query-level documents
2. Pre-computing context using the transcript_replay baseline
3. Scoring model responses against ground truth
"""

from __future__ import annotations

from typing import Any

from datasets import Dataset

# Lazy imports to avoid circular dependencies
_baseline = None
_judge = None


def get_baseline():
    """Get or create the transcript_replay baseline."""
    global _baseline
    if _baseline is None:
        from statebench.baselines import get_baseline as sb_get_baseline
        _baseline = sb_get_baseline("transcript_replay", token_budget=8000)
    return _baseline


def get_judge():
    """Get or create the response judge (deterministic only for reproducibility)."""
    global _judge
    if _judge is None:
        from statebench.evaluation import create_judge
        _judge = create_judge(use_llm=False)  # Deterministic for reproducibility
    return _judge


def process_docs(dataset: Dataset) -> list[dict[str, Any]]:
    """Flatten timelines to query-level documents with pre-computed context.

    This is the core transformation that enables lm-eval compatibility.
    Each timeline may have multiple queries; we create one doc per query
    with the context computed at that point in the conversation.

    Args:
        dataset: HuggingFace dataset with timeline rows

    Returns:
        List of flattened docs, one per query
    """
    from statebench.huggingface import hf_row_to_timeline
    from statebench.schema.timeline import Query

    processed = []
    baseline = get_baseline()

    for row in dataset:
        # Convert HF row back to Timeline
        timeline = hf_row_to_timeline(dict(row))
        baseline.reset()

        query_idx = 0
        for event in timeline.events:
            if isinstance(event, Query):
                # Build context at this point
                context_result = baseline.build_context(event.prompt)

                processed.append({
                    # Identifiers
                    "timeline_id": timeline.id,
                    "query_idx": query_idx,
                    "track": timeline.track,
                    "domain": timeline.domain,
                    "difficulty": timeline.difficulty,

                    # Context and query
                    "context": context_result.context,
                    "query": event.prompt,

                    # Ground truth
                    "expected_decision": event.ground_truth.decision,
                    "decision_type": event.ground_truth.decision_type,
                    "must_mention": [
                        m if isinstance(m, str) else m.phrase
                        for m in event.ground_truth.must_mention
                    ],
                    "must_not_mention": [
                        m if isinstance(m, str) else m.phrase
                        for m in event.ground_truth.must_not_mention
                    ],

                    # Raw ground truth for detailed scoring
                    "ground_truth_json": event.ground_truth.model_dump_json(),
                })
                query_idx += 1
            else:
                # Process non-query events to update state
                baseline.process_event(event)

    return processed


def doc_to_text(doc: dict[str, Any]) -> str:
    """Format context + query for LLM input.

    Creates a prompt that includes:
    1. The conversation context (from transcript_replay baseline)
    2. A separator
    3. The query

    Args:
        doc: Flattened document from process_docs

    Returns:
        Formatted prompt string
    """
    context = doc["context"]
    query = doc["query"]

    # Build prompt similar to StateBench harness
    prompt = f"""Here is the conversation history and relevant context:

{context}

---

Based on the above context, please answer the following question:

{query}

Answer:"""

    return prompt


def doc_to_target(doc: dict[str, Any]) -> str:
    """Extract expected decision as the target.

    For binary decisions (yes/no), this is straightforward.
    For categorical/freeform, we use the decision value directly.

    Args:
        doc: Flattened document from process_docs

    Returns:
        Expected decision string
    """
    return doc["expected_decision"]


def decision_accuracy(predictions: list[str], references: list[dict[str, Any]]) -> float:
    """Compute decision accuracy using StateBench's deterministic judge.

    This measures whether the model's response contains the correct decision,
    using fuzzy matching for semantic equivalence.

    Args:
        predictions: Model output strings
        references: List of doc dicts containing ground truth

    Returns:
        Accuracy as a float between 0 and 1
    """
    judge = get_judge()
    correct = 0

    for pred, ref in zip(predictions, references):
        expected = ref["expected_decision"]
        decision_type = ref["decision_type"]

        # Extract decision from model response
        extracted = judge.extract_decision(pred, decision_type)

        # Compare with expected
        if judge.decisions_match(extracted, expected, decision_type):
            correct += 1

    return correct / len(predictions) if predictions else 0.0


def filter_supersession_tracks(doc: dict[str, Any]) -> bool:
    """Filter to include only supersession-related tracks.

    Used by the statebench_supersession task variant.

    Args:
        doc: Document to filter

    Returns:
        True if document is from a supersession track
    """
    return doc["track"] in ["supersession", "supersession_detection"]


def filter_scope_tracks(doc: dict[str, Any]) -> bool:
    """Filter to include only scope-related tracks.

    Args:
        doc: Document to filter

    Returns:
        True if document is from a scope track
    """
    return doc["track"] in ["scope_permission", "scope_leak", "enterprise_privacy"]
