"""Judge calibration for StateBench.

Compares LLM judge outputs to human annotations to measure agreement
and establish judge reliability.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from statebench.evaluation.judge import create_judge
from statebench.schema.timeline import GroundTruth, MentionRequirement


def _get_phrase_str(item: str | MentionRequirement) -> str:
    """Extract phrase string from either a string or MentionRequirement."""
    if isinstance(item, str):
        return item
    return item.phrase


class HumanLabels(BaseModel):
    """Human annotation labels for a query-response pair."""
    decision_correct: bool
    must_mention_hits: list[str]
    must_not_mention_violations: list[str]
    annotator: str
    timestamp: str


class AuditItem(BaseModel):
    """A single item in the calibration audit set."""
    timeline_id: str
    query_idx: int
    response: str
    ground_truth: GroundTruth
    human_labels: HumanLabels


class CalibrationResult(BaseModel):
    """Results from calibration comparison."""
    total_items: int
    decision_agreement: float  # % of decisions where judge matches human
    decision_kappa: float  # Cohen's kappa for decision
    must_mention_precision: float
    must_mention_recall: float
    must_not_mention_precision: float
    must_not_mention_recall: float

    # Detailed breakdowns
    decision_confusion: dict[str, dict[str, int]]  # human -> judge -> count
    must_mention_details: list[dict[str, Any]]
    must_not_mention_details: list[dict[str, Any]]


def load_audit_set(path: Path) -> Iterator[AuditItem]:
    """Load audit set from JSONL file."""
    with open(path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                yield AuditItem.model_validate(data)


def calculate_cohens_kappa(confusion: dict[str, dict[str, int]]) -> float:
    """Calculate Cohen's kappa from a confusion matrix.

    Args:
        confusion: Nested dict of human_label -> judge_label -> count

    Returns:
        Cohen's kappa coefficient (-1 to 1, higher is better agreement)
    """
    # Get all labels
    labels = list(set(confusion.keys()) | {k for v in confusion.values() for k in v.keys()})
    if len(labels) < 2:
        return 1.0  # Perfect agreement if only one label

    # Build counts
    n = sum(sum(v.values()) for v in confusion.values())
    if n == 0:
        return 0.0

    # Calculate observed agreement
    p_o = sum(confusion.get(lbl, {}).get(lbl, 0) for lbl in labels) / n

    # Calculate expected agreement
    p_e = 0.0
    for label in labels:
        human_count = sum(confusion.get(label, {}).values())
        judge_count = sum(v.get(label, 0) for v in confusion.values())
        p_e += (human_count / n) * (judge_count / n)

    # Kappa
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def run_calibration(
    audit_set_path: Path,
    use_llm_judge: bool = True,
    provider: str = "openai",
) -> CalibrationResult:
    """Run calibration comparing judge to human labels.

    Args:
        audit_set_path: Path to audit set JSONL
        use_llm_judge: Whether to use LLM judge
        provider: LLM provider

    Returns:
        CalibrationResult with agreement metrics
    """
    judge = create_judge(use_llm=use_llm_judge, provider=provider)
    items = list(load_audit_set(audit_set_path))

    # Track agreement
    decision_correct_agree = 0
    decision_confusion: dict[str, dict[str, int]] = {}

    must_mention_tp = 0  # Judge says hit, human says hit
    must_mention_fp = 0  # Judge says hit, human says miss
    must_mention_fn = 0  # Judge says miss, human says hit

    must_not_mention_tp = 0
    must_not_mention_fp = 0
    must_not_mention_fn = 0

    must_mention_details = []
    must_not_mention_details = []

    for item in items:
        # Run judge
        result = judge.judge(
            response=item.response,
            ground_truth=item.ground_truth,
            timeline_id=item.timeline_id,
            query_idx=item.query_idx,
            track="calibration",
            domain="calibration",
        )

        # Decision agreement
        judge_correct = result.decision_correct
        human_correct = item.human_labels.decision_correct

        human_key = "correct" if human_correct else "incorrect"
        judge_key = "correct" if judge_correct else "incorrect"

        if human_key not in decision_confusion:
            decision_confusion[human_key] = {}
        current = decision_confusion[human_key].get(judge_key, 0)
        decision_confusion[human_key][judge_key] = current + 1

        if judge_correct == human_correct:
            decision_correct_agree += 1

        # Must mention agreement
        human_hits = set(p.lower() for p in item.human_labels.must_mention_hits)
        judge_hits = set(p.lower() for p in result.must_mention_hits)

        all_phrases = set(_get_phrase_str(p).lower() for p in item.ground_truth.must_mention)
        for phrase in all_phrases:
            in_human = phrase in human_hits
            in_judge = phrase in judge_hits

            if in_human and in_judge:
                must_mention_tp += 1
            elif in_judge and not in_human:
                must_mention_fp += 1
            elif in_human and not in_judge:
                must_mention_fn += 1

        must_mention_details.append({
            "timeline_id": item.timeline_id,
            "human_hits": list(human_hits),
            "judge_hits": list(judge_hits),
            "agreement": human_hits == judge_hits,
        })

        # Must not mention agreement
        human_violations = set(p.lower() for p in item.human_labels.must_not_mention_violations)
        judge_violations = set(p.lower() for p in result.must_not_mention_violations)

        all_forbidden = set(_get_phrase_str(p).lower() for p in item.ground_truth.must_not_mention)
        for phrase in all_forbidden:
            in_human = phrase in human_violations
            in_judge = phrase in judge_violations

            if in_human and in_judge:
                must_not_mention_tp += 1
            elif in_judge and not in_human:
                must_not_mention_fp += 1
            elif in_human and not in_judge:
                must_not_mention_fn += 1

        must_not_mention_details.append({
            "timeline_id": item.timeline_id,
            "human_violations": list(human_violations),
            "judge_violations": list(judge_violations),
            "agreement": human_violations == judge_violations,
        })

    # Calculate metrics
    total = len(items)
    decision_agreement = decision_correct_agree / total if total > 0 else 0.0
    decision_kappa = calculate_cohens_kappa(decision_confusion)

    mm_total = must_mention_tp + must_mention_fp
    mm_precision = must_mention_tp / mm_total if mm_total > 0 else 1.0
    mm_total_recall = must_mention_tp + must_mention_fn
    mm_recall = must_mention_tp / mm_total_recall if mm_total_recall > 0 else 1.0

    mnm_total = must_not_mention_tp + must_not_mention_fp
    mnm_precision = must_not_mention_tp / mnm_total if mnm_total > 0 else 1.0
    mnm_total_recall = must_not_mention_tp + must_not_mention_fn
    mnm_recall = must_not_mention_tp / mnm_total_recall if mnm_total_recall > 0 else 1.0

    return CalibrationResult(
        total_items=total,
        decision_agreement=decision_agreement,
        decision_kappa=decision_kappa,
        must_mention_precision=mm_precision,
        must_mention_recall=mm_recall,
        must_not_mention_precision=mnm_precision,
        must_not_mention_recall=mnm_recall,
        decision_confusion=decision_confusion,
        must_mention_details=must_mention_details,
        must_not_mention_details=must_not_mention_details,
    )


def create_audit_template(
    release_dir: Path,
    output_path: Path,
    sample_size: int = 50,
    seed: int = 42,
) -> int:
    """Create a template audit set for human annotation.

    Samples from the dev split and creates empty human_labels
    for annotators to fill in.

    Args:
        release_dir: Path to release directory
        output_path: Path to write audit template
        sample_size: Number of items to sample
        seed: Random seed for sampling

    Returns:
        Number of items created
    """
    import random

    from statebench.release import load_split
    from statebench.schema.timeline import Query

    # Load dev split
    timelines = list(load_split(release_dir, "dev"))

    # Extract all query events with their context
    query_items: list[dict[str, Any]] = []
    for timeline in timelines:
        for idx, event in enumerate(timeline.events):
            if isinstance(event, Query):
                query_items.append({
                    "timeline": timeline,
                    "query_idx": idx,
                    "query": event,
                })

    # Sample
    rng = random.Random(seed)
    sampled = rng.sample(query_items, min(sample_size, len(query_items)))

    # Write template
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in sampled:
            timeline_obj = item["timeline"]
            query_obj = item["query"]
            audit_item = {
                "timeline_id": timeline_obj.id,
                "query_idx": item["query_idx"],
                "response": "[PLACEHOLDER - Run model to get response]",
                "ground_truth": query_obj.ground_truth.model_dump(),
                "human_labels": {
                    "decision_correct": None,  # Annotator fills this
                    "must_mention_hits": [],   # Annotator fills this
                    "must_not_mention_violations": [],  # Annotator fills this
                    "annotator": "",
                    "timestamp": "",
                },
            }
            f.write(json.dumps(audit_item) + "\n")

    return len(sampled)
