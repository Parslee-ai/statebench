# StateBench Lighteval Task

This directory contains a custom [Lighteval](https://github.com/huggingface/lighteval) task for evaluating LLMs on [StateBench](https://huggingface.co/datasets/parslee/statebench).

## Quick Start

```bash
# Install dependencies
pip install lighteval statebench

# Run evaluation on the full benchmark
lighteval accelerate \
    "model_name=meta-llama/Llama-2-7b-hf" \
    "statebench|0|0" \
    --custom-tasks lighteval_tasks/statebench_task.py

# Run on a specific track
lighteval accelerate \
    "model_name=meta-llama/Llama-2-7b-hf" \
    "statebench:supersession|0|0" \
    --custom-tasks lighteval_tasks/statebench_task.py
```

## How It Works

StateBench is a **multi-turn stateful benchmark**. Unlike single-turn Q&A tasks, each timeline contains:
- Conversation turns that establish facts
- State changes (writes, supersessions)
- Query events with ground truth

### Multi-Turn Handling

The Lighteval adapter handles this by:

1. **Pre-computing context**: For each timeline, we process events sequentially using the `transcript_replay` baseline to build conversation context at each query point.

2. **Flattening to queries**: Each query becomes a separate evaluation doc with its pre-computed context included in the prompt.

3. **Rich ground truth**: The ground truth includes not just the expected decision, but also `must_mention` and `must_not_mention` constraints for comprehensive scoring.

## Available Tasks

| Task | Description |
|------|-------------|
| `statebench` | Full benchmark (all 13 tracks) |
| `statebench:supersession` | Facts invalidated by newer info |
| `statebench:commitment_durability` | Commitments survive interruptions |
| `statebench:interruption_resumption` | Context survives topic switches |
| `statebench:scope_permission` | Role-based access control |
| `statebench:environmental_freshness` | Time-sensitive expiration |
| `statebench:hallucination_resistance` | Only assert established state |
| `statebench:scope_leak` | Task-local stays local |
| `statebench:causality` | Multi-constraint dependencies |
| `statebench:repair_propagation` | Fixes propagate |
| `statebench:brutal_realistic` | Real-world complexity |
| `statebench:supersession_detection` | Infer supersession from NL |
| `statebench:authority_hierarchy` | Respect authority levels |
| `statebench:enterprise_privacy` | Cross-tenant isolation |

## Metrics

The task reports three metrics:

| Metric | Description | Higher is Better |
|--------|-------------|------------------|
| `decision_accuracy` | Correct answers to queries | Yes |
| `must_mention_rate` | Coverage of required facts | Yes |
| `must_not_mention_violations` | Mentions of forbidden facts | No |

## Dataset Revision

This task evaluates against `parslee/statebench` on HuggingFace Hub:
- **Version**: 1.0
- **Test split**: 209 timelines
- **Tracks**: 13

## Limitations

The Lighteval adapter uses a simplified evaluation mode:

1. **Fixed baseline**: Uses `transcript_replay` for context building (not configurable)
2. **Deterministic scoring**: No LLM judge for paraphrase detection
3. **Single query per doc**: Each doc contains one query (multi-query timelines are flattened)

For full evaluation with multiple baselines and LLM-assisted scoring, use the native StateBench harness:

```bash
pip install statebench
statebench evaluate -d data/releases/v1.0/test.jsonl -b state_based -m gpt-4o
```

## Files

- `statebench_task.py` - Main task implementation
- `requirements.txt` - Dependencies
- `README.md` - This file
