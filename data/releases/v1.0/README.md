---
license: mit
task_categories:
  - question-answering
  - text-generation
language:
  - en
tags:
  - benchmark
  - llm-evaluation
  - state-management
  - memory
  - multi-turn
  - conversational-ai
pretty_name: StateBench
size_categories:
  - 1K<n<10K
---

# StateBench: A Benchmark for LLM State Correctness

StateBench measures how well LLMs maintain accurate state over multi-turn conversations. It tests whether models can correctly track facts, respect supersessions (when new information invalidates old), and avoid "resurrecting" outdated information.

## Key Features

- **Multi-turn stateful evaluation**: Each timeline contains a sequence of conversation turns and state changes, followed by queries with ground truth
- **13 evaluation tracks**: Testing different aspects of state management
- **Provenance-aware scoring**: Ground truth includes which facts must/must-not be mentioned
- **Adversarial cases**: Designed to defeat shallow heuristics

## Dataset Structure

Each example is a **timeline** containing:
- `id`: Unique timeline identifier
- `track`: Evaluation track (see below)
- `domain`: Business domain (procurement, sales, project, hr, support)
- `difficulty`: easy, medium, hard, or adversarial
- `events`: JSON-serialized list of conversation turns and state changes
- `initial_state`: JSON-serialized initial state (identity, facts, working set, environment)

## Evaluation Tracks

| Track | Description |
|-------|-------------|
| `supersession` | Facts invalidated by newer information |
| `commitment_durability` | Commitments survive interruptions |
| `interruption_resumption` | Context survives topic switches |
| `scope_permission` | Role-based access control |
| `environmental_freshness` | Time-sensitive state expiration |
| `hallucination_resistance` | Only assert established state |
| `scope_leak` | Task-local state stays local |
| `causality` | Multi-constraint dependencies |
| `repair_propagation` | Fixes propagate to dependent facts |
| `brutal_realistic` | Real-world complexity scenarios |
| `supersession_detection` | Infer supersession from natural language |
| `authority_hierarchy` | Respect authority levels |
| `enterprise_privacy` | Cross-tenant isolation |

## Usage

```python
from datasets import load_dataset

# Load the full dataset
ds = load_dataset("parslee/statebench")

# Access splits
train = ds["train"]      # 839 timelines
val = ds["validation"]   # 209 timelines
test = ds["test"]        # 209 timelines

# Filter by track
supersession = ds["test"].filter(lambda x: x["track"] == "supersession")

# Parse events for evaluation
import json
from statebench.huggingface import hf_row_to_timeline

timeline = hf_row_to_timeline(ds["test"][0])
for event in timeline.events:
    print(event.type, event)
```

## Evaluation

### With Lighteval (HuggingFace)

```bash
pip install lighteval statebench

# Clone the task file
git clone https://github.com/Parslee-ai/statebench.git
cd statebench

# Run evaluation
lighteval accelerate \
    "model_name=meta-llama/Llama-2-7b-hf" \
    "statebench|0|0" \
    --custom-tasks lighteval_tasks/statebench_task.py

# Run on specific track
lighteval accelerate \
    "model_name=meta-llama/Llama-2-7b-hf" \
    "statebench:supersession|0|0" \
    --custom-tasks lighteval_tasks/statebench_task.py
```

### With Native StateBench Harness

For full evaluation with multiple memory baselines:

```bash
pip install statebench

# Evaluate a baseline on the test split
statebench evaluate -d test.jsonl -b state_based -m gpt-4o -p openai
```

See the [StateBench repository](https://github.com/Parslee-ai/statebench) for full documentation.

## Key Metrics

- **Decision Accuracy**: Correct answers to queries
- **SFRR (Superseded Fact Resurrection Rate)**: How often the model mentions facts that have been superseded (lower is better)
- **Must Mention Rate**: Coverage of required facts
- **Must Not Mention Violation Rate**: Mentions of forbidden/superseded facts

## Splits

| Split | Count | Description |
|-------|-------|-------------|
| train | 839 | Training data (60%) |
| validation | 209 | Development/validation (15%) |
| test | 209 | Held-out test (15%) |

## Baseline Scoreboard

Results on the test split with different memory strategies.

> **Dataset revision**: `ffb2d1ab314ba6c2f92195e5e642ddffadee8df4`

### GPT-5.2

| Baseline | Decision Acc | SFRR ↓ | Must Mention | MNM Violations ↓ |
|----------|-------------|--------|--------------|------------------|
| **state_based** | **80.3%** | 34.4% | 79.8% | 18.5% |
| state_based_no_supersession | 75.4% | 23.0% | 84.0% | 14.0% |
| rolling_summary | 72.1% | 21.3% | 66.4% | 10.8% |
| fact_extraction_with_supersession | 72.1% | 26.2% | 63.9% | 12.7% |
| rag_transcript | 68.9% | 29.5% | 62.2% | 15.3% |
| fact_extraction | 63.9% | 27.9% | 56.3% | 13.4% |
| transcript_replay | 60.7% | 24.6% | 67.2% | 12.1% |
| transcript_latest_wins | 60.7% | 21.3% | 42.0% | 9.6% |
| no_memory | 26.2% | 19.7% | 5.0% | 9.6% |

### Claude Opus 4.5

| Baseline | Decision Acc | SFRR ↓ | Must Mention | MNM Violations ↓ |
|----------|-------------|--------|--------------|------------------|
| state_based_no_supersession | **62.9%** | 38.2% | 86.0% | 23.9% |
| **state_based** | 58.2% | 41.0% | 87.4% | 23.6% |
| transcript_replay | 53.0% | 33.5% | 74.8% | 20.6% |
| rolling_summary | 51.4% | 45.8% | 73.6% | 28.7% |
| fact_extraction_with_supersession | 51.4% | 39.0% | 71.0% | 22.9% |
| rag_transcript | 51.0% | 44.2% | 76.9% | 27.7% |
| fact_extraction | 49.0% | 37.5% | 68.8% | 21.1% |
| transcript_latest_wins | 36.7% | 24.3% | 48.1% | 16.5% |
| no_memory | 13.5% | 7.6% | 7.9% | 3.8% |

**Key findings:**
- `state_based` achieves highest decision accuracy on GPT-5.2 (80.3%)
- Higher must-mention rates correlate with higher SFRR (accuracy-safety tradeoff)
- Claude Opus 4.5 shows different baseline rankings than GPT-5.2

## Citation

```bibtex
@software{statebench2024,
  title = {StateBench: A Benchmark for LLM State Correctness},
  author = {Liotta, Matt},
  year = {2024},
  url = {https://github.com/parslee-ai/statebench},
  version = {1.0}
}
```

## License

MIT License
