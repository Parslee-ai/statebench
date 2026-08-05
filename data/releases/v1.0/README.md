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

## ⚠️ Scoring was corrected in statebench 2.0.0

**The scoreboard below has been re-derived.** Results published before August 2026
used an evaluation instrument with six defects, most consequentially that
phrase-list scoring could not distinguish *naming a value in order to reject it*
from *asserting it*. A response like "the meeting is **not** Friday — that was
superseded" was scored as a resurrection.

Across 400 queries from this release, **100% of correct answers phrased as
explicit rejections were flagged as violations**. Under corrected scoring, 0%.

Practical consequence: **SFRR figures published before 2.0.0 are roughly 2× the
true resurrection rate, and the ordering between baselines does not survive
correction.** They cannot be rescaled — the artifact share differs per system.
Decision accuracy and must-mention are largely unaffected.

Use `statebench >= 2.0.0` to reproduce the numbers below. See
[CHANGELOG](https://github.com/Parslee-ai/statebench/blob/main/CHANGELOG.md).

## Baseline Scoreboard (corrected scoring)

`gpt-5.2-2025-12-11`, judge `gpt-4o-mini` pinned globally, 3 runs, mean ± std.

> **Dataset revision**: `ffb2d1ab314ba6c2f92195e5e642ddffadee8df4`
> **Instrument**: statebench 2.0.0

### Validation split (dev)

| Baseline | Decision Acc | SFRR ↓ | Leakage ↓ | Must Mention |
|----------|-------------|--------|-----------|--------------|
| **memgine** | **96.8% ± 0.0%** | 14.0% ± 0.2% | **1.9%** | 81.0% ± 0.1% |
| state_based_no_supersession | 91.5% ± 0.6% | **9.5% ± 1.2%** | 5.4% | 80.8% ± 0.5% |
| state_based | 87.6% ± 0.5% | 12.9% ± 0.9% | 7.4% | 79.3% ± 0.2% |
| fact_extraction_with_supersession | 84.0% ± 1.1% | 13.4% ± 0.8% | 4.6% | 65.2% ± 1.5% |
| rolling_summary | 83.9% ± 0.7% | 6.9% ± 0.6% | 5.1% | 67.1% ± 1.2% |
| rag_transcript | 83.5% ± 1.0% | 10.8% ± 0.5% | 4.8% | 71.0% ± 0.3% |
| transcript_replay | 83.2% ± 0.2% | 7.9% ± 0.8% | 4.6% | 70.6% ± 0.5% |
| fact_extraction | 77.6% ± 2.0% | 14.1% ± 1.1% | 4.4% | 64.3% ± 0.7% |
| transcript_latest_wins | 69.5% ± 1.3% | 7.4% ± 0.5% | 2.3% | 42.7% ± 1.1% |
| no_memory | 26.3% ± 1.1% | 4.7% ± 0.5% | 2.3% | 9.3% ± 0.4% |

### Test split (held out)

| Baseline | Decision Acc | SFRR ↓ | Leakage ↓ | Must Mention |
|----------|-------------|--------|-----------|--------------|
| **memgine** | **94.2% ± 1.0%** | 12.7% ± 1.0% | 4.0% | 77.6% ± 1.9% |
| state_based_no_supersession | 90.3% ± 0.7% | 9.2% ± 0.7% | 8.2% | **81.1% ± 0.4%** |
| state_based | 86.9% ± 0.9% | 9.6% ± 0.6% | 8.9% | 76.9% ± 0.8% |
| transcript_replay | 85.1% ± 0.7% | **6.8% ± 0.9%** | 7.4% | 67.2% ± 0.3% |
| rolling_summary | 84.7% ± 0.5% | 6.8% ± 0.6% | 8.5% | 65.4% ± 1.2% |
| rag_transcript | 83.1% ± 0.7% | 12.1% ± 0.5% | 8.0% | 67.3% ± 0.3% |
| fact_extraction | 78.0% ± 0.8% | 11.2% ± 1.0% | 6.9% | 59.8% ± 0.7% |
| transcript_latest_wins | 68.5% ± 0.3% | 8.2% ± 0.5% | 5.4% | 40.5% ± 0.1% |
| no_memory | 25.5% ± 0.6% | 2.4% ± 0.0% | 3.3% | 7.4% ± 0.1% |

**Key findings under corrected scoring:**

- `memgine` leads decision accuracy on both splits by a wide margin, and has the
  **lowest leakage of any baseline** (1.9% dev) — a result the previously blended
  SFRR metric could not express.
- The **accuracy–safety tradeoff claimed in earlier versions of this card does not
  hold as stated.** It rested on SFRR figures that were substantially artifact;
  the systems said to trade accuracy for resurrection are not the ones that do.
- SFRR now counts resurrection only. Restricted-data leaks and fabrications are
  reported separately as `leakage_rate` and `fabrication_rate`.
- A refresh on a current-generation model (`gpt-5.6-sol`) shows the architectural
  accuracy premium narrowing sharply, from 9.1pp to 0.7pp over `state_based`.

Full analysis, including what the correction invalidated in the associated
papers, is in the repository's `docs/`.

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
