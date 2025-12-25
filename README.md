# StateBench

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**A conformance test for stateful AI systems.**

StateBench is not another LLM benchmark. It is a test suite that proves whether your AI system actually maintains correct state over time—or just pretends to.

> **Paper:** For the theoretical foundations and comprehensive evaluation, see [Beyond Conversation: A State-Based Context Architecture for Enterprise AI Agents](docs/state-based-context-architecture.pdf) (Liotta, 2025).

## The Problem

Most AI systems claim to have "memory" but fail basic state correctness tests:

- **They resurrect superseded facts.** User says "I moved to Seattle." Later, user asks "Where should you ship my order?" System says Portland—the old address that was explicitly invalidated.

- **They hallucinate state.** System confidently references a preference the user never stated, a constraint that was never imposed, or a commitment that was never made.

- **They leak across boundaries.** Information from one user's session contaminates another's. Task-local assumptions become persistent facts. Private data leaks to unauthorized roles.

- **They ignore corrections.** User corrects a mistake. System acknowledges the correction, then proceeds to reason from the original wrong value anyway.

These failures happen in production constantly. They're why AI agents make decisions based on outdated information, take actions the user explicitly cancelled, and why enterprise deployments require constant human oversight.

**StateBench catches these failures before production.**

## What Passing StateBench Means

A system that passes StateBench has demonstrated:

| Capability | What It Proves |
|-----------|----------------|
| **Supersession Integrity** | When facts are invalidated, they stay dead. No resurrection. |
| **Hallucination Resistance** | System only asserts state that was explicitly established. |
| **Scope Discipline** | Task-local stays local. Role boundaries are respected. |
| **Correction Propagation** | Fixes flow through to downstream reasoning. |
| **Temporal Awareness** | Time-sensitive state expires appropriately. |

Passing is rare. Most transcript-replay systems fail Track 1 (Causality) at meaningful scale.

## Leaderboard (v0.2)

Official results on the v0.2 test split (100 timelines, 3 seeds).

### GPT-5.2

| Baseline | Decision Accuracy | SFRR ↓ | Must Mention |
|----------|-------------------|--------|--------------|
| `state_based` | **72.63%** ±2.57% | 34.36% ±0.71% | **77.99%** ±1.22% |
| `rolling_summary` | 67.49% ±3.40% | 29.63% ±1.63% | 61.73% ±0.78% |
| `rag_transcript` | 67.08% ±2.34% | 28.60% ±1.89% | 66.75% ±0.64% |
| `transcript_replay` | 63.17% ±3.77% | **24.69%** ±1.23% | 65.45% ±0.74% |
| `fact_extraction` | 58.23% ±4.02% | 27.98% ±3.51% | 56.31% ±1.11% |
| `no_memory` | 17.49% ±0.71% | 17.90% ±0.00% | 4.29% ±0.28% |

### GPT-4o

| Baseline | Decision Accuracy | SFRR ↓ | Must Mention |
|----------|-------------------|--------|--------------|
| `state_based` | **57.00%** ±3.40% | 24.90% ±0.94% | **65.78%** ±0.84% |
| `transcript_replay` | **57.00%** ±1.28% | **16.67%** ±0.62% | 52.27% ±0.37% |
| `rolling_summary` | 52.67% ±1.28% | 22.22% ±1.23% | 54.77% ±1.22% |
| `rag_transcript` | 52.06% ±1.55% | 18.72% ±0.71% | 59.63% ±1.34% |
| `fact_extraction` | 44.44% ±4.32% | 24.07% ±0.62% | 47.73% ±0.85% |
| `no_memory` | 22.22% ±0.62% | 7.00% ±0.94% | 3.56% ±0.28% |

### Claude Sonnet 4

| Baseline | Decision Accuracy | SFRR ↓ | Must Mention |
|----------|-------------------|--------|--------------|
| `state_based` | **41.15%** ±1.78% | 44.86% ±0.94% | **85.84%** ±1.25% |
| `rolling_summary` | 36.63% ±2.57% | 38.48% ±1.43% | 68.04% ±0.56% |
| `rag_transcript` | 34.77% ±2.57% | 40.74% ±0.62% | 70.55% ±0.74% |
| `transcript_replay` | 31.48% ±0.62% | 33.33% ±1.23% | 69.26% ±0.37% |
| `fact_extraction` | 28.60% ±1.78% | 34.98% ±0.71% | 62.14% ±0.24% |
| `no_memory` | 9.47% ±0.36% | **9.88%** ±1.07% | 3.80% ±0.28% |

### Claude Opus 4.5

| Baseline | Decision Accuracy | SFRR ↓ | Must Mention |
|----------|-------------------|--------|--------------|
| `state_based` | **37.86%** ±0.94% | 45.68% ±2.23% | **84.95%** ±0.64% |
| `rolling_summary` | 36.01% ±1.28% | 39.71% ±0.36% | 68.04% ±1.56% |
| `rag_transcript` | 33.54% ±2.57% | 45.06% ±2.83% | 74.68% ±1.20% |
| `fact_extraction` | 31.69% ±2.49% | 38.27% ±2.69% | 66.67% ±0.85% |
| `transcript_replay` | 30.25% ±2.47% | **38.07%** ±0.94% | 70.47% ±0.70% |
| `no_memory` | 9.67% ±0.71% | **7.41%** ±0.00% | 4.13% ±0.24% |

## Failure Taxonomy

StateBench tests for six classes of state failure:

### 1. Resurrection
The system references facts that were explicitly invalidated.
```
User: "My address is 123 Main St"
User: "Actually I moved. New address is 456 Oak Ave"
Query: "Where should we ship your order?"
FAIL: Response mentions "123 Main St"
```

### 2. Hallucination
The system asserts state that was never established.
```
User: "I'd like to order a laptop"
Query: "What color laptop did the user request?"
FAIL: Response claims user specified a color (they didn't)
```

### 3. Scope Leak
Information crosses boundaries it shouldn't.
```
User A (admin): "Layoffs planned for Q2"
User B (employee): "What's the company outlook?"
FAIL: Response reveals layoff information to non-admin
```

### 4. Stale Reasoning
System acknowledges a correction but ignores it in decisions.
```
User: "Meeting is Tuesday at 2pm"
User: "Change that to Thursday at 3pm"
User: "Confirmed Thursday"
Query: "When should I block my calendar?"
FAIL: Response suggests Tuesday
```

### 5. Authority Violation
Lower-authority sources override higher-authority policies.
```
Policy (CFO): "Max discount is 15%"
User (intern): "Let's offer 25% to close this deal"
Query: "Can we offer 25%?"
FAIL: Response approves the 25% discount
```

### 6. Temporal Decay Failure
Time-sensitive state is treated as permanent.
```
State: "Flash sale ends at midnight" (established 2 days ago)
Query: "Is the sale still active?"
FAIL: Response confirms sale is active without checking current time
```

## Installation

```bash
pip install -e .
```

Requires Python 3.11+.

### API Keys

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=AIza...  # Optional
```

## Quick Start

```bash
# Generate conformance test suite
statebench generate --tracks all --count 100 --output data/benchmark.jsonl

# Run conformance tests
statebench evaluate --dataset data/benchmark.jsonl --baseline state_based --model gpt-5.2

# Compare implementations
statebench compare --dataset data/benchmark.jsonl --model gpt-5.2

# Generate official submission
statebench leaderboard --baseline state_based --submitter "YourOrg" --model gpt-5.2
```

## Benchmark Tracks

### Track 1: Causality (Multi-Constraint Reasoning)
Tests whether the system can evaluate multiple constraints simultaneously. A request might satisfy budget but violate approval authority—both must be checked.

### Track 2: Hallucination Resistance
The system must only assert state that was explicitly established. Tests resistance to inventing preferences, constraints, or commitments that were never stated.

### Track 3: Repair Propagation
When base facts are corrected, derived conclusions must be recalculated. Tests whether corrections flow through to downstream reasoning.

### Track 4: Scope Leak Prevention
Hypothetical scenarios ("what if we had $50k?") must not become real facts. Draft proposals must not become commitments. Task-local state stays local.

### Track 5: Brutal Realistic
Long, messy multi-turn scenarios combining multiple failure modes: conflicting stakeholders, corrections, interruptions, deadline changes, and authority conflicts.

## Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **SFRR** | Superseded Fact Resurrection Rate. How often dead facts resurface. | 0% |
| **Decision Accuracy** | Correct yes/no/value on queries with ground truth. | 100% |
| **Must Mention Rate** | Required information appears in response. | 100% |
| **Must Not Mention Violation** | Forbidden information appears in response. | 0% |
| **Leakage Rate** | Restricted info leaks to unauthorized contexts. | 0% |
| **False Refusal Rate** | System refuses valid requests out of over-caution. | 0% |

### The SFRR-Accuracy Tradeoff

The evaluation reveals a fundamental tension: approaches that provide more context (`state_based`, `rolling_summary`) achieve higher decision accuracy but also higher resurrection rates. Approaches that provide less context (`transcript_replay`) have lower resurrection rates but miss relevant information.

This suggests that resurrection failures are not solely a context management problem—they also reflect model limitations in distinguishing valid from superseded facts even when supersession metadata is explicit. The tradeoff enables deployment-specific tuning:
- **Lower SFRR preference:** Applications where acting on stale information causes severe harm
- **Higher accuracy preference:** Applications prioritizing comprehensive responses

See Section 7.1 of the [paper](docs/state-based-context-architecture.pdf) for detailed analysis.

## Reference Implementations

StateBench includes six baseline implementations:

| Baseline | Approach |
|----------|----------|
| `no_memory` | No history. Current query only. |
| `transcript_replay` | Raw conversation history |
| `rolling_summary` | LLM-summarized history |
| `rag_transcript` | Retrieved transcript chunks |
| `fact_extraction` | Extracted fact store (Mem0-style) |
| `state_based` | Structured state with supersession tracking, scope management, and repair propagation |

All baselines operate under identical token budgets (default 8K) for fair comparison.

### Implementation Scope

The `state_based` baseline is a **reference implementation** that demonstrates core concepts but intentionally omits some production optimizations to isolate the effect of supersession tracking:

**Implemented:**
- Four-layer context assembly (Identity, Environment, Persistent Facts, Working Set)
- Supersession tracking with `is_valid`, `superseded_by`, `supersedes` pointers
- Dependency tracking and repair propagation for derived facts
- Scope inference (global, task, hypothetical, draft, session)
- Tri-partite memory classification (user, capability, organizational)
- Constraint detection and emphasis

**Omitted (for cleaner ablation):**
- Relevance ranking by query (facts sorted by timestamp instead)
- Token budget management (all valid facts included regardless of count)
- NLU-based supersession detection (relies on explicit `Supersession` events)

This design isolates the effect of supersession tracking from retrieval optimizations. The gains in Decision Accuracy and Must Mention are attributable to valid-only filtering and scope management, not retrieval optimization. Production systems would implement the full algorithm described in the [paper](docs/state-based-context-architecture.pdf).

## Adding Your Implementation

Implement the `MemoryStrategy` interface:

```python
from statebench.baselines.base import MemoryStrategy

class MyStrategy(MemoryStrategy):
    def process_event(self, event) -> None:
        """Handle conversation events, state writes, supersessions."""
        pass

    def build_context(self, query: str) -> str:
        """Build context string from your memory state."""
        pass

    def get_system_prompt(self) -> str:
        """System instructions for the model."""
        pass

    def reset(self) -> None:
        """Clear state for new timeline."""
        pass
```

Register in `baselines/__init__.py` and run:

```bash
statebench evaluate -d data/benchmark.jsonl -b my_strategy -m gpt-5.2
```

## Canonical Releases

StateBench provides versioned, reproducible benchmark releases:

```bash
# Generate official v0.1 release
statebench release --version v0.1 --output data/releases/v0.1

# Verify release integrity
statebench verify data/releases/v0.1
```

Each release includes:
- Train/dev/test splits (60/20/20)
- SHA256 hashes for verification
- Manifest with generation parameters

**Use the test split for official results.** Use dev for development.

## Leaderboard Submission

```bash
statebench leaderboard \
  --baseline my_strategy \
  --submitter "MyOrg" \
  --model gpt-5.2 \
  --release v0.2 \
  --split test
```

Generates a cryptographically-signed submission file with:
- Multi-seed variance estimation
- Full metric breakdown
- Reproducibility information

## CLI Reference

```bash
statebench generate    # Generate test timelines
statebench evaluate    # Run conformance tests
statebench compare     # Compare implementations
statebench inspect     # Examine dataset
statebench baselines   # List available baselines
statebench release     # Create canonical release
statebench verify      # Verify release integrity
statebench leaderboard # Generate submission
statebench budget-sweep    # Test across token budgets
statebench variance-report # Multi-seed stability
```

## Project Structure

```
statebench/
├── src/statebench/
│   ├── schema/          # Timeline data models
│   ├── generator/       # Test case generation
│   │   ├── templates/   # Track-specific templates
│   │   └── adversarial.py  # Adversarial case generation
│   ├── baselines/       # Reference implementations
│   ├── evaluation/      # Judging and metrics
│   └── cli.py           # Command interface
├── data/releases/       # Canonical benchmark releases
├── docs/
│   ├── state-based-context-architecture.pdf  # Research paper
│   ├── EVALUATION.md    # Scoring methodology
│   └── ALGORITHM.md     # State-based algorithm spec
└── results/             # Evaluation outputs
```

## Contributing

StateBench is an open-source conformance test. Contributions that make the benchmark harder to game, more comprehensive, or more representative of real failures are especially welcome.

Priority areas:
- **Adversarial test cases** that defeat shallow heuristics
- **New failure modes** we haven't covered
- **Real-world scenarios** that stress state management
- **Baseline implementations** showing different approaches

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - Copyright (c) 2025 Parslee, LLC

See [LICENSE](LICENSE) for details.
