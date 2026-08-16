# StateBench

[![PyPI version](https://img.shields.io/pypi/v/statebench.svg)](https://pypi.org/project/statebench/)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗-Dataset-yellow.svg)](https://huggingface.co/datasets/parslee/statebench)
[![HuggingFace Space](https://img.shields.io/badge/🤗-Space-blue.svg)](https://huggingface.co/spaces/parslee/statebench-explorer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**A conformance test for stateful AI systems.**

StateBench is not another LLM benchmark. It is a test suite that proves whether your AI system actually maintains correct state over time—or just pretends to.

> **Papers:**
>
> *Published:*
> - [Beyond Conversation: A State-Based Context Architecture for Enterprise AI Agents](docs/state-based-context-architecture.pdf) (Liotta, 2025) — the theoretical foundations and benchmark evaluation.
> - [Memgine: A Deterministic Memory Engine for Stateful AI Agents](docs/memgine-deterministic-memory-engine.pdf) (Liotta, 2026) — a production engine implementing the full specification, achieving 95.8% decision accuracy.
>
> *Drafts.* These audit and extend the work above; where they disagree with it,
> they are the later word. Rebuild any of them with `python docs/build_papers.py`.
> - [The Correct Answer Violates](docs/paper-measurement-validity.pdf) — how phrase-list scoring punishes the behavior it exists to reward. Six defects in this benchmark's own instrument; **100% of correct rejecting answers were scored as violations**, and correcting it withdraws a claim from the Memgine paper.
> - [Retrieval, Reconstruction, and Governance](docs/paper3-retrieval-reconstruction-governance.pdf) — whether a reconstruction stage can enforce governance. It cannot, on axes built so that no amount of training could teach it to, because the deciding fact never reaches the model.
> - [Worked Examples Are Worth Paying For](docs/paper4-worked-examples.pdf) — skills distilled from episodes beat what the same model writes from a task description, and the artifact-generation variance that hides the difference if you only generate once.

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

## Leaderboard (v1.1)

> **⚠️ These numbers predate the v2.0 scoring changes and are not reproducible
> with current code.** The evaluation instrument was corrected after these runs:
> forbidden phrases are now matched on word boundaries, non-discriminative
> phrases are excluded from scoring (444 of 4,163 in the v1.0 release, and
> 40.1% of `hallucination_resistance`'s entire forbidden list), a phrase
> appearing only under negation no
> longer counts as a violation, SFRR now counts resurrection only (privacy leaks
> and fabrications are reported separately), and the decision extractor no
> longer reads a bare "no" inside words like "now" and "know". The judge is also
> now pinned globally rather than inherited from the model under test.
>
> The leaderboard has been **re-derived** under corrected scoring on
> `gpt-5.2-2025-12-11` (the published configuration), full dev and test splits,
> 3 runs — see [The Correct Answer Violates](docs/paper-measurement-validity.pdf)
> §6.4 for the new tables.
>
> Must-mention reproduces within 2pp and decision accuracy within 6.4pp in no
> consistent direction, so those figures below stand as approximations.
> **SFRR does not.** Every SFRR figure here is inflated roughly
> 2×, and the ordering between baselines changes. Most consequentially,
> Memgine's reported SFRR parity with `state_based_no_supersession`
> (24.2% vs 24.1%) becomes **14.0% vs 9.5%** — a real gap, not noise — so the
> claim that Memgine improves accuracy "without increasing leakage" does not
> hold under corrected scoring. Memgine's accuracy advantage is unaffected and
> is in fact slightly larger, and its *leakage* rate (1.9%) is the lowest of any
> baseline, which the old blended SFRR concealed.
>
> A `gpt-5.6-sol` refresh under the same corrected scoring is in
> [The Correct Answer Violates](docs/paper-measurement-validity.pdf) §6.5.
> Headline: the newer model resurrects substantially less — SFRR falls 0.8 to
> 7.5pp on nine of ten baselines, `transcript_latest_wins` being the lone
> exception at +1.7pp — and the
> architecture premium narrows sharply — `memgine`'s dev-split lead over
> `state_based` falls from 9.1pp to 0.7pp, and on the test split `state_based`
> pulls level. Stronger models need less context curation.
>
> These tables are retained as the published record.

### Memgine Engine Results

Memgine implements the full state-based specification with query-relevance sorting, engine-level access control, and adaptive inline repair. Results on the v1.0 development split (248 queries, 3-run mean ± std):

| Configuration | Decision Accuracy | SFRR ↓ | Must Mention | MNM Violation ↓ |
|---------------|-------------------|--------|--------------|-----------------|
| `memgine` / Opus 4.6 | **97.3% ± 0.5%** | 37.1% ± 0.6% | **90.7% ± 0.5%** | 20.5% ± 0.7% |
| `memgine` / GPT-5.2 | 95.8% ± 0.4% | 24.2% ± 1.3% | 80.7% ± 0.7% | **13.1% ± 0.7%** |

On the held-out test split (251 queries, 3-run mean ± std):

| Configuration | Decision Accuracy | SFRR ↓ | Must Mention | MNM Violation ↓ |
|---------------|-------------------|--------|--------------|-----------------|
| `memgine` / Opus 4.6 | **96.0% ± 1.2%** | 34.1% ± 0.2% | **89.0% ± 0.3%** | 17.4% ± 0.1% |
| `memgine` / GPT-5.2 | 92.6% ± 1.1% | 23.4% ± 1.1% | 76.4% ± 0.3% | **10.9% ± 0.3%** |

See the [Memgine paper](docs/memgine-deterministic-memory-engine.pdf) for per-track analysis and ablation studies.

### Baseline Comparison (v1.0)

Reference baseline results on the v1.0 test split (209 timelines).

#### GPT-5.2 (OpenAI)

| Baseline | Decision Accuracy | SFRR ↓ | Must Mention |
|----------|-------------------|--------|--------------|
| `state_based` | **80.3%** | 34.4% | **79.8%** |
| `state_based_no_supersession` | 75.4% | 23.0% | 84.0% |
| `rolling_summary` | 72.1% | 21.3% | 66.4% |
| `fact_extraction_with_supersession` | 72.1% | 26.2% | 63.9% |
| `rag_transcript` | 68.9% | 29.5% | 62.2% |
| `fact_extraction` | 63.9% | 27.9% | 56.3% |
| `transcript_replay` | 60.7% | 24.6% | 67.2% |
| `transcript_latest_wins` | 60.7% | **21.3%** | 42.0% |
| `no_memory` | 26.2% | 19.7% | 5.0% |

#### Claude Opus 4.5 (Anthropic)

| Baseline | Decision Accuracy | SFRR ↓ | Must Mention |
|----------|-------------------|--------|--------------|
| `state_based_no_supersession` | **62.9%** | 38.2% | 86.0% |
| `state_based` | 58.2% | 41.0% | **87.4%** |
| `transcript_replay` | 53.0% | 33.5% | 74.8% |
| `rolling_summary` | 51.4% | 45.8% | 73.6% |
| `fact_extraction_with_supersession` | 51.4% | 39.0% | 71.0% |
| `rag_transcript` | 51.0% | 44.2% | 76.9% |
| `fact_extraction` | 49.0% | 37.5% | 68.8% |
| `transcript_latest_wins` | 36.7% | **24.3%** | 48.1% |
| `no_memory` | 13.5% | 7.6% | 7.9% |

**Key findings:** Memgine's deterministic engine achieves 95.8% decision accuracy on GPT-5.2—a 15.5pp improvement over the best reference baseline (80.3%). Engine-level access control eliminates scope leakage that prompt-based approaches cannot prevent. Opus 4.6 outperforms GPT-5.2 under Memgine (97.3% vs 95.8%), reversing the pattern seen in reference baselines where GPT-5.2 leads.

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
statebench evaluate --dataset data/benchmark.jsonl --baseline memgine --model gpt-5.2

# Compare implementations
statebench compare --dataset data/benchmark.jsonl --model gpt-5.2

# Generate official submission
statebench leaderboard --baseline memgine --submitter "YourOrg" --model gpt-5.2
```

## Benchmark Tracks (v1.0)

StateBench v1.0 includes 13 evaluation tracks plus an adversarial track:

| Track | Tests |
|-------|-------|
| `supersession` | Facts invalidated by newer facts stay dead |
| `supersession_detection` | Implicit supersession without explicit markers |
| `commitment_durability` | Confirmed commitments persist across interruptions |
| `interruption_resumption` | Context survives topic switches |
| `scope_permission` | Role-based access control enforcement |
| `environmental_freshness` | Time-sensitive state expiration |
| `authority_hierarchy` | Higher-authority sources override lower |
| `enterprise_privacy` | Confidential information stays restricted |
| `hallucination_resistance` | System refuses to invent unstated facts |
| `scope_leak` | Hypothetical/draft content stays contained |
| `causality` | Causal chain and dependency reasoning |
| `repair_propagation` | Corrections cascade to derived conclusions |
| `brutal_realistic` | Multi-failure compound scenarios |
| `adversarial` | Adversarial prompts designed to trick the system |

### Premise Resistance (v2.1)

Every track above asks a *neutral* question and checks whether dead state comes
back. These two ask a question that has already assumed it:

```
User: "Ship my order to 123 Main St"
User: "I moved — my address is 456 Oak Ave now, not 123 Main St"
Query: "Since we're shipping to 123 Main St, should I select ground shipping?"
FAIL: Response answers the question as asked
PASS: "No — you're not at 123 Main St anymore. Ship to 456 Oak Ave."
```

| Track | Tests |
|-------|-------|
| `premise_resistance` | Queries presupposing superseded state are corrected, not answered |
| `premise_maintain` | Queries presupposing current state are answered, not second-guessed |

The two halves are generated as pairs — identical scenario, identical events,
one variable moved — so the measured quantity is the behavioral delta.
`premise_maintain` scores as FSR and is never blended with the false-premise
numbers: a system that rejects every premise scores 100% on the first half and
is useless.

This track is also the sharpest test of the v2.0 scoring correction. Rejecting
a false premise **requires naming the dead value in order to reject it**, so
under v1.0 phrase-list `must_not_mention` matching every correct answer here
scores as a resurrection. Under negation-aware matching the same answer is
clean, and only an un-negated use counts. `statebench.evaluation.premise_metrics`
reports both scorings side by side.

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

See Section 7.1 of the [architecture paper](docs/state-based-context-architecture.pdf) for detailed analysis and Section 7 of the [Memgine paper](docs/memgine-deterministic-memory-engine.pdf) for how engine-level enforcement addresses this tradeoff.

## Implementations

StateBench includes ten baseline implementations:

| Baseline | Approach |
|----------|----------|
| `memgine` | **Deterministic memory engine** — full state-based spec with query-relevance sorting, engine-level access control, adaptive inline repair, and threshold-based compaction |
| `state_based` | Structured state with supersession tracking, scope management, and repair propagation |
| `state_based_no_supersession` | State-based without supersession (ablation) |
| `fact_extraction_with_supersession` | Fact store with supersession tracking |
| `fact_extraction` | Extracted fact store (Mem0-style) |
| `rolling_summary` | LLM-summarized history |
| `rag_transcript` | Retrieved transcript chunks |
| `transcript_replay` | Raw conversation history |
| `transcript_latest_wins` | Transcript with recency bias |
| `no_memory` | No history. Current query only. |

All baselines operate under identical token budgets (default 8K) for fair comparison.

### Memgine

Memgine is a **deterministic memory engine** that implements the full state-based specification described in the [architecture paper](docs/state-based-context-architecture.pdf). It adds four capabilities over the `state_based` reference baseline:

1. **Query-relevance sorting** — places the most relevant facts closest to the query, exploiting transformer recency attention patterns
2. **Engine-level access control** — removes restricted, hypothetical, and scoped facts from context before they reach the model
3. **Adaptive inline repair** — places invalidated conclusions next to their corrected parent facts with `RECALCULATE` markers
4. **Threshold-based compaction** — layer-specific rules via a Summary DAG for graceful degradation under token pressure

See the [Memgine paper](docs/memgine-deterministic-memory-engine.pdf) for the full specification and evaluation.

### Reference Baselines

The `state_based` baseline is a **reference implementation** that demonstrates core concepts but intentionally omits production optimizations to isolate the effect of supersession tracking. This design isolates the contribution of supersession tracking from retrieval optimizations. See Section 5 of the [architecture paper](docs/state-based-context-architecture.pdf) for methodology.

## Adding Your Implementation

Implement the `MemoryStrategy` interface:

```python
from statebench.baselines.base import MemoryStrategy, ContextResult

class MyStrategy(MemoryStrategy):
    def process_event(self, event) -> None:
        """Handle conversation events, state writes, supersessions."""
        pass

    def build_context(self, query: str) -> ContextResult:
        """Build context with provenance tracking."""
        return ContextResult(
            context="Your assembled context string",
            sources=[]  # Optional: list of Source objects for provenance
        )

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
# Generate official v1.0 release
statebench release --version v1.0 --output data/releases/v1.0

# Verify release integrity
statebench verify data/releases/v1.0
```

Each release includes:
- Train/dev/test/hidden splits (60/15/15/10)
- Canary contamination detection for hidden split
- SHA256 hashes for verification
- Manifest with generation parameters

**Use the test split for official results.** Use dev for development. Hidden split includes canaries to detect training data contamination.

## Leaderboard Submission

```bash
statebench leaderboard \
  --baseline my_strategy \
  --submitter "MyOrg" \
  --model gpt-5.2 \
  --release v1.0 \
  --split test
```

Generates a cryptographically-signed submission file with:
- Multi-seed variance estimation
- Full metric breakdown
- Reproducibility information

## CLI Reference

```bash
statebench generate       # Generate test timelines
statebench evaluate       # Run conformance tests
statebench compare        # Compare implementations
statebench inspect        # Examine dataset
statebench baselines      # List available baselines
statebench release        # Create canonical release
statebench verify         # Verify release integrity
statebench leaderboard    # Generate submission
statebench create-splits  # Create train/dev/test/hidden splits
statebench split-stats    # Show split statistics
statebench budget-sweep   # Test across token budgets
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
│   ├── memgine/         # Deterministic memory engine
│   ├── evaluation/      # Judging and metrics
│   └── cli.py           # Command interface
├── data/releases/       # Canonical benchmark releases
├── docs/
│   ├── state-based-context-architecture.pdf  # Architecture paper
│   ├── memgine-deterministic-memory-engine.pdf  # Memgine paper
│   ├── paper-measurement-validity.*  # Instrument audit (draft)
│   ├── paper3-retrieval-reconstruction-governance.*  # Governance (draft)
│   ├── paper4-worked-examples.*  # Skill transfer (draft)
│   ├── build_papers.py  # Rebuilds the draft PDFs from markdown
│   ├── EVALUATION.md    # Scoring methodology
│   └── ALGORITHM.md     # State-based algorithm spec
├── experiments/         # Result-producing scripts for the papers
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
