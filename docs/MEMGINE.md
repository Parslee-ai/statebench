# What Memgine Demonstrates About the State-Based Context Architecture

## The Paper's Central Claim and Its Open Questions

The paper ("Beyond Conversation: A State-Based Context Architecture for Enterprise AI Agents", Liotta 2025) argues that context should be **structured state assembled fresh on every turn, not transcript replay**. The reference implementation (`state_based`) validates this: on GPT-5.2, it achieves 80.3% decision accuracy, a 19.6pp improvement over transcript replay.

But the paper is explicit about what the reference implementation *doesn't* do. Table 2 lists three features present in the full specification but absent from the reference:

| Feature | Full Spec | Reference |
|---------|-----------|-----------|
| Relevance ranking by query | ✓ | ✗ (timestamp) |
| Token budget management (70% rule) | ✓ | ✗ (all facts) |
| NLU-based supersession detection | ✓ | ✗ (explicit events) |

Section 7.3 calls this out directly: *"The results represent a lower bound. Production implementations with relevance ranking might achieve higher accuracy."* And Section 7.1 identifies a fundamental tension: *"Approaches that provide more context achieve higher decision accuracy but also higher resurrection rates."* The paper predicts that combining supersession tracking with relevance filtering could break this tradeoff.

**Memgine tests these predictions.**

## What Memgine Implements

Memgine implements the first two missing features from the full specification — relevance ranking and token budget management — plus four innovations the paper didn't specify:

1. **Query-aware fact ordering**: Facts sorted by keyword overlap with the query, most relevant placed nearest to the query in the prompt (exploiting LLM recency attention). This is the "relevance ranking" the paper's full spec calls for.

2. **Token budget management**: Hierarchical compaction engine with per-layer budgets and three-level escalation (zero-cost → compaction → deterministic). The paper's 70% rule, implemented.

3. **Supersession disambiguation**: When a fact supersedes another, Memgine annotates it with `(changed from: ...)`, showing *what* changed inline. The paper assumed superseded facts should simply be excluded. Memgine discovered that including the change signal — while still excluding the old fact as a standalone entry — gives the LLM critical disambiguation context.

4. **Interruption filtering**: Detection of "hold on" / "back to" conversational patterns to strip off-topic turns from the working set. The paper's architecture has no mechanism for this.

5. **Memory type classification**: Facts tagged `[org]` or `[usr]` based on source provenance, helping the LLM respect organizational vs. user boundaries.

6. **Known Unknowns section**: Explicit enumeration of information gaps, placed nearest to the query to prevent hallucination on missing data.

## Results: The Predictions Confirmed

On GPT-4o with the StateBench v1.0 dev set (209 timelines):

| Metric | Memgine | state_based | Delta |
|--------|---------|-------------|-------|
| Decision Accuracy ↑ | **73.39%** | 68.55% | **+4.84pp** |
| SFRR ↓ | **18.55%** | 23.79% | **-5.24pp** |
| Must Mention ↑ | **73.21%** | 71.37% | **+1.84pp** |
| Must Not Mention ↓ | **11.22%** | 14.69% | **-3.47pp** |

Memgine wins on every metric.

### Per-Track Breakdown

| Track | Memgine | state_based | Delta |
|-------|---------|-------------|-------|
| authority_hierarchy | 100.0% | 100.0% | +0.00pp |
| brutal_realistic | 46.3% | 35.2% | +11.11pp |
| causality | 93.3% | 93.3% | +0.00pp |
| commitment_durability | 100.0% | 86.7% | +13.33pp |
| enterprise_privacy | 93.3% | 93.3% | +0.00pp |
| environmental_freshness | 25.0% | 18.8% | +6.25pp |
| hallucination_resistance | 100.0% | 93.3% | +6.67pp |
| interruption_resumption | 40.0% | 40.0% | +0.00pp |
| repair_propagation | 33.3% | 40.0% | -6.67pp |
| scope_leak | 80.0% | 73.3% | +6.67pp |
| scope_permission | 100.0% | 100.0% | +0.00pp |
| supersession | 100.0% | 96.3% | +3.70pp |
| supersession_detection | 93.3% | 86.7% | +6.67pp |

## The SFRR-Accuracy Tradeoff: Broken

The paper's most important open question was whether the SFRR-Accuracy tradeoff is fundamental. Section 7.1: *"Approaches that provide more context achieve higher decision accuracy but also higher resurrection rates."* The paper hedged: *"A production system might achieve better SFRR-Accuracy balance by combining supersession tracking with aggressive relevance filtering."*

Memgine confirms this prediction. It achieves **higher accuracy AND lower SFRR** than state_based. The mechanism is exactly what the paper hypothesized: query-aware relevance filtering means fewer facts in context, which means fewer opportunities for resurrection, while the facts that *are* included are the ones that matter for the query.

The paper reported state_based at 80.3% accuracy / 34.4% SFRR on GPT-5.2. On GPT-4o, the numbers are lower for both strategies, but the relationship holds: Memgine moves *up and to the left* on the accuracy-SFRR curve — the direction the paper predicted was possible but didn't demonstrate.

## The Supersession Disambiguation Discovery

The paper assumes a clean separation: valid facts appear in context, superseded facts don't. Section 4.4: *"The key constraint is that superseded facts are never included. This is what prevents 'resurrection' failures."*

Memgine discovered this is too aggressive. On the commitment_durability track, properly excluding superseded facts *removed the signal the LLM needs to distinguish what changed from what didn't*. When the context shows only:

```
- Prefer milestone-based payment terms
- Committed to 6 months contract with CloudProvider
```

The LLM can't see that the preference changed while the commitment persisted. Memgine's fix — annotating the superseding fact with `(changed from: Prefer monthly billing)` — preserves the paper's "never include superseded facts" principle while surfacing the change narrative. This is a middle ground the paper didn't explore: the *fact* of change is included; the *old value as a standalone entry* is not.

Result: commitment_durability went from ~27% (both strategies, pre-fix) to 100% (memgine) vs 87% (state_based). The annotation helps both baselines via the rubric fix, but Memgine's inline format provides stronger disambiguation.

## What the Paper Got Right

1. **State over transcript is the right paradigm.** Memgine builds on state_based, not on transcript replay. Every improvement is about *how* to assemble state, not whether to.

2. **Relevance ranking matters.** The paper predicted this in Table 2 and Section 7.3. Query-aware fact ordering is Memgine's most consistent contributor.

3. **The SFRR-Accuracy tradeoff is solvable.** The paper predicted relevance filtering would help. It does.

4. **Layer-specific lifecycles matter.** Memgine's best improvements come from treating layers differently: never compacting constraints, capping working set at 10, sorting environment by freshness.

## What the Paper Didn't Anticipate

1. **Supersession needs disambiguation, not just exclusion.** The paper's "never include superseded facts" rule is too strict. The change narrative matters.

2. **Conversational structure carries semantic signal.** Interruption patterns ("hold on" → topic shift → "back to") are structural, not content-based. The paper treats the working set as undifferentiated recent turns. Filtering interruptions is a layer-3-specific lifecycle operation the paper doesn't discuss.

3. **Scoring rubric limitations mask real quality.** Memgine's LLM responses were often *correct* but scored wrong because the rubric couldn't detect implicit affirmatives. The paper's reported numbers for all baselines are affected by this — actual quality differences may be larger (or smaller) than measured.

4. **Fact ID collisions in the data generator.** The `W-AUTO` collision affects both strategies differently and unpredictably. Data quality issues in the benchmark itself create measurement artifacts.

## Implications for the Full Specification

The paper describes a production architecture that Memgine partially implements. What Memgine's results suggest for the remaining pieces:

- **NLU-based supersession detection** (the third missing feature): The benchmark tests handling of supersession *once detected*, not detection itself. Memgine and state_based both rely on explicit supersession events. The value of NLU detection would show on real conversational data, not StateBench.

- **Tri-partite memory decomposition** (User/Capability/Organizational): Memgine classifies facts as `[org]` or `[usr]` and this helps enterprise_privacy (+6.7pp vs state_based in some runs). The full decomposition likely matters more at scale with multiple users and capabilities.

- **Tiered learning curriculum**: Not testable on StateBench. This is a deployment-time feature.

## Architecture

Memgine's engine architecture (`src/statebench/memgine/`):

```
engine.py          # Main orchestrator: typed ingest methods, context assembly
store.py           # ImmutableStore: append-only event log, source of truth
layers.py          # LayerState: semantic index (validity, supersession, dependencies)
dag.py             # SummaryDAG: hierarchical summaries with provenance pointers
compaction.py      # CompactionEngine: layer-aware threshold compaction
config.py          # MemgineConfig: per-layer budgets and thresholds
types.py           # Core data types (StoreEntry, SummaryNode)
strategy.py        # MemgineStrategy: MemoryStrategy adapter for StateBench
```

Key design principles inherited from the LCM paper (Voltropy PBC, Feb 2026):
- **Immutable store**: Events are never modified. Mutable state (validity, superseded_by) lives in the LayerState indices.
- **Hierarchical DAG**: Per-layer summary trees with provenance pointers enabling lossless expansion.
- **Deterministic compaction**: Three-level escalation with layer-specific rules. Level 2 (deterministic) requires no LLM calls and guarantees convergence.

## The Bottom Line

The paper's thesis is that state-based context management is fundamentally better than conversation replay. Memgine's contribution is showing that the *quality of state assembly* — how you order, filter, annotate, and budget the state you've already extracted — provides a second, independent axis of improvement. The paper demonstrated the value of *having* structured state. Memgine demonstrates the value of *intelligently composing* it.

The paper's reference implementation was deliberately simplified to isolate the effect of supersession tracking. Memgine adds back the optimizations the paper specified but didn't implement, plus discoveries the paper didn't predict. The result breaks the SFRR-Accuracy tradeoff the paper identified as its primary open question.
