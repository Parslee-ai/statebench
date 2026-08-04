# Changelog

## 2.0.0 — 2026-08-04

### ⚠️ Breaking: scoring semantics changed. Your numbers will move.

**Do not compare 1.x results to 2.x results.** The evaluation instrument had six
defects, and correcting them changes what the metrics measure. On the v1.0
release scored with `gpt-5.2`, SFRR falls on **all ten reference baselines** by
8.8–17.6 percentage points, and the ordering between baselines does not survive.

If you have published numbers produced with 1.x, they were computed with the
defects below and need re-deriving rather than rescaling — the artifact share
differs per system, so no constant correction exists.

### The defect that matters most

Phrase-list scoring could not distinguish *naming a value in order to reject it*
from *asserting it*. A response like:

> "The meeting is **not** Friday — that was superseded. It moved to Thursday."

demonstrates exactly the state discrimination the benchmark exists to measure,
and was scored as a resurrection because it contains "Friday". Across 400 real
queries, **100% of correct answers phrased as explicit rejections were flagged as
violations**, on every track. Under 2.0 scoring, 0%.

This was not a verbosity effect — generic filler of 319 words trips the old
scorer 0.3% of the time. The bias tracked *engagement with the scenario*, so the
metric rewarded evasion.

### All six fixes

1. **Judge is now pinned globally.** It was inherited from the model under test,
   so OpenAI arms were graded by `gpt-4o-mini` and Anthropic arms by
   `claude-3-haiku` — every cross-model comparison mixed two graders. Set with
   `judge_provider` / `judge_model`, or `$STATEBENCH_JUDGE`. The judge identity
   is recorded on every result via `judge.descriptor`.
2. **Phrase matching is boundary-aware.** `"by"` no longer matches "nearby",
   `"HP"` no longer matches "PHP", `"100"` no longer matches "1000". Per-edge
   handling means `"$45"` still does not fire inside `"$450"`.
3. **Non-discriminative forbidden phrases are excluded from scoring and from the
   denominator.** 979 of 8,299 in the v1.0 release could not distinguish correct
   from incorrect behavior. On `hallucination_resistance` the forbidden list
   contained the vocabulary the question asks about, so the correct answer
   violated by construction. Skipped phrases are recorded in
   `QueryResult.skipped_phrases` for audit.
4. **Negation is recognised.** A forbidden phrase appearing only under a negation
   cue is recorded in `QueryResult.negated_mentions`, not counted as a violation.
   Conservative: one un-negated occurrence anywhere still counts.
5. **SFRR counts resurrection only.** It previously fired on any
   must-not-mention violation, so privacy leaks and fabrications counted as
   resurrections. `leakage_rate` and `fabrication_rate` now report those
   separately, and `violations_by_kind` carries the breakdown.
6. **`extract_decision` matches on word boundaries.** A bare `"no"` inside
   "now"/"know"/"noted" no longer flips an affirmative answer — and because
   extraction previously *succeeded*, the LLM fallback never ran to correct it.

### Migration

Most users need no code changes. If you construct a judge or harness directly:

```python
# 1.x — judge silently inherited from the model under test
harness = EvaluationHarness(model="gpt-5.2", provider="openai")

# 2.x — judge is independent, and recorded with results
harness = EvaluationHarness(
    model="gpt-5.2", provider="openai",
    judge_provider="openai", judge_model="gpt-4o-mini",
)
print(harness.judge.descriptor)  # "openai:gpt-4o-mini"
```

Results are now `(system, model, instrument)` triples. Record all three; treat an
instrument change as invalidating comparisons across it.

### Added

- **Paired-counterfactual tracks** (`cf_*`) over seven governance axes. Each pair
  holds entity, wording, event count and query constant and moves exactly one
  governance variable, so the measured quantity is the behavioral delta.
- **`applicability` track** — an 18-cell governance × experience-applicability
  factorial.
- **Governance Bypass Rate** and **Unsupported Reconstruction Rate**, scored
  against the engine's own audit record rather than phrase lists, so they need no
  judge.
- **Reconstruction baselines** (`reconstructive_rag_*`) implementing a
  critique-and-reconstruct memory stage.
- **Skill transfer** (`statebench.skills`) — distil reusable procedures from
  frontier episodes and serve them to a smaller model.

### Fixed

- `memgine` applied scope-based exclusions correctly but never recorded them in
  `facts_excluded`, leaving the audit record complete for access control and
  silently empty for scope containment. Provenance only; context is unchanged.
- Bounded retry on transient CAR gateway failures.

### Documentation

Three drafts in `docs/` document the correction and what it invalidated,
including in our own prior published work.

---

## 1.0.2 and earlier

See git history. Note that all results produced with these versions used the
scoring described above as defective.
