# Worked Examples Beat Task Descriptions

### What distillation actually adds, and why the variance most people measure is the wrong one

**Matt Liotta** · August 2026

---

## Abstract

A frontier model can write a procedure that lifts a small model's accuracy substantially.
The practical question is what that procedure needs to be *derived from*: worked episodes
of the task, or merely a description of it. Episodes are ~12× more expensive to distil
from, so if a one-line description suffices, distillation is not worth doing.

We find episodes are worth **+18.7pp** over the best description-only alternative
(64.0% ± 3.3% vs 45.3% ± 7.8%, against a 38.7% ± 5.0% floor), a gap **3.1× the pooled
generation noise**. Episodes also produce *more stable* artifacts — the description-only
condition varies nearly twice as widely across generations, meaning a practitioner gets
whichever procedure their single call happened to produce.

Reaching that result required correcting a methodological error we made twice, and which
we believe is common. Our first two experiments generated **one** artifact and evaluated
it three times, reporting `± 0.0%` error bars. Those bars described *evaluation* noise
while the quantity that actually moved between experiments was *generation* noise. The two
passes returned opposite answers (+0.0pp and +8.9pp for the same comparison). Measured
properly — N independent generations, one evaluation each — artifact-generation variance
(6.0pp) **exceeds** model-evaluation variance (5.0pp). Any evaluation of LLM-authored
artifacts that repeats evaluation over a fixed artifact is measuring the smaller of two
noise sources and reporting confident intervals about the wrong quantity.

Three secondary results. A *generic* procedure actively harms (−8.9pp), so the benefit
requires task-appropriateness rather than mere structure. Governance bypass rises from 0%
to 13.3% for **both** artifact conditions equally, so the safety cost attaches to having a
procedure at all, not to distillation. And a capability-gradient hypothesis — that curation
helps small models more — **failed**: of five models across three families, only one
showed any premium.

Scope is narrow and we do not overstate it: one weak model, one task family, 15 evaluation
timelines.

---

## 1. The question

Give a small model a procedure and it does better. That much is uncontroversial and easy
to demonstrate. The question with economic content is *where the procedure comes from*.

Two sources, differing by an order of magnitude in cost:

- **Distillation from worked episodes.** Show a frontier model N solved instances and ask
  for the transferable procedure. Cost: N episodes rendered into context, plus a
  generation call.
- **Description-only authoring.** Show the same frontier model a one-line description of
  the task and ask for the same thing. Cost: one short call.

If these produce equivalent procedures, distillation is theatre. Nobody should pay 12×
for it, and the framing of memory-as-transferable-experience loses its empirical support.

We set out to test the first and found — twice, with confident-looking error bars — that
it added nothing. Both results were artifacts of a sampling error described in §3.

## 2. Setup

**Task.** `repair_propagation` from StateBench: a value is corrected, and conclusions
derived from the old value must be recomputed rather than reused. Chosen because the weak
model has genuine headroom there (≈39% unaided) rather than the ceiling it hits on other
tracks, where no lift is measurable.

**Models.** Weak: `mlx/qwen3-4b:4bit`, local. Author/distiller: `gpt-5.6-sol`. Judge:
`gpt-4o-mini`, pinned globally and never inherited from the system under test.

**Contamination control.** Distillation draws from the **train** split and evaluation from
**dev**, so episode sets are disjoint by construction. A scrub pass then rejects any
artifact containing an evaluation-set literal — entity, amount, or date — because templates
reuse entity pools across splits, and disjoint episodes do not guarantee disjoint content.
Without this, "transfer" could be answer leakage.

**Conditions.**

| Condition | Artifact source |
|---|---|
| `weak_alone` | none |
| `distilled` | frontier model, 12 worked episodes |
| `description_only` | frontier model, one-line task description, no episodes |
| `generic` (§5.1) | hand-written, task-agnostic |
| `targeted` (§5.1) | hand-written, task-specific |

## 3. The methodological correction

Our first two experiments each ran **three evaluation passes over one generated artifact**
and reported the spread as an error bar. Results:

| Pass | `distilled` | `description_only` | Conclusion drawn |
|---|---|---|---|
| A | 53.3% | 53.3% ± 0.0% | episodes add **+0.0pp** |
| B | 55.6% | 46.7% ± 0.0% | episodes add **+8.9pp** |

Two passes, opposite conclusions, and both reported `± 0.0%` on the description-only arm.

That `± 0.0%` was the diagnostic and we initially read it as precision. Three evaluations
of a *fixed* artifact measure only weak-model sampling. The artifact itself is drawn from a
distribution, and it is the artifact that differed between passes: pass B's generation was
simply worse.

The corrected design points the same compute at the right variance — **N independent
generations, one evaluation each** — and compares distributions rather than point
estimates. This is not a refinement. The naive design returned the wrong sign.

**The general claim.** For any evaluation of LLM-authored artifacts — prompts, procedures,
plans, distilled skills, generated rubrics — the artifact is a random variable. Repeating
*evaluation* while holding the artifact fixed produces error bars that describe the wrong
quantity, and here that quantity was the *smaller* of the two (5.0pp evaluation vs 6.0pp
generation). Confident intervals over the smaller source, applied to a comparison driven by
the larger, is how both of our early passes reached the wrong answer with narrow bars.

## 4. Result

Five independent generations per condition, one evaluation each, 15 dev timelines.

| Condition | Accuracy | Range | SFRR | GBR |
|---|---|---|---|---|
| `weak_alone` | 38.7% ± 5.0% | 33.3–46.7% | 50.7% | **0.0%** |
| **`distilled`** | **64.0% ± 3.3%** | 60.0–66.7% | 69.3% | 13.3% |
| `description_only` | 45.3% ± 7.8% | 40.0–60.0% | 52.0% | 13.3% |

**Episodes are worth +18.7pp**, 3.1× the pooled generation standard deviation of 6.0pp.
The distributions barely touch: distilled's worst generation (60.0%) equals
description-only's best.

**Episodes also stabilise.** Description-only spreads ±7.8% against distilled's ±3.3% —
roughly twice the variance. For a deployment this matters more than the mean: you get one
generation, not a distribution, and the description-only condition's 20pp range means the
procedure you happen to receive may be barely better than nothing.

That stability result is only visible under the §3 design. The naive design cannot see it
at all, because it never generates a second artifact.

## 5. Secondary results

### 5.1 Structure is not enough; task-appropriateness is required

Two hand-written controls, three evaluation runs each:

| Control | Accuracy | vs floor |
|---|---|---|
| generic ("answer from currently valid state…") | 26.7% ± 5.4% | **−8.9pp** |
| targeted (repair-specific) | 60.0% ± 0.0% | +24.4pp |

A generic procedure is **worse than no procedure**. So the effect is not "text in the
context helps" — a procedure must fit the task or it actively misleads. The generic one
advises using the most recent value, which is close to wrong advice on a track that
requires recomputation.

The targeted control scores well, but we authored it *after* seeing the distilled
artifacts and treat it as contaminated in its own favour. It bounds what a domain expert
might achieve; it does not establish it.

### 5.2 The governance cost is not distillation-specific

Governance Bypass Rate — asserting content the state layer excluded, scored against the
engine's audit record with no judge — is **0.0% unaided and 13.3% for both artifact
conditions, identically**.

So procedures cause a weak model to surface withheld state regardless of their provenance.
A procedure instructing the model to trace corrections through their dependents evidently
also induces it to reach for state it should not have. This is a cost of the technique, not
of distillation, and it is worth stating plainly: the accuracy gain comes with a real
governance regression.

### 5.3 A failed hypothesis: no capability gradient

We predicted curation would help in inverse proportion to model capability, which would
have given artifact transfer an economic motivation. Across five models and three families,
matched subset, judge held constant:

| Model | Family | Premium | runs |
|---|---|---|---|
| `qwen3-4b` | Qwen | **+16.3pp** | 3 |
| `qwen3-8b` | Qwen | −3.3pp | 3 |
| `gemma-4-12b` | Google | +3.3pp | 3 |
| `gpt-5.4-mini` | OpenAI | +0.7pp | 3 |
| `gpt-5.6-sol` | OpenAI | −0.0pp | 3 |

Four of five sit within ±3.3pp of zero, and only `qwen3-4b` clears its own noise
(5.6× pooled σ; the next largest is `gemma-4-12b` at 1.6×, which is not significant).
**Only the 4B model shows a premium**, and a second small local model gets nothing. There
is no gradient and no threshold — there is one outlier. Whether that reflects scale, that
specific model, or an interaction with the context format is unresolved, and a single
model cannot distinguish them.

The tiers also illustrate §3 from the other direction. Every one of them shrank toward zero
as runs accumulated: `gemma-4-12b` read +5.9pp at n=1, +2.0pp at n=2 and +3.3pp at n=3;
`qwen3-8b` read +3.9pp at n=1 and −3.3pp at n=3, reversing sign. Single-run tier results
were uniformly more encouraging than replicated ones.

We report this because the hypothesis motivated the work, and because a single supporting
datapoint would have been easy to present as a trend.

## 6. Limitations

**One model, one task, small n.** `qwen3-4b` on `repair_propagation` over 15 timelines.
Whether +18.7pp generalises across tasks or models is untested. Given §5.3, we would not
assume it.

**Description quality is a confound we could not fully remove.** `repair_propagation`'s
description — "corrections cascade to derived conclusions" — nearly *is* the procedure. We
attempted an impoverished-description arm, but the frontier model inferred the method
anyway, producing guidance about "treating corrections as superseding earlier information"
from a description that only said information changes over time. A task whose method cannot
be inferred from any honest description would be a cleaner test; we did not find one.

**Over-generalization is untested.** The intended control — applying skills to a family
whose correct behavior conflicts — was degenerate: the weak model scores 0.0% on that
family with or without skills, so there is no headroom for harm to appear.

**No cost measurement.** The local inference path reports `usage: null`, so token
accounting for the weak arm reads zero and the cost ratio is unavailable. The 12×
generation-cost difference is by construction, not measured end-to-end.

**Judge and distiller share a provider family**, though not a model. A cross-family judge
control was not run.

## 7. Conclusion

Worked examples are worth paying for. They produce procedures that are ~19 points better
and about twice as stable as what the same model writes from a task description alone.

The methodological result may be the more portable one: when the thing under test is
generated by a model, the generation is the experiment. We measured the wrong variance
twice, with narrow error bars both times, and got opposite answers. The fix costs nothing —
the same compute, redistributed — and it is the difference between a result and its
negation.

Finally, the technique carries a governance cost that its accuracy numbers conceal. A
procedure that helps a weak model reason also helps it reach for state it was not given.
That cost is identical whether the procedure came from episodes or a description, and it
should be measured with an instrument the procedure cannot talk its way past.

---

## Appendix: reproduction

```bash
# The corrected design: N generations, one evaluation each
uv run python experiments/with_car_secret.py -- \
    python experiments/artifact_variance.py --samples 5 --limit 15

# Controls, three evaluation runs each
uv run python experiments/with_car_secret.py -- \
    python experiments/skill_transfer.py --track repair_propagation --runs 3
```

Results: `experiments/results/artifact_variance.json`,
`skill_transfer_v2.json`, `capability_curve.json`.
