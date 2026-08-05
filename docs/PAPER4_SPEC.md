# Paper 4 Specification — Frontier-Distilled Skills for Weak Models

**Working title:** *Capability Transfer Without Weight Transfer: Frontier-Derived Memory
Artifacts as Governed Augmentation for Small Models*

Status: specification with a feasibility probe. Numbers marked HYPOTHESIS are not measured.

---

## 1. The observation that motivates this

Paper 3 §9.4 measured the accuracy premium of a state-based memory engine over a simpler
structured baseline, on two model generations (full dev split, 3 runs):

| Model | `memgine` − `state_based` (dev) |
|---|---|
| `gpt-5.2-2025-12-11` | **+9.1pp** |
| `gpt-5.6-sol` | **+0.7pp** |

The obvious reading is pessimistic: context curation is being absorbed by model capability,
so the engine's accuracy contribution is a wasting asset.

There is a second reading, and it is the one worth pursuing. If curation contributes more
where reasoning is weaker, then its value has not disappeared — it has **migrated
downward**. The engine is worth most exactly where the model is worst. That predicts a
*capability gradient*, which §7 tests by adding a third, much weaker tier. It is the
empirical premise on which everything else depends, and it is the gate for this work.

If the gradient holds, an economic question follows. Frontier inference is expensive and
weak-model inference is nearly free. A memory architecture that lets a weak model behave
like a strong one on *recurring* tasks converts a recurring frontier cost into a one-time
one.

---

## 2. The proposal

**Distill reusable artifacts from frontier episodes; serve them to a weak model on later
instances of the same abstract task.**

```
Phase 1 (once, expensive)   frontier model runs N episodes of task family T
                            → distill artifacts: skills, constraints, procedures
Phase 2 (M ≫ N, cheap)      weak model runs new instances of T
                            → retrieves artifacts, no frontier call at inference
```

The frontier model leaves the loop. Cost amortizes over M.

### 2.1 What distinguishes this from three neighbours

**Classic distillation** trains a small model on large-model outputs. Weight updates are
opaque, unrevocable, and unattributable; a distilled behavior cannot be audited, scoped to
an organization, or withdrawn when the underlying policy changes. Artifact-mediated
transfer keeps the small model's weights fixed and puts the transferred capability in an
inspectable, revocable, governable object.

**Retrieval-augmented generation** retrieves *documents*. What is retrieved here is a
distilled *procedure with its source conditions* — the experience-plane object of Paper 3
§3, `(principle, action, σ(e))` — not a passage.

**`memgine_distill`** (this repo) already pairs a frontier extractor with a local
generator, and establishes the critical precondition: *extraction quality is the whole
game*. With a `gpt-5.4` extractor the front-end is net-positive on implicit supersession
(SFRR 0.60 → 0.47, accuracy held); with a local `qwen3-4b`/`8b` extractor it is actively
harmful (accuracy 0.93 → 0.60, SFRR 0.47 → 0.87). Weak extractors emit inconsistent keys
and miss authority-based supersession.

But `memgine_distill` calls the frontier model **on every episode**. Cost scales linearly
with usage and the frontier model never leaves the loop. The proposal here is the
amortizing version: pay the frontier cost N times, serve M ≫ N episodes. That difference
is the contribution — the existing negative-then-positive result establishes that
frontier-quality distillation works, and this asks whether its product is *reusable*.

### 2.2 Why StateBench is an unusually good testbed

StateBench timelines are generated from **templates**: one template produces many
timelines that differ in entity, amounts, dates, and actor while sharing causal structure.
"The same abstract task recurring with different specifics" is not a scenario we have to
construct — it is the generator's design.

This gives a clean and honest split:

- **In-family transfer.** Distill from template T instances 1..N, evaluate on T instances
  N+1..M. Same abstract task, no shared surface content.
- **Cross-family transfer.** Evaluate T-derived artifacts on template U. Measures whether
  artifacts generalize or merely memorize.
- **Over-generalization.** The above, where U's correct behavior *differs* from T's. A
  skill that fires here is worse than no skill.

The third case is the one that makes this a real experiment rather than a demo.

---

## 3. Artifact schema

An artifact is an experience-plane object with explicit applicability conditions, so that
Paper 3's machinery applies unchanged.

```json
{
  "skill_id": "SK-0007",
  "task_family": "supersession/vendor_pricing",
  "principle": "when a later authoritative write targets the same key, answer from it and
                do not restate the prior value",
  "procedure": ["identify the key the question asks about",
                "find the most recent write to that key",
                "check whether an explicit supersession marks the earlier one dead",
                "answer from the live value only"],
  "preconditions": ["two or more writes to one key", "explicit supersession event"],
  "counter_indications": ["writes target different keys", "later event is an update that
                          does not invalidate"],
  "source_episodes": ["SUP-0031", "SUP-0044"],
  "source_scope": {"org": "acme_corp", "actor_authority": "manager"},
  "derived_by": "gpt-5.6-sol",
  "confidence": 0.8
}
```

Three fields are load-bearing and are the reason this is not just a prompt library:

- **`counter_indications`** — the artifact says when *not* to fire. Paper 3 §7.2 found
  that a reconstruction stage never declines; an artifact that carries its own negative
  conditions moves that judgment out of the model. Whether it actually helps is RQ3.
- **`source_scope`** — the artifact inherits the governance context of the episodes it was
  distilled from. This is what makes §5.3 checkable.
- **`source_episodes`** — provenance, so any behavior can be traced to the episodes that
  produced it and the artifact revoked if those episodes are withdrawn.

---

## 4. Research questions

**RQ1 — Is there a capability gradient?** Does the curation premium rise as the generating
model weakens? *Premise; if false, stop.* **Measured — directionally supported (§7).**

**RQ2 — Does artifact transfer lift a weak model?** **Measured: +20.0pp, 33% gap closure
(§7b).**

**RQ3 — Does it close the gap, and at what cost ratio?** **Gap closure measured (33%);
cost ratio not yet measured.** Report gain against inference cost: a 33% closure at 3% of
frontier cost is a different claim from the same closure at 60%.

**RQ4 — Do artifacts over-generalize?** **Inconclusive (§7b)** — off-family skills helped
rather than harmed, which may mean the artifacts generalize or may mean any structured
procedure helps. Needs an off-family track whose correct behavior actively conflicts.

**RQ5 — Do artifacts leak across governance boundaries?** An artifact distilled from Org A
episodes carries Org A specifics in its `principle` and `procedure` text. Retrieved for
Org B, that is a governance violation *through the artifact*. Paper 3 §3 established that
`Γ` must apply to the experience plane; a skill is an experience, so this is the same
failure with a longer fuse. Measurable with the existing Governance Bypass Rate.

RQ5 is where this paper and Paper 3 genuinely compose rather than merely cite each other.

---

## 5. Design

### 5.1 Arms

| Arm | Generator | Artifacts | Isolates |
|---|---|---|---|
| `weak_alone` | weak | none | floor |
| `weak_plus_skills` | weak | frontier-derived, in-family | the effect |
| `weak_plus_own_skills` | weak | **weak**-derived | is frontier quality necessary? |
| `weak_plus_offfamily` | weak | frontier-derived, wrong family | over-generalization (RQ4) |
| `frontier_alone` | frontier | none | ceiling |
| `frontier_per_episode` | frontier extractor + weak generator | per-episode (`memgine_distill`) | the non-amortizing comparator |

`weak_plus_own_skills` is the arm that decides whether this is a *frontier* transfer story
or merely a *structure* story. If a weak model distilling its own artifacts captures most
of the gain, the frontier model is not doing the work and the paper's framing is wrong.
`memgine_distill`'s prior negative result predicts it will not — but that was per-turn fact
extraction, and abstract skill distillation may be easier or harder. It must be measured.

### 5.2 Contamination control

The artifact must not encode the answer. Two guards:

1. **Disjoint instances.** Distillation and evaluation use non-overlapping timeline sets
   from the same template.
2. **Entity scrubbing.** Distilled artifacts are checked for evaluation-set entities,
   amounts, and dates; any artifact containing them is rejected. Without this, "transfer"
   could be answer leakage, and the result would be worthless.

The measurement-validity paper's lesson applies directly: the cheap proxy for "the artifact
generalized" is "the score went up," and that proxy fails exactly when the artifact
memorized. Guard 2 is not optional.

### 5.3 Governance conditions (RQ5)

Distill artifacts from episodes scoped to Org A, including episodes containing
`[RESTRICTED:]` facts. Then evaluate for an Org B actor. Correct behavior: the artifact's
*principle* may transfer (it is procedural) while its Org A *specifics* must not.

Scored with the existing GBR against a reference resolver — no judge, no phrase list.

---

## 6. Metrics

Existing, reused unchanged: decision accuracy, SFRR (resurrection only), leakage,
fabrication, GBR, URR.

New:

- **Gap closure** = (skills − weak_alone) / (frontier_alone − weak_alone). The headline.
  Undefined when the denominator is small; report the raw gap alongside.
- **Cost ratio** = inference cost of the skills arm ÷ frontier arm, including amortized
  distillation at stated M. Must be reported *with* gap closure; either alone is
  uninterpretable.
- **Over-generalization delta** = off-family arm − weak_alone. Negative means artifacts
  actively harm.
- **Artifact firing rate** and **precision**: how often an artifact was retrieved, and how
  often that was correct. Separates "did not help" from "never fired."

Report the amortization crossover: the M at which total cost of (N frontier distillation
episodes + M weak episodes) drops below M frontier episodes. This is arithmetic, not a
result, but it is the number a practitioner needs.

---

## 7. RQ1 result: the gradient holds (weakly)

**Measured.** `memgine` versus `state_based` on a matched 40-timeline dev subset (51
queries), judge pinned to `gpt-4o-mini` across every tier so the only variable is the
generator.

| Model | `memgine` | `state_based` | Premium | runs |
|---|---|---|---|---|
| `mlx/qwen3-4b:4bit` | 79.7% ± 3.3% | 63.4% | **+16.3pp** | 3 |
| `gpt-5.2-2025-12-11` | 94.1% | 88.2% | **+5.9pp** | 1 |
| `gpt-5.4-mini` | 87.6% ± 1.8% | 86.9% | **+0.7pp** | 3 |
| `gpt-5.6-sol` | 88.2% | 88.2% | **+0.0pp** | 1 |

The premise holds in its weak form: **curation is worth ~16pp to a 4B local model and
essentially nothing to any frontier-family model.** That is the gap artifact transfer aims
to exploit, and it is large.

**But it is not a smooth capability gradient, and §1's framing overreached.**
`gpt-5.4-mini` is a *smaller* model than `gpt-5.2` yet shows less premium (+0.7 vs +5.9).
Baseline competence does not explain it either: the two have near-identical `state_based`
scores (86.9% vs 88.2%) and very different premiums. What the data support is a
**threshold**, not a gradient — below some capability level curation is worth a great deal,
above it almost nothing — and the 4B tier is the only one clearly below it.

Anyone extending this should not assume "smaller model ⇒ bigger premium." A small *recent*
model may need curation no more than a large one. The economic argument survives, because
it only requires that some deployable-cheap model sit below the threshold, and qwen3-4b
does by 16pp.

Note also that `gpt-5.6-sol` + `memgine` (88.2%) scores *below* `gpt-5.2` + `memgine`
(94.1%) on this subset, consistent with the full-split regression reported in Paper 3
§9.4. Whatever is hurting Memgine on the newer model is visible here too.

**Headroom for RQ2/RQ3.** The gap artifact transfer would need to close:

```
qwen3-4b + memgine        76.5%   ← floor for the weak arm
gpt-5.2  + memgine        94.1%   ← ceiling
                          -----
                          17.6pp  of headroom
```

Curation alone already bought the weak model +9.8pp of that. The question RQ3 asks is what
fraction of the remaining ~18pp frontier-derived artifacts can recover, and at what cost
ratio.

**Revised interpretation of the motivating claim.** §1 framed this as curation value
"migrating downward" as models improve. The data support the weaker statement — *curation
is worth much more below the frontier* — but not a specific functional form. The spec
should not claim a law from three points.

---

## 7b. RQ2/RQ3/RQ4 result: transfer works, and the deflationary control fails

**Setup.** Track `repair_propagation` — chosen because the weak model has real headroom
there (40% alone) rather than the ceiling it hits on `supersession` (100%, unmeasurable).
Distilled from 12 **train** episodes with `gpt-5.6-sol`; evaluated on 15 **dev** timelines.
Splits are disjoint by construction, and all 4 skills passed the literal-scrub against 52
evaluation-set values. Weak model `mlx/qwen3-4b:4bit`, judge `gpt-4o-mini`, single run.

| Arm | Accuracy | SFRR | Skill firing |
|---|---|---|---|
| `weak_alone` | 40.0% | 33.3% | — |
| **`weak_plus_skills`** | **60.0%** | 53.3% | 100% |
| `weak_plus_hand` (control) | 26.7% | 53.3% | 86.7% |
| `weak_plus_offfamily` (control) | 53.3% | 86.7% | 100% |
| `frontier_alone` | 100.0% | 20.0% | — |

**RQ2 — transfer lifts the weak model: +20.0pp**, closing **33%** of the 60pp weak→frontier
gap. Skills fired on every query, so the effect is not a retrieval artifact.

**RQ3 — the deflationary explanation is rejected.** A hand-written generic procedure scores
26.7%, *below* the no-skill floor: it does not merely fail to help, it actively harms. The
distilled artifacts beat it by 33.3pp. Whatever the frontier model produced is not
substitutable with an engineer's generic prompt.

*Caveat that limits this.* The hand-written control is a **generic** state-tracking
procedure, not one targeted at repair propagation. It therefore tests "does generic
guidance suffice?" and answers no — emphatically, since generic guidance about using "the
most recent value" is close to wrong advice for a track requiring recomputation. It does
**not** test whether a domain expert writing a repair-propagation-specific procedure could
match distillation. That control should be run before publication; it is the strongest
remaining deflationary account.

**RQ4 — over-generalization did not appear, and this is suspicious.** Off-family skills
scored 53.3%, *above* the floor rather than below it. Two readings, and we cannot separate
them here: either the skills are general enough to help across families (interesting), or
merely having any structured procedure in context helps this weak model regardless of
content (deflationary, and consistent with the +13.3pp being mostly a formatting effect).
The 6.7pp shortfall against in-family skills is one query at n=15. **A cleanly negative
over-generalization result requires an off-family track whose correct behavior actively
conflicts**, which `scope_permission` does not.

**RQ5 (partial) — skills introduce real governance bypass.** Every skill arm shows
elevated SFRR, which the companion measurement paper warns fires on *engagement* rather
than leakage. We therefore re-measured with the engine-grounded Governance Bypass Rate,
which is judge-free, phrasing-insensitive, and scored against what state resolution
actually excluded:

| Arm | SFRR | **GBR** | URR | mean words |
|---|---|---|---|---|
| `weak_alone` | 53.3% | **0.0%** | 46.7% | 90 |
| `weak_plus_skills` | 60.0% | **13.3%** | 53.3% | 88 |

**The leakage is real.** GBR rises from 0% to 13.3% — the skill-augmented model asserted
content the engine had withheld, on 2 of 15 queries. Response length is unchanged (90 vs
88 words), so this is not the verbosity story either. Adding a procedure that instructs the
model to trace corrections through their dependents evidently also induces it to surface
values the state layer had excluded.

So the honest summary of the safety picture is: **+20pp accuracy, and a real 13.3pp
governance-bypass cost**, appearing spontaneously rather than under a deliberate cross-org
probe. RQ5's dedicated experiment is now higher priority, not lower.

**Two methodological notes.** First, this is where the three papers compose: the SFRR
signal alone was uninterpretable, and only the judge-free metric from Paper 3 could
adjudicate it. Second, `weak_alone` SFRR was 33.3% in the §7b run and 53.3% here under an
identical configuration — the local model is nondeterministic and n=15 is small. **Every
number in §7b and §7c is single-run at n=15 and should be treated as directional.**
Three runs are required before any of this is publishable.

---

## 7d. RQ3 RESULT: the episodes add nothing. The framing is wrong.

Three runs per arm, `repair_propagation`, 15 dev timelines, weak model
`mlx/qwen3-4b:4bit`, judge `gpt-4o-mini`.

| Arm | Accuracy | Gain over floor |
|---|---|---|
| `weak_alone` | 35.6% ± 3.1% | — |
| **`weak_plus_skills`** (distilled from 12 episodes) | 53.3% ± 5.4% | **+17.8pp** |
| **`weak_plus_desconly`** (frontier, *one-line description, no episodes*) | 53.3% ± 0.0% | **+17.8pp** |
| `weak_plus_targeted` (hand-written, task-specific) | 60.0% ± 0.0% | +24.4pp |
| `weak_plus_generic` (hand-written, generic) | 26.7% ± 5.4% | **−8.9pp** |
| `frontier_alone` | 100.0% ± 0.0% | +64.4pp |

**The deflationary control succeeds, and the paper's thesis does not.** A
procedure the frontier model wrote from nothing but the string *"Corrections
cascade to derived conclusions"* matches distillation from twelve worked
episodes **exactly** — 53.3% both ways. The episodes contribute **+0.0pp**.

A hand-written task-targeted procedure does *better* than distillation
(+24.4pp), though that control is contaminated: it was authored after seeing the
distilled skills. The description-only control has no such exposure and is the
one to weight. It already ties.

**What survives, and it is worth having.** Procedures lift a weak model
substantially — +17.8pp, closing 28% of a 64pp gap — and the generic control
(−8.9pp) shows this is not "any text in the context helps": a procedure must be
*task-appropriate* or it actively harms. But the source of that appropriateness
is the task description, not worked examples.

**The honest reframing.** This is not capability transfer from experience. It is:
*a frontier model can write a procedure that substantially lifts a weak model,
from a one-line task description, at the cost of a single call.* That is a
cheaper and more practical claim than the one we set out to test — 1 API call
rather than 12 episodes — but it is a different claim, and "amortized experience
distillation" is the wrong frame for it.

**RQ4 is unmeasurable as designed.** On the conflicting family
(`supersession_maintain`, where the correct answer is to *affirm* the still-valid
fact) the weak model scores **0.0% with or without skills**. It cannot do the
task at all, so there is no headroom for skills to harm it and the
over-generalization test is degenerate at this floor. It needs either a stronger
weak model or a conflicting family the weak model can partly do. Separately,
0.0% unaided is itself notable: this model over-supersedes universally.

**RQ3 cost.** Token accounting is not usable — the CAR local path reports
`usage: null`, so the weak arm's token count is 0 and the ratio is meaningless.
Cost comparison needs wall-clock or price-weighted accounting instead.

### What this means for the project

The interesting question moved. "Do distilled experiences transfer?" is answered
*no, not measurably beyond the task description*. The live questions are now:

1. Does the description-only result hold on tasks where the description
   under-determines the procedure? `repair_propagation`'s description
   ("corrections cascade to derived conclusions") nearly *is* the procedure,
   which may be why episodes added nothing. A task whose description does not
   telegraph its method is the real test, and is the single experiment most
   worth running next.
2. Does the +17.8pp survive on a weak model that is not at a floor?
3. Is the governance cost (GBR 0% → 13.3%, §7c) attached to procedures in
   general, or to distilled ones specifically? The description-only arm makes
   this separable and it was not measured.

Question 1 could rescue the original framing or bury it. It should run before
anything else.

---

## 8. Sequencing

1. **RQ1 gradient** — running. Gate on it.
2. **Artifact schema + distiller** — frontier model summarizes N episodes of one template
   into artifacts; entity-scrubbing check.
3. **Retrieval + injection** — artifacts enter the experience plane; `Γ` applies (Paper 3
   §3 established this is required and that our first implementation got it wrong).
4. **RQ2/RQ3 on one track** — cheapest decisive test. If gap closure is negligible, stop.
5. **RQ4 over-generalization** — the result most likely to be negative and most worth
   knowing.
6. **RQ5 governance** — composes with Paper 3; needs no new metric.

Steps 1 and 4 are the ones that can end the project early, which is why they come first and
second-to-last rather than last.

---

## 9. Risks

**The gradient may not hold.** §7. Gate.

**Artifacts may be prompt engineering by another name.** If a hand-written system prompt
for the task family captures the same gain, the distillation is doing nothing. **Add a
`weak_plus_handwritten_prompt` arm.** This is the most likely deflationary explanation and
the paper must run it.

**Gains may be memorization.** §5.2 guards, and the over-generalization arm cross-checks.

**Task families may be too narrow.** StateBench templates are procedurally generated and
may support transfer that natural task recurrence would not. A second, non-synthetic domain
would strengthen the claim considerably and is out of scope for a first paper — but the
limitation must be stated, not buried.

**The weak model may be unable to follow artifacts.** Paper 3 §7.2 found a reconstruction
stage that had a rejection vocabulary and never used it. An artifact carrying
`counter_indications` may be ignored the same way. The artifact firing-rate metric exists
to distinguish "ignored" from "followed and unhelpful."
