# Research Review: Agent Memory, August 2026

A review of *Rethinking Memory Mechanisms of Foundation Agents in the Second Half:
A Survey* (Huang et al., arXiv:2602.06052) and the cluster of 2026 work it points at,
read against StateBench's design, claims, and open questions.

---

## 0. Provenance and confidence — read this first

**I was not able to obtain the paper's primary text.** This session's egress proxy
denies `arxiv.org`, `openreview.net`, `alphaxiv.org`, `huggingface.co`,
`researchgate.net`, and every other mirror attempted — CONNECT returns 403 at the
gateway, and the fetch tool reports `EGRESS_BLOCKED` for the same hosts. The only
working channel was the web-search tool, which returns model-written summaries over
search results rather than source text.

Everything below is therefore **second-hand**. Each claim carries a confidence tag:

- **[direct]** — appears verbatim or near-verbatim in abstract text surfaced by search;
  high confidence it reflects the source.
- **[reported]** — asserted by a search summary, consistent across more than one query,
  but not read in the original.
- **[single]** — asserted once, by one summary. Treat as a lead, not a fact.
- **[ours]** — my inference or judgement, not a claim about any paper.

**No number or claim in this document may be cited in a StateBench paper until it has
been checked against the primary source.** That rule matters more here than usual:
the whole point of `paper-measurement-validity` is that instruments get believed
without being checked. A citation chain that terminates in a search snippet is the
same failure at a different layer. §6 lists what to verify and how.

Also note a title discrepancy: the PDF the request was made against is titled
*A Survey of Agent Memory in the Second Half: Towards Self-Evolving and Long-Horizon
Agents* (v4, 4 Aug 2026), while search indexes v1–v3 under *Rethinking Memory
Mechanisms of Foundation Agents in the Second Half: A Survey* (Jan 14 2026, revised
Feb 10 2026). Same arXiv ID, same author block, retitled at v4. Everything retrieved
below describes **v3 or earlier**. The v4 retitling — foregrounding *self-evolving* and
*long-horizon* — is itself a signal about where the authors think the field moved
between February and August, and the v4-only material is exactly the material I could
not read.

---

## 1. The survey in brief

**Scope.** ~60 authors across ~27 institutions, synthesizing 200+ papers on memory in
LLM agents. [reported]

**Framing.** The field is moving "from prioritizing model innovations over benchmark
scores towards emphasizing problem definition and rigorous real-world evaluation." In
the "second half," the central challenge is *real utility in long-horizon, dynamic,
user-dependent environments*, where agents face context explosion and must
continuously accumulate, manage, and selectively reuse information across extended
interactions. [direct]

**The three-dimensional taxonomy.** [direct]

| Dimension | Values |
|---|---|
| Memory substrate | internal (parametric) / external (non-parametric) / mixed |
| Cognitive mechanism | working, episodic, semantic, procedural, sensory |
| Memory subject | agent-centric / user-centric |

**Memory operations (single-agent).** Storage & index; loading & retrieval; update &
refresh; compression & summarization; forgetting & retention. [reported]

**Memory operations (multi-agent).** Memory architecture definition; routing protocols;
isolation and conflict-resolution strategies. [reported]

**Evaluation.** Benchmarks split into user-centric (MSC, MemoryBank — persona
retention, preference recall, cross-session consistency) and agent-centric (OSWorld,
WebArena — task success under multi-hop reasoning and tool use). The survey notes that
"designing benchmarks to evaluate agents in real environments has become one of the
most important challenges." [reported]

**Open challenges.** Six of them, oriented toward "reliable, scalable, self-evolving,
and trustworthy memory infrastructures." I could not recover the enumeration. [single]

---

## 2. Where StateBench sits in the survey's coordinate system

Mapping our work onto their axes, because being locatable in a field's taxonomy is how
work gets cited: [ours]

| Axis | StateBench / Memgine |
|---|---|
| Substrate | **External**, non-parametric, explicitly so. Memgine is an external engine; we test no parametric-memory baseline at all. |
| Cognitive mechanism | **Semantic** (facts, constraints, policies) and **episodic** (commitments, corrections, interruption/resumption). No working-memory, no sensory, no procedural coverage. |
| Subject | **Mixed**, and unusually so. `scope_permission`, `enterprise_privacy`, and `authority_hierarchy` are *multi-principal* — several parties writing one memory under different roles. Most of the field's user-centric work assumes a single user. |
| Operations | Memgine covers storage & index, loading & retrieval (query-relevance sorting), update & refresh (supersession, adaptive inline repair), compression (threshold-based compaction / Summary DAG). **Forgetting & retention: absent.** |

Three observations fall out of this.

**(a) Multi-principal is our differentiator, and we under-sell it.** The README leads
with resurrection. But the survey's own framing treats multi-principal governance as
an open frontier, and GateMem (§3.6) exists specifically because "memory benchmarks
for LLM agents largely assume single-user settings." Our `scope_permission`,
`enterprise_privacy`, and `authority_hierarchy` tracks predate that framing and are
directly on it. Memgine's engine-level access control is a *governance* claim, not a
memory-quality claim, and it should be pitched that way.

**(b) Procedural memory is the name for what paper 4 is about.** *Worked Examples Are
Worth Paying For* — skills distilled from episodes beating skills written from a task
description — is, in the survey's vocabulary, an experiment about **consolidating
episodic memory into procedural memory**. The survey describes procedural memory as
"abstracting complex action sequences into reusable patterns" and notes that "as
execution experience accumulates, short-lived action states can be consolidated into
reusable skills or routines." [direct] That is paper 4's thesis in the field's own
words. Adopting the term costs nothing and makes the paper findable.

**(c) Forgetting is a real hole.** We have compaction under token pressure. We have no
notion of *retention policy* — deletion on request, decay, or the right of a fact to
be actively removed rather than merely superseded. Superseding a fact keeps it in the
store with a tombstone; deleting it means it must not be recoverable at all. GateMem
evaluates exactly this and finds nobody does it well. See §5, P3.

---

## 3. Seven things worth acting on

### 3.1 We are no longer alone on supersession, and that is mostly good news

Between February and August 2026 the field independently converged on StateBench's
core thesis. Instruments and metrics now in the literature that target our failure
modes: [reported; IDs unverified]

| Work | arXiv | Overlaps |
|---|---|---|
| STALE — *Can LLM Agents Know When Their Memories Are No Longer Valid?* | 2605.06527 | supersession, implicit detection, stale-premise queries |
| Supersede — *Diagnosing and Training the Memory-Update Gap* | 2606.27472 | supersession under bounded memory; RL training env |
| GateMem — *Memory Governance in Multi-Principal Shared-Memory Agents* | 2606.18829 | scope, access control, deletion/forgetting |
| *Don't Ask the LLM to Track Freshness* | 2606.01435 | deterministic conflict resolution |
| TEPA — *Revoking Stale Memories for Conflict-Robust Language Agents* | 2608.07429 | revocation |
| SubtleMemory | 2606.05761 | fine-grained relational discrimination |
| MemSyco-Bench | 2607.01071 | sycophancy in memory (an authority-hierarchy cousin) |
| MemTX — *Transactional Belief Commit* | 2607.23929 | state-machine memory semantics |
| Memora / **FAMA** metric | — | penalizes reliance on obsolete or invalidated memory — an SFRR analogue |
| BEAM | — | 10 categories incl. knowledge update, contradiction resolution, abstention |

**Implication.** Two things change. First, our papers need a real related-work section;
"nobody tests this" is no longer true and reviewers will know it. Second, SFRR is now
one of several metrics measuring the same construct, and **FAMA got there in the
literature**. We should state the relationship explicitly — SFRR is a per-response
violation rate over `must_not_mention` constraints, which is a stricter and more
brittle instrument than a penalized-accuracy metric, as our own measurement-validity
audit demonstrates at length. Being the strict instrument is defensible. Being the
strict instrument *without saying so* looks like ignorance of the alternative.

### 3.2 STALE has a probing dimension we lack, and it collides with our scoring bug

STALE's framing: the hard case is **Implicit Conflict** — "a later observation
invalidates an earlier memory without explicit negation." It probes three dimensions:
[direct]

1. **State Resolution** — detecting that a prior belief is outdated.
2. **Premise Resistance** — rejecting queries that falsely presuppose a stale state.
3. **Implicit Policy Adaptation** — proactively applying updated state downstream.

Map to us: (1) ≈ `supersession_detection`, (3) ≈ `repair_propagation`.
**(2) has no StateBench equivalent.**

And a premise-resistance track is not merely missing — it is the exact case our own
instrument mis-scores. To reject a false presupposition, a correct answer must *name
the stale fact in order to refuse it*: "You haven't moved back to Portland — your
address is 456 Oak Ave." That answer contains the forbidden phrase. Under phrase-list
`must_not_mention` scoring it is a resurrection. This is precisely the defect
`paper-measurement-validity` documents — **100% of correct rejecting answers scored as
violations** — arrived at from the opposite direction: they designed the question type
that our instrument cannot score, while we found that our instrument cannot score it.

**Implication.** A `premise_resistance` track is the highest-value single addition
available, *and* it is a natural second empirical section for the measurement-validity
paper: a track built specifically to break naive phrase-list scoring, scored correctly
under the v2.0 negation-aware rules. It converts a defect writeup into a contribution.

### 3.3 Supersede's result is in tension with our gpt-5.6 conclusion

Supersede reports, on LongMemEval's knowledge-update subset: [reported]

- Replacing full context with a bounded self-maintained memory drops accuracy
  **92% → 77%** on a frontier model, McNemar p<0.005.
- The gap **persists across model scale**, while full-context accuracy saturates ~92%.
- Scaling the conversation **24×** drops accuracy **68% → 28%**.
- Granting proportionally more memory budget yields **no recovery** (28% → 28%).
- Conclusion: "the bottleneck is memory maintenance, not comprehension, and is not
  closed by a stronger model."

Our README, from the gpt-5.6-sol refresh, says the opposite-ish: "Stronger models need
less context curation" — Memgine's dev-split lead over `state_based` falling 9.1pp →
0.7pp, and `state_based` pulling level on test.

These are not necessarily contradictory. Ours is measured at a **fixed, short timeline
length**; Supersede's persistence claim is measured **across a length sweep**. If the
architecture premium is length-dependent, both are true and ours is the special case.
But as written, our sentence is a general claim about models, supported by a
single-length experiment. [ours]

**Implication.** This is a live risk to a published claim, and it is testable with code
we already have — `budget-sweep` exists, and the generator can emit longer timelines.
See §5, P1.

### 3.4 "Tenure crossover" may be the mechanism, and it is a bigger deal than the fix

*Ground Truth First* (2607.21962) reports what its title calls a **tenure crossover in
memory-architecture rankings**: which architecture wins depends on how long the agent
has been running. [single — title-level, mechanism unverified]

If that holds, then §3.3 is not a caveat to bolt onto our claim — it is the finding.
"Architectures rank differently at different interaction lengths, and every published
memory leaderboard reports a single length" is a stronger, more useful result than
either "Memgine wins" or "stronger models need less curation." StateBench is unusually
well positioned to test it: we control timeline generation, so we can sweep tenure
directly rather than inferring it.

The same paper independently validates our generator design. It "inverts the pipeline":
a seeded sampler emits facts with **validity intervals**, volatility classes, and
source channels *before any text exists*; an LLM renders the text from per-event fact
manifests; a fidelity verifier confirms every planted fact; questions are instantiated
mechanically from the script. Its stated motivation is that generate-then-extract
pipelines have "documented label-error and contamination problems." [direct]

That is StateBench's architecture, described by someone else, with two features we
lack: **per-fact validity intervals** (we have `environmental_freshness` as a track,
not as a schema primitive) and **an explicit fidelity-verification pass** confirming
every planted fact actually landed in the rendered text. The second is cheap and would
close a real hole — right now nothing checks that our generator's facts survive
rendering.

### 3.5 Deterministic conflict resolution: independent corroboration of Memgine

*Don't Ask the LLM to Track Freshness* (2606.01435) tests on MemoryAgentBench's
`FactConsolidation` task, where facts carry serial numbers and **agents are explicitly
told that newer facts have larger serials**. Reported results: [reported — numbers
unverified]

| System | Single-hop accuracy |
|---|---|
| HippoRAG-v2 | 54% |
| BM25 | 48% |
| Mem0 | 18% |
| Zep / Graphiti | 7% |

Their diagnosis: "the bottleneck is the assembly step" — baselines leave conflict
resolution to LLM-mediated retrieval or generation instead of explicit version-aware
aggregation. Their fix: retrieve with BM25, extract matching candidates with an LLM,
then pick the winner with a deterministic `max(serial)` in Python.

**This is Memgine's thesis, replicated externally, on a different benchmark, by an
unrelated group.** Our claim — that determinism at the engine layer beats asking the
model to sort it out — now has outside support. The 7–18% figures for popular
production memory systems, if they hold up, are also the strongest available answer to
"why not just use Mem0/Zep?"

### 3.6 GateMem is the external validity test for Memgine's access-control claim

GateMem jointly evaluates three things: utility on legitimate long-horizon requests
with state updates; **access control across contextual authorization boundaries**; and
**agent-facing active forgetting after explicit deletion requests**. Domains: medical,
office, education, household. Design: long-form multi-party episodes, incremental
memory injection, hidden checkpoints, structured judging, leak-target annotations.
Headline: **"no method simultaneously achieves strong utility, robust access control,
and reliable forgetting."** [direct]

We claim Memgine does two of those three. Our README: "Engine-level access control
eliminates scope leakage that prompt-based approaches cannot prevent," with a 1.9%
leakage rate — the lowest of any baseline.

**Implication.** GateMem is a ready-made external test of our strongest architectural
claim, run by people with no stake in it. If Memgine clears it, that is worth more than
any number produced on our own benchmark — self-evaluated benchmarks are exactly the
thing a skeptical reviewer discounts. If Memgine does *not* clear it, we need to know
before someone else finds out.

Caveat worth being honest about: our leakage claim is architecturally near-tautological
on our own data. If the engine deletes restricted facts before the model sees them, and
our leak test asks whether restricted facts appear, low leakage is close to a property
of the construction, not a discovery. GateMem's contextual authorization boundaries are
harder than our role flags, and that is the point.

### 3.7 The benchmark-to-deployment gap cuts both ways

MemoryArena (2602.16313) evaluates memory inside multi-session **agentic loops**, where
memorization and action are coupled — agents must distill experience from earlier
sessions to satisfy constraints in later ones. Reported: systems scoring ~95% on LoCoMo
fall to **40–60%** on MemoryArena. [reported]

Read one way, this is support: passive recall benchmarks overstate real competence,
which is StateBench's founding complaint. Read another way, it is a warning aimed at
us. **StateBench is a passive-query benchmark.** We build context, ask a question,
score the text. Nothing is *executed*; no action is taken; no downstream state changes
as a result of the answer. Our `repair_propagation` track propagates corrections into
*a stated conclusion*, not into *a taken action*.

**Implication.** The honest framing is that StateBench measures state correctness in
the *reasoning* substrate, and that this is a necessary but not sufficient condition
for agentic competence. Worth saying explicitly in the papers rather than waiting to be
told. An action-coupled track is a v2 conversation, not a quick fix.

---

## 4. Terminology alignment

Cheap, high-leverage, no experiments required. Our vocabulary is idiosyncratic and it
is costing us citations. [ours]

| StateBench term | Field term(s) | Where |
|---|---|---|
| Supersession | knowledge update; memory-update gap; belief revision | LongMemEval, Supersede |
| Implicit supersession | **implicit conflict** | STALE |
| SFRR | forgetting-aware accuracy family; **FAMA** | Memora |
| Scope leak / enterprise privacy | **memory governance**; contextual authorization | GateMem |
| Stale reasoning | failure of **implicit policy adaptation** | STALE |
| Repair propagation | cascading invalidation; propagated conflict | STALE, TEPA |
| Environmental freshness | **validity intervals**; volatility class | Ground Truth First |
| Compaction / Summary DAG | compression & summarization | survey op taxonomy |
| (absent) | **forgetting & retention** | survey op taxonomy, GateMem |
| Worked-example skills (paper 4) | **procedural memory**; experience distillation | survey |

Keep our terms as the primary vocabulary — they are more precise and the papers are
already written around them — but gloss each one against the field term on first use.

---

## 5. Recommended actions, in priority order

**P0 — Verify before citing.** Nothing here is source-checked. Request an egress
allowlist entry for `arxiv.org` (and ideally `openreview.net`) for this environment,
then re-read the survey's §Evaluation and §Open Challenges, plus the six papers in
§3, from primary text. Confirm every arXiv ID; search-returned IDs are frequently
plausible and wrong. Until then this document is a lead list.

**P1 — Length-scaling / tenure study.** Sweep timeline length (say 1×, 4×, 12×, 24×)
against Memgine, `state_based`, and `transcript_replay`, on gpt-5.6-sol. Directly tests
whether "stronger models need less context curation" is a general claim or an artifact
of short timelines, and whether a tenure crossover exists in our rankings. We own the
generator, so this is a generation-parameter change and a `budget-sweep`-shaped run,
not new machinery. **This one can retract or qualify a published claim, which makes it
the most important experiment on the list.**

**P1 — `premise_resistance` track.** Queries that falsely presuppose superseded state;
correct answers must reject the premise, which necessarily names the stale fact. Scored
under v2.0 negation-aware rules. Doubles as the second empirical section of
`paper-measurement-validity` — a question type designed so that the naive instrument
scores every correct answer as a failure.

**P2 — Related-work sections.** All five papers currently read as if the field is
empty. It isn't, as of about March 2026. Position SFRR against FAMA, StateBench
against STALE and GateMem, Memgine against *Don't Ask the LLM to Track Freshness*
(as corroboration, not competition).

**P2 — External validity runs.** Memgine on GateMem, and on MemoryAgentBench
`FactConsolidation`. Two benchmarks we did not build, testing the two claims we care
most about. Cheap relative to their evidentiary value.

**P2 — Generator fidelity verifier.** A post-render pass asserting every planted fact
is present and every superseded fact is superseded in the rendered text. Closes a hole
`paper-measurement-validity` did not look at: the audit checked the *scorer*, not the
*generator*. If facts silently fail to land, every downstream number is off and nothing
currently would tell us.

**P3 — Forgetting & retention track.** Explicit deletion requests, with the requirement
that deleted facts are unrecoverable rather than tombstoned. Distinct from supersession;
GateMem finds it unsolved across the board; it maps to a survey operation we do not
implement at all.

**P3 — Frame paper 4 as procedural memory.** Retitle or re-abstract *Worked Examples
Are Worth Paying For* in the survey's terms: episodic-to-procedural consolidation via
experience distillation. Same content, findable by anyone reading the survey.

---

## 6. Verification checklist

Before any of this reaches a paper:

- [ ] Survey primary text read — confirm the three-dimensional taxonomy wording, the
      five memory operations, the enumeration of the six open challenges, and whether
      v4's *self-evolving / long-horizon* retitling reflects new sections.
- [ ] Confirm all arXiv IDs in §3.1 resolve to the titles given.
- [ ] Supersede: confirm 92→77, 68→28, 28→28, and the McNemar p-value, from the paper.
- [ ] *Don't Ask the LLM to Track Freshness*: confirm the 54/48/18/7 table and that the
      task grants explicit serial-number semantics.
- [ ] GateMem: confirm the "no method achieves all three" claim and check whether a
      Memgine-shaped baseline was already evaluated.
- [ ] STALE: confirm the three probing dimensions and the 400-scenario / 1,200-query
      scale.
- [ ] *Ground Truth First*: confirm what "tenure crossover" actually measures — this
      is the load-bearing citation for P1 and it is currently **[single]**.
- [ ] FAMA: find the defining paper (Memora?) and read its formulation before comparing
      it to SFRR.

---

## 7. Bottom line

The survey itself is a map, not a result — its value to us is vocabulary, positioning,
and the citation graph it exposes. The real finding of this review is what sits in that
graph: **between February and August 2026 the field independently converged on
StateBench's thesis**, produced at least four instruments overlapping our tracks, and
externally replicated Memgine's core design claim on a benchmark we did not build.

That is validating and it is a deadline. Our differentiators — multi-principal
governance, deterministic engine-level enforcement, generate-state-then-render — are
real and are now contested territory. Two specific exposures need attention: the
"stronger models need less curation" claim rests on a single timeline length, and
tenure-crossover evidence suggests that is exactly the variable that decides it; and
our strongest architectural claim (engine-level access control) has never been tested
on data we did not author, while a benchmark purpose-built to test it now exists and
reports that nobody passes.
