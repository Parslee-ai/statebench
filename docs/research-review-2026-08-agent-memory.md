# Research Review: Agent Memory, August 2026

A review of **Huang et al., *A Survey of Agent Memory in the Second Half: Towards
Self-Evolving and Long-Horizon Agents***, TMLR 07/2026 (arXiv:2602.06052 v4; v1–v3
titled *Rethinking Memory Mechanisms of Foundation Agents in the Second Half*), read
against StateBench's tracks, metrics, and claims.

Read from the primary PDF: 90 pages, ~60 authors across 27 institutions, TMLR
camera-ready. Section and table references below are to that document. §8 is the only
section not read closely.

> Provenance note. §7 of this document lists adjacent 2026 work found via web search
> and **not** read in the original — that section is flagged, and nothing in it may be
> cited until source-checked. Everything in §1–§6 comes from the survey PDF itself.

---

## 1. What the survey actually is

**Thesis.** AI has moved "from prioritizing model innovations and benchmark scores
towards emphasizing problem definition and rigorous real-world evaluation." In the
"second half," the central challenge is *real utility in long-horizon, dynamic, and
user-dependent settings such as agentic coding, deep research, and computer use*,
where agents face "context explosion beyond fixed context windows." Memory is framed
not as passive storage but as **the substrate through which agents self-evolve**.

**Structure.**

| § | Content |
|---|---|
| 3 | Taxonomy: substrates (§3.1), cognitive mechanisms (§3.2), subjects (§3.3) |
| 4 | Operations: five single-agent ops (§4.1), multi-agent architecture / routing / isolation (§4.2) |
| 5 | Learning policies over memory: prompting, fine-tuning, RL |
| 6 | Context-limited vs context-exploded environments |
| 7 | Evaluation: metrics (Table 3), user-centric benchmarks (Table 4), agent-centric (Table 5) |
| 8 | Applications by domain |
| 9 | Six open challenges (9.1–9.6) |

**The three dimensions.** Substrate (internal parametric / external non-parametric /
mixed); cognitive mechanism (sensory, working, episodic, semantic, procedural);
subject (user-centric personalization / agent-centric experience).

**Five single-agent operations** (Figure 6): storage & index; loading & retrieval;
update & refresh; compression & summarization; forgetting & retention.

**Six open challenges** (§9, Figure 9): continual learning and self-evolving agents;
multi-human-agent memory organization; memory infrastructure and efficiency; lifelong
personalization and trustworthy memory; multimodal/embodied/world-model memory;
real-world benchmarking and evaluation.

For our purposes the survey is a **map, not a result**. Its value is vocabulary,
positioning, and one section — §9.6 — that reads like a specification for StateBench.

---

## 2. The headline finding: our central term does not appear in the field's survey

Word counts across all 90 pages:

| Term | Occurrences |
|---|---|
| "supersed*" (supersession, superseded, supersede) | **0** |
| "authority" | **0** |
| "provenance" | 10 |
| "conflict" | 25 |
| "governance" | 5 |
| "access control" | 3 |
| "stale" | 3 |

Zero. In a 90-page, 200+-paper synthesis of exactly our problem area, the word
StateBench is built around never occurs. The field's term for the same operation is
**Update & Refresh (UR)**, one of ten defined user-centric memory abilities (§7.2.1),
and §4.1.3 "Update and Refresh" describes precisely what we call supersession:
"previously stored memory may become incomplete, outdated, or misaligned with newly
observed information, making static or append-only memory representations
insufficient."

Two readings, both true. **We are invisible to keyword search from the field's
mainstream.** Nobody surveying agent memory will find StateBench by searching their
own vocabulary. And **we are not a rediscovery** — the survey's UR is a one-line
ability tag inside a benchmark table; our whole instrument decomposes what they treat
as a single checkbox.

"Authority" scoring zero is sharper still. The survey's multi-agent treatment (§4.2.3)
covers *isolation* and *conflict* between peer agents, and §9.2 raises "questions of
ownership, access, responsibility." But **the idea that memory writes carry differing
authority, and that a lower-authority write must not override a higher-authority
policy, does not appear anywhere.** Our `authority_hierarchy` track has no counterpart
in the field's survey of the field.

**Action.** Keep our vocabulary — it is more precise — but gloss it against theirs on
first use in every paper, and put "update and refresh," "memory conflict," and
"knowledge update" in the abstracts and keywords. This is a pure-upside edit.

---

## 3. Where StateBench sits, by the survey's own axes

| Axis | StateBench / Memgine |
|---|---|
| Substrate (§3.1) | **External**, non-parametric. We test no parametric-memory baseline at all — the survey devotes substantial space to internal/parametric and mixed substrates, so this is a stated scope, not an oversight to hide. |
| Cognitive mechanism (§3.2) | **Semantic** (facts, constraints, policies) and **episodic** (commitments, corrections, interruption/resumption). No sensory, no working, no procedural. |
| Subject (§3.3) | **Mixed and multi-principal.** The survey notes user-centric and agent-centric "are conceptual orientations rather than mutually exclusive," and that "current benchmarks rarely require both simultaneously." Ours requires both, plus multiple *human* principals with distinct roles. |
| Operations (§4.1) | Memgine covers storage & index, loading & retrieval (query-relevance sorting), update & refresh (supersession, adaptive inline repair), compression & summarization (Summary DAG compaction). **Forgetting & retention: absent.** |

### 3.1 Map onto the survey's ten user-centric abilities

§7.2.1 defines ten abilities. This is the best available instrument for stating what
StateBench covers, in language the field already uses:

| Ability | StateBench coverage |
|---|---|
| **UR** — Update & Refresh: "explicitly revising memory when new evidence contradicts old content (overwriting outdated facts and following the latest state under conflicts)" | **Our core.** `supersession`, `supersession_detection`, `repair_propagation`. We decompose UR into detection, handling, and downstream propagation; the survey treats it as one tag. |
| **TR** — Temporal Reasoning, incl. "selecting the correct state when information changes" | `environmental_freshness`, and the temporal half of supersession. |
| **AB** — Abstain & Boundary Handling: "recognizing unknown or unanswerable cases, conflicts, **or false premises** and avoiding fabrication" | `hallucination_resistance` is an AB instrument. **False premises we do not test** — see §5. |
| **FR** — Forgetting & Retention | **Not covered.** We supersede (tombstone); we never delete. |
| **CS** — Compression & Summarization | Only as a baseline mechanism (`rolling_summary`), never as a scored ability. |
| FE, MR, UP, AS, IC | Partial and incidental; not what we are for. |
| *(none)* | **Scope/permission, authority hierarchy, privacy containment — four of our fourteen tracks map to no ability in their list at all.** |

Two observations from the survey's own analysis of Table 4:

- "**CS and FR remain comparatively under-evaluated**... selective forgetting or
  retention is frequently partial or absent, despite being essential for long-horizon
  assistants operating under finite memory budgets and evolving user states."
- "**AB is also inconsistently required.** Only a few benchmarks explicitly reward
  abstention under missing evidence, leaving a gap for evaluating safe memory behavior
  that prevents confident hallucinations."

Our `hallucination_resistance` track sits squarely in the second gap. Worth saying so.

*Caveat on Table 4:* the table distinguishes fully-covered from partially-covered with
two different check glyphs, and **that distinction does not survive text extraction** —
both render identically. Do not quote per-benchmark coverage from this review; read the
rendered table.

---

## 4. §9.6 is, effectively, StateBench's problem statement

The Real-World Benchmarking challenge (§9.6) is worth quoting at length because it
names our failure taxonomy almost item for item. On user-centric benchmarks:

> Benchmarks such as LoCoMo emphasize long-context retrieval accuracy, yet implicitly
> assume stationary user intent and unambiguous ground truth, **overlooking critical
> failure modes such as stale preference reuse, incorrect overwriting of long-term user
> state, or unsafe retention of sensitive information.** Even PersonaMem, which
> explicitly targets evolving preferences, evaluates them over fully simulated sessions
> and **does not assess memory update or refresh**.

Stale reuse → our `supersession`. Incorrect overwriting → `commitment_durability`,
`repair_propagation`. Unsafe retention of sensitive information → `enterprise_privacy`,
`scope_permission`.

On agent-centric benchmarks:

> agents may optimize for short-horizon success while silently failing at
> memory-critical competencies such as **provenance tracking, contradiction resolution,
> and long-term policy consistency**.

Contradiction resolution → supersession. Long-term policy consistency →
`authority_hierarchy`. Provenance tracking → our `Source` objects in `ContextResult`.

And on what to build:

> Execution-based frameworks like OSWorld can be extended with **memory-sensitive
> invariants, requiring agents to version, audit, and roll back persistent state, and
> to attach provenance metadata to stored knowledge.** [...] benchmarks should
> explicitly quantify **resource–utility trade-offs, measuring memory quality as a
> function of token budget**, storage cost, and latency.

That last clause is `statebench budget-sweep`, which already exists and which we have
never made a headline result. The survey asks for exactly it.

**This is the single most valuable paragraph in the survey for us.** It is a
top-venue, 60-author, 27-institution statement that the failure modes StateBench
measures are unmeasured by the field's standard benchmarks — written by people who
have never heard of us. Every StateBench paper's introduction should cite §9.6.

The equivalent endorsement for Memgine's architecture is in §6.2:

> Externalized memory enables **schema-aware retrieval, versioning, targeted edits, and
> access control — operations that are difficult or infeasible within prompt-based
> context alone.**

That is Memgine's engine-level thesis, in the survey's words, cited to third parties.

---

## 5. Six things to do

### P1 — Add a `premise_resistance` track

The survey's **AB** ability explicitly includes "**false premises**," and the survey
says only a few benchmarks reward abstention at all. We test fabrication
(`hallucination_resistance`) but never the harder case: a query that *presupposes*
superseded state. "Since you're back in Portland, should I ship there?"

This is the highest-value single addition, and it lands on a defect we have already
documented. To reject a false premise, a correct answer **must name the stale fact in
order to refuse it** — "You're not in Portland; your address is 456 Oak Ave." Under
phrase-list `must_not_mention` scoring that is a resurrection. This is precisely the
finding in `paper-measurement-validity` (100% of correct rejecting answers scored as
violations), arrived at from the opposite direction: the field has defined a question
type our v1 instrument structurally cannot score, and our v2.0 negation-aware scoring
is what makes it scoreable.

So the track is simultaneously (a) coverage of a named field ability we lack, (b) a
second empirical section for the measurement-validity paper, and (c) a demonstration
that the v2.0 correction was load-bearing rather than cosmetic.

**Status: built.** `premise_resistance` and its `premise_maintain` guardrail twin ship
in `generator/templates/premise.py`, with paired generation, `superseded`-tagged
forbidden phrases, and `evaluation/premise_metrics.py` reporting PRR, False Rejection
Rate, and the v1.0-vs-v2.0 instrument comparison. 47 tests, no model calls. What
remains is running it — see the P0 note about API access.

### P1 — Report dependency distance, and sweep it

§7.2.2 closes by naming two dimensions "crucial" for memory-centric analysis:

> (1) **dependency distance** — how far apart the required information and its later
> use occur, such as within-turn, cross-turn, or cross-session, and (2) **memory
> correctness under interaction** — whether stored items remain faithful,
> non-contradictory, and policy-consistent as the environment evolves.

We are a pure (2) instrument and we do not report (1) at all. Every StateBench result
is quoted at a single, short timeline length.

This matters beyond taxonomy. Our published claim from the gpt-5.6-sol refresh —
"stronger models need less context curation," with Memgine's dev-split lead over
`state_based` collapsing 9.1pp → 0.7pp — is measured at one dependency distance. If
the architecture premium is a function of distance, that sentence is a special case
stated as a general one. Since we own the generator, sweeping timeline length is a
generation-parameter change plus a `budget-sweep`-shaped run.

**This is the experiment most likely to qualify a published claim, which makes it the
most important one on the list.** Report every future result as a curve over
dependency distance, not a scalar.

**Status: harness built, not yet run.** `generator/distance.py` defines four profiles
(`within_turn` → `long_horizon`, roughly 7 → 200 events per timeline); `statebench
generate --distance` and `statebench distance-sweep` drive them, and every timeline now
records the profile it was padded to in a `dependency_distance` field, so a padded
dataset can never be mistaken for an unpadded one. Padding inserts conversation only,
is checked against every ground-truth phrase with the judge's own matcher so it cannot
manufacture must-mention hits or must-not-mention violations, and holds the clock still
on time-sensitive tracks rather than silently rewriting their expiry ground truth. 30
tests, no model calls; `--generate-only` builds and inspects the datasets offline. The
README caveat on the gpt-5.6 claim now states that it was measured at `within_turn`.
Running the sweep needs API access — see P0.

### P2 — Related-work sections in all five papers

**Correction to this item.** It originally said our papers "read as though the field is
empty." Having since read them, that was wrong and unfair. `paper-measurement-validity`
§2 covers LoCoMo, LongMemEval and MemDelta with a scoring-family analysis; paper 3 §2
covers MemHarness, the Always-On Agents survey, MemGPT, Zep/Graphiti and Mem0; paper 4
§2 covers instruction retrieval, step-by-step distillation and ReliableEval. These are
substantive sections.

The real gap was narrower and entirely citable: **no paper cited the field's own
survey**, and none positioned our metrics against their nearest published relatives. The
citation spine is §4.1.3 for update/refresh, §7.2.1 for the ability taxonomy, §7.2.2 for
dependency distance, §9.6 for the evaluation gap, §6.2 for externalized memory.

Correct one thing while doing it: **the survey's memory-integrity metrics are Memory
Integrity (MI) and False Memory Rate (FMR), from HaluMem** (Table 3) — not the "FAMA"
metric an earlier draft of this review attributed to the field. FMR — "rate of
introducing hallucinated memories, including fabricated or incorrect updates" — is the
closest published relative of SFRR, and MI is the closest relative of Must-Mention.
Position SFRR against **FMR**, and note the difference: FMR scores the *memory store*,
SFRR scores the *response*. Both are needed; they fail differently.

**Status: done for the three drafts.** `paper-measurement-validity` §2 gains the survey's
three-family metric taxonomy, a HaluMem row in its scoring-family table, and an explicit
SFRR-vs-FMR positioning (store-level metrics are immune to the negation defect; response-
level metrics measure something store-level ones cannot; reporting either alone leaves a
system unfalsified in one of the two places it can fail). Paper 3 §2 gains the survey's
statement that memory architecture determines "whether systemic issues such as
information leakage may arise" — the architectural claim it tests — plus the observation
that governance is filed under *future directions* there, and that *authority* appears
nowhere in 90 pages. Paper 4 §2 gains the procedural-memory framing and the survey's
independent statement of its result. All three PDFs rebuilt.

**Deliberately not done for the two published papers.** `build_papers.py` treats
`memgine-...pdf` and `state-based-...pdf` as historical artifacts that are never
re-rendered — for the 2025 paper no markdown source exists at all. Editing their
markdown would produce a version of a published paper saying things the published PDF
does not. The Memgine positioning went into `docs/MEMGINE.md` instead: the four of five
survey operations it implements, the forgetting-and-retention gap, §6.2 as an external
statement of its architectural thesis, and its placement in the survey's *write-control*
conflict family.

### P2 — Name paper 4 as procedural memory, and cite the survey's own version of its finding

§3.2's procedural-memory subsection describes skills "packaged as composable bundles of
instructions, code, and resources that agents load on demand," citing Anthropic's Agent
Skills, and frames the abstract's "explicit, portable, and shareable agent skills
surfaced through agent harnesses, context engineering, and standardized tool-mediation
protocols."

Then this:

> Empirically, **self-generated skills still underperform human-curated ones**,
> suggesting that fully autonomous skill induction remains an open challenge for
> self-evolving agents (Li et al., 2026).

That is *Worked Examples Are Worth Paying For*'s thesis, stated independently in a
survey that does not know we exist. Reframing paper 4 as **episodic→procedural
consolidation via experience distillation** costs an abstract rewrite and buys
placement in a live debate with an independent corroborating citation.

### P3 — Forgetting & retention track

The one survey operation Memgine does not implement, and one of the two the survey
calls under-evaluated. Distinct from supersession: superseding tombstones a fact,
deletion must make it unrecoverable. Explicit user deletion requests, with the
requirement that the fact is gone from context *and* from the store, are a natural
extension of our scope machinery and land on a named gap.

**Status: built.** `deletion_compliance` and `deletion_maintain`, five paired scenarios
across support/HR/sales/project/procurement, in `generator/templates/deletion.py`. Three
design points were not obvious going in and are worth recording:

1. **A leak here is `restricted`, not `superseded`.** Reusing revoked data is a
   governance failure; counting it in SFRR would re-create the "SFRR means any
   violation" defect the measurement-validity paper exists to fix. It feeds
   `leakage_rate`.
2. **Negation credit had to be switched off.** `all_mentions_negated` is right for
   supersession — naming a dead value to reject it is the rewarded behavior — and wrong
   here, because *"I no longer have card ending 4471 on file"* has displayed the card. A
   new opt-in `MentionRequirement.negation_exempt` handles it; it defaults to `False`
   and bare strings are never exempt, so no existing release changes. This is a real
   refinement of the v2.0 rule and belongs in the measurement-validity paper: negation
   credit is a property of *what the phrase is*, not of the scorer.
3. **The revoked value stays in the transcript.** The user had to name it to ask for its
   removal, so a replay baseline can still see it and only a system that honors the
   revocation withholds it — the same shape as `scope_leak`.

49 tests, no model calls, and the track audits clean under `statebench audit-dataset`
(zero errors, zero warnings). Memgine still does not implement the operation; that gap
is now documented in `docs/MEMGINE.md` rather than merely true.

### P3 — Generator fidelity verifier

§9.6's call for "provenance metadata" and the survey's repeated emphasis on
attribution point at a hole we have not checked: `paper-measurement-validity` audited
the *scorer*, never the *generator*. Nothing currently asserts that every planted fact
actually appears in the rendered timeline text, or that every superseded fact is
actually superseded there. If facts silently fail to land, every downstream number is
wrong and nothing would tell us.

**Status: built, and it found things.** `generator/fidelity.py` plus `statebench
audit-dataset` (non-zero exit on errors, so it can gate a release). Over the shipped
`v1.0/full.jsonl` — 1,400 timelines, 1,667 queries — it reports **107 errors and 3,352
warnings**:

- **2,026 unreachable forbidden phrases.** The phrase appears nowhere in its own
  timeline. Must-not-mention scoring is deterministic with no paraphrase fallback, so
  nothing can ever violate them — yet they sit in the denominator and deflate the
  phrase-level violation rate. SFRR is computed per query, so SFRR is *not* affected;
  `must_not_mention_violation_rate` is.
- **889 unsupported must-mention phrases**, which a correct answer can only reach by
  paraphrase, making the score a function of the judge rather than of the system.
- **105 non-monotonic event sequences**, all on `supersession_detection`: a baseline
  that sorts events by timestamp sees a different timeline than one that uses list
  order.
- **2 unpassable queries.** In `DET-001030` and `DET-001091` the hourly rate "changes"
  from $150 to $150, so `$150` is both the required answer and a forbidden superseded
  value. Every response scores as a resurrection.

**Both error classes are now fixed at the generator** (see the README's Generator
Fixes section): the detection track's two-clock event assembly, the overlapping
paired-value pools behind the unpassable queries, and a red-herring insertion that
did not shift subsequent events. Newly generated data audits with zero errors on
every track and a test holds that line. The shipped `v1.0` files are left alone on
purpose — they are the published record the leaderboard was computed against.

The dominant pattern is *near miss, not absent fact*: the timeline says "Reset MFA for
CEO" while the forbidden list says `"MFA reset"`; it says "They have 500 active users"
while `"500 users"` is required. Same species as the v1.0 forbidden-phrase defects, one
stage earlier in the pipeline — which is a second, independent instance of the
measurement-validity paper's thesis and probably belongs in it.

Two caveats before any of this is cited. The checker's first run produced 145 errors,
of which 124 were **its own** false positives — it did not know that `Write.supersedes`
holds a fact key in some generators and a fact ID in others, nor that
`cf_temporal_validity` invalidates by an expiry embedded in the value text rather than
by a supersession event. Both are fixed and tested. And warnings are warnings: a
must-mention phrase the model is meant to infer rather than quote is legitimate, so the
889 needs sampling before it becomes a number in a paper.

---

## 6. Smaller notes worth keeping

- **§4.2.3 classifies conflict handling into exactly two families**: *write control*
  (Memory-R1's ADD/UPDATE/DELETE/NOOP, where "the memory manager agent is the only
  agent allowed to mutate memory") and *feedback-loop consistency* (EvoMem's iterative
  verifier). **Memgine is a write-control system**, and saying so places it in a named
  family rather than presenting it as sui generis.
- **§4.2.1 states that memory-architecture choices determine "whether systemic issues
  such as information leakage may arise."** Direct support for our claim that scope
  leakage is architectural, not promptable-away.
- **§9.2 raises "memory governance... ownership, access, responsibility, and how
  divergent perspectives or human corrections should be handled"** as an open challenge
  for multi-human-agent settings. This is the closest the survey comes to our
  authority/scope tracks — and it is filed under *future work*, not *existing
  benchmarks*.
- **§9.6 cites InterruptBench** (Zou et al., 2026), which augments WebArena-Lite with
  "mid-task additions, revisions, and retractions of the user's goal," reporting these
  remain hard for strong backbones. That is our `interruption_resumption` and
  `commitment_durability` in an execution-grounded setting — the nearest published
  neighbor to two of our tracks, and a natural comparison point.
- **§7.2.2's four memory-sensitive measurements** are worth adopting as a reporting
  checklist: retrieval faithfulness/coverage; error modes in state tracking (drift,
  omission, contradiction); **persistence under interruptions (resume after long
  gaps)**; efficiency trade-offs. We have the third as a track and the fourth as a CLI;
  we report neither prominently.
- **§8, Legal & Consulting** anticipates "verifiable memory architectures that link
  every retrieved insight back to a cryptographically signed source document." Our
  `Source` provenance objects and signed leaderboard submissions are early moves in
  that direction, and the enterprise framing is a ready-made application section.
- **StateBench appears in neither Table 4 nor Table 5.** Neither does any dedicated
  update/refresh instrument — UR is covered only as one ability inside general suites
  (LongMemEval, MemoryAgentBench, HaluMem). The niche is real and currently open.

---

## 7. Adjacent 2026 work — leads only, NOT verified

Found via web search while the PDF was unavailable. **None of these were read in the
original; titles, IDs, and numbers are unverified and must not be cited until
source-checked.** They do not appear in the survey's reference list as far as I
checked, so several likely postdate it.

| Work | arXiv (unverified) | Why it matters to us |
|---|---|---|
| STALE — *Can LLM Agents Know When Their Memories Are No Longer Valid?* | 2605.06527 | Reported to probe State Resolution / **Premise Resistance** / Implicit Policy Adaptation — the closest published thing to our track list, and the origin of the §5 premise-resistance idea |
| Supersede — *Diagnosing and Training the Memory-Update Gap* | 2606.27472 | Reported: bounded memory drops knowledge-update accuracy 92%→77%; gap persists across model scale; degrades with conversation length. Bears directly on our "stronger models need less curation" claim |
| GateMem — *Memory Governance in Multi-Principal Shared-Memory Agents* | 2606.18829 | Reported to jointly test utility + access control + deletion, and to find no method achieves all three. Would be an **external** test of Memgine's access-control claim on data we did not author |
| *Don't Ask the LLM to Track Freshness* | 2606.01435 | Reported to show deterministic version-aware aggregation beats LLM-mediated conflict resolution — Memgine's thesis, replicated elsewhere |
| *Ground Truth First* (tenure crossover) | 2607.21962 | Reported to find architecture rankings flip with interaction length — the mechanism behind the P1 dependency-distance sweep |
| MemoryArena | 2602.16313 | Reported benchmark→deployment gap: ~95% on LoCoMo → 40–60% in agentic loops |

Verification order: STALE and Supersede first (they touch published claims), then
GateMem.

---

## 8. Bottom line

The survey does not contain a result that changes what StateBench is. It contains
something more useful: **a 60-author, 27-institution statement, in §9.6, that the exact
failure modes StateBench measures are not measured by the field's standard
benchmarks** — stale reuse, incorrect overwriting, unsafe retention, contradiction
resolution, long-term policy consistency — together with a call for memory-sensitive
invariants over versioned, audited, provenance-carrying state under measured token
budgets. That is our product specification, written by strangers, and it should anchor
every introduction we write from here.

Two gaps of ours are named by the survey's own analysis: forgetting/retention, which we
do not implement, and abstention/false-premise handling, which we half-implement. Two
of our differentiators — authority hierarchy and multi-principal scope — have no
counterpart in the survey at all, appearing only as §9.2 future work.

The one live risk is §5's second item. "Stronger models need less context curation" is
published, and it rests on a single dependency distance in a survey that names
dependency distance a crucial analysis dimension. Sweep it before someone else does.
