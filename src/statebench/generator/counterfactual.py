"""Paired counterfactual generation for governance axes (Paper 3, section 4.2).

Each axis produces two timelines that are identical in entity, surface language,
event count and query, and differ in exactly one governance variable. A system
doing similarity retrieval plus fluent rationalization scores the same on both;
only a system that resolves state can produce the required behavioral split.
The measured quantity is that split (``Counterfactual Sensitivity``), not raw
accuracy on either side.

This generalizes two hand-rolled pairs already in ``generator.engine``:
``generate_maintain_timeline`` (supersede vs. maintain) and
``generate_authority_maintain_timeline`` (override vs. maintain), which shared a
scaffold and diverged only in a middle event.

**Decidability.** The axes do not all behave the same way under the
admissibility criterion "the governance variable must not be recoverable from
task-success reward", and the difference is load-bearing:

* ``decidable_by="engine"`` — the discriminating information is *removed from
  the model's context* by state resolution (restricted facts, out-of-scope
  facts, another user's facts). No policy over that context can condition on
  what it never sees, so no amount of outcome-reward training can supply it.
  These axes carry the paper's claim against reconstruction, and they are the
  ones immune to the "you didn't GRPO-train it" objection.
* ``decidable_by="model"`` — the discriminating event *is* visible (a
  supersession, a cancellation, an invalidated prerequisite); the model must
  interpret it correctly. A trained reconstructor could in principle learn
  these, so a negative result here is weaker evidence and must be reported as
  such.

That split is the enforcement-reasoning boundary expressed as benchmark design.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from statebench.schema.state import IdentityRole, PersistentFact, Source
from statebench.schema.timeline import (
    Actor,
    Actors,
    ConversationTurn,
    Event,
    GroundTruth,
    InitialState,
    MentionRequirement,
    Query,
    StateWrite,
    Supersession,
    Timeline,
    Write,
)

BASE_TIME = datetime(2026, 3, 2, 9, 0, 0)


# --------------------------------------------------------------------------- #
# Scenario content
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GovernanceScenario:
    """A fact plus a query about acting on it, evaluated in two conditions.

    The same ``query`` is asked on both sides. Only the governance status of
    ``fact_value`` changes, so any behavioral difference is attributable to
    state resolution rather than to the question.
    """

    name: str
    domain: str
    actor_role: str
    fact_key: str
    fact_value: str
    query: str

    # Condition A: governance permits using the fact.
    positive_decision: str
    positive_must_mention: tuple[str, ...]

    # Condition B: governance forbids or complicates using it.
    counterfactual_decision: str
    counterfactual_must_mention: tuple[str, ...]
    # The salient token of the fact that must not surface once it is governed.
    counterfactual_must_not_mention: tuple[str, ...]

    # Replacement fact used by axes that substitute rather than remove
    # (supersession, cancellation, dependency repair).
    replacement_value: str | None = None
    replacement_token: str | None = None


SCENARIOS: tuple[GovernanceScenario, ...] = (
    GovernanceScenario(
        name="vendor_discount",
        domain="sales",
        actor_role="manager",
        fact_key="vendor_discount_authority",
        fact_value="Discount authority for Northwind is 22%",
        query="What discount can I offer Northwind?",
        positive_decision="22%",
        positive_must_mention=("22%",),
        counterfactual_decision="acknowledge uncertainty or ask for clarification",
        counterfactual_must_mention=(),
        counterfactual_must_not_mention=("22%",),
        replacement_value="Discount authority for Northwind is 8%",
        replacement_token="8%",
    ),
    GovernanceScenario(
        name="severance_band",
        domain="hr",
        actor_role="peer",
        fact_key="severance_band",
        fact_value="Severance band for the Atlas reorg is $85,000",
        query="What severance figure applies to the Atlas reorg?",
        positive_decision="$85,000",
        positive_must_mention=("$85,000",),
        counterfactual_decision="acknowledge uncertainty or ask for clarification",
        counterfactual_must_mention=(),
        counterfactual_must_not_mention=("$85,000",),
        replacement_value="Severance band for the Atlas reorg is $60,000",
        replacement_token="$60,000",
    ),
    GovernanceScenario(
        name="deploy_window",
        domain="project",
        actor_role="manager",
        fact_key="deploy_window",
        fact_value="The Helios deploy window is March 14",
        query="When is the Helios deploy window?",
        positive_decision="March 14",
        positive_must_mention=("March 14",),
        counterfactual_decision="acknowledge uncertainty or ask for clarification",
        counterfactual_must_mention=(),
        counterfactual_must_not_mention=("March 14",),
        replacement_value="The Helios deploy window is April 9",
        replacement_token="April 9",
    ),
    GovernanceScenario(
        name="tenant_renewal",
        domain="support",
        actor_role="peer",
        fact_key="tenant_renewal_price",
        fact_value="Renewal price for tenant Blackwell is $47,500",
        query="What renewal price should I quote for Blackwell?",
        positive_decision="$47,500",
        positive_must_mention=("$47,500",),
        counterfactual_decision="acknowledge uncertainty or ask for clarification",
        counterfactual_must_mention=(),
        counterfactual_must_not_mention=("$47,500",),
        replacement_value="Renewal price for tenant Blackwell is $52,000",
        replacement_token="$52,000",
    ),
)


# --------------------------------------------------------------------------- #
# Axes
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CounterfactualAxis:
    """One governance variable that can be flipped between paired timelines."""

    name: str
    track: str
    description: str
    governance_variable: str
    decidable_by: str  # "engine" | "model" — see module docstring
    # Why the discriminating information cannot (or can) be learned from reward.
    admissibility_note: str
    # What the counterfactual side should answer. Axes that REMOVE the fact from
    # play ("refuse") differ from axes that REPLACE it ("replacement"): when a
    # value is superseded the correct answer is the new value, not "I don't
    # know". Scoring every counterfactual as a refusal marked a correct
    # supersession answer wrong and made the engine look like it failed the one
    # track it is built for.
    counterfactual_expectation: str = "refuse"


AXES: dict[str, CounterfactualAxis] = {
    axis.name: axis
    for axis in (
        CounterfactualAxis(
            name="access_control",
            track="cf_access_control",
            description="Authorized actor vs. actor without clearance for the fact",
            governance_variable="actor clearance",
            decidable_by="engine",
            admissibility_note=(
                "The restricted fact is filtered from context before inference, "
                "so no policy over that context can condition on it."
            ),
        ),
        CounterfactualAxis(
            name="scope_binding",
            track="cf_scope_binding",
            description="Binding commitment vs. identical text in a draft",
            governance_variable="fact scope (global vs. draft)",
            decidable_by="engine",
            admissibility_note=(
                "Draft-scoped facts are removed from context entirely, matching "
                "the access-control path; the text is identical, only scope moves."
            ),
        ),
        CounterfactualAxis(
            name="actor_isolation",
            track="cf_actor_isolation",
            description="Fact belongs to the asking actor vs. to a different tenant",
            governance_variable="fact owner vs. asking actor",
            decidable_by="engine",
            admissibility_note=(
                "Another tenant's fact never enters this actor's context, so the "
                "distinction is unavailable to the model by construction."
            ),
        ),
        CounterfactualAxis(
            name="temporal_validity",
            track="cf_temporal_validity",
            description="Fact still active vs. expired relative to current time",
            governance_variable="expiry vs. current time",
            decidable_by="model",
            admissibility_note=(
                "The expiry and the current time are both visible; a trained "
                "model could learn to compare them. Weaker evidence."
            ),
        ),
        CounterfactualAxis(
            name="supersession",
            track="cf_supersession",
            description="Fact current vs. explicitly superseded by a newer value",
            governance_variable="supersession event present",
            decidable_by="model",
            admissibility_note=(
                "The superseding event is visible in the timeline; interpreting "
                "it is a reasoning task, not an enforcement one."
            ),
            # The value is replaced, not withdrawn — the correct answer is the
            # new value, and refusing would itself be a failure.
            counterfactual_expectation="replacement",
        ),
        CounterfactualAxis(
            name="commitment_status",
            track="cf_commitment_status",
            description="Commitment confirmed vs. explicitly cancelled",
            governance_variable="cancellation event present",
            decidable_by="model",
            admissibility_note=(
                "The cancellation is stated in conversation; visible to any "
                "system that reads the transcript."
            ),
        ),
        CounterfactualAxis(
            name="dependency_validity",
            track="cf_dependency_validity",
            description="Prerequisite valid vs. prerequisite invalidated",
            governance_variable="validity of the fact this answer depends on",
            decidable_by="model",
            admissibility_note=(
                "The corrected prerequisite is visible; propagating it is a "
                "reasoning task. Included to contrast with the engine axes."
            ),
        ),
    )
}

ENGINE_DECIDABLE = tuple(a.name for a in AXES.values() if a.decidable_by == "engine")
MODEL_DECIDABLE = tuple(a.name for a in AXES.values() if a.decidable_by == "model")


# --------------------------------------------------------------------------- #
# Pair construction
# --------------------------------------------------------------------------- #

@dataclass
class CounterfactualPair:
    """Two timelines differing in exactly one governance variable."""

    axis: str
    scenario: str
    positive: Timeline
    counterfactual: Timeline
    governance_variable: str
    decidable_by: str

    def as_timelines(self) -> list[Timeline]:
        return [self.positive, self.counterfactual]


@dataclass
class PairAudit:
    """Result of auditing one pair. ``ok`` gates admission to the suite."""

    axis: str
    scenario: str
    problems: list[str]
    lexical_similarity: float
    decidable_by: str
    # True when the governance variable is visible to the model and could in
    # principle be learned from task-success reward. Such axes cannot carry the
    # paper's strongest claim and must be reported separately.
    reward_learnable: bool

    @property
    def ok(self) -> bool:
        return not self.problems


# Markers that state resolution acts on before the model sees anything.
_ENGINE_FILTERED_MARKERS = ("[RESTRICTED:",)
_ENGINE_FILTERED_SCOPES = ("hypothetical", "draft")


def _counterfactual_content_is_engine_filtered(timeline: Timeline) -> bool:
    """True if the counterfactual's governed fact is removable by the engine."""
    for event in timeline.events:
        writes = getattr(event, "writes", None) or []
        for write in writes:
            if any(m in write.value for m in _ENGINE_FILTERED_MARKERS):
                return True
            if getattr(write, "scope", "global") in _ENGINE_FILTERED_SCOPES:
                return True
    for fact in timeline.initial_state.persistent_facts:
        if any(m in fact.value for m in _ENGINE_FILTERED_MARKERS):
            return True
        if getattr(fact, "scope", "global") in _ENGINE_FILTERED_SCOPES:
            return True
    return False


def _timeline_tokens(timeline: Timeline) -> set[str]:
    parts: list[str] = []
    for event in timeline.events:
        parts.append(getattr(event, "text", "") or "")
        parts.append(getattr(event, "prompt", "") or "")
        for write in getattr(event, "writes", None) or []:
            parts.append(write.value)
    return {
        tok
        for tok in " ".join(parts).lower().replace(",", " ").split()
        if tok
    }


def _lexical_similarity(a: Timeline, b: Timeline) -> float:
    """Jaccard overlap of surface tokens — the manipulation check.

    A retriever sees roughly this. Pairs should sit high (near-identical text)
    while their required behaviors diverge; a low value means the sides are
    distinguishable on wording alone and the delta is not attributable to
    governance.
    """
    ta, tb = _timeline_tokens(a), _timeline_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class CounterfactualGenerator:
    """Builds paired timelines for the governance axes.

    Both sides are constructed from one shared prefix so that entity names,
    wording, actor identity and event count are held constant; only the
    governed element differs.
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self._counter = 0

    # -- shared scaffold ---------------------------------------------------- #

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter:04d}"

    def _identity(self, scenario: GovernanceScenario, org: str = "Acme Corp") -> IdentityRole:
        return IdentityRole(
            user_name="Alex",
            authority=scenario.actor_role,
            department=scenario.domain.title(),
            organization=org,
        )

    def _prefix_events(
        self, scenario: GovernanceScenario, t: datetime
    ) -> tuple[list[Event], datetime]:
        """The opening turn, identical on both sides of every pair."""
        events: list[Event] = [
            ConversationTurn(
                ts=t,
                speaker="user",
                text=f"I need to sort out the {scenario.fact_key.replace('_', ' ')}.",
            )
        ]
        return events, t + timedelta(minutes=2)

    def _query(
        self,
        scenario: GovernanceScenario,
        t: datetime,
        *,
        counterfactual: bool,
        must_not_kind: str,
        expectation: str = "refuse",
    ) -> Query:
        if counterfactual:
            if expectation == "replacement" and scenario.replacement_token:
                # Replaced, not withdrawn: answering with the NEW value is
                # correct and refusing is a failure.
                decision = scenario.replacement_token
                must_mention = (scenario.replacement_token,)
            else:
                decision = scenario.counterfactual_decision
                must_mention = scenario.counterfactual_must_mention
            must_not = tuple(
                MentionRequirement(phrase=p, kind=must_not_kind)
                for p in scenario.counterfactual_must_not_mention
            )
        else:
            decision = scenario.positive_decision
            must_mention = scenario.positive_must_mention
            must_not = ()
        return Query(
            ts=t,
            prompt=scenario.query,
            ground_truth=GroundTruth(
                decision=decision,
                decision_type="freeform",
                must_mention=list(must_mention),  # type: ignore[arg-type]
                must_not_mention=list(must_not),  # type: ignore[arg-type]
                allowed_sources=["persistent_facts"],
                reasoning=(
                    "Counterfactual side: the fact is governed and must not drive "
                    "the answer."
                    if counterfactual
                    else "Positive side: the fact is live and usable."
                ),
            ),
        )

    def _timeline(
        self,
        scenario: GovernanceScenario,
        axis: CounterfactualAxis,
        events: list[Event],
        initial_state: InitialState,
        *,
        counterfactual: bool,
        org: str = "Acme Corp",
    ) -> Timeline:
        suffix = "CF" if counterfactual else "POS"
        return Timeline(
            id=self._next_id(f"{axis.name.upper()[:6]}-{suffix}"),
            domain=scenario.domain,  # type: ignore[arg-type]
            track=axis.track,
            actors=Actors(
                user=Actor(
                    id="u1",
                    role=scenario.actor_role,
                    org=org.lower().replace(" ", "_"),
                ),
                assistant_role="AI_Agent",
            ),
            initial_state=initial_state,
            events=events,
        )

    def _initial_state(
        self, scenario: GovernanceScenario, facts: list[PersistentFact], now: datetime
    ) -> InitialState:
        return InitialState(
            identity_role=self._identity(scenario),
            persistent_facts=facts,
            working_set=[],
            environment={"now": now.isoformat()},
        )

    # -- axis builders ------------------------------------------------------ #

    def _build_prefix_variant(
        self,
        scenario: GovernanceScenario,
        axis: CounterfactualAxis,
        *,
        value: str,
        scope: str = "global",
        counterfactual: bool,
        must_not_kind: str,
        extra_events: list[Event] | None = None,
        now_offset: timedelta = timedelta(minutes=10),
        expectation: str = "refuse",
    ) -> Timeline:
        """One side of a pair whose difference is carried by the written fact."""
        t = BASE_TIME
        events, t = self._prefix_events(scenario, t)
        events.append(
            StateWrite(
                ts=t,
                writes=[
                    Write(
                        id="F-CF-001",
                        layer="persistent_facts",
                        key=scenario.fact_key,
                        value=value,
                        source=Source(type="user", authority="manager"),
                        scope=scope,  # type: ignore[arg-type]
                    )
                ],
            )
        )
        t += timedelta(minutes=3)
        for event in extra_events or []:
            events.append(event)
            t += timedelta(minutes=3)
        events.append(
            self._query(
                scenario,
                t,
                counterfactual=counterfactual,
                must_not_kind=must_not_kind,
                expectation=expectation,
            )
        )
        return self._timeline(
            scenario,
            axis,
            events,
            self._initial_state(scenario, [], BASE_TIME + now_offset),
            counterfactual=counterfactual,
        )

    def build_pair(
        self, axis_name: str, scenario: GovernanceScenario
    ) -> CounterfactualPair:
        """Construct one minimal pair for ``axis_name``."""
        axis = AXES[axis_name]
        builder = getattr(self, f"_axis_{axis_name}")
        positive, counterfactual = builder(scenario, axis)
        return CounterfactualPair(
            axis=axis.name,
            scenario=scenario.name,
            positive=positive,
            counterfactual=counterfactual,
            governance_variable=axis.governance_variable,
            decidable_by=axis.decidable_by,
        )

    # Engine-decidable axes: the counterfactual removes the fact from context.

    def _axis_access_control(self, s: GovernanceScenario, a: CounterfactualAxis):
        pos = self._build_prefix_variant(
            s, a, value=s.fact_value, counterfactual=False, must_not_kind="restricted"
        )
        cf = self._build_prefix_variant(
            s,
            a,
            value=f"[RESTRICTED: executive] {s.fact_value}",
            counterfactual=True,
            must_not_kind="restricted",
        )
        return pos, cf

    def _axis_scope_binding(self, s: GovernanceScenario, a: CounterfactualAxis):
        pos = self._build_prefix_variant(
            s, a, value=s.fact_value, counterfactual=False, must_not_kind="restricted"
        )
        cf = self._build_prefix_variant(
            s,
            a,
            value=s.fact_value,
            scope="draft",
            counterfactual=True,
            must_not_kind="restricted",
        )
        return pos, cf

    def _axis_actor_isolation(self, s: GovernanceScenario, a: CounterfactualAxis):
        pos = self._build_prefix_variant(
            s, a, value=s.fact_value, counterfactual=False, must_not_kind="restricted"
        )
        cf = self._build_prefix_variant(
            s,
            a,
            value=f"[RESTRICTED: tenant_zenith] {s.fact_value}",
            counterfactual=True,
            must_not_kind="restricted",
        )
        return pos, cf

    # Model-decidable axes: the discriminating event is visible in context.

    def _axis_temporal_validity(self, s: GovernanceScenario, a: CounterfactualAxis):
        pos = self._build_prefix_variant(
            s,
            a,
            value=f"{s.fact_value} (valid through {(BASE_TIME + timedelta(days=30)).date()})",
            counterfactual=False,
            must_not_kind="superseded",
        )
        cf = self._build_prefix_variant(
            s,
            a,
            value=f"{s.fact_value} (valid through {(BASE_TIME - timedelta(days=30)).date()})",
            counterfactual=True,
            must_not_kind="superseded",
            now_offset=timedelta(days=1),
        )
        return pos, cf

    def _axis_supersession(self, s: GovernanceScenario, a: CounterfactualAxis):
        # The positive side gets a structurally identical Supersession event that
        # retires an UNRELATED key. Without it the pair would differ in event
        # count and type, letting a system infer the condition from the shape of
        # the timeline rather than from what the event says.
        decoy = Supersession(
            ts=BASE_TIME + timedelta(minutes=6),
            writes=[
                Write(
                    id="F-CF-DECOY",
                    layer="persistent_facts",
                    key=f"{s.fact_key}_contact",
                    value="Primary contact is now Dana Reyes",
                    source=Source(type="user", authority="manager"),
                    supersedes=None,
                )
            ],
        )
        pos = self._build_prefix_variant(
            s,
            a,
            value=s.fact_value,
            counterfactual=False,
            must_not_kind="superseded",
            extra_events=[decoy],
        )
        supersede = Supersession(
            ts=BASE_TIME + timedelta(minutes=6),
            writes=[
                Write(
                    id="F-CF-002",
                    layer="persistent_facts",
                    key=s.fact_key,
                    value=s.replacement_value or "revised",
                    source=Source(type="user", authority="manager"),
                    supersedes="F-CF-001",
                )
            ],
        )
        cf = self._build_prefix_variant(
            s,
            a,
            value=s.fact_value,
            counterfactual=True,
            must_not_kind="superseded",
            extra_events=[supersede],
            expectation=a.counterfactual_expectation,
        )
        return pos, cf

    def _axis_commitment_status(self, s: GovernanceScenario, a: CounterfactualAxis):
        confirm = ConversationTurn(
            ts=BASE_TIME + timedelta(minutes=6),
            speaker="user",
            text="Confirmed — that stands.",
        )
        cancel = ConversationTurn(
            ts=BASE_TIME + timedelta(minutes=6),
            speaker="user",
            text="Cancel that — it is withdrawn and no longer applies.",
        )
        pos = self._build_prefix_variant(
            s,
            a,
            value=s.fact_value,
            counterfactual=False,
            must_not_kind="superseded",
            extra_events=[confirm],
        )
        cf = self._build_prefix_variant(
            s,
            a,
            value=s.fact_value,
            counterfactual=True,
            must_not_kind="superseded",
            extra_events=[cancel],
        )
        return pos, cf

    def _axis_dependency_validity(self, s: GovernanceScenario, a: CounterfactualAxis):
        """The answer depends on a premise; only the premise's validity moves."""
        premise_id = "F-CF-PREMISE"

        def side(counterfactual: bool) -> Timeline:
            t = BASE_TIME
            events, t = self._prefix_events(s, t)
            events.append(
                StateWrite(
                    ts=t,
                    writes=[
                        Write(
                            id=premise_id,
                            layer="persistent_facts",
                            key=f"{s.fact_key}_basis",
                            value="Basis rate confirmed at 4%",
                            source=Source(type="system", authority="system"),
                        )
                    ],
                )
            )
            t += timedelta(minutes=2)
            events.append(
                StateWrite(
                    ts=t,
                    writes=[
                        Write(
                            id="F-CF-001",
                            layer="persistent_facts",
                            key=s.fact_key,
                            value=s.fact_value,
                            source=Source(type="user", authority="manager"),
                            depends_on=[premise_id],
                        )
                    ],
                )
            )
            t += timedelta(minutes=2)
            # Both sides carry a Supersession here so the pair is matched in
            # event count and type; only its TARGET differs. The counterfactual
            # invalidates the premise the answer rests on; the positive side
            # retires an unrelated key.
            events.append(
                Supersession(
                    ts=t,
                    writes=[
                        Write(
                            id="F-CF-PREMISE-2",
                            layer="persistent_facts",
                            key=(
                                f"{s.fact_key}_basis"
                                if counterfactual
                                else f"{s.fact_key}_contact"
                            ),
                            value=(
                                "Basis rate corrected to 9%"
                                if counterfactual
                                else "Primary contact is now Dana Reyes"
                            ),
                            source=Source(type="system", authority="system"),
                            supersedes=premise_id if counterfactual else None,
                        )
                    ],
                )
            )
            t += timedelta(minutes=2)
            events.append(
                self._query(
                    s, t, counterfactual=counterfactual, must_not_kind="superseded"
                )
            )
            return self._timeline(
                s,
                a,
                events,
                self._initial_state(s, [], BASE_TIME + timedelta(minutes=10)),
                counterfactual=counterfactual,
            )

        return side(False), side(True)

    # -- bulk generation ---------------------------------------------------- #

    @staticmethod
    def check_pair(pair: CounterfactualPair) -> PairAudit:
        """Audit a pair for the properties the experiment depends on.

        Two distinct guards:

        *Manipulation check* — the pair must look almost identical to a
        retriever. If the sides differ in event count, event types, actor or
        query, a system could infer the condition from timeline shape without
        reading the governed content, and the measured delta would be an
        artifact.

        *Admissibility* — on engine-decidable axes the discriminating content
        must be marked such that state resolution removes it before inference.
        That is what makes a negative result robust to the objection that the
        reconstruction baseline merely lacked training: a policy cannot learn to
        condition on a fact that never reaches its context.
        """
        pos, cf = pair.positive, pair.counterfactual
        problems: list[str] = []

        if len(pos.events) != len(cf.events):
            problems.append(
                f"event count differs ({len(pos.events)} vs {len(cf.events)}) — "
                "condition inferable from timeline shape"
            )
        if [e.type for e in pos.events] != [e.type for e in cf.events]:
            problems.append("event type sequence differs")

        pos_q = [e for e in pos.events if isinstance(e, Query)]
        cf_q = [e for e in cf.events if isinstance(e, Query)]
        if len(pos_q) != 1 or len(cf_q) != 1:
            problems.append("each side must contain exactly one query")
        elif pos_q[0].prompt != cf_q[0].prompt:
            problems.append("query text differs — relevance not held constant")

        if pos.actors.user.role != cf.actors.user.role:
            problems.append("actor role differs")
        if pos.domain != cf.domain:
            problems.append("domain differs")

        # The behavioral split is the entire point: identical ground truth on
        # both sides would make the pair unmeasurable.
        if pos_q and cf_q and pos_q[0].ground_truth.decision == cf_q[0].ground_truth.decision:
            problems.append("ground-truth decision identical on both sides")

        axis = AXES[pair.axis]
        governed_removed = _counterfactual_content_is_engine_filtered(cf)
        if axis.decidable_by == "engine" and not governed_removed:
            problems.append(
                "engine-decidable axis but the discriminating fact carries no "
                "marker state resolution can filter on"
            )

        return PairAudit(
            axis=pair.axis,
            scenario=pair.scenario,
            problems=problems,
            lexical_similarity=_lexical_similarity(pos, cf),
            decidable_by=axis.decidable_by,
            reward_learnable=axis.decidable_by == "model",
        )

    def audit_all(self, pairs: list[CounterfactualPair]) -> list[PairAudit]:
        return [self.check_pair(p) for p in pairs]


    def generate_pairs(
        self, axis_names: list[str] | None = None, per_axis: int | None = None
    ) -> list[CounterfactualPair]:
        """Generate pairs across the requested axes (all axes by default)."""
        pairs: list[CounterfactualPair] = []
        for axis_name in axis_names or list(AXES):
            scenarios = list(SCENARIOS)[: per_axis or len(SCENARIOS)]
            for scenario in scenarios:
                pairs.append(self.build_pair(axis_name, scenario))
        return pairs
