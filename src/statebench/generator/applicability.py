"""The applicability suite: governance status × experience applicability (§4.1).

This is the factorial that separates the two things the literature collapses.
One dimension is whether the *state* may be used at all; the other is whether
the *experience* transfers. They are independent, and a system can pass either
alone while failing their combination.

    governance:   valid | superseded | expired | unauthorized | outranked | draft
    experience:   applicable | adaptable | unusable

Reading the matrix:

* **The diagonal** (valid × {applicable, adaptable, unusable}) is MemHarness's
  territory: governance permits the fact, and the only question is whether the
  past experience still fits. Reconstruction should win here.
* **The off-diagonal** (governed × semantically-relevant) is where a
  reconstruction-only system should fail: the experience looks applicable, and
  nothing in it signals that the underlying state may not be used as-is.
* **The cross-cells** (governed × adaptable) are the paper. The experience
  genuinely does transfer in principle, *and* the state is governed. Handling
  one without the other is a distinct, attributable failure — which is why
  ``expected_applicability`` is scored separately from decision accuracy.

Rejection is a first-class correct outcome here. A system that answers
everything scores badly on REJECT cells; a system that refuses everything
scores badly on KEEP cells and shows up in the false-refusal rate. Neither can
be hidden by a single blended number.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import timedelta
from itertools import product

from statebench.generator.counterfactual import (
    BASE_TIME,
    SCENARIOS,
    GovernanceScenario,
)
from statebench.schema.state import IdentityRole, Source
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

GOVERNANCE_STATUSES = (
    "valid",
    "superseded",
    "expired",
    "unauthorized",
    "outranked",
    "draft",
)
EXPERIENCE_STATUSES = ("applicable", "adaptable", "unusable")


@dataclass(frozen=True)
class Cell:
    """One cell of the factorial and the behavior it requires."""

    governance: str
    experience: str
    expected_applicability: str
    decision: str
    # Which layer must act for this cell to be answered correctly.
    responsible_plane: str
    rationale: str


# Governance statuses divide by what they leave behind. WITHDRAWING statuses
# remove the value from play, so the correct answer is a refusal. SUBSTITUTING
# statuses replace it, so a usable value still exists and the correct answer is
# the NEW one — refusing there is itself a failure, and the experience should be
# ADAPTed (its principle transfers, its value does not) rather than rejected.
#
# Collapsing both into "REJECT + acknowledge uncertainty" marked a correct
# supersession answer wrong, and inflated the suite's REJECT share to 15/18.
WITHDRAWING = ("unauthorized", "draft", "expired")
SUBSTITUTING = ("superseded", "outranked")


def _cell(governance: str, experience: str) -> Cell:
    """Correct behavior for a (governance, experience) combination.

    Governance dominates experience applicability: how well a past experience
    transfers cannot make governed state usable. But dominating is not the same
    as forbidding an answer — see WITHDRAWING vs SUBSTITUTING above.
    """
    if governance in WITHDRAWING:
        return Cell(
            governance=governance,
            experience=experience,
            expected_applicability="REJECT",
            decision="acknowledge uncertainty or ask for clarification",
            responsible_plane="state resolution",
            rationale=(
                f"state is {governance}; no usable value remains, so the "
                "experience must not drive an answer"
            ),
        )
    if governance in SUBSTITUTING:
        return Cell(
            governance=governance,
            experience=experience,
            expected_applicability="ADAPT",
            decision="use the replacement value",
            responsible_plane="state resolution",
            rationale=(
                f"state is {governance}; the old value is dead but a replacement "
                "is live, so the principle transfers and the value is rebound"
            ),
        )
    if experience == "applicable":
        return Cell(
            governance, experience, "KEEP", "use the prior value",
            "retrieval", "state is live and the experience still fits",
        )
    if experience == "adaptable":
        return Cell(
            governance, experience, "ADAPT", "use the current value",
            "reconstruction",
            "principle transfers but the source-state detail has moved",
        )
    return Cell(
        governance, experience, "REJECT",
        "acknowledge uncertainty or ask for clarification",
        "reconstruction", "experience does not transfer to this situation",
    )


MATRIX: tuple[Cell, ...] = tuple(
    _cell(g, e) for g, e in product(GOVERNANCE_STATUSES, EXPERIENCE_STATUSES)
)


class ApplicabilityGenerator:
    """Builds the factorial suite from the shared governance scenarios."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"APPL-{self._counter:04d}"

    def _govern(
        self, scenario: GovernanceScenario, cell: Cell
    ) -> tuple[str, str, list[Event]]:
        """Return (fact_value, scope, extra events) realizing the governance status."""
        value, scope, extra = scenario.fact_value, "global", []
        if cell.governance == "unauthorized":
            value = f"[RESTRICTED: executive] {scenario.fact_value}"
        elif cell.governance == "draft":
            scope = "draft"
        elif cell.governance == "expired":
            value = (
                f"{scenario.fact_value} "
                f"(valid through {(BASE_TIME - timedelta(days=30)).date()})"
            )
        elif cell.governance == "superseded":
            extra = [
                Supersession(
                    ts=BASE_TIME + timedelta(minutes=6),
                    writes=[
                        Write(
                            id="F-APPL-002",
                            layer="persistent_facts",
                            key=scenario.fact_key,
                            value=scenario.replacement_value or "revised",
                            source=Source(type="user", authority="manager"),
                            supersedes="F-APPL-001",
                        )
                    ],
                )
            ]
        elif cell.governance == "outranked":
            extra = [
                StateWrite(
                    ts=BASE_TIME + timedelta(minutes=6),
                    writes=[
                        Write(
                            id="F-APPL-POLICY",
                            layer="persistent_facts",
                            key=scenario.fact_key,
                            value=scenario.replacement_value or "policy limit applies",
                            source=Source(type="policy", authority="policy"),
                        )
                    ],
                )
            ]
        return value, scope, extra

    def _experience_turn(self, scenario: GovernanceScenario, cell: Cell) -> ConversationTurn:
        """The turn that seeds the retrievable past experience.

        Its wording is held constant across experience statuses so that what
        varies is applicability, not retrievability.
        """
        text = {
            "applicable": (
                f"Last time we handled {scenario.fact_key.replace('_', ' ')} "
                "the same way and it worked."
            ),
            "adaptable": (
                f"Last time we handled {scenario.fact_key.replace('_', ' ')} the "
                "same way, though the figure has moved since."
            ),
            "unusable": (
                f"Last time we handled {scenario.fact_key.replace('_', ' ')} the "
                "same way, but that was a different product line entirely."
            ),
        }[cell.experience]
        return ConversationTurn(ts=BASE_TIME + timedelta(minutes=1), speaker="user", text=text)

    def build(self, scenario: GovernanceScenario, cell: Cell) -> Timeline:
        value, scope, extra = self._govern(scenario, cell)
        t = BASE_TIME

        events: list[Event] = [self._experience_turn(scenario, cell)]
        t += timedelta(minutes=2)
        events.append(
            StateWrite(
                ts=t,
                writes=[
                    Write(
                        id="F-APPL-001",
                        layer="persistent_facts",
                        key=scenario.fact_key,
                        value=value,
                        source=Source(type="user", authority="manager"),
                        scope=scope,  # type: ignore[arg-type]
                    )
                ],
            )
        )
        t += timedelta(minutes=2)
        events.extend(extra)
        if extra:
            t += timedelta(minutes=2)

        # On every governed cell the ORIGINAL value must not drive the answer —
        # whether it was withdrawn or replaced. On substituting cells the
        # replacement is what should appear instead.
        forbidden: list[MentionRequirement] = []
        if cell.governance != "valid":
            forbidden = [
                MentionRequirement(
                    phrase=p,
                    kind="restricted"
                    if cell.governance in ("unauthorized", "draft")
                    else "superseded",
                )
                for p in scenario.counterfactual_must_not_mention
            ]

        decision = cell.decision
        must_mention: list[str] = []
        if cell.governance in SUBSTITUTING and scenario.replacement_token:
            decision = scenario.replacement_token
            must_mention = [scenario.replacement_token]
        elif cell.governance == "valid" and cell.expected_applicability != "REJECT":
            decision = scenario.positive_decision
            must_mention = list(scenario.positive_must_mention)

        events.append(
            Query(
                ts=t,
                prompt=scenario.query,
                ground_truth=GroundTruth(
                    decision=decision,
                    decision_type="freeform",
                    must_mention=must_mention,  # type: ignore[arg-type]
                    must_not_mention=list(forbidden),  # type: ignore[arg-type]
                    allowed_sources=["persistent_facts"],
                    expected_applicability=cell.expected_applicability,  # type: ignore[arg-type]
                    transferable_principle=(
                        f"how to handle {scenario.fact_key.replace('_', ' ')}"
                        if cell.expected_applicability != "REJECT"
                        else None
                    ),
                    obsolete_details=list(scenario.counterfactual_must_not_mention)
                    if cell.expected_applicability == "ADAPT"
                    else [],
                    reasoning=cell.rationale,
                ),
            )
        )

        return Timeline(
            id=self._next_id(),
            domain=scenario.domain,  # type: ignore[arg-type]
            track="applicability",
            actors=Actors(
                user=Actor(id="u1", role=scenario.actor_role, org="acme_corp"),
                assistant_role="AI_Agent",
            ),
            initial_state=InitialState(
                identity_role=IdentityRole(
                    user_name="Alex",
                    authority=scenario.actor_role,
                    department=scenario.domain.title(),
                    organization="Acme Corp",
                ),
                persistent_facts=[],
                working_set=[],
                environment={"now": (BASE_TIME + timedelta(days=1)).isoformat()},
            ),
            events=events,
        )

    def generate(self, per_cell: int = 1, valid_oversample: int | None = None) -> list[Timeline]:
        """Generate the suite.

        ``valid_oversample`` draws more scenarios for the governance=valid row.
        That row holds every KEEP and ADAPT cell — the rest of the matrix is
        REJECT, because once state is governed the correct disposition does not
        depend on how well the experience transfers.

        The resulting suite is genuinely REJECT-heavy, and that is a property of
        the domain rather than a sampling flaw: in production most governed
        states really do forbid use. So the imbalance is not papered over.
        Instead the KEEP/ADAPT row is given enough samples to measure, and
        ``ApplicabilityMetrics`` always reports the blanket-refusal baseline
        alongside classification accuracy plus a false-refusal rate over the
        use-cells, so a system that simply declines everything is visible
        immediately rather than scoring in the high eighties.
        """
        oversample = valid_oversample or len(SCENARIOS)
        out: list[Timeline] = []
        for cell in MATRIX:
            n = oversample if cell.governance == "valid" else per_cell
            for scenario in list(SCENARIOS)[:n]:
                out.append(self.build(scenario, cell))
        return out
