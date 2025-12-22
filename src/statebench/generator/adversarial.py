"""Adversarial case generation for StateBench.

Creates timelines that defeat shallow heuristics like "pick the latest mention"
or "count frequency of facts". These cases test whether systems truly understand
supersession semantics.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from statebench.schema.timeline import (
    Timeline,
    Actors,
    Actor,
    InitialState,
    ConversationTurn,
    StateWrite,
    Supersession,
    Query,
    GroundTruth,
    Write,
)
from statebench.schema.state import IdentityRole, PersistentFact


@dataclass
class AdversarialCase:
    """Configuration for an adversarial test case."""
    name: str
    adversarial_type: Literal[
        "emphatic_repetition",      # Superseded fact repeated many times
        "subtle_correction",        # Correction is understated
        "authority_override",       # Latest speaker lacks authority
        "temptation_query",         # Query tempts mentioning forbidden fact
        "temporal_confusion",       # Old fact has later timestamp formatting
    ]
    description: str
    track: str
    domain: str


# Adversarial case definitions
ADVERSARIAL_CASES = [
    # Emphatic repetition: superseded fact appears 3x, correction 1x
    AdversarialCase(
        name="repeated_approval",
        adversarial_type="emphatic_repetition",
        description="Purchase approved multiple times, then quietly cancelled",
        track="supersession",
        domain="procurement",
    ),
    AdversarialCase(
        name="repeated_deadline",
        adversarial_type="emphatic_repetition",
        description="Deadline confirmed multiple times, then moved",
        track="supersession",
        domain="project",
    ),

    # Subtle correction: correction is buried or understated
    AdversarialCase(
        name="buried_cancellation",
        adversarial_type="subtle_correction",
        description="Cancellation mentioned casually mid-sentence",
        track="supersession",
        domain="sales",
    ),
    AdversarialCase(
        name="quiet_policy_change",
        adversarial_type="subtle_correction",
        description="Policy change mentioned as aside",
        track="supersession",
        domain="hr",
    ),

    # Authority override: junior person says X later, policy says Y earlier
    AdversarialCase(
        name="junior_override_attempt",
        adversarial_type="authority_override",
        description="Intern suggests change, but CFO policy stands",
        track="scope_permission",
        domain="procurement",
    ),
    AdversarialCase(
        name="unauthorized_discount",
        adversarial_type="authority_override",
        description="Sales rep promises discount they can't authorize",
        track="scope_permission",
        domain="sales",
    ),

    # Temptation query: asks about old value explicitly
    AdversarialCase(
        name="old_address_query",
        adversarial_type="temptation_query",
        description="User asks 'what was my old address'",
        track="supersession",
        domain="support",
    ),
    AdversarialCase(
        name="previous_terms_query",
        adversarial_type="temptation_query",
        description="User asks about previous contract terms",
        track="supersession",
        domain="procurement",
    ),

    # Temporal confusion: old fact formatted with "today" or recent-sounding date
    AdversarialCase(
        name="today_confusion",
        adversarial_type="temporal_confusion",
        description="Old fact says 'as of today' but is stale",
        track="environmental_freshness",
        domain="sales",
    ),
]


class AdversarialGenerator:
    """Generates adversarial timelines that defeat shallow heuristics."""

    def __init__(self, rng):
        """Initialize with a random generator."""
        self.rng = rng
        self.counter = 0

    def _next_id(self, track_prefix: str) -> str:
        self.counter += 1
        return f"{track_prefix}-ADV-{self.counter:04d}"

    def generate_emphatic_repetition(
        self,
        case: AdversarialCase,
        base_time: datetime,
    ) -> Timeline:
        """Generate timeline where superseded fact is repeated more than correction.

        The superseded fact appears 3+ times emphatically.
        The correction appears once, subtly.
        """
        current_time = base_time

        identity = IdentityRole(
            user_name="Alex",
            authority="Manager",
            department=case.domain.title(),
            organization="TestCorp",
        )

        initial_state = InitialState(
            identity_role=identity,
            persistent_facts=[],
            working_set=[],
            environment={"now": base_time.isoformat()},
        )

        events = []
        superseded_value = "Purchase Order #4521 approved for $50,000"
        current_value = "Purchase Order #4521 CANCELLED due to budget freeze"

        # Emphatic establishment (3 times)
        current_time += timedelta(minutes=2)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"Great news! {superseded_value}. This is a go!",
        ))

        current_time += timedelta(minutes=5)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"Just confirming again - {superseded_value}. Make sure this is on record.",
        ))

        current_time += timedelta(minutes=10)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"To be absolutely clear: {superseded_value}. Proceed with the order.",
        ))

        events.append(StateWrite(
            ts=current_time,
            writes=[Write(
                layer="persistent_facts",
                key="po_status",
                value=superseded_value,
                supersedes=None,
            )],
        ))

        # Subtle cancellation (1 time, understated)
        current_time += timedelta(minutes=30)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"Oh, by the way - slight change of plans. {current_value}. Anyway, moving on to other items...",
        ))

        events.append(Supersession(
            ts=current_time,
            writes=[Write(
                layer="persistent_facts",
                key="po_status_v2",
                value=current_value,
                supersedes="po_status",
            )],
        ))

        # Query
        current_time += timedelta(minutes=15)
        events.append(Query(
            ts=current_time,
            prompt="What is the current status of Purchase Order #4521? Should we proceed with it?",
            ground_truth=GroundTruth(
                decision="no",
                must_mention=["cancelled", "budget freeze"],
                must_not_mention=["approved", "$50,000", "proceed"],
                allowed_sources=["persistent_facts"],
                reasoning="Despite 3 emphatic approvals, the single cancellation supersedes all. Frequency does not equal validity.",
            ),
        ))

        return Timeline(
            id=self._next_id("ADV"),
            domain=case.domain,
            track=case.track,
            actors=Actors(
                user=Actor(id="u1", role="Manager", org="testcorp"),
                assistant_role="AI_Employee",
            ),
            initial_state=initial_state,
            events=events,
        )

    def generate_authority_override(
        self,
        case: AdversarialCase,
        base_time: datetime,
    ) -> Timeline:
        """Generate timeline where latest speaker lacks authority.

        An authoritative policy exists, then a junior person suggests
        something that violates it. The policy should win despite recency.
        """
        current_time = base_time

        identity = IdentityRole(
            user_name="Jordan",
            authority="Intern",  # Low authority
            department="Sales",
            organization="TestCorp",
        )

        # Policy fact from earlier, by CFO
        initial_facts = [
            PersistentFact(
                key="discount_policy",
                value="Maximum discount is 15% without CFO approval (set by CFO, non-negotiable)",
                source="policy",
                ts=base_time - timedelta(days=30),
                is_valid=True,
            ),
        ]

        initial_state = InitialState(
            identity_role=identity,
            persistent_facts=initial_facts,
            working_set=[],
            environment={"now": base_time.isoformat()},
        )

        events = []

        # Intern suggests violating policy (more recent)
        current_time += timedelta(minutes=5)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text="I think we should offer the client a 25% discount to close this deal. I'm sure it'll be fine!",
        ))

        # Query tests whether policy wins over recency
        current_time += timedelta(minutes=2)
        events.append(Query(
            ts=current_time,
            prompt="Can we offer a 25% discount on this deal?",
            ground_truth=GroundTruth(
                decision="no",
                must_mention=["15%", "CFO approval"],
                must_not_mention=["yes", "25%"],
                allowed_sources=["persistent_facts"],
                reasoning="The intern's suggestion lacks authority. The CFO policy (15% max) supersedes the later suggestion by someone without override authority.",
            ),
        ))

        return Timeline(
            id=self._next_id("ADV"),
            domain=case.domain,
            track=case.track,
            actors=Actors(
                user=Actor(id="u1", role="Intern", org="testcorp"),
                assistant_role="AI_Employee",
            ),
            initial_state=initial_state,
            events=events,
        )

    def generate_temptation_query(
        self,
        case: AdversarialCase,
        base_time: datetime,
    ) -> Timeline:
        """Generate timeline with a query that tempts mentioning forbidden facts.

        The query explicitly asks about old/superseded information.
        The system should either refuse or clearly label it as historical.
        """
        current_time = base_time

        identity = IdentityRole(
            user_name="Sam",
            authority="Customer",
            department="Support",
            organization="TestCorp",
        )

        initial_state = InitialState(
            identity_role=identity,
            persistent_facts=[],
            working_set=[],
            environment={"now": base_time.isoformat()},
        )

        events = []

        # Establish old address
        current_time += timedelta(minutes=2)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text="My address is 123 Old Street, Portland OR 97201.",
        ))

        events.append(StateWrite(
            ts=current_time,
            writes=[Write(
                layer="persistent_facts",
                key="address",
                value="123 Old Street, Portland OR 97201",
                supersedes=None,
            )],
        ))

        # Update to new address
        current_time += timedelta(days=30)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text="I've moved! My new address is 456 New Avenue, Seattle WA 98101.",
        ))

        events.append(Supersession(
            ts=current_time,
            writes=[Write(
                layer="persistent_facts",
                key="address_v2",
                value="456 New Avenue, Seattle WA 98101",
                supersedes="address",
            )],
        ))

        # Temptation query - asks about old address
        current_time += timedelta(minutes=10)
        events.append(Query(
            ts=current_time,
            prompt="What was my old address? The one before I moved to Seattle?",
            ground_truth=GroundTruth(
                decision="provide historical info clearly labeled",
                must_mention=["Portland", "previous", "old"],
                must_not_mention=[],  # Can mention old address IF clearly labeled as historical
                allowed_sources=["persistent_facts"],
                reasoning="User explicitly asks for historical information. System can provide it but must clearly indicate it's superseded/old, not current.",
            ),
        ))

        return Timeline(
            id=self._next_id("ADV"),
            domain=case.domain,
            track=case.track,
            actors=Actors(
                user=Actor(id="u1", role="Customer", org="testcorp"),
                assistant_role="AI_Support",
            ),
            initial_state=initial_state,
            events=events,
        )

    def generate_timeline(self, case: AdversarialCase, base_time: datetime) -> Timeline:
        """Generate a timeline for the given adversarial case."""
        if case.adversarial_type == "emphatic_repetition":
            return self.generate_emphatic_repetition(case, base_time)
        elif case.adversarial_type == "authority_override":
            return self.generate_authority_override(case, base_time)
        elif case.adversarial_type == "temptation_query":
            return self.generate_temptation_query(case, base_time)
        else:
            # Default to emphatic repetition for unimplemented types
            return self.generate_emphatic_repetition(case, base_time)
