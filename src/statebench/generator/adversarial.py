"""Adversarial case generation for StateBench.

Creates timelines that defeat shallow heuristics like "pick the latest mention"
or "count frequency of facts". These cases test whether systems truly understand
supersession semantics.

v1.0 Additions:
- generate_subtle_correction: Correction without explicit markers
- generate_temporal_confusion: Multiple time references create ambiguity
- TimelinePerturbator: Generate adversarial variants of existing timelines
"""

import random
import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from statebench.schema.state import IdentityRole, PersistentFact, Source
from statebench.schema.timeline import (
    Actor,
    Actors,
    ConversationTurn,
    Event,
    GroundTruth,
    InitialState,
    Query,
    StateWrite,
    Supersession,
    Timeline,
    Write,
)


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

    def __init__(self, rng: random.Random) -> None:
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

        events: list[Event] = []
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
                id="F-PO-STATUS",
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
            text=(
                f"Oh, by the way - slight change of plans. {current_value}. "
                "Anyway, moving on to other items..."
            ),
        ))

        events.append(Supersession(
            ts=current_time,
            writes=[Write(
                id="F-PO-STATUS-V2",
                layer="persistent_facts",
                key="po_status_v2",
                value=current_value,
                supersedes="F-PO-STATUS",
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
                reasoning=(
                    "Despite 3 emphatic approvals, the single cancellation "
                    "supersedes all. Frequency does not equal validity."
                ),
            ),
        ))

        return Timeline(
            id=self._next_id("ADV"),
            domain=case.domain,  # type: ignore[arg-type]
            track=case.track,  # type: ignore[arg-type]
            actors=Actors(
                user=Actor(id="u1", role="Manager", org="testcorp"),
                assistant_role="AI_Agent",
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
                id="F-DISCOUNT-POLICY",
                key="discount_policy",
                value="Maximum discount is 15% without CFO approval (set by CFO, non-negotiable)",
                source=Source(type="policy", authority="executive"),
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

        events: list[Event] = []

        # Intern suggests violating policy (more recent)
        current_time += timedelta(minutes=5)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=(
                "I think we should offer the client a 25% discount to close "
                "this deal. I'm sure it'll be fine!"
            ),
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
                reasoning=(
                    "The intern's suggestion lacks authority. The CFO policy "
                    "(15% max) supersedes the later suggestion by someone "
                    "without override authority."
                ),
            ),
        ))

        return Timeline(
            id=self._next_id("ADV"),
            domain=case.domain,  # type: ignore[arg-type]
            track=case.track,  # type: ignore[arg-type]
            actors=Actors(
                user=Actor(id="u1", role="Intern", org="testcorp"),
                assistant_role="AI_Agent",
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

        events: list[Event] = []

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
                id="F-ADDRESS",
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
                id="F-ADDRESS-V2",
                layer="persistent_facts",
                key="address_v2",
                value="456 New Avenue, Seattle WA 98101",
                supersedes="F-ADDRESS",
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
                # Can mention old address IF clearly labeled as historical
                must_not_mention=[],
                allowed_sources=["persistent_facts"],
                reasoning=(
                    "User explicitly asks for historical information. "
                    "System can provide it but must clearly indicate "
                    "it's superseded/old, not current."
                ),
            ),
        ))

        return Timeline(
            id=self._next_id("ADV"),
            domain=case.domain,  # type: ignore[arg-type]
            track=case.track,  # type: ignore[arg-type]
            actors=Actors(
                user=Actor(id="u1", role="Customer", org="testcorp"),
                assistant_role="AI_Support",
            ),
            initial_state=initial_state,
            events=events,
        )

    def generate_subtle_correction(
        self,
        case: AdversarialCase,
        base_time: datetime,
    ) -> Timeline:
        """Generate timeline with subtle correction (no explicit markers).

        v1.0: The correction is subtle - user just states the new value without
        saying "actually", "instead", "change that to", etc.
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

        events: list[Event] = []

        # Initial value establishment
        initial_value = "Seattle office, Building A, Room 302"
        current_time += timedelta(minutes=2)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"The meeting will be in {initial_value}.",
        ))

        current_time += timedelta(minutes=1)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="assistant",
            text=f"Got it, meeting location: {initial_value}.",
        ))

        events.append(StateWrite(
            ts=current_time,
            writes=[Write(
                id="F-MEETING-LOC",
                layer="persistent_facts",
                key="meeting_location",
                value=initial_value,
                supersedes=None,
            )],
        ))

        # Filler conversation (makes it harder)
        current_time += timedelta(minutes=5)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text="Make sure to send calendar invites to everyone.",
        ))

        current_time += timedelta(minutes=1)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="assistant",
            text="I'll send out the invites right away.",
        ))

        # Subtle correction - just states new value, no correction markers
        new_value = "Portland office, Building C, Conference Room 1"
        current_time += timedelta(minutes=8)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"Meeting is in {new_value}.",  # No "actually", "instead", etc.
        ))

        events.append(Supersession(
            ts=current_time,
            writes=[Write(
                id="F-MEETING-LOC-V2",
                layer="persistent_facts",
                key="meeting_location_v2",
                value=new_value,
                supersedes="F-MEETING-LOC",
            )],
        ))

        # Query
        current_time += timedelta(minutes=5)
        events.append(Query(
            ts=current_time,
            prompt="Where is the meeting being held?",
            ground_truth=GroundTruth(
                decision=new_value,
                must_mention=["Portland", "Conference Room 1"],
                must_not_mention=["Seattle", "Room 302"],
                allowed_sources=["persistent_facts"],
                reasoning=(
                    f"Subtle correction: user said 'Meeting is in {new_value}' "
                    f"without explicit correction language. This supersedes "
                    f"the earlier '{initial_value}'."
                ),
            ),
        ))

        return Timeline(
            id=self._next_id("ADV-SUB"),
            domain=case.domain,  # type: ignore[arg-type]
            track=case.track,  # type: ignore[arg-type]
            version="1.0",
            difficulty="adversarial",
            actors=Actors(
                user=Actor(id="u1", role="Manager", org="testcorp"),
                assistant_role="AI_Agent",
            ),
            initial_state=initial_state,
            events=events,
        )

    def generate_temporal_confusion(
        self,
        case: AdversarialCase,
        base_time: datetime,
    ) -> Timeline:
        """Generate timeline with temporal confusion.

        v1.0: Multiple time references create ambiguity.
        Example: "The original deadline was Friday, then we moved to Thursday,
        but after discussion, we're going with Monday."
        """
        current_time = base_time

        identity = IdentityRole(
            user_name="Jordan",
            authority="Project Manager",
            department="Engineering",
            organization="TestCorp",
        )

        initial_state = InitialState(
            identity_role=identity,
            persistent_facts=[],
            working_set=[],
            environment={"now": base_time.isoformat()},
        )

        events: list[Event] = []

        # The confusing multi-date statement
        dates = ["Friday March 15th", "Thursday March 14th", "Monday March 18th"]
        final_date = dates[-1]

        current_time += timedelta(minutes=2)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text="Let me update you on the launch timeline.",
        ))

        current_time += timedelta(minutes=1)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="assistant",
            text="Sure, what's the current status?",
        ))

        # Confusing statement with multiple dates
        current_time += timedelta(minutes=2)
        confusion_text = (
            f"So the original launch date was {dates[0]}. "
            f"Then we moved it to {dates[1]} because of the dependency issue. "
            f"But after the stakeholder meeting, the final launch date is {dates[2]}."
        )
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=confusion_text,
        ))

        # Record final state
        events.append(StateWrite(
            ts=current_time,
            writes=[Write(
                id="F-LAUNCH-DATE",
                layer="persistent_facts",
                key="launch_date",
                value=final_date,
                supersedes=None,
            )],
        ))

        # Add more emphasis on old dates (adversarial)
        current_time += timedelta(minutes=3)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=(
                f"To recap: we originally planned {dates[0]}, "
                f"and {dates[1]} was considered too. "
                f"But {final_date} is the one."
            ),
        ))

        # Query
        current_time += timedelta(minutes=5)
        events.append(Query(
            ts=current_time,
            prompt="When is the launch?",
            ground_truth=GroundTruth(
                decision=final_date,
                must_mention=["Monday", "March 18"],
                must_not_mention=["Friday", "Thursday"],
                allowed_sources=["persistent_facts"],
                reasoning=(
                    f"Temporal confusion: multiple dates mentioned "
                    f"({', '.join(dates)}). The final/current date is {final_date}. "
                    f"Earlier dates were mentioned more times but are superseded."
                ),
            ),
        ))

        return Timeline(
            id=self._next_id("ADV-TMP"),
            domain=case.domain,  # type: ignore[arg-type]
            track=case.track,  # type: ignore[arg-type]
            version="1.0",
            difficulty="adversarial",
            actors=Actors(
                user=Actor(id="u1", role="Project Manager", org="testcorp"),
                assistant_role="AI_Agent",
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
        elif case.adversarial_type == "subtle_correction":
            return self.generate_subtle_correction(case, base_time)
        elif case.adversarial_type == "temporal_confusion":
            return self.generate_temporal_confusion(case, base_time)
        else:
            # Default to emphatic repetition for unimplemented types
            return self.generate_emphatic_repetition(case, base_time)


# =============================================================================
# v1.0: Perturbation Pipeline
# =============================================================================

@dataclass
class PerturbationResult:
    """Result of applying perturbations to a timeline."""
    original_id: str
    variant_id: str
    perturbations_applied: list[str]
    timeline: Timeline


class TimelinePerturbator:
    """Generate adversarial variants of existing timelines.

    Applies various perturbations that should not change the
    correct answer but may confuse context management systems.
    """

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng if rng is not None else random.Random()
        self._id_counter = 0

        # Name substitution pools
        self._name_pools = {
            "person": ["Alex", "Jordan", "Casey", "Morgan", "Taylor", "Riley"],
            "company": ["Acme", "Globex", "Initech", "Hooli", "Pied Piper"],
            "project": ["Phoenix", "Apollo", "Mercury", "Atlas", "Titan"],
        }

        # Red herring templates
        self._red_herrings = [
            "By the way, I also need to check on {topic}.",
            "Unrelated, but did you see the update about {topic}?",
            "Oh, and remind me later about {topic}.",
            "Speaking of which, {topic} is also pending.",
        ]

        self._red_herring_topics = [
            "the quarterly report",
            "the team meeting",
            "the client feedback",
            "the budget review",
            "the performance metrics",
        ]

    def _next_variant_id(self, original_id: str) -> str:
        self._id_counter += 1
        return f"{original_id}-V{self._id_counter:03d}"

    def paraphrase(self, timeline: Timeline) -> Timeline:
        """Apply paraphrasing to conversation turns.

        Same semantics, different wording.
        """
        variant = deepcopy(timeline)

        paraphrase_map = {
            "got it": ["understood", "noted", "okay", "sure thing"],
            "please": ["kindly", "would you", "can you"],
            "thanks": ["thank you", "appreciate it", "great"],
            "i need": ["i require", "i want", "i'd like"],
            "send": ["ship", "deliver", "dispatch"],
            "schedule": ["set up", "arrange", "book"],
        }

        for event in variant.events:
            if isinstance(event, ConversationTurn):
                text = event.text.lower()
                for phrase, alternatives in paraphrase_map.items():
                    if phrase in text:
                        replacement = self.rng.choice(alternatives)
                        # Case-preserving replacement
                        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                        event.text = pattern.sub(replacement, event.text, count=1)
                        break

        return variant

    def temporal_shuffle(self, timeline: Timeline) -> Timeline:
        """Reorder non-dependent conversation turns.

        Only shuffles adjacent filler conversations, not state-changing events.
        """
        variant = deepcopy(timeline)

        # Find pairs of adjacent ConversationTurns that are both filler
        filler_pairs: list[tuple[int, int]] = []
        events = variant.events

        for i in range(len(events) - 1):
            ev_i = events[i]
            ev_j = events[i + 1]
            if (isinstance(ev_i, ConversationTurn) and
                isinstance(ev_j, ConversationTurn) and
                self._is_filler(ev_i) and self._is_filler(ev_j)):
                filler_pairs.append((i, i + 1))

        # Swap one random pair if any exist
        if filler_pairs:
            i, j = self.rng.choice(filler_pairs)
            events[i], events[j] = events[j], events[i]

        return variant

    def _is_filler(self, turn: ConversationTurn) -> bool:
        """Check if a turn is filler conversation."""
        filler_markers = [
            "thanks", "thank you", "got it", "okay", "sure",
            "of course", "happy to", "no problem", "sounds good",
        ]
        text_lower = turn.text.lower()
        return any(marker in text_lower for marker in filler_markers)

    def name_substitute(self, timeline: Timeline) -> Timeline:
        """Change entity names consistently throughout the timeline."""
        variant = deepcopy(timeline)

        # Find names to substitute (look for capitalized words)
        name_pattern = re.compile(r'\b[A-Z][a-z]+\b')
        all_names: set[str] = set()

        for event in variant.events:
            if isinstance(event, ConversationTurn):
                all_names.update(name_pattern.findall(event.text))

        # Create substitution map
        substitutions: dict[str, str] = {}
        used_names: set[str] = set()

        for name in all_names:
            # Pick a pool based on context
            pool = self._name_pools.get("person", self._name_pools["person"])
            available = [n for n in pool if n not in used_names and n != name]
            if available:
                new_name = self.rng.choice(available)
                substitutions[name] = new_name
                used_names.add(new_name)

        # Apply substitutions
        for event in variant.events:
            if isinstance(event, ConversationTurn):
                for old_name, new_name in substitutions.items():
                    event.text = event.text.replace(old_name, new_name)

        return variant

    def emphasis_invert(self, timeline: Timeline) -> Timeline:
        """Add emphasis to superseded facts (makes detection harder).

        This is adversarial because it adds more mentions of the OLD value.
        """
        variant = deepcopy(timeline)

        # Find supersession events to identify old values
        old_values: list[str] = []
        for event in variant.events:
            if isinstance(event, Supersession):
                for write in event.writes:
                    if write.supersedes:
                        old_values.append(write.supersedes)

        # If we have old values, add emphasis to them in earlier conversation
        if old_values:
            emphasis_templates = [
                "Remember, {value} is important.",
                "As I mentioned, {value} was the original decision.",
                "We discussed {value} earlier.",
            ]

            # Insert emphasis before the supersession
            for i, event in enumerate(variant.events):
                if isinstance(event, Supersession):
                    old_val = old_values[0] if old_values else "that"
                    emphasis = self.rng.choice(emphasis_templates).format(value=old_val)

                    # Insert a turn before this event
                    emphasis_turn = ConversationTurn(
                        ts=event.ts - timedelta(minutes=1),
                        speaker="user",
                        text=emphasis,
                    )
                    variant.events.insert(i, emphasis_turn)
                    break

        return variant

    def add_red_herrings(self, timeline: Timeline) -> Timeline:
        """Add irrelevant distractors to the conversation."""
        variant = deepcopy(timeline)

        # Find a good insertion point (after initial conversation, before query)
        insertion_idx = None
        for i, event in enumerate(variant.events):
            if isinstance(event, ConversationTurn) and i > 2:
                insertion_idx = i
                break

        if insertion_idx is not None:
            topic = self.rng.choice(self._red_herring_topics)
            herring = self.rng.choice(self._red_herrings).format(topic=topic)

            # Get timestamp from nearby event
            nearby = variant.events[insertion_idx]
            herring_turn = ConversationTurn(
                ts=nearby.ts + timedelta(seconds=30),
                speaker="user",
                text=herring,
            )
            variant.events.insert(insertion_idx + 1, herring_turn)

            # Add assistant response
            response_turn = ConversationTurn(
                ts=nearby.ts + timedelta(seconds=45),
                speaker="assistant",
                text="I'll make a note of that.",
            )
            variant.events.insert(insertion_idx + 2, response_turn)

        return variant

    def generate_variants(
        self,
        timeline: Timeline,
        n_variants: int = 5,
        perturbations: list[str] | None = None,
    ) -> list[PerturbationResult]:
        """Generate n adversarial variants of a timeline.

        Args:
            timeline: The base timeline to perturb
            n_variants: Number of variants to generate
            perturbations: Optional list of perturbation names to use.
                          If None, uses random selection.

        Returns:
            List of PerturbationResult with the variant timelines.
        """
        perturbation_funcs: dict[str, Callable[[Timeline], Timeline]] = {
            "paraphrase": self.paraphrase,
            "temporal_shuffle": self.temporal_shuffle,
            "name_substitute": self.name_substitute,
            "emphasis_invert": self.emphasis_invert,
            "add_red_herrings": self.add_red_herrings,
        }

        if perturbations is None:
            available = list(perturbation_funcs.keys())
        else:
            available = [p for p in perturbations if p in perturbation_funcs]

        results: list[PerturbationResult] = []

        for _ in range(n_variants):
            variant = deepcopy(timeline)

            # Apply random subset of perturbations (1-3)
            n_perturbs = self.rng.randint(1, min(3, len(available)))
            selected = self.rng.sample(available, k=n_perturbs)

            applied: list[str] = []
            for perturb_name in selected:
                variant = perturbation_funcs[perturb_name](variant)
                applied.append(perturb_name)

            # Update variant ID
            variant.id = self._next_variant_id(timeline.id)

            results.append(PerturbationResult(
                original_id=timeline.id,
                variant_id=variant.id,
                perturbations_applied=applied,
                timeline=variant,
            ))

        return results


# =============================================================================
# Convenience Functions
# =============================================================================

def get_adversarial_cases() -> list[AdversarialCase]:
    """Get all adversarial case definitions."""
    return ADVERSARIAL_CASES.copy()


def get_cases_by_type(
    adversarial_type: str,
) -> list[AdversarialCase]:
    """Get adversarial cases by type."""
    return [c for c in ADVERSARIAL_CASES if c.adversarial_type == adversarial_type]
