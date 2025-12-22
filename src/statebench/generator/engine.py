"""Parameterized generation engine for StateBench.

This engine takes templates and generates JSONL timelines with
controlled randomization of names, dates, amounts, and supersession patterns.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

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
from statebench.schema.state import IdentityRole, PersistentFact, WorkingSetItem, EnvironmentSignal
from statebench.generator.templates.supersession import SupersessionTemplate, SUPERSESSION_TEMPLATES
from statebench.generator.templates.commitment import CommitmentTemplate, COMMITMENT_TEMPLATES
from statebench.generator.templates.interruption import InterruptionTemplate, INTERRUPTION_TEMPLATES
from statebench.generator.templates.permission import PermissionTemplate, PERMISSION_TEMPLATES
from statebench.generator.templates.environmental import EnvironmentalTemplate, ENVIRONMENTAL_TEMPLATES
# v0.2 tracks
from statebench.generator.templates.hallucination import HallucinationTemplate, HALLUCINATION_TEMPLATES
from statebench.generator.templates.scope_leak import ScopeLeakTemplate, SCOPE_LEAK_TEMPLATES
from statebench.generator.templates.causality import (
    CausalityTemplate,
    MultiConstraintTemplate,
    EdgeCaseTemplate,
    ConflictingConstraintTemplate,
    ChainDependencyTemplate,
    AggregationTemplate,
    CAUSALITY_TEMPLATES,
    HARD_CAUSALITY_TEMPLATES,
)
from statebench.generator.templates.repair import RepairChain, REPAIR_CHAIN_TEMPLATES
from statebench.generator.templates.brutal import BrutalScenario, BrutalEvent, BRUTAL_SCENARIOS


# --- Name and Value Pools ---

FIRST_NAMES = [
    "Matt", "Sarah", "David", "Emily", "Michael", "Jessica", "James", "Ashley",
    "Robert", "Jennifer", "John", "Amanda", "William", "Stephanie", "Daniel", "Nicole",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
]

ORGANIZATIONS = [
    "Acme Corp", "TechStart Inc", "Global Industries", "Innovation Labs",
    "Enterprise Solutions", "Digital Dynamics", "Future Systems", "Prime Consulting",
]

ROLES = {
    "procurement": ["Procurement Manager", "Vendor Manager", "Operations Director", "CFO"],
    "sales": ["Sales Rep", "Account Executive", "Sales Manager", "VP Sales"],
    "project": ["Project Manager", "Tech Lead", "Engineering Director", "Product Manager"],
    "hr": ["HR Manager", "HR Director", "People Ops", "CHRO"],
    "support": ["Support Agent", "Support Lead", "Customer Success Manager", "Support Director"],
}

APPROVERS = ["CFO", "VP", "Director", "Manager", "Legal"]

DURATIONS = ["1 year", "6 months", "2 years", "18 months", "3 years"]

AMOUNTS = [5000, 10000, 25000, 50000, 100000, 250000]

DISCOUNT_PCTS = [10, 15, 20, 25, 30]

DATES = [
    "January 15, 2026", "January 20, 2026", "January 31, 2026",
    "February 1, 2026", "February 15, 2026", "February 28, 2026",
    "March 1, 2026", "March 15, 2026", "March 31, 2026",
]


class TimelineGenerator:
    """Generates timelines from templates with controlled randomization."""

    def __init__(self, seed: int | None = None):
        """Initialize the generator with an optional random seed."""
        self.rng = random.Random(seed)
        self.counter = 0

    def _next_id(self, track_prefix: str) -> str:
        """Generate the next timeline ID."""
        self.counter += 1
        return f"{track_prefix}-{self.counter:06d}"

    def _random_name(self) -> str:
        """Generate a random person name."""
        return f"{self.rng.choice(FIRST_NAMES)} {self.rng.choice(LAST_NAMES)}"

    def _random_org(self) -> str:
        """Generate a random organization name."""
        return self.rng.choice(ORGANIZATIONS)

    def _random_role(self, domain: str) -> str:
        """Generate a random role for a domain."""
        return self.rng.choice(ROLES.get(domain, ROLES["project"]))

    def _base_time(self) -> datetime:
        """Generate a base timestamp for the timeline."""
        # Random time on a weekday in the past month
        base = datetime(2025, 12, 21, 9, 0, 0)
        offset_days = self.rng.randint(0, 30)
        offset_hours = self.rng.randint(0, 8)
        return base - timedelta(days=offset_days) + timedelta(hours=offset_hours)

    def generate_supersession_timeline(
        self,
        template: SupersessionTemplate,
        adversarial: bool = False,
    ) -> Timeline:
        """Generate a single timeline from a supersession template."""
        # Generate random values
        entity = self.rng.choice(template.entity_names)
        entity_id = entity.lower().replace(" ", "_").replace("-", "_")
        user_name = self._random_name()
        org = self._random_org()
        role = self._random_role(template.domain)

        # Determine number of supersessions
        num_supersessions = self.rng.randint(
            template.min_supersessions,
            template.max_supersessions
        )

        # Generate base time
        base_time = self._base_time()
        current_time = base_time

        # Build initial state
        initial_facts: list[PersistentFact] = []
        if template.policy_key and template.policy_template:
            approver = self.rng.choice(APPROVERS)
            policy_value = template.policy_template.format(
                approver=approver,
                max_discount=self.rng.choice([10, 15, 20]),
            )
            initial_facts.append(PersistentFact(
                key=template.policy_key,
                value=policy_value,
                source="policy",
                ts=base_time - timedelta(days=180),
                is_valid=True,
            ))

        identity = IdentityRole(
            user_name=user_name.split()[0],  # First name only
            authority=role,
            department=template.domain.title(),
            organization=org,
        )

        initial_state = InitialState(
            identity_role=identity,
            persistent_facts=initial_facts,
            working_set=[],
            environment={"now": base_time.isoformat()},
        )

        # Build events
        events = []

        # Initial user request
        current_time += timedelta(minutes=2)
        duration = self.rng.choice(DURATIONS)
        amount = self.rng.choice(AMOUNTS)
        discount_pct = self.rng.choice(DISCOUNT_PCTS)

        initial_text = self._generate_initial_request(template, entity, duration, amount, discount_pct)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=initial_text,
        ))

        # Initial state write
        current_time += timedelta(minutes=1)

        # Build format context with all possible placeholders
        format_ctx = {
            "entity": entity,
            "entity_id": entity_id,
            "duration": duration,
            "amount": amount,
            "discount_pct": discount_pct,
            "date": DATES[0],
            "initial_terms": "net-30, standard pricing",
            "initial_rule": "3 days per week maximum",
            "initial_resolution": "resolved with workaround",
            "project": "Project Phoenix",
        }

        initial_fact_value = template.initial_fact_template.format(**format_ctx)
        fact_key = template.initial_fact_key.format(**format_ctx)

        events.append(StateWrite(
            ts=current_time,
            writes=[Write(
                layer="persistent_facts",
                key=fact_key,
                value=initial_fact_value,
                supersedes=None,
            )],
        ))

        # Track all fact values for ground truth
        all_fact_values = [initial_fact_value]
        current_fact_key = fact_key

        # Generate supersession chain
        for i in range(num_supersessions):
            # Time gap before supersession
            gap_minutes = self.rng.randint(5, 60)
            current_time += timedelta(minutes=gap_minutes)

            # Pick a reason
            reason = self.rng.choice(template.supersession_reasons)

            # Generate new fact value
            # Extend format context for supersession
            supersession_ctx = format_ctx.copy()
            supersession_ctx["reason"] = reason
            supersession_ctx["new_date"] = DATES[min(i + 1, len(DATES) - 1)]
            supersession_ctx["new_terms"] = f"net-60, volume discount {10 + i * 5}%"
            supersession_ctx["new_rule"] = f"updated policy v{i + 2}"
            supersession_ctx["new_status"] = f"escalated to tier {i + 2}"
            supersession_ctx["new_project"] = f"Project {'Alpha' if i == 0 else 'Beta'}"

            try:
                new_value = template.supersession_fact_template.format(**supersession_ctx)
            except KeyError:
                # Fallback for simple templates
                new_value = template.supersession_fact_template.format(
                    entity=entity,
                    reason=reason,
                )

            new_fact_key = f"{fact_key}_v{i+2}"

            # Supersession event
            events.append(Supersession(
                ts=current_time,
                writes=[Write(
                    layer="persistent_facts",
                    key=new_fact_key,
                    value=new_value,
                    supersedes=current_fact_key,
                )],
            ))

            # User explanation turn (makes it adversarial if emphatic)
            current_time += timedelta(seconds=30)
            explanation = self._generate_supersession_explanation(
                template, entity, reason, adversarial, i
            )
            events.append(ConversationTurn(
                ts=current_time,
                speaker="user",
                text=explanation,
            ))

            all_fact_values.append(new_value)
            current_fact_key = new_fact_key

        # Final query
        current_time += timedelta(minutes=self.rng.randint(5, 30))
        query_text = template.query_template.format(entity=entity)

        # Build ground truth
        must_mention = self._extract_must_mention(template, all_fact_values[-1])
        must_not_mention = self._extract_must_not_mention(template, all_fact_values[:-1])

        # Resolve decision - may contain placeholders
        decision = template.correct_decision
        if "{" in decision:
            # Build decision context from final state
            decision_ctx = supersession_ctx.copy()
            decision_ctx["final_terms"] = supersession_ctx.get("new_terms", "current terms")
            decision_ctx["final_date"] = supersession_ctx.get("new_date", DATES[-1])
            decision_ctx["final_rule"] = supersession_ctx.get("new_rule", "current policy")
            decision_ctx["final_status"] = supersession_ctx.get("new_status", "current status")
            decision_ctx["final_project"] = supersession_ctx.get("new_project", "current project")
            decision_ctx["superseded_terms"] = format_ctx.get("initial_terms", "")
            decision_ctx["superseded_dates"] = format_ctx.get("date", "")
            decision_ctx["superseded_rules"] = format_ctx.get("initial_rule", "")
            decision_ctx["superseded_statuses"] = format_ctx.get("initial_resolution", "")
            decision_ctx["superseded_projects"] = format_ctx.get("project", "")
            try:
                decision = decision.format(**decision_ctx)
            except KeyError:
                decision = "see current facts"

        ground_truth = GroundTruth(
            decision=decision,
            must_mention=must_mention,
            must_not_mention=must_not_mention,
            allowed_sources=["persistent_facts", "environment"],
            reasoning=f"Original decision was superseded {num_supersessions} time(s). "
                      f"Only the final state is valid. Referencing superseded facts is an error.",
        )

        events.append(Query(
            ts=current_time,
            prompt=query_text,
            ground_truth=ground_truth,
        ))

        # Build timeline
        return Timeline(
            id=self._next_id("S1"),
            domain=template.domain,
            track="supersession",
            actors=Actors(
                user=Actor(id="u1", role=role, org=org.lower().replace(" ", "_")),
                assistant_role="AI_Employee",
            ),
            initial_state=initial_state,
            events=events,
        )

    def _generate_initial_request(
        self,
        template: SupersessionTemplate,
        entity: str,
        duration: str,
        amount: int,
        discount_pct: int,
    ) -> str:
        """Generate the initial user request text."""
        if template.name == "vendor_renewal":
            return f"Renew {entity} for {duration}. Keep it simple."
        elif template.name == "purchase_approval":
            return f"I need to purchase {entity}. Budget is ${amount:,}. Please approve."
        elif template.name == "discount_approval":
            return f"I'm working on the {entity} deal. They want a {discount_pct}% discount. I got VP approval."
        elif template.name == "deadline_change":
            return f"We're launching {entity} on {DATES[0]}. Mark that as our target."
        elif template.name == "resource_allocation":
            return f"Allocate {entity} to Project Phoenix for the next sprint."
        elif template.name == "policy_change":
            return f"Please note the {entity} has been updated."
        elif template.name == "ticket_status":
            return f"I'm resolving {entity} - marked as fixed with workaround."
        else:
            return f"Let's proceed with {entity}."

    def _generate_supersession_explanation(
        self,
        template: SupersessionTemplate,
        entity: str,
        reason: str,
        adversarial: bool,
        iteration: int,
    ) -> str:
        """Generate explanation for why fact was superseded."""
        if adversarial and iteration == 0:
            # Make the explanation subtle/buried
            return f"Oh by the way, {reason.lower()}. No rush on this."
        else:
            if template.name == "vendor_renewal":
                return f"Actually, hold off on {entity}. We need to {reason}."
            elif template.name == "purchase_approval":
                return f"Cancel that purchase. {reason.title()}."
            elif template.name == "discount_approval":
                return f"Bad news - {reason}. We're back to standard policy."
            elif template.name == "deadline_change":
                return f"Update on {entity} - stakeholders moved the date. {reason.title()}."
            else:
                return f"Change of plans: {reason}."

    def _extract_must_mention(
        self,
        template: SupersessionTemplate,
        final_value: str,
    ) -> list[str]:
        """Extract phrases that must be mentioned from final value."""
        # Extract key phrases from the final value
        phrases = []
        words = final_value.lower().split()

        # Look for key action words
        action_words = ["not", "cancel", "revoke", "stop", "hold", "wait", "updated", "moved", "changed"]
        for word in action_words:
            if word in words:
                phrases.append(word)
                break

        # Add reason-like phrases
        if "renegotiate" in final_value.lower():
            phrases.append("renegotiate")
        if "revoked" in final_value.lower():
            phrases.append("revoked")
        if "cancelled" in final_value.lower():
            phrases.append("cancelled")

        return phrases[:3]  # Limit to 3 phrases

    def _extract_must_not_mention(
        self,
        template: SupersessionTemplate,
        superseded_values: list[str],
    ) -> list[str]:
        """Extract phrases that must NOT be mentioned from superseded values."""
        phrases = []

        for value in superseded_values:
            # Extract positive/approval phrases that are now invalid
            if "approved" in value.lower():
                phrases.append("approved")
            if "proceed" in value.lower():
                phrases.append("proceed")
            # Look for specific amounts/percentages
            import re
            amounts = re.findall(r'\$[\d,]+', value)
            pcts = re.findall(r'\d+%', value)
            phrases.extend(amounts[:1])
            phrases.extend(pcts[:1])

        return list(set(phrases))[:4]  # Dedupe and limit

    def generate_commitment_timeline(
        self,
        template: CommitmentTemplate,
    ) -> Timeline:
        """Generate a single timeline from a commitment template."""
        entity = self.rng.choice(template.entity_names)
        entity_id = entity.lower().replace(" ", "_").replace("-", "_")
        user_name = self._random_name()
        org = self._random_org()
        role = self._random_role(template.domain)
        base_time = self._base_time()
        current_time = base_time

        # Initial preference
        initial_pref = self.rng.choice(template.preference_values)
        new_pref = self.rng.choice(template.new_preference_values)

        identity = IdentityRole(
            user_name=user_name.split()[0],
            authority=role,
            department=template.domain.title(),
            organization=org,
        )

        # Initial facts: commitment + preference
        # Build format context for templates with various placeholders
        commit_ctx = {
            "entity": entity,
            "entity_id": entity_id,
            "date": DATES[0],
            "sla_hours": self.rng.choice([1, 2, 4, 8]),
            "duration": self.rng.choice(DURATIONS),
        }
        initial_facts = [
            PersistentFact(
                key=template.commitment_key.format(**commit_ctx),
                value=template.commitment_template.format(**commit_ctx),
                source="commitment",
                ts=base_time,
                is_valid=True,
            ),
            PersistentFact(
                key=template.preference_key,
                value=template.preference_template.format(style=initial_pref, time=initial_pref,
                    review_type=initial_pref, location=initial_pref, payment_type=initial_pref, tool=initial_pref),
                source="preference",
                ts=base_time,
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

        # User states the commitment and preference
        current_time += timedelta(minutes=2)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"I'm committed to {entity}. I prefer {initial_pref} for now.",
        ))

        # Preference changes
        current_time += timedelta(minutes=30)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"Actually, let's switch to {new_pref} instead. That works better for me.",
        ))

        events.append(Supersession(
            ts=current_time,
            writes=[Write(
                layer="persistent_facts",
                key=f"{template.preference_key}_v2",
                value=template.preference_template.format(style=new_pref, time=new_pref,
                    review_type=new_pref, location=new_pref, payment_type=new_pref, tool=new_pref),
                supersedes=template.preference_key,
            )],
        ))

        # Query about commitment (should persist)
        current_time += timedelta(minutes=15)
        query_text = template.commitment_query.format(entity=entity)

        ground_truth = GroundTruth(
            decision="yes" if template.commitment_persists else "no",
            must_mention=[entity] if template.commitment_persists else [],
            must_not_mention=[],
            allowed_sources=["persistent_facts", "environment"],
            reasoning="Commitment persists even though preference changed.",
        )

        events.append(Query(
            ts=current_time,
            prompt=query_text,
            ground_truth=ground_truth,
        ))

        return Timeline(
            id=self._next_id("S2"),
            domain=template.domain,
            track="commitment_durability",
            actors=Actors(
                user=Actor(id="u1", role=role, org=org.lower().replace(" ", "_")),
                assistant_role="AI_Employee",
            ),
            initial_state=initial_state,
            events=events,
        )

    def generate_interruption_timeline(
        self,
        template: InterruptionTemplate,
    ) -> Timeline:
        """Generate a single timeline from an interruption template."""
        user_name = self._random_name()
        org = self._random_org()
        role = self._random_role(template.domain)
        base_time = self._base_time()
        current_time = base_time

        identity = IdentityRole(
            user_name=user_name.split()[0],
            authority=role,
            department=template.domain.title(),
            organization=org,
        )

        initial_state = InitialState(
            identity_role=identity,
            persistent_facts=[],
            working_set=[],
            environment={"now": base_time.isoformat()},
        )

        events = []

        # Primary task steps
        for step in template.primary_steps:
            current_time += timedelta(minutes=3)
            events.append(ConversationTurn(
                ts=current_time,
                speaker="user",
                text=step,
            ))

        # Record commitment
        current_time += timedelta(minutes=1)
        events.append(StateWrite(
            ts=current_time,
            writes=[Write(
                layer="persistent_facts",
                key="primary_commitment",
                value=template.primary_commitment,
                supersedes=None,
            )],
        ))

        # Interruption trigger
        current_time += timedelta(minutes=10)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"Hold on - {template.interruption_trigger}",
        ))

        # Interruption context
        for ctx in template.interruption_context:
            current_time += timedelta(minutes=2)
            events.append(ConversationTurn(
                ts=current_time,
                speaker="user",
                text=ctx,
            ))

        # Resumption
        current_time += timedelta(minutes=15)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=template.resumption_trigger,
        ))

        # Query
        current_time += timedelta(minutes=2)
        ground_truth = GroundTruth(
            decision="recall primary task",
            must_mention=template.should_remember[:2],
            must_not_mention=template.should_forget[:2],
            allowed_sources=["persistent_facts", "working_set"],
            reasoning="Must remember primary task commitment, forget interruption context.",
        )

        events.append(Query(
            ts=current_time,
            prompt=template.resumption_query,
            ground_truth=ground_truth,
        ))

        return Timeline(
            id=self._next_id("S3"),
            domain=template.domain,
            track="interruption_resumption",
            actors=Actors(
                user=Actor(id="u1", role=role, org=org.lower().replace(" ", "_")),
                assistant_role="AI_Employee",
            ),
            initial_state=initial_state,
            events=events,
        )

    def generate_permission_timeline(
        self,
        template: PermissionTemplate,
    ) -> Timeline:
        """Generate a single timeline from a permission template."""
        base_time = self._base_time()
        current_time = base_time

        # Use the low-permission user for the query
        user = template.user_low
        identity = IdentityRole(
            user_name=user["name"],
            authority=user["role"],
            department=template.domain.title(),
            organization=template.org_name,
        )

        # Initial facts include both shared and restricted (to test filtering)
        initial_facts = []
        for i, fact in enumerate(template.shared_facts):
            initial_facts.append(PersistentFact(
                key=f"shared_fact_{i}",
                value=fact,
                source="system",
                ts=base_time,
                is_valid=True,
            ))

        # Restricted facts - marked with access control
        for i, fact in enumerate(template.restricted_facts):
            initial_facts.append(PersistentFact(
                key=f"restricted_fact_{i}",
                value=f"[RESTRICTED: {template.restriction_reason}] {fact}",
                source="system",
                ts=base_time,
                is_valid=True,
            ))

        initial_state = InitialState(
            identity_role=identity,
            persistent_facts=initial_facts,
            working_set=[],
            environment={"now": base_time.isoformat()},
        )

        events = []

        # Context setting
        current_time += timedelta(minutes=2)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"I'm {user['name']}, {user['role']} at {template.org_name}.",
        ))

        # Query that tests permission boundary
        current_time += timedelta(minutes=5)
        ground_truth = GroundTruth(
            decision="use only permitted information",
            must_mention=template.low_user_can_mention[:2],
            must_not_mention=template.low_user_cannot_mention[:3],
            allowed_sources=["persistent_facts"],
            reasoning=f"{template.restriction_reason}. Low-permission user cannot access restricted facts.",
        )

        events.append(Query(
            ts=current_time,
            prompt=template.query_template,
            ground_truth=ground_truth,
        ))

        return Timeline(
            id=self._next_id("S4"),
            domain=template.domain,
            track="scope_permission",
            actors=Actors(
                user=Actor(id=user["id"], role=user["role"], org=template.org_name.lower().replace(" ", "_")),
                assistant_role="AI_Employee",
            ),
            initial_state=initial_state,
            events=events,
        )

    def generate_environmental_timeline(
        self,
        template: EnvironmentalTemplate,
    ) -> Timeline:
        """Generate a single timeline from an environmental template."""
        user_name = self._random_name()
        org = self._random_org()
        role = self._random_role(template.domain)
        base_time = self._base_time()
        current_time = base_time

        identity = IdentityRole(
            user_name=user_name.split()[0],
            authority=role,
            department=template.domain.title(),
            organization=org,
        )

        initial_state = InitialState(
            identity_role=identity,
            persistent_facts=[],
            working_set=[],
            environment={
                "now": base_time.isoformat(),
                template.initial_signal["type"]: template.initial_signal_text,
            },
        )

        events = []

        # Initial signal mentioned
        current_time += timedelta(minutes=2)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"Note: {template.initial_signal_text}",
        ))

        # Time passes, signal updates
        current_time += timedelta(minutes=template.time_gap_minutes)

        # Updated signal
        events.append(StateWrite(
            ts=current_time,
            writes=[Write(
                layer="environment",
                key=template.updated_signal["type"],
                value=template.updated_signal_text,
                supersedes=None,
            )],
        ))

        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"Update: {template.updated_signal_text}",
        ))

        # Query that requires fresh environmental data
        current_time += timedelta(minutes=5)
        ground_truth = GroundTruth(
            decision="use fresh signal",
            must_mention=template.should_use_fresh[:2],
            must_not_mention=template.should_not_use_stale[:2],
            allowed_sources=["environment", "persistent_facts"],
            reasoning=f"{template.time_sensitivity}",
        )

        events.append(Query(
            ts=current_time,
            prompt=template.query_template,
            ground_truth=ground_truth,
        ))

        return Timeline(
            id=self._next_id("S5"),
            domain=template.domain,
            track="environmental_freshness",
            actors=Actors(
                user=Actor(id="u1", role=role, org=org.lower().replace(" ", "_")),
                assistant_role="AI_Employee",
            ),
            initial_state=initial_state,
            events=events,
        )

    # --- v0.2 Track Generators ---

    def generate_hallucination_timeline(
        self,
        template: HallucinationTemplate,
    ) -> Timeline:
        """Generate a timeline testing hallucination resistance."""
        user_name = self._random_name()
        org = self._random_org()
        role = self._random_role(template.domain)
        base_time = self._base_time()
        current_time = base_time

        identity = IdentityRole(
            user_name=user_name.split()[0],
            authority=role,
            department=template.domain.title(),
            organization=org,
        )

        initial_state = InitialState(
            identity_role=identity,
            persistent_facts=[],
            working_set=[],
            environment={"now": base_time.isoformat()},
        )

        events = []

        # User states only what they stated - nothing more
        for i, fact in enumerate(template.stated_facts):
            current_time += timedelta(minutes=2)
            events.append(ConversationTurn(
                ts=current_time,
                speaker="user",
                text=fact,
            ))

            # Record as persistent fact
            events.append(StateWrite(
                ts=current_time,
                writes=[Write(
                    layer="persistent_facts",
                    key=f"stated_fact_{i}",
                    value=fact,
                    supersedes=None,
                )],
            ))

        # Query that tempts hallucination
        current_time += timedelta(minutes=5)
        ground_truth = GroundTruth(
            decision="acknowledge uncertainty or ask for clarification",
            must_mention=template.must_mention,
            must_not_mention=template.hallucination_triggers,
            allowed_sources=["persistent_facts"],
            reasoning=f"User never specified: {', '.join(template.unstated_facts[:3])}. "
                      f"System should not fabricate these details.",
        )

        events.append(Query(
            ts=current_time,
            prompt=template.query,
            ground_truth=ground_truth,
        ))

        return Timeline(
            id=self._next_id("S6"),
            domain=template.domain,
            track="hallucination_resistance",
            actors=Actors(
                user=Actor(id="u1", role=role, org=org.lower().replace(" ", "_")),
                assistant_role="AI_Employee",
            ),
            initial_state=initial_state,
            events=events,
        )

    def generate_scope_leak_timeline(
        self,
        template: ScopeLeakTemplate,
    ) -> Timeline:
        """Generate a timeline testing scope leak detection."""
        user_name = self._random_name()
        org = self._random_org()
        role = self._random_role(template.domain)
        base_time = self._base_time()
        current_time = base_time

        identity = IdentityRole(
            user_name=user_name.split()[0],
            authority=role,
            department=template.domain.title(),
            organization=org,
        )

        # Set up scoped context in working set (temporary)
        working_set = []
        if template.scoped_context:
            for key, value in template.scoped_context.items():
                if isinstance(value, list):
                    value = "; ".join(value)
                working_set.append(WorkingSetItem(
                    item_type="context",
                    content=f"[SCOPE: {template.scope_boundary}] {key}: {value}",
                    ts=base_time,
                ))

        # Set up global context in persistent facts
        initial_facts = []
        if template.global_context:
            for key, value in template.global_context.items():
                initial_facts.append(PersistentFact(
                    key=f"global_{key}",
                    value=str(value),
                    source="system",
                    ts=base_time,
                    is_valid=True,
                ))

        initial_state = InitialState(
            identity_role=identity,
            persistent_facts=initial_facts,
            working_set=working_set,
            environment={"now": base_time.isoformat()},
        )

        events = []

        # Establish the scoped context
        current_time += timedelta(minutes=2)
        scope_desc = template.scoped_context.get("task", template.scope_boundary)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"Let's do a {scope_desc}. This is just exploratory.",
        ))

        # Add some scoped content
        scoped_items = template.scoped_context.get(
            "ideas_discussed",
            template.scoped_context.get("hypotheticals",
            template.scoped_context.get("draft_terms",
            template.scoped_context.get("tentative", [])))
        )
        if isinstance(scoped_items, list):
            for item in scoped_items[:2]:
                current_time += timedelta(minutes=3)
                events.append(ConversationTurn(
                    ts=current_time,
                    speaker="user",
                    text=item,
                ))

        # End the scoped context
        current_time += timedelta(minutes=5)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"OK, that's enough {scope_desc} for now. Let's talk about real commitments.",
        ))

        # Query that tests scope boundary
        current_time += timedelta(minutes=5)
        ground_truth = GroundTruth(
            decision="use only global/committed information",
            must_mention=template.valid_sources[:2] if template.valid_sources else [],
            must_not_mention=template.leak_indicators[:4],
            allowed_sources=["persistent_facts"],
            reasoning=f"Scoped context ({template.scope_boundary}) should not leak. "
                      f"Leak type: {template.leak_type}",
        )

        events.append(Query(
            ts=current_time,
            prompt=template.query,
            ground_truth=ground_truth,
        ))

        return Timeline(
            id=self._next_id("S7"),
            domain=template.domain,
            track="scope_leak",
            actors=Actors(
                user=Actor(id="u1", role=role, org=org.lower().replace(" ", "_")),
                assistant_role="AI_Employee",
            ),
            initial_state=initial_state,
            events=events,
        )

    def generate_causality_timeline(
        self,
        template: CausalityTemplate,
    ) -> Timeline:
        """Generate a timeline testing causal dependency on state."""
        user_name = self._random_name()
        org = self._random_org()
        role = self._random_role(template.domain)
        base_time = self._base_time()
        current_time = base_time

        identity = IdentityRole(
            user_name=user_name.split()[0],
            authority=role,
            department=template.domain.title(),
            organization=org,
        )

        # Map template source to valid schema source
        source_map = {
            "finance_system": "system",
            "client_email": "user",
            "policy": "policy",
            "service_policy": "policy",
            "inventory_system": "system",
            "resource_planner": "system",
            "compliance_policy": "policy",
            "legal_policy": "policy",
            "contract_db": "system",
        }
        schema_source = source_map.get(template.constraint["source"], "system")

        # Establish the constraint as a persistent fact
        initial_facts = [
            PersistentFact(
                key=template.constraint["key"],
                value=template.constraint["value"],
                source=schema_source,
                ts=base_time - timedelta(days=7),
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

        # Remind of the constraint
        current_time += timedelta(minutes=2)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"Just to confirm: {template.constraint_description}",
        ))

        # Present the scenario
        current_time += timedelta(minutes=5)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=template.scenario,
        ))

        # Query for decision
        current_time += timedelta(minutes=2)
        ground_truth = GroundTruth(
            decision=template.expected_with_constraint,
            must_mention=template.constraint_indicators[:3],
            must_not_mention=[],  # We want to see the constraint was considered
            allowed_sources=["persistent_facts", "environment"],
            reasoning=f"Decision must depend on constraint: {template.constraint['value']}. "
                      f"Without this constraint, answer would be: {template.expected_without_constraint}",
        )

        events.append(Query(
            ts=current_time,
            prompt=template.decision_query,
            ground_truth=ground_truth,
        ))

        return Timeline(
            id=self._next_id("S8"),
            domain=template.domain,
            track="causality",
            actors=Actors(
                user=Actor(id="u1", role=role, org=org.lower().replace(" ", "_")),
                assistant_role="AI_Employee",
            ),
            initial_state=initial_state,
            events=events,
        )

    # -------------------------------------------------------------------------
    # Hard Mode Causality Generators
    # -------------------------------------------------------------------------

    def _map_source(self, source: str) -> str:
        """Map template source names to valid schema sources."""
        source_map = {
            "finance_system": "system",
            "finance": "system",
            "client_email": "user",
            "policy": "policy",
            "service_policy": "policy",
            "security_policy": "policy",
            "eng_policy": "policy",
            "hr_policy": "policy",
            "procurement_policy": "policy",
            "sales_policy": "policy",
            "inventory_system": "system",
            "resource_planner": "system",
            "resource_mgr": "system",
            "compliance_policy": "policy",
            "legal_policy": "policy",
            "legal": "policy",
            "contract_db": "system",
            "crm": "system",
            "cfo": "user",
            "product_manager": "user",
            "program_manager": "user",
            "tech_lead": "user",
            "rfp": "user",
            "wms": "system",
            "price_book": "system",
            "finance_policy": "policy",
            "hr_finance": "system",
            "talent_acquisition": "system",
            "pmo": "system",
            "sales": "user",
        }
        return source_map.get(source, "system")

    def generate_multi_constraint_timeline(
        self,
        template: MultiConstraintTemplate,
    ) -> Timeline:
        """Generate a timeline requiring multiple constraints to be satisfied."""
        user_name = self._random_name()
        org = self._random_org()
        role = self._random_role(template.domain)
        base_time = self._base_time()
        current_time = base_time

        identity = IdentityRole(
            user_name=user_name.split()[0],
            authority=role,
            department=template.domain.title(),
            organization=org,
        )

        # Establish ALL constraints as persistent facts
        initial_facts = []
        for i, constraint in enumerate(template.constraints):
            initial_facts.append(PersistentFact(
                key=constraint["key"],
                value=constraint["value"],
                source=self._map_source(constraint["source"]),
                ts=base_time - timedelta(days=7),
                is_valid=True,
            ))

        initial_state = InitialState(
            identity_role=identity,
            persistent_facts=initial_facts,
            working_set=[],
            environment={"now": base_time.isoformat()},
        )

        events = []

        # Remind of each constraint
        for constraint in template.constraints:
            current_time += timedelta(minutes=2)
            events.append(ConversationTurn(
                ts=current_time,
                speaker="user",
                text=f"Note: {constraint['value']}",
            ))

        # Present the scenario
        current_time += timedelta(minutes=5)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=template.scenario,
        ))

        # Query - must identify WHICH constraint(s) block
        current_time += timedelta(minutes=2)
        constraint_keys = [c["key"] for c in template.constraints]
        ground_truth = GroundTruth(
            decision=template.expected_decision,
            must_mention=template.blocking_constraints,  # Must identify the blocking constraint
            must_not_mention=[template.partial_satisfaction_trap[:20]],  # Avoid the trap answer
            allowed_sources=["persistent_facts", "environment"],
            reasoning=f"Must check ALL constraints. Trap answer: {template.partial_satisfaction_trap}",
        )

        events.append(Query(
            ts=current_time,
            prompt=template.decision_query,
            ground_truth=ground_truth,
        ))

        return Timeline(
            id=self._next_id("S8"),
            domain=template.domain,
            track="causality",
            actors=Actors(
                user=Actor(id="u1", role=role, org=org.lower().replace(" ", "_")),
                assistant_role="AI_Employee",
            ),
            initial_state=initial_state,
            events=events,
        )

    def generate_edge_case_timeline(
        self,
        template: EdgeCaseTemplate,
    ) -> Timeline:
        """Generate a timeline with edge case/threshold values."""
        user_name = self._random_name()
        org = self._random_org()
        role = self._random_role(template.domain)
        base_time = self._base_time()
        current_time = base_time

        identity = IdentityRole(
            user_name=user_name.split()[0],
            authority=role,
            department=template.domain.title(),
            organization=org,
        )

        initial_facts = [
            PersistentFact(
                key=template.constraint["key"],
                value=template.constraint["value"],
                source=self._map_source(template.constraint["source"]),
                ts=base_time - timedelta(days=7),
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

        # State the constraint
        current_time += timedelta(minutes=2)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"Current status: {template.constraint['value']}",
        ))

        # Present the edge case request
        current_time += timedelta(minutes=5)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=template.edge_case_request,
        ))

        # Query
        current_time += timedelta(minutes=2)
        ground_truth = GroundTruth(
            decision=template.expected_decision,
            must_mention=[template.violation_detail.split()[0]],  # Key part of the detail
            must_not_mention=[template.wrong_reasoning[:15]],  # Avoid wrong reasoning
            allowed_sources=["persistent_facts", "environment"],
            reasoning=f"Edge case: {template.violation_detail}. Wrong answer: {template.wrong_reasoning}",
        )

        events.append(Query(
            ts=current_time,
            prompt=template.decision_query,
            ground_truth=ground_truth,
        ))

        return Timeline(
            id=self._next_id("S8"),
            domain=template.domain,
            track="causality",
            actors=Actors(
                user=Actor(id="u1", role=role, org=org.lower().replace(" ", "_")),
                assistant_role="AI_Employee",
            ),
            initial_state=initial_state,
            events=events,
        )

    def generate_conflicting_constraint_timeline(
        self,
        template: ConflictingConstraintTemplate,
    ) -> Timeline:
        """Generate a timeline with two conflicting constraints."""
        user_name = self._random_name()
        org = self._random_org()
        role = self._random_role(template.domain)
        base_time = self._base_time()
        current_time = base_time

        identity = IdentityRole(
            user_name=user_name.split()[0],
            authority=role,
            department=template.domain.title(),
            organization=org,
        )

        initial_facts = [
            PersistentFact(
                key=template.constraint_a["key"],
                value=template.constraint_a["value"],
                source=self._map_source(template.constraint_a["source"]),
                ts=base_time - timedelta(days=7),
                is_valid=True,
            ),
            PersistentFact(
                key=template.constraint_b["key"],
                value=template.constraint_b["value"],
                source=self._map_source(template.constraint_b["source"]),
                ts=base_time - timedelta(days=7),
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

        # State both constraints
        current_time += timedelta(minutes=2)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"Requirement 1: {template.constraint_a['value']}",
        ))

        current_time += timedelta(minutes=2)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"Requirement 2: {template.constraint_b['value']}",
        ))

        # Present the scenario
        current_time += timedelta(minutes=5)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=template.scenario,
        ))

        # Query - must recognize the conflict
        current_time += timedelta(minutes=2)
        ground_truth = GroundTruth(
            decision=template.expected_decision,
            must_mention=["cannot", "both"],  # Must acknowledge conflict
            must_not_mention=[template.if_only_a[:10], template.if_only_b[:10]],  # Can't just pick one
            allowed_sources=["persistent_facts", "environment"],
            reasoning=f"Conflicting constraints. If only A: {template.if_only_a}. If only B: {template.if_only_b}",
        )

        events.append(Query(
            ts=current_time,
            prompt=template.decision_query,
            ground_truth=ground_truth,
        ))

        return Timeline(
            id=self._next_id("S8"),
            domain=template.domain,
            track="causality",
            actors=Actors(
                user=Actor(id="u1", role=role, org=org.lower().replace(" ", "_")),
                assistant_role="AI_Employee",
            ),
            initial_state=initial_state,
            events=events,
        )

    def generate_chain_dependency_timeline(
        self,
        template: ChainDependencyTemplate,
    ) -> Timeline:
        """Generate a timeline with chained fact dependencies."""
        user_name = self._random_name()
        org = self._random_org()
        role = self._random_role(template.domain)
        base_time = self._base_time()
        current_time = base_time

        identity = IdentityRole(
            user_name=user_name.split()[0],
            authority=role,
            department=template.domain.title(),
            organization=org,
        )

        initial_facts = [
            PersistentFact(
                key=template.initial_fact["key"],
                value=template.initial_fact["value"],
                source=self._map_source(template.initial_fact["source"]),
                ts=base_time - timedelta(days=7),
                is_valid=True,
            ),
            PersistentFact(
                key=template.derived_fact["key"],
                value=template.derived_fact["value"],
                source=self._map_source(template.derived_fact["source"]),
                ts=base_time - timedelta(days=7),
                is_valid=True,
            ),
            PersistentFact(
                key=template.final_constraint["key"],
                value=template.final_constraint["value"],
                source=self._map_source(template.final_constraint["source"]),
                ts=base_time - timedelta(days=7),
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

        # State the chain of facts
        current_time += timedelta(minutes=2)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"Base fact: {template.initial_fact['value']}",
        ))

        current_time += timedelta(minutes=2)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"Policy: {template.derived_fact['value']}",
        ))

        current_time += timedelta(minutes=2)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"Also: {template.final_constraint['value']}",
        ))

        # Present the scenario
        current_time += timedelta(minutes=5)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=template.scenario,
        ))

        # Query - must follow the chain
        current_time += timedelta(minutes=2)
        # Extract key reasoning steps
        reasoning_keywords = []
        for step in template.reasoning_chain:
            words = step.split()[:2]
            reasoning_keywords.extend(words)

        ground_truth = GroundTruth(
            decision=template.expected_decision,
            must_mention=reasoning_keywords[:3],  # Key parts of the chain
            must_not_mention=[template.chain_break_error[:15]],  # Avoid breaking the chain
            allowed_sources=["persistent_facts", "environment"],
            reasoning=f"Chain: {' → '.join(template.reasoning_chain)}",
        )

        events.append(Query(
            ts=current_time,
            prompt=template.decision_query,
            ground_truth=ground_truth,
        ))

        return Timeline(
            id=self._next_id("S8"),
            domain=template.domain,
            track="causality",
            actors=Actors(
                user=Actor(id="u1", role=role, org=org.lower().replace(" ", "_")),
                assistant_role="AI_Employee",
            ),
            initial_state=initial_state,
            events=events,
        )

    def generate_aggregation_timeline(
        self,
        template: AggregationTemplate,
    ) -> Timeline:
        """Generate a timeline requiring aggregation across items."""
        user_name = self._random_name()
        org = self._random_org()
        role = self._random_role(template.domain)
        base_time = self._base_time()
        current_time = base_time

        identity = IdentityRole(
            user_name=user_name.split()[0],
            authority=role,
            department=template.domain.title(),
            organization=org,
        )

        # The constraint
        initial_facts = [
            PersistentFact(
                key=template.constraint["key"],
                value=template.constraint["value"],
                source=self._map_source(template.constraint["source"]),
                ts=base_time - timedelta(days=30),
                is_valid=True,
            ),
        ]

        # Add each item as a fact
        for i, item in enumerate(template.items):
            initial_facts.append(PersistentFact(
                key=f"item_{i}",
                value=f"{item['name']}: {item['value']}",
                source="system",
                ts=base_time - timedelta(days=30-i*10),
                is_valid=True,
            ))

        initial_state = InitialState(
            identity_role=identity,
            persistent_facts=initial_facts,
            working_set=[],
            environment={"now": base_time.isoformat()},
        )

        events = []

        # State the constraint
        current_time += timedelta(minutes=2)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"Policy reminder: {template.constraint['value']}",
        ))

        # List each item
        for item in template.items:
            current_time += timedelta(minutes=2)
            events.append(ConversationTurn(
                ts=current_time,
                speaker="user",
                text=f"{item['name']}: {item['value']}",
            ))

        # Query - must aggregate
        current_time += timedelta(minutes=5)
        ground_truth = GroundTruth(
            decision=template.expected_decision,
            must_mention=[template.aggregation_detail.split("=")[0].strip()],  # The sum
            must_not_mention=[template.per_item_trap[:15]],  # Avoid per-item reasoning
            allowed_sources=["persistent_facts", "environment"],
            reasoning=f"Must aggregate: {template.aggregation_detail}. Trap: {template.per_item_trap}",
        )

        events.append(Query(
            ts=current_time,
            prompt=template.decision_query,
            ground_truth=ground_truth,
        ))

        return Timeline(
            id=self._next_id("S8"),
            domain=template.domain,
            track="causality",
            actors=Actors(
                user=Actor(id="u1", role=role, org=org.lower().replace(" ", "_")),
                assistant_role="AI_Employee",
            ),
            initial_state=initial_state,
            events=events,
        )

    def generate_repair_timeline(
        self,
        template: RepairChain,
    ) -> Timeline:
        """Generate a timeline testing repair/correction propagation."""
        user_name = self._random_name()
        org = self._random_org()
        role = self._random_role(template.domain)
        base_time = self._base_time()
        current_time = base_time

        # Map template sources to valid schema sources
        source_map = {
            "price_list": "system",
            "sales_rep": "user",
            "project_plan": "system",
            "backend_team": "user",
            "resource_plan": "system",
            "finance": "system",
            "calendar": "system",
        }

        identity = IdentityRole(
            user_name=user_name.split()[0],
            authority=role,
            department=template.domain.title(),
            organization=org,
        )

        # Start with the initial (wrong) fact
        template_source = template.initial_fact["source"]
        mapped_source = source_map.get(template_source, "user")
        initial_facts = [
            PersistentFact(
                key=template.initial_fact["key"],
                value=template.initial_fact["value"],
                source=mapped_source,
                ts=base_time,
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

        # Establish the initial fact
        current_time += timedelta(minutes=2)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"Based on {template.initial_fact['source']}: {template.initial_fact['value']}",
        ))

        # Make decision based on wrong fact
        current_time += timedelta(minutes=5)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"So: {template.initial_decision}",
        ))

        events.append(StateWrite(
            ts=current_time,
            writes=[Write(
                layer="persistent_facts",
                key="derived_decision",
                value=template.initial_decision,
                supersedes=None,
            )],
        ))

        # Correction arrives
        current_time += timedelta(minutes=30)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"⚠️ CORRECTION: {template.correction['reason']}. "
                 f"The value was: {template.initial_fact['value']}. "
                 f"The ACTUAL value is: {template.correction['new_value']}",
        ))

        # Supersession of the wrong fact
        events.append(Supersession(
            ts=current_time,
            writes=[Write(
                layer="persistent_facts",
                key=f"{template.correction['key']}_corrected",
                value=template.correction["new_value"],
                supersedes=template.initial_fact["key"],
            )],
        ))

        # Explicit recalculation prompt
        current_time += timedelta(minutes=2)
        events.append(ConversationTurn(
            ts=current_time,
            speaker="user",
            text=f"⚠️ This means our previous conclusion '{template.initial_decision}' "
                 f"was based on incorrect information and needs to be RECALCULATED.",
        ))

        # Supersede the derived decision with explicit invalidation
        events.append(Supersession(
            ts=current_time,
            writes=[Write(
                layer="persistent_facts",
                key="derived_decision_corrected",
                value=f"[INVALIDATED - was based on wrong data: {template.initial_fact['value']}] "
                      f"Original conclusion: {template.initial_decision}",
                supersedes="derived_decision",
            )],
        ))

        # Query that tests propagation
        current_time += timedelta(minutes=10)
        ground_truth = GroundTruth(
            decision=template.expected_updated_decision,
            must_mention=template.corrected_must_mention[:3],
            must_not_mention=template.original_must_not_mention[:3],
            allowed_sources=["persistent_facts"],
            reasoning=f"Correction must propagate: {template.correction['reason']}. "
                      f"Original decision was based on wrong value.",
        )

        events.append(Query(
            ts=current_time,
            prompt=template.propagation_query,
            ground_truth=ground_truth,
        ))

        return Timeline(
            id=self._next_id("S9"),
            domain=template.domain,
            track="repair_propagation",
            actors=Actors(
                user=Actor(id="u1", role=role, org=org.lower().replace(" ", "_")),
                assistant_role="AI_Employee",
            ),
            initial_state=initial_state,
            events=events,
        )

    def generate_brutal_timeline(
        self,
        scenario: BrutalScenario,
    ) -> Timeline:
        """Generate a timeline from a brutal realistic scenario."""
        base_time = self._base_time()

        # Use first speaker's info for identity
        first_event = scenario.events[0]
        identity = IdentityRole(
            user_name=first_event.speaker,
            authority=first_event.role,
            department=scenario.domain.title(),
            organization="TestCorp",
        )

        initial_state = InitialState(
            identity_role=identity,
            persistent_facts=[],
            working_set=[],
            environment={"now": base_time.isoformat()},
        )

        events = []
        current_time = base_time
        fact_counter = 0

        # Process each event in the brutal scenario
        for brutal_event in scenario.events:
            # Advance time based on day
            day_offset = timedelta(days=brutal_event.day - 1)
            # Parse time like "9:00 AM" -> hours
            time_parts = brutal_event.time.replace(":", " ").replace("AM", "").replace("PM", "").split()
            hour = int(time_parts[0])
            if "PM" in brutal_event.time and hour != 12:
                hour += 12

            current_time = base_time + day_offset + timedelta(hours=hour - 9)

            if brutal_event.event_type == "query":
                # This is a final query - skip for now, we'll add final queries at end
                continue

            # Add conversation turn
            events.append(ConversationTurn(
                ts=current_time,
                speaker="user",
                text=f"[{brutal_event.speaker} ({brutal_event.role})]: {brutal_event.content}",
            ))

            # For instructions and corrections, record state
            if brutal_event.event_type in ("instruction", "correction"):
                fact_counter += 1
                if brutal_event.event_type == "correction":
                    # This supersedes something
                    events.append(Supersession(
                        ts=current_time,
                        writes=[Write(
                            layer="persistent_facts",
                            key=f"fact_{fact_counter}",
                            value=brutal_event.content,
                            supersedes=f"fact_{fact_counter - 1}" if fact_counter > 1 else None,
                        )],
                    ))
                else:
                    events.append(StateWrite(
                        ts=current_time,
                        writes=[Write(
                            layer="persistent_facts",
                            key=f"fact_{fact_counter}",
                            value=brutal_event.content,
                            supersedes=None,
                        )],
                    ))

        # Add final queries from the scenario
        for i, query_spec in enumerate(scenario.final_queries):
            current_time += timedelta(minutes=10 + i * 5)

            ground_truth = GroundTruth(
                decision="see critical state",
                must_mention=query_spec.get("must_mention", [])[:3],
                must_not_mention=query_spec.get("must_not_mention", [])[:3],
                allowed_sources=["persistent_facts"],
                reasoning=f"After all events, critical state is: {list(scenario.critical_state.items())[:2]}",
            )

            events.append(Query(
                ts=current_time,
                prompt=query_spec["query"],
                ground_truth=ground_truth,
            ))

        return Timeline(
            id=self._next_id("S10"),
            domain=scenario.domain,
            track="brutal_realistic",
            actors=Actors(
                user=Actor(id="u1", role="Multi-stakeholder", org="testcorp"),
                assistant_role="AI_Employee",
            ),
            initial_state=initial_state,
            events=events,
        )

    def generate_track(
        self,
        track: str,
        count: int = 100,
        adversarial_ratio: float = 0.3,
    ) -> Iterator[Timeline]:
        """Generate timelines for a track.

        Args:
            track: Track name (e.g., "supersession")
            count: Number of timelines to generate
            adversarial_ratio: Fraction of timelines to make adversarial
        """
        if track == "supersession":
            templates = SUPERSESSION_TEMPLATES
            for i in range(count):
                template = self.rng.choice(templates)
                adversarial = self.rng.random() < adversarial_ratio
                yield self.generate_supersession_timeline(template, adversarial=adversarial)

        elif track == "commitment_durability":
            templates = COMMITMENT_TEMPLATES
            for i in range(count):
                template = self.rng.choice(templates)
                yield self.generate_commitment_timeline(template)

        elif track == "interruption_resumption":
            templates = INTERRUPTION_TEMPLATES
            for i in range(count):
                template = self.rng.choice(templates)
                yield self.generate_interruption_timeline(template)

        elif track == "scope_permission":
            templates = PERMISSION_TEMPLATES
            for i in range(count):
                template = self.rng.choice(templates)
                yield self.generate_permission_timeline(template)

        elif track == "environmental_freshness":
            templates = ENVIRONMENTAL_TEMPLATES
            for i in range(count):
                template = self.rng.choice(templates)
                yield self.generate_environmental_timeline(template)

        # v0.2 tracks
        elif track == "hallucination_resistance":
            templates = HALLUCINATION_TEMPLATES
            for i in range(count):
                template = self.rng.choice(templates)
                yield self.generate_hallucination_timeline(template)

        elif track == "scope_leak":
            templates = SCOPE_LEAK_TEMPLATES
            for i in range(count):
                template = self.rng.choice(templates)
                yield self.generate_scope_leak_timeline(template)

        elif track == "causality":
            # Mix simple and hard mode templates (50% hard)
            from statebench.generator.templates.causality import (
                MULTI_CONSTRAINT_TEMPLATES,
                EDGE_CASE_TEMPLATES,
                CONFLICTING_TEMPLATES,
                CHAIN_DEPENDENCY_TEMPLATES,
                AGGREGATION_TEMPLATES,
            )

            simple_templates = CAUSALITY_TEMPLATES
            hard_templates = (
                MULTI_CONSTRAINT_TEMPLATES +
                EDGE_CASE_TEMPLATES +
                CONFLICTING_TEMPLATES +
                CHAIN_DEPENDENCY_TEMPLATES +
                AGGREGATION_TEMPLATES
            )

            for i in range(count):
                # 50% chance of hard mode
                if self.rng.random() < 0.5 and hard_templates:
                    template = self.rng.choice(hard_templates)
                    # Dispatch based on template type
                    if isinstance(template, MultiConstraintTemplate):
                        yield self.generate_multi_constraint_timeline(template)
                    elif isinstance(template, EdgeCaseTemplate):
                        yield self.generate_edge_case_timeline(template)
                    elif isinstance(template, ConflictingConstraintTemplate):
                        yield self.generate_conflicting_constraint_timeline(template)
                    elif isinstance(template, ChainDependencyTemplate):
                        yield self.generate_chain_dependency_timeline(template)
                    elif isinstance(template, AggregationTemplate):
                        yield self.generate_aggregation_timeline(template)
                else:
                    template = self.rng.choice(simple_templates)
                    yield self.generate_causality_timeline(template)

        elif track == "repair_propagation":
            templates = REPAIR_CHAIN_TEMPLATES
            for i in range(count):
                template = self.rng.choice(templates)
                yield self.generate_repair_timeline(template)

        elif track == "brutal_realistic":
            # For brutal, we cycle through scenarios since there are only 3
            scenarios = BRUTAL_SCENARIOS
            for i in range(count):
                scenario = scenarios[i % len(scenarios)]
                yield self.generate_brutal_timeline(scenario)

        else:
            raise ValueError(
                f"Unknown track: {track}. Available: supersession, commitment_durability, "
                f"interruption_resumption, scope_permission, environmental_freshness, "
                f"hallucination_resistance, scope_leak, causality, repair_propagation, brutal_realistic"
            )


def generate_dataset(
    output_path: Path,
    tracks: list[str],
    count_per_track: int = 100,
    seed: int | None = None,
) -> int:
    """Generate a complete benchmark dataset.

    Args:
        output_path: Path to write JSONL output
        tracks: List of track names to generate
        count_per_track: Number of timelines per track
        seed: Random seed for reproducibility

    Returns:
        Total number of timelines generated
    """
    generator = TimelineGenerator(seed=seed)
    total = 0

    with open(output_path, "w") as f:
        for track in tracks:
            for timeline in generator.generate_track(track, count=count_per_track):
                f.write(timeline.model_dump_json() + "\n")
                total += 1

    return total
