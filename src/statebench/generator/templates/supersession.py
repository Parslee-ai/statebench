"""Templates for Track 1: Supersession Integrity.

This track tests whether systems correctly handle fact invalidation.
Each template defines a scenario where a decision/fact becomes invalid
and the system must not resurrect it.
"""

from dataclasses import dataclass, field


@dataclass
class SupersessionTemplate:
    """A template for generating supersession test cases."""
    name: str
    domain: str
    description: str

    # Entity generators (called with random seed)
    entity_names: list[str]
    entity_type: str  # e.g., "vendor", "customer", "project"

    # Initial fact pattern
    initial_fact_key: str
    initial_fact_template: str  # e.g., "approved {amount} for {entity}"
    initial_fact_source: str

    # Supersession pattern
    supersession_fact_template: str  # e.g., "do not proceed with {entity}; {reason}"
    supersession_reasons: list[str]

    # Query pattern
    query_template: str  # e.g., "Should I proceed with {entity}?"
    correct_decision: str

    # Constraint patterns
    must_mention_pattern: str  # What the superseded state requires mentioning
    must_not_mention_pattern: str  # What the original (superseded) state said

    # Optional: policy context
    policy_key: str | None = None
    policy_template: str | None = None

    # Number of supersessions (1-4)
    min_supersessions: int = 1
    max_supersessions: int = 2

    # --- "Maintain" (should-NOT-supersede) variant (Belief-R guardrail) ---
    # A later event that is UPDATE-FLAVORED and shares the same entity (tempting a
    # supersession detector to misfire) but does NOT invalidate the original fact —
    # and crucially does not state the answer. Measures False Supersession Rate
    # (FSR): wrongly retiring a still-valid fact. Templates without these fields
    # generate no maintain cases. FSR is scored behaviorally (did the system still
    # affirm the fact? decision="yes"), not by forbidden vocabulary.
    maintain_event_template: str | None = None  # update-flavored, non-invalidating NL, e.g. "finalized the {entity} renewal paperwork"
    maintain_reasons: list[str] = field(default_factory=list)
    maintain_query_template: str | None = None  # asks about the ORIGINAL (still-valid) fact
    maintain_must_mention: str | None = None  # surviving value the system should still affirm


# --- Procurement Domain Templates ---

VENDOR_RENEWAL = SupersessionTemplate(
    name="vendor_renewal",
    domain="procurement",
    description="Vendor renewal approval that gets revoked",
    entity_names=["VendorX", "Acme Supplies", "TechCorp", "DataServices", "CloudPro"],
    entity_type="vendor",
    initial_fact_key="renew_{entity_id}",
    initial_fact_template="approved {duration} renewal for {entity}",
    initial_fact_source="decision",
    supersession_fact_template="do not renew {entity}; {reason}",
    supersession_reasons=[
        "renegotiate terms first",
        "budget constraints",
        "switching to competitor",
        "pending legal review",
        "contract terms unfavorable",
    ],
    query_template="Should I send the renewal email to {entity} now?",
    correct_decision="no",
    must_mention_pattern="do not renew|renegotiate|{reason}",
    must_not_mention_pattern="approved|renewal approved|proceed with renewal",
    policy_key="vendor_policy",
    policy_template="No auto-renew without {approver} review",
    min_supersessions=1,
    max_supersessions=2,
    maintain_event_template="finalized the {entity} renewal paperwork and filed it with procurement",
    maintain_reasons=["paperwork filed", "logged in the system", "contact updated"],
    maintain_query_template="Is the {entity} renewal still approved?",
    maintain_must_mention="renewal|approved",
)

PURCHASE_APPROVAL = SupersessionTemplate(
    name="purchase_approval",
    domain="procurement",
    description="Purchase approval that gets cancelled",
    entity_names=["Server Upgrade", "Software License", "Office Equipment", "Training Program"],
    entity_type="purchase",
    initial_fact_key="purchase_{entity_id}",
    initial_fact_template="approved purchase of {entity} for ${amount}",
    initial_fact_source="decision",
    supersession_fact_template="purchase of {entity} cancelled; {reason}",
    supersession_reasons=[
        "budget reallocated",
        "found better alternative",
        "project postponed",
        "duplicate request",
    ],
    query_template="Can I submit the PO for {entity}?",
    correct_decision="no",
    must_mention_pattern="cancelled|not approved|{reason}",
    must_not_mention_pattern="approved purchase|approved for \\$",
    min_supersessions=1,
    max_supersessions=3,
    maintain_event_template="updated the {entity} request with the assigned PO number",
    maintain_reasons=["PO number assigned", "logged in finance", "added to the queue"],
    maintain_query_template="Can I still submit the PO for {entity}?",
    maintain_must_mention="proceed|approved|yes",
)


# --- Sales Domain Templates ---

DISCOUNT_APPROVAL = SupersessionTemplate(
    name="discount_approval",
    domain="sales",
    description="Discount approval that gets revoked",
    entity_names=["Globex", "Initech", "Umbrella Corp", "Wayne Enterprises", "Stark Industries"],
    entity_type="customer",
    initial_fact_key="{entity_id}_discount",
    initial_fact_template="{discount_pct}% discount approved for {entity}",
    initial_fact_source="decision",
    supersession_fact_template="discount for {entity} revoked; {reason}",
    supersession_reasons=[
        "VP rescinded approval",
        "customer credit issue",
        "margin requirements",
        "policy violation",
    ],
    query_template="Can I offer {entity} the special discount we discussed?",
    correct_decision="no",
    must_mention_pattern="revoked|standard policy|{reason}",
    must_not_mention_pattern="approved|special discount|{discount_pct}%",
    policy_key="discount_policy",
    policy_template="Max {max_discount}% discount without VP approval",
    min_supersessions=1,
    max_supersessions=2,
    maintain_event_template="updated the {entity} account notes after a check-in call",
    maintain_reasons=["notes updated", "check-in logged", "account reviewed"],
    maintain_query_template="Can I still offer {entity} the {discount_pct}% discount?",
    maintain_must_mention="{discount_pct}%|approved|yes",
)

DEAL_TERMS = SupersessionTemplate(
    name="deal_terms",
    domain="sales",
    description="Deal terms that change multiple times",
    entity_names=["Q4 Enterprise Deal", "Government Contract", "Partnership Agreement"],
    entity_type="deal",
    initial_fact_key="{entity_id}_terms",
    initial_fact_template="{entity} terms: {initial_terms}",
    initial_fact_source="decision",
    supersession_fact_template="{entity} terms updated: {new_terms}",
    supersession_reasons=[
        "customer requested change",
        "legal review required change",
        "pricing update",
    ],
    query_template="What are the current terms for {entity}?",
    correct_decision="{final_terms}",
    must_mention_pattern="{final_terms}",
    must_not_mention_pattern="{superseded_terms}",
    min_supersessions=2,
    max_supersessions=4,
)


# --- Project Domain Templates ---

DEADLINE_CHANGE = SupersessionTemplate(
    name="deadline_change",
    domain="project",
    description="Project deadline that changes multiple times",
    entity_names=["Feature Launch", "API Migration", "Security Audit", "Platform Upgrade"],
    entity_type="project",
    initial_fact_key="{entity_id}_deadline",
    initial_fact_template="{entity} deadline: {date}",
    initial_fact_source="decision",
    supersession_fact_template="{entity} deadline moved to {new_date}",
    supersession_reasons=[
        "stakeholder request",
        "resource constraints",
        "dependency delay",
        "scope change",
    ],
    query_template="When is {entity} due?",
    correct_decision="{final_date}",
    must_mention_pattern="{final_date}",
    must_not_mention_pattern="{superseded_dates}",
    min_supersessions=2,
    max_supersessions=4,
)

RESOURCE_ALLOCATION = SupersessionTemplate(
    name="resource_allocation",
    domain="project",
    description="Resource allocation that changes",
    entity_names=["Alpha Team", "Backend Squad", "Mobile Team", "Data Team"],
    entity_type="team",
    initial_fact_key="{entity_id}_allocation",
    initial_fact_template="{entity} allocated to {project}",
    initial_fact_source="decision",
    supersession_fact_template="{entity} reallocated to {new_project}",
    supersession_reasons=[
        "priority shift",
        "project cancelled",
        "emergency request",
    ],
    query_template="Which project is {entity} working on?",
    correct_decision="{final_project}",
    must_mention_pattern="{final_project}",
    must_not_mention_pattern="{superseded_projects}",
    min_supersessions=1,
    max_supersessions=3,
)


# --- HR Domain Templates ---

POLICY_CHANGE = SupersessionTemplate(
    name="policy_change",
    domain="hr",
    description="HR policy that gets updated",
    entity_names=["Remote Work Policy", "PTO Policy", "Expense Policy", "Travel Policy"],
    entity_type="policy",
    initial_fact_key="{entity_id}",
    initial_fact_template="{entity}: {initial_rule}",
    initial_fact_source="policy",
    supersession_fact_template="{entity} updated: {new_rule}",
    supersession_reasons=[
        "executive decision",
        "legal requirement",
        "budget change",
    ],
    query_template="What is the current {entity}?",
    correct_decision="{final_rule}",
    must_mention_pattern="{final_rule}",
    must_not_mention_pattern="{superseded_rules}",
    min_supersessions=1,
    max_supersessions=2,
)


# --- Support Domain Templates ---

TICKET_STATUS = SupersessionTemplate(
    name="ticket_status",
    domain="support",
    description="Support ticket resolution that changes",
    entity_names=["TICKET-1234", "CASE-5678", "INC-9012", "SR-3456"],
    entity_type="ticket",
    initial_fact_key="{entity_id}_status",
    initial_fact_template="{entity} resolution: {initial_resolution}",
    initial_fact_source="decision",
    supersession_fact_template="{entity} status changed: {new_status}",
    supersession_reasons=[
        "customer reported issue persists",
        "root cause identified",
        "escalation required",
    ],
    query_template="What is the status of {entity}?",
    correct_decision="{final_status}",
    must_mention_pattern="{final_status}",
    must_not_mention_pattern="{superseded_statuses}",
    min_supersessions=1,
    max_supersessions=3,
)


# --- All Templates ---

SUPERSESSION_TEMPLATES = [
    VENDOR_RENEWAL,
    PURCHASE_APPROVAL,
    DISCOUNT_APPROVAL,
    DEAL_TERMS,
    DEADLINE_CHANGE,
    RESOURCE_ALLOCATION,
    POLICY_CHANGE,
    TICKET_STATUS,
]


def get_templates_by_domain(domain: str) -> list[SupersessionTemplate]:
    """Get all supersession templates for a given domain."""
    return [t for t in SUPERSESSION_TEMPLATES if t.domain == domain]
