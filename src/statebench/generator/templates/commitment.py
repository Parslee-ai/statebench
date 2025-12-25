"""Templates for Track 2: Commitment and Preference Durability.

This track tests whether systems correctly distinguish between:
- Preferences: Can change over time (e.g., "I prefer bullet points")
- Commitments: Durable promises that persist (e.g., "I'll have it done by Friday")

The system must:
- Update preferences when they change
- Maintain commitments even when preferences change
- Not confuse preference changes for commitment changes
"""

from dataclasses import dataclass


@dataclass
class CommitmentTemplate:
    """A template for generating commitment durability test cases."""
    name: str
    domain: str
    description: str

    # Entity info
    entity_names: list[str]
    entity_type: str

    # Initial commitment (durable)
    commitment_key: str
    commitment_template: str
    commitment_source: str

    # Initial preference (changeable)
    preference_key: str
    preference_template: str
    preference_values: list[str]  # Different preference options

    # Preference change pattern
    new_preference_values: list[str]

    # Query patterns
    commitment_query: str  # Asks about the commitment
    preference_query: str  # Asks about the preference

    # Ground truth
    commitment_persists: bool  # Should commitment remain after preference change?


# --- Sales Domain ---

COMMUNICATION_STYLE = CommitmentTemplate(
    name="communication_style",
    domain="sales",
    description="Communication preference changes but delivery commitment persists",
    entity_names=["Q4 Proposal", "Enterprise Deal", "Partnership Deck", "RFP Response"],
    entity_type="deliverable",
    commitment_key="{entity_id}_deadline",
    commitment_template="Will deliver {entity} by {date}",
    commitment_source="commitment",
    preference_key="format_preference",
    preference_template="Prefer {style} format for documents",
    preference_values=["bullet points", "detailed paragraphs", "executive summary"],
    new_preference_values=["visual slides", "one-pager", "formal report"],
    commitment_query="When will {entity} be ready?",
    preference_query="What format should I use for {entity}?",
    commitment_persists=True,
)

MEETING_CADENCE = CommitmentTemplate(
    name="meeting_cadence",
    domain="sales",
    description="Meeting time preference changes but deal commitment persists",
    entity_names=["Globex", "Initech", "Umbrella Corp", "Stark Industries"],
    entity_type="customer",
    commitment_key="{entity_id}_deal",
    commitment_template="Committed to close {entity} deal by end of quarter",
    commitment_source="commitment",
    preference_key="meeting_preference",
    preference_template="Prefer {time} meetings",
    preference_values=["morning", "afternoon", "end of day"],
    new_preference_values=["lunch meetings", "async updates", "weekly syncs"],
    commitment_query="Are we still on track to close {entity} this quarter?",
    preference_query="When should we schedule the next {entity} meeting?",
    commitment_persists=True,
)


# --- Project Domain ---

REVIEW_PROCESS = CommitmentTemplate(
    name="review_process",
    domain="project",
    description="Review preference changes but milestone commitment persists",
    entity_names=["Alpha Release", "Beta Launch", "Security Audit", "API v2"],
    entity_type="milestone",
    commitment_key="{entity_id}_milestone",
    commitment_template="Committed to complete {entity} by {date}",
    commitment_source="commitment",
    preference_key="review_preference",
    preference_template="Prefer {review_type} for code reviews",
    preference_values=["async PR reviews", "pair programming", "weekly review meetings"],
    new_preference_values=["AI-assisted review", "mob programming", "daily standups"],
    commitment_query="When is {entity} scheduled to complete?",
    preference_query="How should we handle reviews for {entity}?",
    commitment_persists=True,
)

TOOL_PREFERENCE = CommitmentTemplate(
    name="tool_preference",
    domain="project",
    description="Tool preference changes but project scope commitment persists",
    entity_names=["Dashboard Redesign", "Mobile App", "Analytics Pipeline", "Auth System"],
    entity_type="project",
    commitment_key="{entity_id}_scope",
    commitment_template="Committed to deliver {entity} with all P0 features",
    commitment_source="commitment",
    preference_key="tool_preference",
    preference_template="Prefer using {tool} for this project",
    preference_values=["React", "Vue", "Svelte"],
    new_preference_values=["Next.js", "Nuxt", "SvelteKit"],
    commitment_query="What features are committed for {entity}?",
    preference_query="What framework should we use for {entity}?",
    commitment_persists=True,
)


# --- HR Domain ---

WORK_ARRANGEMENT = CommitmentTemplate(
    name="work_arrangement",
    domain="hr",
    description="Work location preference changes but role commitment persists",
    entity_names=["Engineering Lead", "Product Manager", "Senior Developer", "Tech Lead"],
    entity_type="role",
    commitment_key="{entity_id}_role",
    commitment_template="Committed to {entity} role for minimum 18 months",
    commitment_source="commitment",
    preference_key="location_preference",
    preference_template="Prefer {location} work arrangement",
    preference_values=["fully remote", "hybrid 2 days", "in-office"],
    new_preference_values=["hybrid 3 days", "remote with monthly onsite", "flexible"],
    commitment_query="How long is the commitment for {entity} role?",
    preference_query="What's the work arrangement for {entity}?",
    commitment_persists=True,
)


# --- Support Domain ---

RESPONSE_STYLE = CommitmentTemplate(
    name="response_style",
    domain="support",
    description="Response style preference changes but SLA commitment persists",
    entity_names=["Enterprise Support", "Premium Tier", "Standard Support", "VIP Account"],
    entity_type="support_tier",
    commitment_key="{entity_id}_sla",
    commitment_template="Committed to {sla_hours}-hour response time for {entity}",
    commitment_source="commitment",
    preference_key="response_style",
    preference_template="Prefer {style} responses",
    preference_values=["detailed technical", "concise summary", "step-by-step"],
    new_preference_values=["video walkthroughs", "annotated screenshots", "live chat"],
    commitment_query="What's the SLA for {entity}?",
    preference_query="How should responses be formatted for {entity}?",
    commitment_persists=True,
)


# --- Procurement Domain ---

VENDOR_TERMS = CommitmentTemplate(
    name="vendor_terms",
    domain="procurement",
    description="Payment preference changes but contract commitment persists",
    entity_names=["CloudProvider", "DataVendor", "SaaS Platform", "Infrastructure Co"],
    entity_type="vendor",
    commitment_key="{entity_id}_contract",
    commitment_template="Committed to {duration} contract with {entity}",
    commitment_source="commitment",
    preference_key="payment_preference",
    preference_template="Prefer {payment_type} payment terms",
    preference_values=["monthly billing", "annual prepay", "quarterly"],
    new_preference_values=["usage-based", "milestone-based", "hybrid billing"],
    commitment_query="What's the contract length with {entity}?",
    preference_query="How should we structure payments to {entity}?",
    commitment_persists=True,
)


COMMITMENT_TEMPLATES = [
    COMMUNICATION_STYLE,
    MEETING_CADENCE,
    REVIEW_PROCESS,
    TOOL_PREFERENCE,
    WORK_ARRANGEMENT,
    RESPONSE_STYLE,
    VENDOR_TERMS,
]


def get_commitment_templates_by_domain(domain: str) -> list[CommitmentTemplate]:
    """Get all commitment templates for a given domain."""
    return [t for t in COMMITMENT_TEMPLATES if t.domain == domain]
