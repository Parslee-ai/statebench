"""Templates for Track 3: Interruption and Resumption.

This track tests whether systems correctly handle:
- Multi-step tasks that get interrupted by unrelated work
- Resumption of original task after interruption
- Preservation of durable commitments across interruptions
- Clearing of working context that shouldn't persist

The system must:
- Remember the original objective when resuming
- Not contaminate resumed task with interruption context
- Maintain commitments made before interruption
- Clear ephemeral working state from interruption
"""

from dataclasses import dataclass


@dataclass
class InterruptionTemplate:
    """A template for generating interruption/resumption test cases."""
    name: str
    domain: str
    description: str

    # Primary task
    primary_task: str
    primary_entity: str
    primary_steps: list[str]  # Multi-step task
    primary_commitment: str  # Commitment made during primary task

    # Interruption
    interruption_trigger: str  # What causes the interruption
    interruption_task: str
    interruption_entity: str
    interruption_context: list[str]  # Working context from interruption

    # Resumption
    resumption_trigger: str
    resumption_query: str

    # Ground truth
    should_remember: list[str]  # From primary task
    should_forget: list[str]  # From interruption (ephemeral)
    should_not_confuse: list[str]  # Don't mix contexts


# --- Sales Domain ---

DEAL_REVIEW_INTERRUPTED = InterruptionTemplate(
    name="deal_review_interrupted",
    domain="sales",
    description="Deal preparation interrupted by urgent customer call",
    primary_task="Preparing quarterly business review for Globex",
    primary_entity="Globex QBR",
    primary_steps=[
        "Pulled Q3 metrics for Globex",
        "Identified 3 upsell opportunities",
        "Committed to present renewal options at QBR",
    ],
    primary_commitment="Will present 3 renewal tiers at the Globex QBR",
    interruption_trigger="Urgent call from Initech about service outage",
    interruption_task="Handling Initech escalation",
    interruption_entity="Initech",
    interruption_context=[
        "Initech experiencing API timeouts",
        "Escalated to engineering",
        "Offered 15% credit as goodwill",
    ],
    resumption_trigger="Back to Globex prep",
    resumption_query="What were we planning to present at the Globex QBR?",
    should_remember=["3 renewal tiers", "upsell opportunities", "Globex"],
    should_forget=["Initech", "API timeouts", "15% credit"],
    should_not_confuse=["Don't offer Globex the Initech credit"],
)

PROPOSAL_WRITING_INTERRUPTED = InterruptionTemplate(
    name="proposal_writing_interrupted",
    domain="sales",
    description="Proposal writing interrupted by pricing approval request",
    primary_task="Drafting enterprise proposal for Wayne Corp",
    primary_entity="Wayne Corp Proposal",
    primary_steps=[
        "Outlined 5-year strategic partnership",
        "Included custom integration scope",
        "Committed to $2M annual deal value",
    ],
    primary_commitment="Targeting $2M annual contract with Wayne Corp",
    interruption_trigger="Pricing exception request from another rep",
    interruption_task="Reviewing discount request for Stark deal",
    interruption_entity="Stark Industries",
    interruption_context=[
        "Stark wants 40% discount",
        "Non-standard payment terms requested",
        "Recommended declining exception",
    ],
    resumption_trigger="Continuing Wayne proposal",
    resumption_query="What deal value were we targeting for Wayne Corp?",
    should_remember=["$2M", "5-year", "Wayne Corp", "custom integration"],
    should_forget=["40% discount", "Stark", "payment terms"],
    should_not_confuse=["Don't apply Stark's discount request to Wayne"],
)


# --- Project Domain ---

SPRINT_PLANNING_INTERRUPTED = InterruptionTemplate(
    name="sprint_planning_interrupted",
    domain="project",
    description="Sprint planning interrupted by production incident",
    primary_task="Planning Sprint 24 features",
    primary_entity="Sprint 24",
    primary_steps=[
        "Committed to auth refactor as P0",
        "Allocated 3 engineers to API work",
        "Set sprint goal: complete OAuth migration",
    ],
    primary_commitment="OAuth migration must complete in Sprint 24",
    interruption_trigger="P0 production incident",
    interruption_task="Database performance degradation",
    interruption_entity="Production DB",
    interruption_context=[
        "Added emergency index",
        "Scaled up read replicas",
        "Post-mortem scheduled for Thursday",
    ],
    resumption_trigger="Incident resolved, back to planning",
    resumption_query="What's the P0 commitment for Sprint 24?",
    should_remember=["OAuth migration", "auth refactor", "Sprint 24", "3 engineers"],
    should_forget=["index", "read replicas", "post-mortem"],
    should_not_confuse=["DB fixes are not part of Sprint 24 scope"],
)

CODE_REVIEW_INTERRUPTED = InterruptionTemplate(
    name="code_review_interrupted",
    domain="project",
    description="Code review interrupted by architecture discussion",
    primary_task="Reviewing PR #1234 for payment service",
    primary_entity="PR #1234",
    primary_steps=[
        "Found 2 security issues to fix",
        "Requested error handling improvements",
        "Committed to approve after fixes",
    ],
    primary_commitment="Will approve PR #1234 once security issues are fixed",
    interruption_trigger="Architecture review meeting",
    interruption_task="Discussing microservices migration",
    interruption_entity="Migration Plan",
    interruption_context=[
        "Decided to use event-driven architecture",
        "Kafka selected as message broker",
        "Migration starts Q2",
    ],
    resumption_trigger="Back to the PR review",
    resumption_query="What's blocking approval of PR #1234?",
    should_remember=["2 security issues", "error handling", "payment service"],
    should_forget=["Kafka", "event-driven", "Q2 migration"],
    should_not_confuse=["PR doesn't need migration changes"],
)


# --- HR Domain ---

INTERVIEW_LOOP_INTERRUPTED = InterruptionTemplate(
    name="interview_loop_interrupted",
    domain="hr",
    description="Interview debrief interrupted by urgent hiring freeze update",
    primary_task="Evaluating candidate Jane Doe for Senior Engineer",
    primary_entity="Jane Doe",
    primary_steps=[
        "Strong technical performance",
        "Culture fit concerns from one interviewer",
        "Committed to make decision by EOD",
    ],
    primary_commitment="Decision on Jane Doe by end of day",
    interruption_trigger="All-hands about hiring changes",
    interruption_task="Processing new hiring guidelines",
    interruption_entity="Hiring Policy",
    interruption_context=[
        "Freeze on external hires in EU",
        "Internal transfers prioritized",
        "New budget approval process",
    ],
    resumption_trigger="Back to candidate evaluation",
    resumption_query="What's our timeline for the Jane Doe decision?",
    should_remember=["end of day", "Jane Doe", "Senior Engineer", "culture fit concerns"],
    should_forget=["EU freeze", "internal transfers", "budget approval"],
    should_not_confuse=["Jane is US-based, freeze doesn't apply"],
)


# --- Support Domain ---

TICKET_RESOLUTION_INTERRUPTED = InterruptionTemplate(
    name="ticket_resolution_interrupted",
    domain="support",
    description="Complex ticket resolution interrupted by VIP escalation",
    primary_task="Resolving TICKET-5678 data export issue",
    primary_entity="TICKET-5678",
    primary_steps=[
        "Identified bug in export pagination",
        "Workaround: export in batches of 1000",
        "Committed to permanent fix in next release",
    ],
    primary_commitment="Permanent fix for TICKET-5678 in next release",
    interruption_trigger="VIP customer on phone",
    interruption_task="Handling CEO account access issue",
    interruption_entity="VIP-CEO-Account",
    interruption_context=[
        "Reset MFA for CEO",
        "Temporary bypass enabled",
        "Security review scheduled",
    ],
    resumption_trigger="VIP handled, back to ticket",
    resumption_query="What's the workaround we gave for TICKET-5678?",
    should_remember=["batches of 1000", "export pagination", "next release"],
    should_forget=["MFA reset", "temporary bypass", "CEO"],
    should_not_confuse=["Don't mention MFA in export ticket"],
)


# --- Procurement Domain ---

CONTRACT_REVIEW_INTERRUPTED = InterruptionTemplate(
    name="contract_review_interrupted",
    domain="procurement",
    description="Contract negotiation interrupted by budget reallocation",
    primary_task="Negotiating CloudCorp SLA terms",
    primary_entity="CloudCorp Contract",
    primary_steps=[
        "Agreed on 99.95% uptime SLA",
        "Negotiated 2x credits for breaches",
        "Committed to 3-year term for best pricing",
    ],
    primary_commitment="3-year commitment to CloudCorp for preferred pricing",
    interruption_trigger="CFO call about Q4 budget",
    interruption_task="Reviewing department budget cuts",
    interruption_entity="Q4 Budget",
    interruption_context=[
        "15% reduction in software spend",
        "Defer new tools to next year",
        "Headcount freeze",
    ],
    resumption_trigger="Back to CloudCorp negotiation",
    resumption_query="What term length did we commit to for CloudCorp?",
    should_remember=["3-year", "99.95%", "2x credits", "CloudCorp"],
    should_forget=["15% reduction", "headcount freeze", "defer tools"],
    should_not_confuse=["CloudCorp is existing vendor, not new tool"],
)


INTERRUPTION_TEMPLATES = [
    DEAL_REVIEW_INTERRUPTED,
    PROPOSAL_WRITING_INTERRUPTED,
    SPRINT_PLANNING_INTERRUPTED,
    CODE_REVIEW_INTERRUPTED,
    INTERVIEW_LOOP_INTERRUPTED,
    TICKET_RESOLUTION_INTERRUPTED,
    CONTRACT_REVIEW_INTERRUPTED,
]


def get_interruption_templates_by_domain(domain: str) -> list[InterruptionTemplate]:
    """Get all interruption templates for a given domain."""
    return [t for t in INTERRUPTION_TEMPLATES if t.domain == domain]
