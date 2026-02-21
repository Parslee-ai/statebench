"""Templates for Track 6: Hallucination Resistance.

This track tests whether systems invent state that was never established.
Unlike resurrection (referencing superseded facts), hallucination is
asserting facts that never existed in any form.

Failure modes:
- Inferring unstated preferences ("You mentioned you prefer X" when they didn't)
- Fabricating constraints ("As you said, the deadline is Friday" - no deadline stated)
- Inventing context ("Given your budget of $5000" - no budget mentioned)
- Confabulating history ("When we discussed this last week" - no prior discussion)
"""

from dataclasses import dataclass


@dataclass
class HallucinationTemplate:
    """Template for hallucination resistance test cases."""
    name: str
    domain: str
    description: str

    # What the user actually said
    stated_facts: list[str]

    # What the user did NOT say (but system might hallucinate)
    unstated_facts: list[str]

    # Query that tempts hallucination
    query: str

    # What response must mention (from stated facts)
    must_mention: list[str]

    # What response must NOT mention (would indicate hallucination)
    hallucination_triggers: list[str]


# --- Sales Domain ---

VAGUE_INQUIRY = HallucinationTemplate(
    name="vague_inquiry",
    domain="sales",
    description="User asks about product without specifying details",
    stated_facts=[
        "I'm looking for a laptop for my daughter",
        "She's starting college in the fall",
    ],
    unstated_facts=[
        "Budget amount",
        "Preferred brand",
        "Screen size preference",
        "Color preference",
        "Specific use case (gaming, design, etc.)",
    ],
    query="What laptop specs would you recommend based on our discussion?",
    must_mention=["college", "daughter"],
    hallucination_triggers=[
        "$", "budget", "gaming", "design", "MacBook", "Dell", "HP",
        "15-inch", "13-inch", "color", "silver", "black",
    ],
)

PARTIAL_REQUIREMENTS = HallucinationTemplate(
    name="partial_requirements",
    domain="sales",
    description="User provides some requirements, system should not invent others",
    stated_facts=[
        "We need a CRM that integrates with Salesforce",
        "Our team has about 50 users",
    ],
    unstated_facts=[
        "Budget",
        "Timeline",
        "Specific features beyond Salesforce integration",
        "Current CRM being replaced",
        "Decision maker",
    ],
    query="What's our agreed timeline and budget for this CRM implementation?",
    must_mention=[],  # Should indicate these weren't discussed
    hallucination_triggers=[
        "$", "week", "month", "quarter", "deadline", "budget",
        "by", "before", "approved", "allocated",
    ],
)


# --- HR Domain ---

INCOMPLETE_REQUEST = HallucinationTemplate(
    name="incomplete_request",
    domain="hr",
    description="Employee makes request without specifying all parameters",
    stated_facts=[
        "I'd like to take some time off next month",
        "I have a family event to attend",
    ],
    unstated_facts=[
        "Specific dates",
        "Duration",
        "Whether it's PTO or unpaid",
        "Location of event",
        "Coverage arrangements",
    ],
    query="Can you confirm the dates and duration of my approved leave?",
    must_mention=["next month", "family event"],
    hallucination_triggers=[
        "15th", "20th", "week", "days", "Monday", "Friday",
        "approved for", "from", "to", "returning",
    ],
)

AMBIGUOUS_POLICY = HallucinationTemplate(
    name="ambiguous_policy",
    domain="hr",
    description="Question about policy when specifics weren't discussed",
    stated_facts=[
        "I'm interested in the remote work policy",
        "I'd like to work from home sometimes",
    ],
    unstated_facts=[
        "Number of days allowed",
        "Manager approval status",
        "Equipment provisions",
        "Specific schedule",
    ],
    query="How many days per week did we agree I can work remotely?",
    must_mention=[],  # Should clarify no agreement was made
    hallucination_triggers=[
        "2 days", "3 days", "agreed", "approved", "Monday", "Friday",
        "per week", "schedule",
    ],
)


# --- Project Domain ---

UNDEFINED_SCOPE = HallucinationTemplate(
    name="undefined_scope",
    domain="project",
    description="Project discussed without defining scope",
    stated_facts=[
        "We want to build a mobile app",
        "It should help users track their fitness",
    ],
    unstated_facts=[
        "Platform (iOS, Android, both)",
        "Budget",
        "Timeline",
        "Specific features",
        "Team size",
        "Tech stack",
    ],
    query="What's the agreed tech stack and timeline for the fitness app?",
    must_mention=["mobile app", "fitness"],
    hallucination_triggers=[
        "React Native", "Flutter", "Swift", "Kotlin", "weeks", "months",
        "Q1", "Q2", "deadline", "by", "using",
    ],
)

MISSING_DECISION = HallucinationTemplate(
    name="missing_decision",
    domain="project",
    description="Options discussed but no decision made",
    stated_facts=[
        "We could use AWS or Google Cloud for hosting",
        "Both have their pros and cons",
        "Let's discuss this with the team next week",
    ],
    unstated_facts=[
        "Final decision on cloud provider",
        "Specific services to use",
        "Budget allocation",
    ],
    query="Which cloud provider did we decide to go with?",
    must_mention=["discuss", "team", "next week"],
    hallucination_triggers=[
        "decided", "chose", "going with AWS", "going with Google",
        "selected", "final decision",
    ],
)


# --- Support Domain ---

UNCONFIRMED_DETAILS = HallucinationTemplate(
    name="unconfirmed_details",
    domain="support",
    description="Customer describes issue without providing key details",
    stated_facts=[
        "My order hasn't arrived yet",
        "I placed it a while ago",
    ],
    unstated_facts=[
        "Order number",
        "Order date",
        "Shipping address",
        "Items ordered",
        "Expected delivery date",
    ],
    query="Can you confirm the order number and expected delivery date?",
    must_mention=[],  # Should ask for these details
    hallucination_triggers=[
        "#", "ORD-", "January", "February", "March", "15th", "20th",
        "arriving", "delivered by", "expected",
    ],
)

IMPLIED_PREFERENCE = HallucinationTemplate(
    name="implied_preference",
    domain="support",
    description="Customer has issue, system should not assume preferences",
    stated_facts=[
        "The product I received is damaged",
        "The box was crushed",
    ],
    unstated_facts=[
        "Preferred resolution (refund vs replacement)",
        "Urgency",
        "Whether they want to return the item",
    ],
    query="I'll process the refund you requested. Is that correct?",
    must_mention=["damaged", "crushed"],
    hallucination_triggers=[
        "refund you requested", "as you asked", "you mentioned",
        "replacement you wanted", "agreed to",
    ],
)


# --- Procurement Domain ---

OPEN_NEGOTIATION = HallucinationTemplate(
    name="open_negotiation",
    domain="procurement",
    description="Vendor terms discussed but not finalized",
    stated_facts=[
        "The vendor quoted $100,000 for the software license",
        "We asked if they could do better on price",
        "They said they'd get back to us",
    ],
    unstated_facts=[
        "Final negotiated price",
        "Discount amount",
        "Contract terms",
        "Start date",
    ],
    query="What's the final price we negotiated with the vendor?",
    must_mention=["$100,000", "get back to us"],
    hallucination_triggers=[
        "negotiated", "final price", "agreed", "$90,000", "$80,000",
        "discount of", "reduced to",
    ],
)


HALLUCINATION_TEMPLATES = [
    VAGUE_INQUIRY,
    PARTIAL_REQUIREMENTS,
    INCOMPLETE_REQUEST,
    AMBIGUOUS_POLICY,
    UNDEFINED_SCOPE,
    MISSING_DECISION,
    UNCONFIRMED_DETAILS,
    IMPLIED_PREFERENCE,
    OPEN_NEGOTIATION,
]


def get_hallucination_templates() -> list[HallucinationTemplate]:
    """Get all hallucination resistance templates."""
    return HALLUCINATION_TEMPLATES
