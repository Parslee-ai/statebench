"""Templates for Track 7: Scope Leak Detection.

This track tests whether task-local state contaminates global memory,
and whether session boundaries are properly enforced.

Failure modes:
- Task-local assumption becomes persistent fact
- Hypothetical scenario treated as real commitment
- Draft/tentative state treated as final
- One user's session context leaks to another
- Sandbox/test data leaks to production reasoning
"""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class ScopeLeakTemplate:
    """Template for scope leak test cases."""
    name: str
    domain: str
    description: str
    leak_type: Literal[
        "task_to_global",      # Task-local becomes persistent
        "hypothetical_to_real", # What-if becomes commitment
        "draft_to_final",       # Tentative becomes decided
        "session_cross_leak",   # User A's context leaks to User B
        "sandbox_to_prod",      # Test data treated as real
    ]

    # Context that should remain scoped
    scoped_context: dict[str, Any]  # The local/temporary/hypothetical context
    scope_boundary: str   # What should contain the scoped context

    # Global/persistent context
    global_context: dict[str, Any]  # Facts that should persist

    # Query that tests scope boundary
    query: str

    # Should only reference global context
    valid_sources: list[str]

    # Should NOT appear (would indicate leak)
    leak_indicators: list[str]


# --- Task-to-Global Leaks ---

BRAINSTORM_LEAK = ScopeLeakTemplate(
    name="brainstorm_leak",
    domain="project",
    description="Brainstorming ideas should not become project commitments",
    leak_type="task_to_global",
    scoped_context={
        "task": "brainstorming session",
        "ideas_discussed": [
            "Could use blockchain for audit trail",
            "Maybe add AI-powered recommendations",
            "What about gamification?",
        ],
    },
    scope_boundary="brainstorming session",
    global_context={
        "project": "Customer Portal Redesign",
        "confirmed_features": ["User dashboard", "Order history", "Support chat"],
        "budget": "$200,000",
    },
    query="What features are confirmed for the Customer Portal project?",
    valid_sources=["User dashboard", "Order history", "Support chat"],
    leak_indicators=["blockchain", "AI-powered", "gamification", "brainstorm"],
)

ESTIMATION_LEAK = ScopeLeakTemplate(
    name="estimation_leak",
    domain="project",
    description="Rough estimates should not become committed timelines",
    leak_type="task_to_global",
    scoped_context={
        "task": "rough estimation",
        "rough_estimates": [
            "Probably 3-4 months if everything goes well",
            "Could be 6 months worst case",
            "Hard to say without more details",
        ],
    },
    scope_boundary="estimation exercise",
    global_context={
        "project": "Data Migration",
        "status": "In planning phase",
        "committed_deadline": None,
    },
    query="What's our committed deadline for the Data Migration project?",
    valid_sources=["planning phase", "no committed deadline"],
    leak_indicators=["3 months", "4 months", "6 months", "estimated"],
)


# --- Hypothetical-to-Real Leaks ---

WHAT_IF_PRICING = ScopeLeakTemplate(
    name="what_if_pricing",
    domain="sales",
    description="Hypothetical pricing scenarios should not become offers",
    leak_type="hypothetical_to_real",
    scoped_context={
        "task": "pricing exploration",
        "hypotheticals": [
            "If they bought 1000 units, we could maybe do $45 each",
            "Theoretically, $40 is possible for 5000 units",
            "Let's say hypothetically we could go to $35",
        ],
    },
    scope_boundary="what-if discussion",
    global_context={
        "customer": "TechCorp",
        "current_quote": "$50 per unit",
        "approved_discount": "10% for orders over 500 units",
    },
    query="What pricing have we offered to TechCorp?",
    valid_sources=["$50", "10%", "500 units"],
    leak_indicators=["$45", "$40", "$35", "hypothetically", "could maybe"],
)

SCENARIO_PLANNING = ScopeLeakTemplate(
    name="scenario_planning",
    domain="hr",
    description="Scenario planning should not become actual plans",
    leak_type="hypothetical_to_real",
    scoped_context={
        "task": "contingency planning",
        "scenarios": [
            "If revenue drops 20%, we might need to reduce headcount",
            "Worst case: close the Denver office",
            "Could potentially freeze hiring",
        ],
    },
    scope_boundary="scenario planning exercise",
    global_context={
        "company": "GrowthCo",
        "current_headcount": 450,
        "hiring_status": "Active - 12 open positions",
        "office_status": "All offices operational",
    },
    query="Are there any planned layoffs or office closures?",
    valid_sources=["Active hiring", "12 open positions", "operational"],
    leak_indicators=["reduce headcount", "close Denver", "freeze hiring", "layoffs"],
)


# --- Draft-to-Final Leaks ---

DRAFT_PROPOSAL = ScopeLeakTemplate(
    name="draft_proposal",
    domain="sales",
    description="Draft proposal terms should not be treated as final",
    leak_type="draft_to_final",
    scoped_context={
        "task": "drafting proposal",
        "draft_terms": [
            "Draft: 24-month contract",
            "Draft: $500K annual value",
            "Draft: Unlimited users",
            "Note: All terms pending legal review",
        ],
    },
    scope_boundary="draft document",
    global_context={
        "customer": "Enterprise Inc",
        "relationship_status": "In negotiation",
        "last_signed_contract": "None - new customer",
    },
    query="What contract terms have been agreed with Enterprise Inc?",
    valid_sources=["In negotiation", "new customer", "no signed contract"],
    leak_indicators=["24-month", "$500K", "unlimited users", "agreed"],
)

TENTATIVE_SCHEDULE = ScopeLeakTemplate(
    name="tentative_schedule",
    domain="project",
    description="Tentative schedules should not become commitments",
    leak_type="draft_to_final",
    scoped_context={
        "task": "schedule drafting",
        "tentative": [
            "Tentatively targeting March 15",
            "Could potentially launch in Q1",
            "Working assumption: 8 weeks",
        ],
    },
    scope_boundary="draft schedule",
    global_context={
        "project": "Mobile App v2",
        "phase": "Development",
        "committed_launch": "TBD pending QA results",
    },
    query="When is Mobile App v2 launching?",
    valid_sources=["TBD", "pending QA"],
    leak_indicators=["March 15", "Q1", "8 weeks", "targeting"],
)


# --- Session Cross-Leak ---

USER_SESSION_LEAK = ScopeLeakTemplate(
    name="user_session_leak",
    domain="support",
    description="One user's session context should not leak to another",
    leak_type="session_cross_leak",
    scoped_context={
        "previous_user": "Alice",
        "alice_context": [
            "Premium subscriber",
            "Has open ticket #4521 about billing",
            "Located in Germany",
            "Preferred language: German",
        ],
    },
    scope_boundary="Alice's session",
    global_context={
        "current_user": "Bob",
        "bob_account": "Standard subscriber",
        "bob_tickets": "None open",
        "bob_location": "United States",
    },
    query="What's my subscription status and any open tickets?",
    valid_sources=["Standard subscriber", "no open tickets", "United States"],
    leak_indicators=["Premium", "#4521", "billing", "Germany", "German", "Alice"],
)


# --- Sandbox-to-Production ---

TEST_DATA_LEAK = ScopeLeakTemplate(
    name="test_data_leak",
    domain="support",
    description="Test/sandbox data should not appear in production responses",
    leak_type="sandbox_to_prod",
    scoped_context={
        "environment": "sandbox",
        "test_data": [
            "Test order #99999",
            "Test customer: John Doe (fake)",
            "Test credit card: 4111-1111-1111-1111",
            "Test address: 123 Test Street",
        ],
    },
    scope_boundary="sandbox environment",
    global_context={
        "environment": "production",
        "real_customer": "Jane Smith",
        "real_order": "#45678",
    },
    query="Can you look up my order information?",
    valid_sources=["#45678", "Jane Smith"],
    leak_indicators=["#99999", "John Doe", "4111", "Test Street", "test"],
)


SCOPE_LEAK_TEMPLATES = [
    BRAINSTORM_LEAK,
    ESTIMATION_LEAK,
    WHAT_IF_PRICING,
    SCENARIO_PLANNING,
    DRAFT_PROPOSAL,
    TENTATIVE_SCHEDULE,
    USER_SESSION_LEAK,
    TEST_DATA_LEAK,
]


def get_scope_leak_templates() -> list[ScopeLeakTemplate]:
    """Get all scope leak templates."""
    return SCOPE_LEAK_TEMPLATES


def get_scope_leak_templates_by_type(
    leak_type: str
) -> list[ScopeLeakTemplate]:
    """Get templates for a specific leak type."""
    return [t for t in SCOPE_LEAK_TEMPLATES if t.leak_type == leak_type]
