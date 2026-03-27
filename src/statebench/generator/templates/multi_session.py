"""Templates for Track: Multi-Session Persistence.

Tests cross-session memory — facts from session 1 must persist into
session 2, with potential contradictions and session-scoped vs global facts.

This is the core value proposition of systems like Honcho: maintaining
stateful context across conversation boundaries.

Current StateBench timelines are single-session. This track adds session
boundaries and tests:
1. Fact persistence across sessions
2. Cross-session supersession (session 2 overrides session 1 fact)
3. Session-scoped facts that should NOT persist
"""

from dataclasses import dataclass, field


@dataclass
class SessionBoundary:
    """Marks a session transition in the timeline."""

    session_id: str
    description: str


@dataclass
class MultiSessionTemplate:
    """Template for multi-session persistence tests."""

    name: str
    domain: str
    description: str

    # Facts established in session 1
    session_1_facts: list[dict[str, str]]
    # key, value, source, scope (global or session)

    # Session boundary
    boundary: SessionBoundary

    # Facts/events in session 2
    session_2_facts: list[dict[str, str]]

    # Query in session 2 about session 1 facts
    query: str

    # Expected behavior
    expected_decision: str
    must_mention: list[str]
    must_not_mention: list[str]


# ── Persistence Tests ────────────────────────────────────────────────

BUDGET_PERSISTS = MultiSessionTemplate(
    name="budget_persists_across_sessions",
    domain="finance",
    description="Budget fact from session 1 should persist to session 2",
    session_1_facts=[
        {"key": "project_budget", "value": "$200,000 approved for Project Alpha", "source": "finance_system", "scope": "global"},
        {"key": "session_note", "value": "Discussed implementation timeline", "source": "user", "scope": "session"},
    ],
    boundary=SessionBoundary(session_id="session_2", description="New day, new conversation"),
    session_2_facts=[],
    query="What's the budget for Project Alpha?",
    expected_decision="$200,000",
    must_mention=["$200,000", "Project Alpha"],
    must_not_mention=[],
)

VENDOR_APPROVAL_PERSISTS = MultiSessionTemplate(
    name="vendor_approval_persists",
    domain="procurement",
    description="Vendor approval from session 1 persists",
    session_1_facts=[
        {"key": "vendor_approved", "value": "Acme Corp approved as vendor for Q4 supplies", "source": "procurement_system", "scope": "global"},
    ],
    boundary=SessionBoundary(session_id="session_2", description="Follow-up meeting"),
    session_2_facts=[],
    query="Is Acme Corp an approved vendor?",
    expected_decision="Yes, approved for Q4 supplies",
    must_mention=["approved", "Acme Corp"],
    must_not_mention=[],
)

# ── Cross-Session Supersession ───────────────────────────────────────

BUDGET_SUPERSEDED_CROSS_SESSION = MultiSessionTemplate(
    name="budget_superseded_cross_session",
    domain="finance",
    description="Session 2 supersedes a budget from session 1",
    session_1_facts=[
        {"key": "project_budget", "value": "$200,000 approved for Project Alpha", "source": "finance_system", "scope": "global"},
    ],
    boundary=SessionBoundary(session_id="session_2", description="Budget review meeting"),
    session_2_facts=[
        {"key": "project_budget", "value": "$150,000 (revised down after Q3 review)", "source": "finance_system", "scope": "global"},
    ],
    query="What's the current budget for Project Alpha?",
    expected_decision="$150,000 (revised)",
    must_mention=["$150,000"],
    must_not_mention=["$200,000"],
)

DEADLINE_CHANGED_CROSS_SESSION = MultiSessionTemplate(
    name="deadline_changed_cross_session",
    domain="project_management",
    description="Deadline from session 1 changed in session 2",
    session_1_facts=[
        {"key": "delivery_deadline", "value": "March 30, 2025", "source": "project_system", "scope": "global"},
    ],
    boundary=SessionBoundary(session_id="session_2", description="Client call"),
    session_2_facts=[
        {"key": "delivery_deadline", "value": "April 15, 2025 (client agreed to extension)", "source": "project_system", "scope": "global"},
    ],
    query="When is the delivery deadline?",
    expected_decision="April 15, 2025",
    must_mention=["April 15"],
    must_not_mention=["March 30"],
)

# ── Session-Scoped Facts Should NOT Persist ─────────────────────────

SESSION_SCOPED_NOT_PERSIST = MultiSessionTemplate(
    name="session_scoped_does_not_persist",
    domain="collaboration",
    description="Session-scoped brainstorm notes should not persist",
    session_1_facts=[
        {"key": "team_size", "value": "Team has 8 engineers", "source": "hr_system", "scope": "global"},
        {"key": "brainstorm_idea", "value": "What if we outsourced the frontend to save $50K?", "source": "user", "scope": "session"},
    ],
    boundary=SessionBoundary(session_id="session_2", description="Planning meeting"),
    session_2_facts=[],
    query="Are we outsourcing the frontend?",
    expected_decision="No such decision has been made",
    must_mention=["not specified", "no"],
    must_not_mention=["outsourc", "$50K"],
)

HYPOTHETICAL_NOT_PERSIST = MultiSessionTemplate(
    name="hypothetical_does_not_persist",
    domain="finance",
    description="Hypothetical discussed in session 1 should not persist as fact",
    session_1_facts=[
        {"key": "actual_headcount", "value": "Current team: 10 engineers", "source": "hr_system", "scope": "global"},
        {"key": "what_if_headcount", "value": "What if we doubled the team to 20?", "source": "user", "scope": "session"},
    ],
    boundary=SessionBoundary(session_id="session_2", description="Status update"),
    session_2_facts=[],
    query="How many engineers are on the team?",
    expected_decision="10 engineers",
    must_mention=["10"],
    must_not_mention=["20", "doubled"],
)


MULTI_SESSION_TEMPLATES: list[MultiSessionTemplate] = [
    # Persistence
    BUDGET_PERSISTS,
    VENDOR_APPROVAL_PERSISTS,
    # Cross-session supersession
    BUDGET_SUPERSEDED_CROSS_SESSION,
    DEADLINE_CHANGED_CROSS_SESSION,
    # Session-scoped non-persistence
    SESSION_SCOPED_NOT_PERSIST,
    HYPOTHETICAL_NOT_PERSIST,
]


def get_multi_session_templates(
    test_type: str | None = None,
) -> list[MultiSessionTemplate]:
    """Get templates, optionally filtered by test type.

    test_type: 'persistence', 'supersession', 'scoped'
    """
    if test_type is None:
        return list(MULTI_SESSION_TEMPLATES)
    type_map = {
        "persistence": [BUDGET_PERSISTS, VENDOR_APPROVAL_PERSISTS],
        "supersession": [BUDGET_SUPERSEDED_CROSS_SESSION, DEADLINE_CHANGED_CROSS_SESSION],
        "scoped": [SESSION_SCOPED_NOT_PERSIST, HYPOTHETICAL_NOT_PERSIST],
    }
    return type_map.get(test_type, [])
