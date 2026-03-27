"""Templates for Track: Contradiction Resolution.

Tests whether systems detect and surface contradictions rather than
silently picking one fact.

Two facts that are individually valid but jointly contradictory.
The system should flag the impossibility rather than hallucinating
a consistent answer.

Inspired by Honcho's emphasis on formal reasoning over static retrieval:
"Static retrieval can't surface implicit connections, struggles when
new information contradicts old data."
"""

from dataclasses import dataclass


@dataclass
class ContradictionTemplate:
    """Template for contradiction detection test cases."""

    name: str
    domain: str
    description: str

    # Two facts that contradict each other
    fact_a: dict[str, str]  # key, value, source
    fact_b: dict[str, str]  # key, value, source

    # What makes them contradictory (human-readable)
    contradiction_explanation: str

    # Query that forces the system to confront the contradiction
    query: str

    # Expected behavior: system should acknowledge the contradiction
    expected_decision: str
    must_mention: list[str]  # Should mention both facts and the conflict
    must_not_mention: list[str]  # Should NOT silently pick one


# ── Resource Contradictions ──────────────────────────────────────────

TEAM_SIZE_VS_ASSIGNMENTS = ContradictionTemplate(
    name="team_vs_assignments",
    domain="project_management",
    description="Team size contradicts project assignments",
    fact_a={"key": "team_size", "value": "Team has 3 engineers", "source": "hr_system"},
    fact_b={"key": "project_assignments", "value": "Each engineer assigned to one of 5 active projects, all projects staffed", "source": "project_system"},
    contradiction_explanation="3 engineers cannot each cover a unique project if there are 5 projects",
    query="Is every project adequately staffed?",
    expected_decision="No — there's a discrepancy between team size (3) and project count (5)",
    must_mention=["3", "5", "discrepancy"],
    must_not_mention=[],
)

BUDGET_VS_SPENDING = ContradictionTemplate(
    name="budget_vs_spending",
    domain="finance",
    description="Remaining budget contradicts spending report",
    fact_a={"key": "quarterly_budget", "value": "Q4 budget: $200,000 total, $50,000 spent so far", "source": "finance_system"},
    fact_b={"key": "spending_report", "value": "Q4 expenditures to date: $175,000", "source": "accounting_system"},
    contradiction_explanation="Two systems disagree on how much has been spent ($50K vs $175K)",
    query="How much budget do we have remaining in Q4?",
    expected_decision="Conflicting data — finance says $150K remaining, accounting says $25K remaining",
    must_mention=["conflict", "discrepancy"],
    must_not_mention=[],
)

# ── Schedule Contradictions ──────────────────────────────────────────

DEADLINE_VS_TIMELINE = ContradictionTemplate(
    name="deadline_vs_timeline",
    domain="project_management",
    description="Hard deadline contradicts estimated completion",
    fact_a={"key": "delivery_deadline", "value": "Client deadline: March 15, 2025 (non-negotiable)", "source": "contracts"},
    fact_b={"key": "completion_estimate", "value": "Estimated completion: April 30, 2025 (based on current velocity)", "source": "project_system"},
    contradiction_explanation="Estimated completion is 6 weeks after the hard deadline",
    query="Will we meet the client deadline?",
    expected_decision="No — current estimate (April 30) exceeds deadline (March 15) by ~6 weeks",
    must_mention=["March 15", "April 30", "exceed"],
    must_not_mention=[],
)

# ── Policy Contradictions ────────────────────────────────────────────

APPROVAL_POLICY_CONFLICT = ContradictionTemplate(
    name="approval_policy_conflict",
    domain="procurement",
    description="Two policies give contradictory approval thresholds",
    fact_a={"key": "dept_policy", "value": "Department policy: purchases over $50,000 require VP approval", "source": "dept_policy"},
    fact_b={"key": "company_policy", "value": "Company policy: all purchases under $100,000 can be approved by department heads", "source": "company_policy"},
    contradiction_explanation="A $75,000 purchase requires VP approval (dept policy) but doesn't (company policy)",
    query="Does a $75,000 purchase need VP approval?",
    expected_decision="Conflicting policies — department says yes, company says no",
    must_mention=["$75,000", "conflict", "department", "company"],
    must_not_mention=[],
)

CAPACITY_VS_COMMITMENT = ContradictionTemplate(
    name="capacity_vs_commitment",
    domain="operations",
    description="Stated capacity contradicts committed workload",
    fact_a={"key": "server_capacity", "value": "Production cluster: 10,000 requests/second max capacity", "source": "infra_system"},
    fact_b={"key": "committed_load", "value": "SLA commitments total 15,000 requests/second across all clients", "source": "sales_system"},
    contradiction_explanation="Committed load (15K rps) exceeds physical capacity (10K rps)",
    query="Can we honor all our SLA commitments?",
    expected_decision="No — committed load (15,000 rps) exceeds capacity (10,000 rps)",
    must_mention=["10,000", "15,000", "exceed"],
    must_not_mention=[],
)


CONTRADICTION_TEMPLATES: list[ContradictionTemplate] = [
    TEAM_SIZE_VS_ASSIGNMENTS,
    BUDGET_VS_SPENDING,
    DEADLINE_VS_TIMELINE,
    APPROVAL_POLICY_CONFLICT,
    CAPACITY_VS_COMMITMENT,
]


def get_contradiction_templates(domain: str | None = None) -> list[ContradictionTemplate]:
    """Get templates, optionally filtered by domain."""
    if domain is None:
        return list(CONTRADICTION_TEMPLATES)
    return [t for t in CONTRADICTION_TEMPLATES if t.domain == domain]
