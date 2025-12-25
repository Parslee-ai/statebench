"""Templates for Track 9: Repair Propagation.

This track tests whether corrections propagate through dependent reasoning.
When a fact is corrected, conclusions that depended on the wrong fact
must also be updated.

This is harder than simple supersession because:
1. The system acknowledged the original fact
2. The system made decisions based on it
3. The fact is later corrected
4. The system must update all downstream conclusions

Failure mode: System corrects the fact but continues reasoning from
the original (now-corrected) value.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class RepairChain:
    """A chain of dependent facts where correction must propagate."""
    name: str
    domain: str
    description: str

    # Initial (wrong) fact
    initial_fact: dict[str, Any]  # key, value, source

    # Decision made based on wrong fact
    initial_decision: str
    decision_reasoning: str

    # Correction applied
    correction: dict[str, Any]  # key, new_value, reason

    # Query after correction - tests propagation
    propagation_query: str

    # What the response should reflect (corrected state)
    corrected_must_mention: list[str]

    # What the response should NOT mention (original wrong values)
    original_must_not_mention: list[str]

    # The correct updated decision
    expected_updated_decision: str


# --- Pricing Chains ---

PRICE_PROPAGATION = RepairChain(
    name="price_propagation",
    domain="sales",
    description="Pricing error must propagate to all quotes",
    initial_fact={
        "key": "unit_price",
        "value": "$100 per unit",
        "source": "price_list",
    },
    initial_decision="Quoted customer 500 units at $100 each = $50,000 total",
    decision_reasoning="Based on $100/unit from price list",
    correction={
        "key": "unit_price",
        "new_value": "$150 per unit (the $100 was last year's pricing)",
        "reason": "Price list was outdated",
    },
    propagation_query="What's the correct total for the 500-unit quote?",
    corrected_must_mention=["$150", "75,000", "$75,000"],
    original_must_not_mention=["$100 per unit", "$50,000 total", "original quote"],
    expected_updated_decision="Updated quote: 500 units at $150 = $75,000",
)

DISCOUNT_PROPAGATION = RepairChain(
    name="discount_propagation",
    domain="sales",
    description="Discount correction propagates to final price",
    initial_fact={
        "key": "applicable_discount",
        "value": "25% volume discount applies",
        "source": "sales_rep",
    },
    initial_decision="Applied 25% discount: $10,000 base - $2,500 = $7,500 final",
    decision_reasoning="Sales rep confirmed 25% discount eligibility",
    correction={
        "key": "applicable_discount",
        "new_value": "Only 10% discount applies (25% requires 3-year contract)",
        "reason": "Discount tier was incorrectly applied",
    },
    propagation_query="What's the actual final price for this order?",
    corrected_must_mention=["10%", "$9,000", "1,000"],
    original_must_not_mention=["25%", "$7,500", "$2,500 discount"],
    expected_updated_decision="Corrected: $10,000 - 10% ($1,000) = $9,000 final",
)


# --- Schedule Chains ---

DEADLINE_PROPAGATION = RepairChain(
    name="deadline_propagation",
    domain="project",
    description="Deadline correction propagates to all dependent dates",
    initial_fact={
        "key": "launch_date",
        "value": "March 1st launch",
        "source": "project_plan",
    },
    initial_decision="Code freeze Feb 15, QA Feb 16-28, Launch March 1",
    decision_reasoning="Working backward from March 1 launch",
    correction={
        "key": "launch_date",
        "new_value": "Launch moved to April 15 (per executive decision)",
        "reason": "Executive pushed launch for trade show alignment",
    },
    propagation_query="When is code freeze and when does QA start?",
    corrected_must_mention=["April", "moved", "trade show"],
    original_must_not_mention=["Feb 15", "Feb 16-28", "March 1"],
    expected_updated_decision="New schedule: Code freeze April 1, QA April 2-14, Launch April 15",
)

DEPENDENCY_PROPAGATION = RepairChain(
    name="dependency_propagation",
    domain="project",
    description="Upstream delay propagates to downstream tasks",
    initial_fact={
        "key": "api_completion",
        "value": "API ready January 15",
        "source": "backend_team",
    },
    initial_decision="Frontend integration starts January 16, complete by January 30",
    decision_reasoning="Frontend needs API to be ready first",
    correction={
        "key": "api_completion",
        "new_value": "API delayed to February 1 (discovered additional requirements)",
        "reason": "Scope expansion in API layer",
    },
    propagation_query="When will frontend integration be complete?",
    corrected_must_mention=["February", "delayed", "after API"],
    original_must_not_mention=["January 16", "January 30"],
    expected_updated_decision="Frontend integration: Feb 2 - Feb 16 (after API ready Feb 1)",
)


# --- Resource Chains ---

HEADCOUNT_PROPAGATION = RepairChain(
    name="headcount_propagation",
    domain="hr",
    description="Headcount correction propagates to capacity planning",
    initial_fact={
        "key": "team_size",
        "value": "8 engineers on Project Alpha",
        "source": "resource_plan",
    },
    initial_decision="Can handle 800 story points this quarter (100 per engineer)",
    decision_reasoning="8 engineers × 100 points = 800 points capacity",
    correction={
        "key": "team_size",
        "new_value": "Only 5 engineers available (3 reassigned to critical bug fixes)",
        "reason": "Emergency reallocation due to production issues",
    },
    propagation_query="What's our actual capacity for Project Alpha this quarter?",
    corrected_must_mention=["5 engineers", "500", "reduced"],
    original_must_not_mention=["8 engineers", "800 points"],
    expected_updated_decision="Revised capacity: 5 engineers × 100 = 500 story points",
)

BUDGET_PROPAGATION = RepairChain(
    name="budget_propagation",
    domain="procurement",
    description="Budget correction propagates to approved spend",
    initial_fact={
        "key": "available_budget",
        "value": "$200,000 available in Q4 budget",
        "source": "finance",
    },
    initial_decision="Approved $150,000 software purchase, $50,000 remaining",
    decision_reasoning="$200K budget, $150K committed = $50K left",
    correction={
        "key": "available_budget",
        "new_value": "$120,000 total budget (CFO reduced allocation)",
        "reason": "Budget reforecast due to revenue shortfall",
    },
    propagation_query="What's our remaining budget after the software purchase?",
    corrected_must_mention=["$120,000", "exceeds", "over budget", "negative"],
    original_must_not_mention=["$50,000 remaining", "$200,000"],
    expected_updated_decision="$120K budget - $150K committed = $30K over budget. Need to renegotiate or find cuts.",
)


# --- Commitment Chains ---

MEETING_PROPAGATION = RepairChain(
    name="meeting_propagation",
    domain="support",
    description="Time change propagates to preparation instructions",
    initial_fact={
        "key": "meeting_time",
        "value": "Client call at 2pm EST Tuesday",
        "source": "calendar",
    },
    initial_decision="Prepare deck by 1:30pm EST Tuesday, dial in at 1:55pm",
    decision_reasoning="Need deck ready 30 min before, join 5 min early",
    correction={
        "key": "meeting_time",
        "new_value": "Moved to 10am EST Tuesday (client conflict)",
        "reason": "Client had conflict with original time",
    },
    propagation_query="When should I have the deck ready and when should I dial in?",
    corrected_must_mention=["9:30am", "9:55am", "10am"],
    original_must_not_mention=["1:30pm", "1:55pm", "2pm"],
    expected_updated_decision="Deck ready by 9:30am, dial in at 9:55am for 10am meeting",
)


REPAIR_CHAIN_TEMPLATES = [
    PRICE_PROPAGATION,
    DISCOUNT_PROPAGATION,
    DEADLINE_PROPAGATION,
    DEPENDENCY_PROPAGATION,
    HEADCOUNT_PROPAGATION,
    BUDGET_PROPAGATION,
    MEETING_PROPAGATION,
]


def get_repair_templates() -> list[RepairChain]:
    """Get all repair chain templates."""
    return REPAIR_CHAIN_TEMPLATES
