"""Templates for Track: Reasoning Depth.

Tests multi-hop inference chains (1-hop, 2-hop, 3-hop).

Most existing tracks test single-hop reasoning:
  "Is fact X valid?" → look up X → answer.

This track tests whether systems can follow dependency chains:
  1-hop: A is stated → query A
  2-hop: A depends on B, B changed → what's A now?
  3-hop: A depends on B depends on C, C changed → what's A now?

Inspired by Honcho's emphasis on reasoning over retrieval — static
retrieval can't surface implicit connections.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReasoningDepthTemplate:
    """Template for a multi-hop reasoning chain."""

    name: str
    domain: str
    depth: int  # Number of hops (1, 2, or 3)
    description: str

    # Chain of facts (ordered from root to leaf)
    # Each fact is {key, value, source, depends_on: list[key]}
    fact_chain: list[dict[str, Any]]

    # The change applied to the root fact
    root_change: dict[str, Any]  # {key, new_value, reason}

    # Query about the leaf fact (furthest from the change)
    query: str

    # Expected answer reflecting propagated change
    expected_decision: str
    must_mention: list[str]
    must_not_mention: list[str]


# ── 1-Hop: Direct fact lookup ─────────────────────────────────────

BUDGET_LOOKUP = ReasoningDepthTemplate(
    name="budget_lookup_1hop",
    domain="finance",
    depth=1,
    description="Direct budget fact retrieval",
    fact_chain=[
        {"key": "project_budget", "value": "$200,000", "source": "finance_system", "depends_on": []},
    ],
    root_change={"key": "project_budget", "new_value": "$175,000 (revised after Q3 cuts)", "reason": "Budget reduction"},
    query="What is the current project budget?",
    expected_decision="$175,000",
    must_mention=["175,000", "$175,000"],
    must_not_mention=["$200,000"],
)

VENDOR_STATUS_LOOKUP = ReasoningDepthTemplate(
    name="vendor_status_1hop",
    domain="procurement",
    depth=1,
    description="Direct vendor status lookup",
    fact_chain=[
        {"key": "vendor_status", "value": "Approved for Phase 1", "source": "procurement_system", "depends_on": []},
    ],
    root_change={"key": "vendor_status", "new_value": "Suspended pending compliance review", "reason": "Compliance issue"},
    query="What is the current vendor status?",
    expected_decision="Suspended pending compliance review",
    must_mention=["suspended", "compliance"],
    must_not_mention=["approved"],
)

# ── 2-Hop: Change propagates through one dependency ──────────────

BUDGET_TO_HEADCOUNT = ReasoningDepthTemplate(
    name="budget_to_headcount_2hop",
    domain="hr",
    depth=2,
    description="Budget change propagates to hiring capacity",
    fact_chain=[
        {"key": "team_budget", "value": "$500,000 annual", "source": "finance_system", "depends_on": []},
        {"key": "hiring_capacity", "value": "Can hire 5 engineers at $100K each", "source": "hr_system", "depends_on": ["team_budget"]},
    ],
    root_change={"key": "team_budget", "new_value": "$300,000 annual (post-restructuring)", "reason": "Org restructuring"},
    query="How many engineers can we hire now?",
    expected_decision="3 engineers",
    must_mention=["3", "300,000", "$300,000"],
    must_not_mention=["5 engineers", "$500,000", "hire 5"],
)

TIMELINE_TO_DELIVERY = ReasoningDepthTemplate(
    name="timeline_to_delivery_2hop",
    domain="project_management",
    depth=2,
    description="Start date change propagates to delivery date",
    fact_chain=[
        {"key": "project_start", "value": "March 1, 2025", "source": "project_system", "depends_on": []},
        {"key": "delivery_date", "value": "June 1, 2025 (3 months after start)", "source": "project_system", "depends_on": ["project_start"]},
    ],
    root_change={"key": "project_start", "new_value": "April 15, 2025 (delayed by resource constraints)", "reason": "Resource delay"},
    query="What is the expected delivery date now?",
    expected_decision="July 15, 2025",
    must_mention=["July 15", "July"],
    must_not_mention=["June 1", "March 1"],
)

RATE_TO_INVOICE = ReasoningDepthTemplate(
    name="rate_to_invoice_2hop",
    domain="billing",
    depth=2,
    description="Hourly rate change propagates to invoice total",
    fact_chain=[
        {"key": "hourly_rate", "value": "$200/hour", "source": "contracts", "depends_on": []},
        {"key": "monthly_invoice", "value": "$32,000 (160 hours at $200/hour)", "source": "billing_system", "depends_on": ["hourly_rate"]},
    ],
    root_change={"key": "hourly_rate", "new_value": "$250/hour (renegotiated)", "reason": "Contract renegotiation"},
    query="What should the monthly invoice be now?",
    expected_decision="$40,000",
    must_mention=["$40,000", "40,000", "$250"],
    must_not_mention=["$32,000", "$200/hour"],
)

# ── 3-Hop: Change propagates through two dependencies ─────────────

EXCHANGE_TO_BUDGET_TO_HEADCOUNT = ReasoningDepthTemplate(
    name="exchange_to_budget_to_headcount_3hop",
    domain="international_ops",
    depth=3,
    description="Exchange rate change propagates through budget to hiring",
    fact_chain=[
        {"key": "exchange_rate", "value": "1 USD = 0.85 EUR", "source": "finance_system", "depends_on": []},
        {"key": "euro_budget", "value": "€425,000 (converted from $500,000 at 0.85)", "source": "finance_system", "depends_on": ["exchange_rate"]},
        {"key": "eu_hiring", "value": "Can hire 5 EU engineers at €85,000 each", "source": "hr_system", "depends_on": ["euro_budget"]},
    ],
    root_change={"key": "exchange_rate", "new_value": "1 USD = 0.75 EUR (currency weakened)", "reason": "Market movement"},
    query="How many EU engineers can we hire now with the same $500K USD budget?",
    expected_decision="4 engineers",
    must_mention=["4", "0.75", "375,000"],
    must_not_mention=["5 engineers", "€425,000", "0.85"],
)

CAPACITY_TO_TIMELINE_TO_COST = ReasoningDepthTemplate(
    name="capacity_to_timeline_to_cost_3hop",
    domain="project_management",
    depth=3,
    description="Team capacity change propagates through timeline to total cost",
    fact_chain=[
        {"key": "team_size", "value": "8 engineers", "source": "hr_system", "depends_on": []},
        {"key": "project_duration", "value": "4 months (32 person-months / 8 engineers)", "source": "project_system", "depends_on": ["team_size"]},
        {"key": "total_cost", "value": "$640,000 (4 months × 8 engineers × $20K/month)", "source": "finance_system", "depends_on": ["project_duration"]},
    ],
    root_change={"key": "team_size", "new_value": "6 engineers (2 reassigned to critical project)", "reason": "Resource reallocation"},
    query="What is the expected total cost now with the reduced team?",
    expected_decision="$640,000 but over ~5.3 months",
    must_mention=["6 engineer", "32 person-month"],
    must_not_mention=["8 engineers", "4 months"],
)

DISCOUNT_TO_PRICE_TO_REVENUE = ReasoningDepthTemplate(
    name="discount_to_price_to_revenue_3hop",
    domain="sales",
    depth=3,
    description="Discount change propagates through unit price to quarterly revenue",
    fact_chain=[
        {"key": "volume_discount", "value": "15% discount for orders over 1000 units", "source": "sales_policy", "depends_on": []},
        {"key": "effective_price", "value": "$85 per unit ($100 list price minus 15%)", "source": "pricing_system", "depends_on": ["volume_discount"]},
        {"key": "q4_revenue_forecast", "value": "$425,000 (5,000 units at $85 each)", "source": "finance_system", "depends_on": ["effective_price"]},
    ],
    root_change={"key": "volume_discount", "new_value": "10% discount (reduced from 15% per new policy)", "reason": "Margin improvement initiative"},
    query="What is the updated Q4 revenue forecast?",
    expected_decision="$450,000",
    must_mention=["$450,000", "450,000", "$90", "10%"],
    must_not_mention=["$425,000", "$85", "15%"],
)


# ── All templates grouped ──────────────────────────────────────────

REASONING_DEPTH_TEMPLATES: list[ReasoningDepthTemplate] = [
    # 1-hop
    BUDGET_LOOKUP,
    VENDOR_STATUS_LOOKUP,
    # 2-hop
    BUDGET_TO_HEADCOUNT,
    TIMELINE_TO_DELIVERY,
    RATE_TO_INVOICE,
    # 3-hop
    EXCHANGE_TO_BUDGET_TO_HEADCOUNT,
    CAPACITY_TO_TIMELINE_TO_COST,
    DISCOUNT_TO_PRICE_TO_REVENUE,
]


def get_reasoning_depth_templates(depth: int | None = None) -> list[ReasoningDepthTemplate]:
    """Get templates, optionally filtered by depth."""
    if depth is None:
        return list(REASONING_DEPTH_TEMPLATES)
    return [t for t in REASONING_DEPTH_TEMPLATES if t.depth == depth]


def get_reasoning_depth_templates_by_domain(domain: str) -> list[ReasoningDepthTemplate]:
    """Get templates filtered by domain."""
    return [t for t in REASONING_DEPTH_TEMPLATES if t.domain == domain]
