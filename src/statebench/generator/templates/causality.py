"""Templates for Track 8: Causality and Dependency.

This track tests whether the system's reasoning actually DEPENDS on state,
not just whether it can recall and recite state.

The key insight: A system can correctly repeat a constraint but then
ignore it when making a decision. That's a causality failure.

Test pattern:
1. Establish a constraint
2. Create a scenario where the constraint should affect the decision
3. Query for a decision (not just recall)
4. Verify the decision respects the constraint

We test this by providing paired scenarios:
- Scenario A: With constraint X, decision should be Y
- Scenario B: Without constraint X (or with opposite), decision should be Z

If the system gives the same answer in both cases, it's not actually
using the state causally.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class CausalityTemplate:
    """Template for causality/dependency test cases."""
    name: str
    domain: str
    description: str

    # The constraint that should affect the decision
    constraint: dict[str, str]  # key, value, source
    constraint_description: str

    # The scenario requiring a decision
    scenario: str

    # Query for a decision (not recall)
    decision_query: str

    # Expected decision WITH the constraint
    expected_with_constraint: str

    # Expected decision WITHOUT the constraint (counterfactual)
    expected_without_constraint: str

    # How to verify causal dependency
    constraint_indicators: list[str]  # Evidence constraint was considered
    decision_indicators: list[str]    # Evidence of correct decision


# --- Budget Constraints ---

BUDGET_DEPENDENCY = CausalityTemplate(
    name="budget_dependency",
    domain="procurement",
    description="Decision should depend on budget constraint",
    constraint={
        "key": "quarterly_budget",
        "value": "$50,000 remaining",
        "source": "finance_system",
    },
    constraint_description="Q4 budget is $50,000 remaining",
    scenario="Vendor proposes a $75,000 software license.",
    decision_query="Should we proceed with this vendor's proposal?",
    expected_with_constraint="No - exceeds budget",
    expected_without_constraint="Yes - within approval authority",
    constraint_indicators=["$50,000", "budget", "remaining", "exceeds"],
    decision_indicators=["cannot", "no", "decline", "exceeds budget"],
)

APPROVAL_THRESHOLD = CausalityTemplate(
    name="approval_threshold",
    domain="procurement",
    description="Approval authority should gate the decision",
    constraint={
        "key": "approval_limit",
        "value": "Manager can approve up to $10,000",
        "source": "policy",
    },
    constraint_description="Your approval authority is $10,000",
    scenario="Team requests approval for a $15,000 training program.",
    decision_query="Can you approve this training program request?",
    expected_with_constraint="No - requires higher approval",
    expected_without_constraint="Yes - within typical manager authority",
    constraint_indicators=["$10,000", "approval", "authority", "limit"],
    decision_indicators=["cannot approve", "escalate", "higher approval"],
)


# --- Time Constraints ---

DEADLINE_DEPENDENCY = CausalityTemplate(
    name="deadline_dependency",
    domain="project",
    description="Decision should depend on deadline",
    constraint={
        "key": "submission_deadline",
        "value": "RFP due Friday 5pm",
        "source": "client_email",
    },
    constraint_description="RFP submission deadline is Friday 5pm",
    scenario="It's Thursday. Proposal is 80% complete but needs legal review (takes 24 hours).",
    decision_query="Can we submit a complete, reviewed proposal on time?",
    expected_with_constraint="No - insufficient time for legal review",
    expected_without_constraint="Yes - plenty of time",
    constraint_indicators=["Friday", "5pm", "deadline", "24 hours"],
    decision_indicators=["cannot", "insufficient time", "not possible"],
)

AVAILABILITY_WINDOW = CausalityTemplate(
    name="availability_window",
    domain="support",
    description="Decision should respect availability constraints",
    constraint={
        "key": "service_hours",
        "value": "Technical support: 9am-5pm EST, Mon-Fri",
        "source": "service_policy",
    },
    constraint_description="Technical support available 9am-5pm EST weekdays only",
    scenario="Customer requests urgent technical support at 8pm Saturday.",
    decision_query="Can we provide immediate technical support?",
    expected_with_constraint="No - outside service hours",
    expected_without_constraint="Yes - 24/7 support available",
    constraint_indicators=["9am-5pm", "weekdays", "Saturday", "outside hours"],
    decision_indicators=["unavailable", "no", "outside", "Monday"],
)


# --- Capacity Constraints ---

INVENTORY_DEPENDENCY = CausalityTemplate(
    name="inventory_dependency",
    domain="sales",
    description="Fulfillment decision depends on inventory",
    constraint={
        "key": "inventory_level",
        "value": "Widget X: 50 units in stock",
        "source": "inventory_system",
    },
    constraint_description="Current inventory of Widget X is 50 units",
    scenario="Customer wants to order 100 units of Widget X for next week delivery.",
    decision_query="Can we fulfill this order with next week delivery?",
    expected_with_constraint="No - insufficient inventory",
    expected_without_constraint="Yes - adequate stock",
    constraint_indicators=["50 units", "inventory", "stock"],
    decision_indicators=["cannot fulfill", "partial", "backorder", "insufficient"],
)

TEAM_CAPACITY = CausalityTemplate(
    name="team_capacity",
    domain="project",
    description="Commitment decision depends on team capacity",
    constraint={
        "key": "team_utilization",
        "value": "Engineering team at 95% capacity through Q1",
        "source": "resource_planner",
    },
    constraint_description="Engineering team is at 95% capacity through Q1",
    scenario="Sales wants to commit to a new implementation starting next month.",
    decision_query="Can we commit to starting this implementation next month?",
    expected_with_constraint="No - team over capacity",
    expected_without_constraint="Yes - resources available",
    constraint_indicators=["95%", "capacity", "Q1"],
    decision_indicators=["cannot commit", "no", "overloaded", "unavailable"],
)


# --- Policy Constraints ---

COMPLIANCE_GATE = CausalityTemplate(
    name="compliance_gate",
    domain="hr",
    description="Action depends on compliance requirement",
    constraint={
        "key": "data_access_policy",
        "value": "Employee salary data requires HR Director approval for access",
        "source": "compliance_policy",
    },
    constraint_description="Salary data access requires HR Director approval",
    scenario="Engineering manager requests salary data for their peer's team.",
    decision_query="Can we provide the requested salary data?",
    expected_with_constraint="No - lacks required approval",
    expected_without_constraint="Yes - managers can access team data",
    constraint_indicators=["HR Director", "approval", "policy"],
    decision_indicators=["cannot", "no", "requires approval", "not authorized"],
)

GEOGRAPHIC_RESTRICTION = CausalityTemplate(
    name="geographic_restriction",
    domain="sales",
    description="Offering depends on geographic constraints",
    constraint={
        "key": "service_regions",
        "value": "Product not available in EU due to GDPR data residency requirements",
        "source": "legal_policy",
    },
    constraint_description="Product unavailable in EU due to data residency",
    scenario="Prospect in Germany wants to purchase our SaaS product.",
    decision_query="Can we sell to this German prospect?",
    expected_with_constraint="No - EU not supported",
    expected_without_constraint="Yes - sell globally",
    constraint_indicators=["EU", "GDPR", "Germany", "unavailable"],
    decision_indicators=["cannot sell", "no", "not available", "unsupported region"],
)


# --- Contractual Constraints ---

EXCLUSIVITY_CLAUSE = CausalityTemplate(
    name="exclusivity_clause",
    domain="sales",
    description="New deal depends on existing exclusivity",
    constraint={
        "key": "exclusivity_agreement",
        "value": "Exclusive distribution rights granted to DistCo for healthcare vertical",
        "source": "contract_db",
    },
    constraint_description="DistCo has exclusive rights for healthcare vertical",
    scenario="Hospital system wants to buy directly from us instead of through DistCo.",
    decision_query="Can we sell directly to this hospital?",
    expected_with_constraint="No - violates exclusivity",
    expected_without_constraint="Yes - direct sales allowed",
    constraint_indicators=["exclusive", "DistCo", "healthcare"],
    decision_indicators=["cannot", "no", "must go through", "exclusivity"],
)


# =============================================================================
# HARD MODE: Multi-Constraint Interactions
# =============================================================================
# These require reasoning about MULTIPLE constraints simultaneously

@dataclass
class MultiConstraintTemplate:
    """Template requiring multiple constraints to be satisfied."""
    name: str
    domain: str
    description: str
    constraints: list[dict[str, str]]  # Multiple constraints
    scenario: str
    decision_query: str
    # The decision depends on ALL constraints being satisfied
    expected_decision: str
    # Which constraint(s) block the decision
    blocking_constraints: list[str]
    # What happens if ANY constraint is removed
    partial_satisfaction_trap: str  # Common wrong answer


BUDGET_AND_APPROVAL = MultiConstraintTemplate(
    name="budget_and_approval",
    domain="procurement",
    description="Must satisfy BOTH budget AND approval authority",
    constraints=[
        {"key": "quarterly_budget", "value": "$30,000 remaining in Q4", "source": "finance_system"},
        {"key": "approval_authority", "value": "Your approval limit is $25,000", "source": "policy"},
    ],
    scenario="Request to purchase $28,000 software license.",
    decision_query="Can you approve this purchase?",
    expected_decision="No - within budget but exceeds your approval authority",
    blocking_constraints=["approval_authority"],
    partial_satisfaction_trap="Yes because it's within budget",  # Common wrong answer
)

CAPACITY_AND_DEADLINE_AND_SKILL = MultiConstraintTemplate(
    name="triple_constraint",
    domain="project",
    description="Must satisfy capacity, deadline, AND skill requirements",
    constraints=[
        {"key": "team_capacity", "value": "2 engineers available next sprint", "source": "resource_planner"},
        {"key": "project_deadline", "value": "Feature must ship by March 15", "source": "client_email"},
        {"key": "skill_requirement", "value": "Requires senior backend expertise (3+ years)", "source": "tech_lead"},
    ],
    scenario="New feature request: Build payment integration. Estimate: 3 engineers for 4 weeks. "
             "Available engineers: 2 mid-level frontend devs.",
    decision_query="Can we commit to delivering this feature on time?",
    expected_decision="No - insufficient capacity AND wrong skillset",
    blocking_constraints=["team_capacity", "skill_requirement"],
    partial_satisfaction_trap="Yes if we just add one more engineer",
)

COMPLIANCE_AND_BUDGET_AND_TIMELINE = MultiConstraintTemplate(
    name="compliance_budget_timeline",
    domain="procurement",
    description="Vendor selection with regulatory, budget, and timeline constraints",
    constraints=[
        {"key": "compliance_req", "value": "Vendor must be SOC2 Type II certified", "source": "security_policy"},
        {"key": "budget_limit", "value": "Maximum $100,000 annual spend", "source": "finance"},
        {"key": "go_live", "value": "Must be operational by April 1", "source": "program_manager"},
    ],
    scenario="Vendor A: $90,000/year, SOC2 certified, 6-month implementation. "
             "Vendor B: $80,000/year, no SOC2, 2-month implementation. "
             "Today is February 1.",
    decision_query="Which vendor should we select?",
    expected_decision="Neither works - A misses deadline, B fails compliance",
    blocking_constraints=["go_live", "compliance_req"],
    partial_satisfaction_trap="Vendor B because it's cheaper and faster",
)


# =============================================================================
# HARD MODE: Edge Cases and Near-Threshold
# =============================================================================

@dataclass
class EdgeCaseTemplate:
    """Template with values right at constraint boundaries."""
    name: str
    domain: str
    description: str
    constraint: dict[str, str]
    # The request is EXACTLY at or just over the threshold
    edge_case_request: str
    decision_query: str
    expected_decision: str
    # The subtle detail that makes it a violation
    violation_detail: str
    # Plausible but wrong reasoning
    wrong_reasoning: str


EXACT_BUDGET_EDGE = EdgeCaseTemplate(
    name="exact_budget_edge",
    domain="procurement",
    description="Request equals budget but ignores fees",
    constraint={"key": "available_budget", "value": "$50,000 remaining", "source": "finance"},
    edge_case_request="Purchase request for $50,000 software license (plus 8% sales tax)",
    decision_query="Does this fit within our remaining budget?",
    expected_decision="No - total cost is $54,000 with tax",
    violation_detail="Sales tax pushes it $4,000 over budget",
    wrong_reasoning="Yes, exactly $50,000 matches the budget",
)

CAPACITY_ROUNDING = EdgeCaseTemplate(
    name="capacity_rounding",
    domain="project",
    description="Capacity calculation with fractional resources",
    constraint={"key": "engineer_allocation", "value": "3 engineers at 50% allocation each", "source": "resource_mgr"},
    edge_case_request="Task requires 2 full-time engineer equivalents for 2 weeks",
    decision_query="Do we have sufficient engineering capacity?",
    expected_decision="No - 1.5 FTE available, need 2 FTE",
    violation_detail="3 engineers × 50% = 1.5 FTE, not 2 FTE",
    wrong_reasoning="Yes, we have 3 engineers available",
)

DEADLINE_TIMEZONE = EdgeCaseTemplate(
    name="deadline_timezone",
    domain="sales",
    description="Deadline with timezone complexity",
    constraint={"key": "submission_deadline", "value": "Proposal due Friday 5pm EST", "source": "rfp"},
    edge_case_request="It's Friday 3pm PST. Proposal needs 1 hour to finalize.",
    decision_query="Can we submit on time?",
    expected_decision="No - 3pm PST = 6pm EST, already past deadline",
    violation_detail="Pacific time is 3 hours behind Eastern",
    wrong_reasoning="Yes, we have 2 hours until 5pm",
)

INVENTORY_RESERVED = EdgeCaseTemplate(
    name="inventory_reserved",
    domain="sales",
    description="Inventory count includes reserved units",
    constraint={
        "key": "inventory",
        "value": "100 units in warehouse (40 reserved for pending orders)",
        "source": "wms",
    },
    edge_case_request="New customer wants to order 80 units immediately",
    decision_query="Can we fulfill this 80-unit order from current stock?",
    expected_decision="No - only 60 units available (100 - 40 reserved)",
    violation_detail="Reserved units cannot be sold twice",
    wrong_reasoning="Yes, we have 100 units in stock",
)


# =============================================================================
# HARD MODE: Conflicting Constraints
# =============================================================================

@dataclass
class ConflictingConstraintTemplate:
    """Template where two valid constraints cannot both be satisfied."""
    name: str
    domain: str
    description: str
    constraint_a: dict[str, str]
    constraint_b: dict[str, str]
    scenario: str
    decision_query: str
    # The correct answer acknowledges the conflict
    expected_decision: str
    # What each constraint alone would suggest
    if_only_a: str
    if_only_b: str


SPEED_VS_QUALITY = ConflictingConstraintTemplate(
    name="speed_vs_quality",
    domain="project",
    description="Deadline conflicts with quality requirement",
    constraint_a={
        "key": "deadline", "value": "Feature must ship Monday", "source": "product_manager"
    },
    constraint_b={
        "key": "quality_gate",
        "value": "All features require 80% test coverage before release",
        "source": "eng_policy",
    },
    scenario="It's Friday. Feature is code-complete but has 30% test coverage. "
             "Writing tests to 80% coverage takes 3 days.",
    decision_query="Should we ship the feature Monday?",
    expected_decision="Cannot satisfy both - must negotiate deadline OR quality waiver",
    if_only_a="Yes, ship Monday as requested",
    if_only_b="No, wait until tests are complete",
)

CUSTOMER_A_VS_CUSTOMER_B = ConflictingConstraintTemplate(
    name="customer_priority_conflict",
    domain="support",
    description="Two customers with conflicting priority claims",
    constraint_a={
        "key": "customer_a_sla",
        "value": "Acme Corp has 4-hour response SLA (Enterprise tier)",
        "source": "crm",
    },
    constraint_b={
        "key": "customer_b_sla",
        "value": "Beta Inc has 4-hour response SLA (Enterprise tier)",
        "source": "crm",
    },
    scenario="Both customers submitted critical tickets at same time. "
             "Only 1 senior engineer available. Each ticket needs 4 hours to resolve.",
    decision_query="How should we prioritize these tickets?",
    expected_decision="Cannot meet both SLAs - must escalate for additional resources or negotiate",
    if_only_a="Prioritize Acme Corp",
    if_only_b="Prioritize Beta Inc",
)

COST_VS_COMPLIANCE = ConflictingConstraintTemplate(
    name="cost_vs_compliance",
    domain="procurement",
    description="Cheapest option violates compliance, compliant option exceeds budget",
    constraint_a={"key": "budget_cap", "value": "IT infrastructure budget capped at $200,000", "source": "cfo"},
    constraint_b={"key": "data_residency", "value": "Customer data must remain in US data centers", "source": "legal"},
    scenario="Cloud Provider A: $150,000/year, EU data centers only. "
             "Cloud Provider B: $250,000/year, US data centers.",
    decision_query="Which cloud provider should we select?",
    expected_decision="Neither works within constraints - need budget increase or compliance waiver",
    if_only_a="Provider A (within budget)",
    if_only_b="Provider B (compliant)",
)


# =============================================================================
# HARD MODE: Chain Dependencies
# =============================================================================

@dataclass
class ChainDependencyTemplate:
    """Template where decision A affects fact B which affects decision C."""
    name: str
    domain: str
    description: str
    # The chain of dependencies
    initial_fact: dict[str, str]
    derived_fact: dict[str, str]  # Depends on initial_fact
    final_constraint: dict[str, str]  # Uses derived_fact
    scenario: str
    decision_query: str
    expected_decision: str
    # The chain that must be followed
    reasoning_chain: list[str]
    # Wrong answer from breaking the chain
    chain_break_error: str


DISCOUNT_MARGIN_APPROVAL = ChainDependencyTemplate(
    name="discount_margin_approval",
    domain="sales",
    description="Discount affects margin which affects approval requirement",
    initial_fact={"key": "list_price", "value": "Product list price: $100,000", "source": "price_book"},
    derived_fact={"key": "min_margin", "value": "Minimum margin requirement: 20%", "source": "finance_policy"},
    final_constraint={"key": "discount_authority", "value": "Sales rep can approve up to 15% discount. "
                      "VP approval required for discounts that drop margin below 20%", "source": "sales_policy"},
    scenario="Customer wants 25% discount. Our cost basis is $75,000.",
    decision_query="Can the sales rep approve this discount?",
    expected_decision="No - 25% discount = $75,000 price = 0% margin, requires VP approval",
    reasoning_chain=[
        "List price $100,000 - 25% = $75,000 sale price",
        "Cost $75,000 / Sale $75,000 = 0% margin",
        "0% margin < 20% minimum → requires VP approval"
    ],
    chain_break_error="Yes, sales rep can approve up to 15% so can handle any discount request",
)

HEADCOUNT_BUDGET_TIMELINE = ChainDependencyTemplate(
    name="headcount_budget_timeline",
    domain="hr",
    description="Hiring decision depends on budget which depends on fiscal year",
    initial_fact={
        "key": "fiscal_year", "value": "Fiscal year ends March 31", "source": "finance"
    },
    derived_fact={
        "key": "remaining_budget",
        "value": "Q4 hiring budget: $150,000 (use-or-lose)",
        "source": "hr_finance",
    },
    final_constraint={
        "key": "hiring_timeline",
        "value": "Average time-to-hire: 8 weeks. New hires must start before budget expires.",
        "source": "talent_acquisition",
    },
    scenario="Today is February 15. Want to hire 2 senior engineers at $180,000 total.",
    decision_query="Can we complete these hires within our Q4 budget?",
    expected_decision="No - 8 weeks from Feb 15 = April 12, after March 31 budget expiry",
    reasoning_chain=[
        "Feb 15 + 8 weeks = April 12 start date",
        "Budget expires March 31",
        "April 12 > March 31 → budget will have expired"
    ],
    chain_break_error="Yes, we have $150,000 budget and only need $180,000... wait, also no on amount",
)

CAPACITY_DEPENDENCY_REVENUE = ChainDependencyTemplate(
    name="capacity_dependency_revenue",
    domain="sales",
    description="New deal depends on capacity which depends on existing commitments",
    initial_fact={
        "key": "current_commitments",
        "value": "Committed to 3 implementations in Q1 (600 hours)",
        "source": "pmo",
    },
    derived_fact={
        "key": "available_capacity",
        "value": "Implementation team: 800 hours/quarter total",
        "source": "resource_mgr",
    },
    final_constraint={
        "key": "new_deal_requirement",
        "value": "New deal requires 300 hours implementation",
        "source": "sales",
    },
    scenario="Sales wants to close a new deal requiring 300 implementation hours in Q1.",
    decision_query="Can we commit to implementing this new deal in Q1?",
    expected_decision="No - only 200 hours available (800 - 600 committed)",
    reasoning_chain=[
        "Total capacity: 800 hours",
        "Already committed: 600 hours",
        "Available: 800 - 600 = 200 hours",
        "Need 300 hours > 200 available"
    ],
    chain_break_error="Yes, we have 800 hours capacity",
)


# =============================================================================
# HARD MODE: Aggregation Problems
# =============================================================================

@dataclass
class AggregationTemplate:
    """Template where multiple small items together violate a constraint."""
    name: str
    domain: str
    description: str
    constraint: dict[str, str]
    # Multiple items that individually seem fine
    items: list[dict[str, str]]  # Each has name, value
    aggregation_query: str
    decision_query: str
    expected_decision: str
    # The aggregation that's missed
    aggregation_detail: str
    # Per-item reasoning that misses the aggregate
    per_item_trap: str


AGGREGATE_SPEND = AggregationTemplate(
    name="aggregate_spend",
    domain="procurement",
    description="Multiple small purchases that together exceed authority",
    constraint={
        "key": "quarterly_limit",
        "value": "Single-vendor quarterly spend limit: $50,000 without RFP",
        "source": "procurement_policy",
    },
    items=[
        {"name": "January order", "value": "$18,000"},
        {"name": "February order", "value": "$22,000"},
        {"name": "March order (proposed)", "value": "$15,000"},
    ],
    aggregation_query="What's our Q1 total spend with VendorX?",
    decision_query="Can we place this $15,000 March order without an RFP?",
    expected_decision="No - aggregate Q1 spend would be $55,000, exceeding $50,000 limit",
    aggregation_detail="$18,000 + $22,000 + $15,000 = $55,000 > $50,000",
    per_item_trap="Yes, $15,000 is well under the $50,000 limit",
)

AGGREGATE_HOURS = AggregationTemplate(
    name="aggregate_hours",
    domain="project",
    description="Multiple task assignments that overload a team member",
    constraint={
        "key": "max_allocation",
        "value": "No engineer should exceed 40 hours/week allocation",
        "source": "hr_policy",
    },
    items=[
        {"name": "Project Alpha", "value": "15 hours/week"},
        {"name": "Project Beta", "value": "12 hours/week"},
        {"name": "Project Gamma", "value": "10 hours/week"},
        {"name": "On-call rotation", "value": "8 hours/week"},
    ],
    aggregation_query="What's Sarah's total weekly allocation?",
    decision_query="Can we add Sarah to Project Gamma for 10 hours/week?",
    expected_decision="No - total would be 45 hours/week, exceeding 40-hour limit",
    aggregation_detail="15 + 12 + 10 + 8 = 45 hours > 40 hour limit",
    per_item_trap="Yes, 10 hours is a reasonable part-time allocation",
)

AGGREGATE_RISK = AggregationTemplate(
    name="aggregate_risk",
    domain="sales",
    description="Multiple deals with same customer exceeding risk threshold",
    constraint={
        "key": "customer_concentration",
        "value": "No single customer can exceed 20% of pipeline",
        "source": "sales_policy",
    },
    items=[
        {"name": "Deal 1 (closed)", "value": "$500,000"},
        {"name": "Deal 2 (in progress)", "value": "$300,000"},
        {"name": "Deal 3 (proposed)", "value": "$400,000"},
    ],
    aggregation_query="What's MegaCorp's share of our $5M pipeline?",
    decision_query="Should we pursue this $400,000 expansion deal with MegaCorp?",
    expected_decision="Caution - MegaCorp would be 24% of pipeline ($1.2M / $5M), exceeding 20% limit",
    aggregation_detail="$500K + $300K + $400K = $1.2M = 24% of $5M pipeline",
    per_item_trap="Yes, $400,000 is a good deal size",
)


# =============================================================================
# All Templates
# =============================================================================

# Original simple templates
SIMPLE_CAUSALITY_TEMPLATES = [
    BUDGET_DEPENDENCY,
    APPROVAL_THRESHOLD,
    DEADLINE_DEPENDENCY,
    AVAILABILITY_WINDOW,
    INVENTORY_DEPENDENCY,
    TEAM_CAPACITY,
    COMPLIANCE_GATE,
    GEOGRAPHIC_RESTRICTION,
    EXCLUSIVITY_CLAUSE,
]

# Hard mode templates
MULTI_CONSTRAINT_TEMPLATES = [
    BUDGET_AND_APPROVAL,
    CAPACITY_AND_DEADLINE_AND_SKILL,
    COMPLIANCE_AND_BUDGET_AND_TIMELINE,
]

EDGE_CASE_TEMPLATES = [
    EXACT_BUDGET_EDGE,
    CAPACITY_ROUNDING,
    DEADLINE_TIMEZONE,
    INVENTORY_RESERVED,
]

CONFLICTING_TEMPLATES = [
    SPEED_VS_QUALITY,
    CUSTOMER_A_VS_CUSTOMER_B,
    COST_VS_COMPLIANCE,
]

CHAIN_DEPENDENCY_TEMPLATES = [
    DISCOUNT_MARGIN_APPROVAL,
    HEADCOUNT_BUDGET_TIMELINE,
    CAPACITY_DEPENDENCY_REVENUE,
]

AGGREGATION_TEMPLATES = [
    AGGREGATE_SPEND,
    AGGREGATE_HOURS,
    AGGREGATE_RISK,
]

# Combined list (simple + hard)
CAUSALITY_TEMPLATES = SIMPLE_CAUSALITY_TEMPLATES

# All hard mode templates combined
HARD_CAUSALITY_TEMPLATES = (
    MULTI_CONSTRAINT_TEMPLATES +
    EDGE_CASE_TEMPLATES +
    CONFLICTING_TEMPLATES +
    CHAIN_DEPENDENCY_TEMPLATES +
    AGGREGATION_TEMPLATES
)


def get_causality_templates(include_hard: bool = False) -> list[Any]:
    """Get causality templates.

    Args:
        include_hard: If True, include hard mode templates

    Returns:
        List of templates (mixed types if include_hard=True)
    """
    if include_hard:
        return SIMPLE_CAUSALITY_TEMPLATES + HARD_CAUSALITY_TEMPLATES
    return list(SIMPLE_CAUSALITY_TEMPLATES)


def get_hard_causality_templates() -> list[Any]:
    """Get only the hard mode causality templates."""
    return list(HARD_CAUSALITY_TEMPLATES)


def get_paired_test(template: CausalityTemplate) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate a paired test for counterfactual verification.

    Returns:
        Tuple of (with_constraint_test, without_constraint_test)
    """
    with_constraint = {
        "constraint": template.constraint,
        "scenario": template.scenario,
        "query": template.decision_query,
        "expected": template.expected_with_constraint,
        "must_mention": template.constraint_indicators,
        "decision_indicators": template.decision_indicators,
    }

    without_constraint: dict[str, Any] = {
        "constraint": None,  # Constraint removed
        "scenario": template.scenario,
        "query": template.decision_query,
        "expected": template.expected_without_constraint,
        "must_mention": [],
        "decision_indicators": [],  # Opposite decision expected
    }

    return with_constraint, without_constraint
