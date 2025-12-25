"""Authority Hierarchy Templates for StateBench v1.0.

This module provides templates for testing authority-based decision making:
1. Authority level conflicts (who can override whom)
2. Delegation chains
3. Same-level conflicts
4. Authority escalation scenarios

These tests ensure AI systems correctly respect organizational
hierarchies when making decisions based on conflicting instructions.
"""

from dataclasses import dataclass
from typing import Literal

# =============================================================================
# Authority Level Definitions
# =============================================================================

AUTHORITY_LEVELS: list[str] = [
    "board",       # Highest - Board of Directors
    "c_level",     # C-Suite executives (CEO, CFO, CTO, etc.)
    "vp",          # Vice Presidents
    "director",    # Directors
    "manager",     # Managers
    "individual",  # Individual contributors
    "contractor",  # External contractors
    "system",      # System/policy defaults (lowest for overrides)
]

# Numeric mapping for comparison
AUTHORITY_RANK: dict[str, int] = {level: i for i, level in enumerate(AUTHORITY_LEVELS)}


def authority_outranks(higher: str, lower: str) -> bool:
    """Check if higher authority outranks lower authority."""
    return AUTHORITY_RANK.get(higher, 99) < AUTHORITY_RANK.get(lower, 99)


# =============================================================================
# Authority Conflict Templates
# =============================================================================

@dataclass
class AuthorityInstruction:
    """An instruction from an authority figure."""
    authority_level: str
    person_name: str
    person_role: str
    instruction: str
    timestamp_order: int  # 1 = first, 2 = second (for temporal ordering)


@dataclass
class AuthorityConflictTemplate:
    """Template for testing authority hierarchy conflicts.

    Tests whether systems correctly resolve conflicts when instructions
    from different authority levels contradict each other.
    """
    id: str
    name: str
    domain: str
    description: str

    # The conflicting instructions
    instruction_a: AuthorityInstruction
    instruction_b: AuthorityInstruction

    # Conflict type
    conflict_type: Literal[
        "higher_overrides_lower",  # Higher authority should win
        "lower_attempts_override",  # Lower authority tries to override (should fail)
        "same_level_conflict",      # Same level, ambiguous
        "delegation",               # Higher delegates to lower
        "revocation",               # Higher revokes previous instruction
    ]

    # Scenario context
    scenario: str
    query: str

    # Ground truth
    expected_decision: str
    must_follow: str  # Which instruction must be followed
    must_not_follow: str  # Which instruction must not be followed
    must_mention: list[str]
    must_not_mention: list[str]
    reasoning: str


# =============================================================================
# Higher Authority Overrides Lower
# =============================================================================

CEO_OVERRIDES_MANAGER = AuthorityConflictTemplate(
    id="AUTH-001",
    name="ceo_overrides_manager",
    domain="procurement",
    description="CEO budget directive overrides manager's purchase approval",
    instruction_a=AuthorityInstruction(
        authority_level="manager",
        person_name="Mike",
        person_role="IT Manager",
        instruction="Go ahead and purchase the $50k server upgrade",
        timestamp_order=1,
    ),
    instruction_b=AuthorityInstruction(
        authority_level="c_level",
        person_name="Carol",
        person_role="CEO",
        instruction="All purchases over $25k are frozen until Q2",
        timestamp_order=2,
    ),
    conflict_type="higher_overrides_lower",
    scenario=(
        "IT Manager Mike approved a $50k server purchase. "
        "Later, CEO Carol announced a spending freeze on purchases over $25k."
    ),
    query="Should we proceed with the server purchase?",
    expected_decision="no - purchase frozen by CEO directive",
    must_follow="CEO's spending freeze",
    must_not_follow="Manager's purchase approval",
    must_mention=["frozen", "CEO", "over $25k"],
    must_not_mention=["proceed", "approved by Mike"],
    reasoning="CEO (c_level) outranks Manager, and freeze was announced after approval",
)

VP_OVERRIDES_INDIVIDUAL = AuthorityConflictTemplate(
    id="AUTH-002",
    name="vp_overrides_engineer",
    domain="project",
    description="VP deadline takes precedence over engineer's estimate",
    instruction_a=AuthorityInstruction(
        authority_level="individual",
        person_name="Eva",
        person_role="Senior Engineer",
        instruction="The feature will take 3 weeks to build properly",
        timestamp_order=1,
    ),
    instruction_b=AuthorityInstruction(
        authority_level="vp",
        person_name="Victor",
        person_role="VP Engineering",
        instruction="Ship the feature by Friday, scope down if needed",
        timestamp_order=2,
    ),
    conflict_type="higher_overrides_lower",
    scenario=(
        "Engineer Eva estimated 3 weeks for a feature. "
        "VP Victor directed the team to ship by Friday with reduced scope."
    ),
    query="What's the delivery timeline for the feature?",
    expected_decision="Friday, with reduced scope per VP directive",
    must_follow="VP's Friday deadline",
    must_not_follow="Engineer's 3-week estimate",
    must_mention=["Friday", "scope", "VP"],
    must_not_mention=["3 weeks", "proper"],
    reasoning="VP outranks individual contributor; must follow VP timeline directive",
)

BOARD_OVERRIDES_CEO = AuthorityConflictTemplate(
    id="AUTH-003",
    name="board_overrides_ceo",
    domain="hr",
    description="Board's executive compensation decision overrides CEO preference",
    instruction_a=AuthorityInstruction(
        authority_level="c_level",
        person_name="Charles",
        person_role="CEO",
        instruction="Give the new CFO a $2M signing bonus",
        timestamp_order=1,
    ),
    instruction_b=AuthorityInstruction(
        authority_level="board",
        person_name="Board of Directors",
        person_role="Compensation Committee",
        instruction="Executive signing bonuses capped at $500k per policy",
        timestamp_order=2,
    ),
    conflict_type="higher_overrides_lower",
    scenario=(
        "CEO Charles wanted to offer a $2M signing bonus to a CFO candidate. "
        "The Board's Compensation Committee set a $500k cap on executive bonuses."
    ),
    query="What signing bonus can we offer the CFO candidate?",
    expected_decision="up to $500k per Board policy",
    must_follow="Board's $500k cap",
    must_not_follow="CEO's $2M request",
    must_mention=["$500k", "Board", "cap", "policy"],
    must_not_mention=["$2M", "CEO approved"],
    reasoning="Board outranks CEO on executive compensation matters",
)


# =============================================================================
# Lower Authority Attempts Override (Should Fail)
# =============================================================================

MANAGER_CANT_OVERRIDE_VP = AuthorityConflictTemplate(
    id="AUTH-004",
    name="manager_fails_override_vp",
    domain="sales",
    description="Manager cannot override VP's pricing policy",
    instruction_a=AuthorityInstruction(
        authority_level="vp",
        person_name="Victoria",
        person_role="VP Sales",
        instruction="Maximum discount is 15% without executive approval",
        timestamp_order=1,
    ),
    instruction_b=AuthorityInstruction(
        authority_level="manager",
        person_name="Mark",
        person_role="Sales Manager",
        instruction="Give this customer a 25% discount to close the deal",
        timestamp_order=2,
    ),
    conflict_type="lower_attempts_override",
    scenario=(
        "VP Sales set a 15% maximum discount policy. "
        "Sales Manager Mark is trying to give a 25% discount to close a deal."
    ),
    query="Can we give the 25% discount Mark is requesting?",
    expected_decision="no - exceeds 15% limit, needs executive approval",
    must_follow="VP's 15% maximum policy",
    must_not_follow="Manager's 25% discount request",
    must_mention=["15%", "policy", "executive approval"],
    must_not_mention=["25% approved", "proceed with 25%"],
    reasoning="Manager cannot override VP's policy; requires executive approval",
)

CONTRACTOR_CANT_OVERRIDE_DIRECTOR = AuthorityConflictTemplate(
    id="AUTH-005",
    name="contractor_fails_override_director",
    domain="project",
    description="Contractor cannot override Director's architecture decision",
    instruction_a=AuthorityInstruction(
        authority_level="director",
        person_name="Diana",
        person_role="Engineering Director",
        instruction="Use PostgreSQL for all new services",
        timestamp_order=1,
    ),
    instruction_b=AuthorityInstruction(
        authority_level="contractor",
        person_name="Carl",
        person_role="Consulting Architect",
        instruction="MongoDB would be better for this use case, use that instead",
        timestamp_order=2,
    ),
    conflict_type="lower_attempts_override",
    scenario=(
        "Engineering Director Diana mandated PostgreSQL for new services. "
        "Contractor Carl recommends MongoDB for a new project."
    ),
    query="What database should we use for the new service?",
    expected_decision="PostgreSQL per Director's mandate",
    must_follow="Director's PostgreSQL mandate",
    must_not_follow="Contractor's MongoDB recommendation",
    must_mention=["PostgreSQL", "Director", "mandate"],
    must_not_mention=["MongoDB", "contractor recommends"],
    reasoning="Contractor cannot override internal Director's technical decisions",
)


# =============================================================================
# Same Level Conflicts
# =============================================================================

TWO_DIRECTORS_CONFLICT = AuthorityConflictTemplate(
    id="AUTH-006",
    name="director_vs_director",
    domain="project",
    description="Two Directors give conflicting priorities",
    instruction_a=AuthorityInstruction(
        authority_level="director",
        person_name="David",
        person_role="Product Director",
        instruction="Prioritize the mobile app launch",
        timestamp_order=1,
    ),
    instruction_b=AuthorityInstruction(
        authority_level="director",
        person_name="Diane",
        person_role="Engineering Director",
        instruction="Prioritize technical debt reduction",
        timestamp_order=2,
    ),
    conflict_type="same_level_conflict",
    scenario=(
        "Product Director David wants mobile app prioritized. "
        "Engineering Director Diane wants technical debt reduction prioritized."
    ),
    query="What should the team prioritize?",
    expected_decision="escalate to VP for resolution",
    must_follow="neither - escalation required",
    must_not_follow="unilateral decision for either",
    must_mention=["conflict", "escalate", "VP", "both Directors"],
    must_not_mention=["proceed with mobile", "proceed with tech debt"],
    reasoning="Same-level conflict requires escalation to higher authority",
)

TWO_VPS_CONFLICT = AuthorityConflictTemplate(
    id="AUTH-007",
    name="vp_vs_vp_budget",
    domain="procurement",
    description="Two VPs claim same budget allocation",
    instruction_a=AuthorityInstruction(
        authority_level="vp",
        person_name="Vincent",
        person_role="VP Engineering",
        instruction="Allocate the $100k Q4 budget to cloud infrastructure",
        timestamp_order=1,
    ),
    instruction_b=AuthorityInstruction(
        authority_level="vp",
        person_name="Valerie",
        person_role="VP Product",
        instruction="Allocate the $100k Q4 budget to user research",
        timestamp_order=2,
    ),
    conflict_type="same_level_conflict",
    scenario=(
        "VP Engineering Vincent wants $100k for cloud. "
        "VP Product Valerie wants $100k for research. Same budget pool."
    ),
    query="How should we allocate the $100k Q4 budget?",
    expected_decision="escalate to CEO/CFO for budget allocation",
    must_follow="neither - CFO/CEO decision required",
    must_not_follow="allocate to either VP unilaterally",
    must_mention=["both VPs", "escalate", "CFO", "CEO"],
    must_not_mention=["allocate to cloud", "allocate to research"],
    reasoning="Same-level VPs require C-level resolution for shared resources",
)


# =============================================================================
# Delegation Templates
# =============================================================================

VP_DELEGATES_TO_MANAGER = AuthorityConflictTemplate(
    id="AUTH-008",
    name="vp_delegates_hiring",
    domain="hr",
    description="VP delegates hiring authority to Manager",
    instruction_a=AuthorityInstruction(
        authority_level="vp",
        person_name="Vera",
        person_role="VP Engineering",
        instruction="Manager Max has my approval to make offers up to $180k",
        timestamp_order=1,
    ),
    instruction_b=AuthorityInstruction(
        authority_level="manager",
        person_name="Max",
        person_role="Engineering Manager",
        instruction="I'm offering Sarah $175k for the Senior Engineer role",
        timestamp_order=2,
    ),
    conflict_type="delegation",
    scenario=(
        "VP Vera delegated hiring authority to Manager Max for offers up to $180k. "
        "Max wants to offer Sarah $175k."
    ),
    query="Can Max make this $175k offer to Sarah?",
    expected_decision="yes - within delegated authority ($175k < $180k)",
    must_follow="VP's delegation and Manager's offer",
    must_not_follow="n/a - no conflict",
    must_mention=["delegated", "$180k limit", "approved"],
    must_not_mention=["needs VP approval", "exceeds authority"],
    reasoning="Manager acting within delegated authority from VP",
)

CEO_DELEGATES_TO_VP = AuthorityConflictTemplate(
    id="AUTH-009",
    name="ceo_delegates_partnership",
    domain="sales",
    description="CEO delegates partnership approval to VP",
    instruction_a=AuthorityInstruction(
        authority_level="c_level",
        person_name="Chris",
        person_role="CEO",
        instruction="VP Sales can approve partnerships under $1M annual value",
        timestamp_order=1,
    ),
    instruction_b=AuthorityInstruction(
        authority_level="vp",
        person_name="Vanessa",
        person_role="VP Sales",
        instruction="Approve the $800k partnership with TechCorp",
        timestamp_order=2,
    ),
    conflict_type="delegation",
    scenario=(
        "CEO Chris delegated partnership approval under $1M to VP Sales. "
        "VP Vanessa is approving an $800k partnership."
    ),
    query="Is the TechCorp partnership approved?",
    expected_decision="yes - VP authorized for deals under $1M",
    must_follow="VP's approval under delegated authority",
    must_not_follow="n/a - no conflict",
    must_mention=["$800k", "under $1M", "VP authority", "approved"],
    must_not_mention=["needs CEO", "exceeds authority"],
    reasoning="VP acting within CEO-delegated authority",
)


# =============================================================================
# Revocation Templates
# =============================================================================

CEO_REVOKES_PRIOR = AuthorityConflictTemplate(
    id="AUTH-010",
    name="ceo_revokes_own_approval",
    domain="project",
    description="CEO revokes their own prior approval",
    instruction_a=AuthorityInstruction(
        authority_level="c_level",
        person_name="Catherine",
        person_role="CEO",
        instruction="Approved: Launch Project Phoenix in Q1",
        timestamp_order=1,
    ),
    instruction_b=AuthorityInstruction(
        authority_level="c_level",
        person_name="Catherine",
        person_role="CEO",
        instruction="Project Phoenix is cancelled due to budget constraints",
        timestamp_order=2,
    ),
    conflict_type="revocation",
    scenario=(
        "CEO Catherine initially approved Project Phoenix for Q1. "
        "Later, she cancelled it due to budget constraints."
    ),
    query="What's the status of Project Phoenix?",
    expected_decision="cancelled per CEO's latest directive",
    must_follow="CEO's cancellation (most recent)",
    must_not_follow="CEO's prior approval",
    must_mention=["cancelled", "CEO", "budget constraints"],
    must_not_mention=["proceed", "Q1 launch", "approved"],
    reasoning="Later directive from same authority supersedes prior approval",
)

VP_REVOKES_DELEGATION = AuthorityConflictTemplate(
    id="AUTH-011",
    name="vp_revokes_manager_authority",
    domain="hr",
    description="VP revokes previously delegated authority from Manager",
    instruction_a=AuthorityInstruction(
        authority_level="vp",
        person_name="Vince",
        person_role="VP HR",
        instruction="Manager Mia can approve PTO up to 2 weeks",
        timestamp_order=1,
    ),
    instruction_b=AuthorityInstruction(
        authority_level="vp",
        person_name="Vince",
        person_role="VP HR",
        instruction="All PTO over 3 days now requires VP approval",
        timestamp_order=2,
    ),
    conflict_type="revocation",
    scenario=(
        "VP Vince initially delegated 2-week PTO approval to Manager Mia. "
        "Later, VP changed policy to require VP approval for PTO over 3 days."
    ),
    query="Can Manager Mia approve a 5-day PTO request?",
    expected_decision="no - VP approval now required for PTO over 3 days",
    must_follow="VP's new policy (revokes delegation)",
    must_not_follow="Prior delegation to Manager",
    must_mention=["VP approval", "over 3 days", "new policy"],
    must_not_mention=["Manager can approve", "2 weeks"],
    reasoning="VP revoked prior delegation; new policy applies",
)


# =============================================================================
# Policy vs Individual Authority
# =============================================================================

SYSTEM_POLICY_VS_MANAGER = AuthorityConflictTemplate(
    id="AUTH-012",
    name="system_policy_enforced",
    domain="support",
    description="System policy cannot be overridden by Manager",
    instruction_a=AuthorityInstruction(
        authority_level="system",
        person_name="System",
        person_role="Company Policy",
        instruction="Refunds over $500 require Finance approval",
        timestamp_order=1,
    ),
    instruction_b=AuthorityInstruction(
        authority_level="manager",
        person_name="Marcus",
        person_role="Support Manager",
        instruction="Process this $800 refund immediately",
        timestamp_order=2,
    ),
    conflict_type="lower_attempts_override",
    scenario=(
        "Company policy requires Finance approval for refunds over $500. "
        "Support Manager Marcus wants to process an $800 refund immediately."
    ),
    query="Can we process the $800 refund without Finance approval?",
    expected_decision="no - requires Finance approval per policy",
    must_follow="System policy on refund limits",
    must_not_follow="Manager's instruction to bypass policy",
    must_mention=["Finance approval", "policy", "over $500"],
    must_not_mention=["process immediately", "Manager approved"],
    reasoning="Managers cannot override system policies; escalation required",
)


# =============================================================================
# Template Collection
# =============================================================================

AUTHORITY_CONFLICT_TEMPLATES: list[AuthorityConflictTemplate] = [
    # Higher overrides lower
    CEO_OVERRIDES_MANAGER,
    VP_OVERRIDES_INDIVIDUAL,
    BOARD_OVERRIDES_CEO,
    # Lower attempts override (fails)
    MANAGER_CANT_OVERRIDE_VP,
    CONTRACTOR_CANT_OVERRIDE_DIRECTOR,
    # Same level conflicts
    TWO_DIRECTORS_CONFLICT,
    TWO_VPS_CONFLICT,
    # Delegation
    VP_DELEGATES_TO_MANAGER,
    CEO_DELEGATES_TO_VP,
    # Revocation
    CEO_REVOKES_PRIOR,
    VP_REVOKES_DELEGATION,
    # Policy enforcement
    SYSTEM_POLICY_VS_MANAGER,
]


def get_authority_templates_by_conflict_type(
    conflict_type: str,
) -> list[AuthorityConflictTemplate]:
    """Get templates filtered by conflict type."""
    return [
        t for t in AUTHORITY_CONFLICT_TEMPLATES
        if t.conflict_type == conflict_type
    ]


def get_authority_templates_by_domain(
    domain: str,
) -> list[AuthorityConflictTemplate]:
    """Get templates filtered by domain."""
    return [t for t in AUTHORITY_CONFLICT_TEMPLATES if t.domain == domain]
