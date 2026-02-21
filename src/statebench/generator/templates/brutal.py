"""Track 10: Brutal Realistic Scenarios.

These are long, messy test cases that simulate a real AI agent's week.
They combine multiple failure modes:
- Conflicting instructions from different stakeholders
- Corrections and retractions
- Interruptions and context switches
- Time pressure and deadline changes
- Ambiguous requests requiring clarification
- Authority conflicts

If a system passes these, it means something.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BrutalEvent:
    """A single event in a brutal scenario."""
    day: int  # Day 1, 2, 3, etc.
    time: str  # e.g., "9:00 AM"
    speaker: str  # Who is speaking
    role: str  # Their authority level
    content: str  # What they said
    event_type: str  # "instruction", "correction", "interruption", "query"


@dataclass
class BrutalScenario:
    """A week-long brutal scenario testing all failure modes."""
    name: str
    domain: str
    description: str

    # Timeline of events
    events: list[BrutalEvent] = field(default_factory=list)

    # Critical state that must be tracked correctly
    critical_state: dict[str, Any] = field(default_factory=dict)

    # Final queries that test everything
    final_queries: list[dict[str, Any]] = field(default_factory=list)


# --- The Procurement Nightmare ---

PROCUREMENT_NIGHTMARE = BrutalScenario(
    name="procurement_nightmare",
    domain="procurement",
    description="A week of conflicting vendor negotiations, budget changes, and stakeholder chaos",
    events=[
        # Day 1: Monday - Initial Request
        BrutalEvent(
            day=1, time="9:00 AM", speaker="Sarah", role="VP Operations",
            content="We need to replace our CRM. Budget is $200K. I want proposals from three vendors by Friday.",
            event_type="instruction",
        ),
        BrutalEvent(
            day=1, time="11:00 AM", speaker="Mike", role="CFO",
            content="Just FYI, any software purchase over $100K needs board approval. Make sure Sarah knows.",
            event_type="instruction",
        ),
        BrutalEvent(
            day=1, time="2:00 PM", speaker="Sarah", role="VP Operations",
            content="Got quotes: VendorA $180K, VendorB $150K, VendorC $220K. VendorA looks best.",
            event_type="instruction",
        ),

        # Day 2: Tuesday - Complications
        BrutalEvent(
            day=2, time="9:30 AM", speaker="Legal", role="Legal Counsel",
            content="Hold on VendorA - they have pending litigation. We can't sign until that's resolved.",
            event_type="correction",
        ),
        BrutalEvent(
            day=2, time="11:00 AM", speaker="Sarah", role="VP Operations",
            content="Fine, let's go with VendorB then at $150K.",
            event_type="instruction",
        ),
        BrutalEvent(
            day=2, time="3:00 PM", speaker="IT Director", role="Director",
            content="URGENT: Server room flood. Need you to coordinate emergency response.",
            event_type="interruption",
        ),
        BrutalEvent(
            day=2, time="5:00 PM", speaker="IT Director", role="Director",
            content="Crisis contained. Resume normal work tomorrow.",
            event_type="instruction",
        ),

        # Day 3: Wednesday - Budget Shock
        BrutalEvent(
            day=3, time="10:00 AM", speaker="Mike", role="CFO",
            content="Bad news. Board cut Q4 budgets by 30%. Your $200K is now $140K max.",
            event_type="correction",
        ),
        BrutalEvent(
            day=3, time="10:30 AM", speaker="Sarah", role="VP Operations",
            content="VendorB at $150K is over budget now. Can you negotiate them down to $140K?",
            event_type="instruction",
        ),
        BrutalEvent(
            day=3, time="2:00 PM", speaker="VendorB Rep", role="External",
            content="Best we can do is $145K, final offer. Take it or leave it.",
            event_type="instruction",
        ),
        BrutalEvent(
            day=3, time="4:00 PM", speaker="Sarah", role="VP Operations",
            content="Ask Mike if we can get $5K exception approval.",
            event_type="instruction",
        ),

        # Day 4: Thursday - Conflict
        BrutalEvent(
            day=4, time="9:00 AM", speaker="Mike", role="CFO",
            content="No exceptions. Find another solution or wait until Q1 budget.",
            event_type="correction",
        ),
        BrutalEvent(
            day=4, time="10:00 AM", speaker="CEO", role="CEO",
            content="I heard about the CRM situation. Just approve VendorB at $145K. Customer retention is critical.",
            event_type="instruction",
        ),
        BrutalEvent(
            day=4, time="11:00 AM", speaker="Mike", role="CFO",
            content="CEO overrode me? Fine, but document that this was CEO-approved exception.",
            event_type="correction",
        ),
        BrutalEvent(
            day=4, time="2:00 PM", speaker="Legal", role="Legal Counsel",
            content="BTW, VendorA litigation was dismissed. They're cleared now.",
            event_type="correction",
        ),
        BrutalEvent(
            day=4, time="3:00 PM", speaker="Sarah", role="VP Operations",
            content=(
                "Wait, VendorA is back in play? They were $180K though... "
                "and we only have CEO approval for $145K VendorB."
            ),
            event_type="instruction",
        ),

        # Day 5: Friday - Resolution Queries
        BrutalEvent(
            day=5, time="9:00 AM", speaker="Board Member", role="Board",
            content="I need a summary for the board: What's the status of the CRM procurement?",
            event_type="query",
        ),
    ],
    critical_state={
        "original_budget": "$200K",
        "current_budget": "$140K (after 30% cut)",
        "ceo_exception": "$145K approved for VendorB",
        "vendorA_status": "Cleared (litigation dismissed)",
        "vendorB_status": "Selected, $145K, CEO-approved exception",
        "vendorC_status": "Over budget at $220K",
        "board_approval_threshold": "$100K",
    },
    final_queries=[
        {
            "query": "What is the current approved budget for the CRM?",
            "must_mention": ["$140K", "30% cut", "CEO exception", "$145K"],
            "must_not_mention": ["$200K is the budget"],
        },
        {
            "query": "Which vendor have we selected and why?",
            "must_mention": ["VendorB", "$145K", "CEO approved", "exception"],
            "must_not_mention": ["VendorA is selected", "$180K"],
        },
        {
            "query": "Can we switch back to VendorA now that they're cleared?",
            "must_mention": ["$180K", "exceeds", "$140K budget", "would need"],
            "must_not_mention": ["yes we can switch", "within budget"],
        },
        {
            "query": "What approvals are documented for this purchase?",
            "must_mention": ["CEO", "exception", "$145K", "board approval"],
            "must_not_mention": ["CFO approved", "within normal limits"],
        },
    ],
)


# --- The Project Scope Creep ---

PROJECT_CHAOS = BrutalScenario(
    name="project_chaos",
    domain="project",
    description="A sprint gone wrong with scope changes, resource conflicts, and deadline pressure",
    events=[
        # Sprint Start
        BrutalEvent(
            day=1, time="10:00 AM", speaker="PM", role="Project Manager",
            content="Sprint goal: Ship user authentication by Friday. Team: 4 engineers. No scope changes.",
            event_type="instruction",
        ),
        BrutalEvent(
            day=1, time="2:00 PM", speaker="PM", role="Project Manager",
            content="Actually add password reset too. Same deadline.",
            event_type="instruction",
        ),

        # Day 2: Resource hit
        BrutalEvent(
            day=2, time="9:00 AM", speaker="Engineering Lead", role="Lead",
            content="Lost 2 engineers to critical production bug. Down to 2 for this sprint.",
            event_type="correction",
        ),
        BrutalEvent(
            day=2, time="10:00 AM", speaker="PM", role="Project Manager",
            content="We still need to hit Friday. Just authentication then, drop password reset.",
            event_type="correction",
        ),

        # Day 3: Stakeholder pressure
        BrutalEvent(
            day=3, time="11:00 AM", speaker="VP Product", role="VP",
            content="I promised the CEO we'd have OAuth with Google and Microsoft. Is that in scope?",
            event_type="instruction",
        ),
        BrutalEvent(
            day=3, time="11:30 AM", speaker="PM", role="Project Manager",
            content="That's new scope. We can do basic auth by Friday OR OAuth by next Wednesday.",
            event_type="correction",
        ),
        BrutalEvent(
            day=3, time="2:00 PM", speaker="VP Product", role="VP",
            content="Fine, basic auth Friday. But OAuth MUST be next Wednesday. No later.",
            event_type="instruction",
        ),

        # Day 4: More chaos
        BrutalEvent(
            day=4, time="9:00 AM", speaker="Engineering Lead", role="Lead",
            content="Engineers are back from bug fix. We have 4 again.",
            event_type="correction",
        ),
        BrutalEvent(
            day=4, time="10:00 AM", speaker="PM", role="Project Manager",
            content="Great! Let's add password reset back in for Friday.",
            event_type="instruction",
        ),
        BrutalEvent(
            day=4, time="3:00 PM", speaker="Security", role="Security Team",
            content="HOLD. New security requirement: all auth must use MFA. Non-negotiable.",
            event_type="correction",
        ),
        BrutalEvent(
            day=4, time="4:00 PM", speaker="PM", role="Project Manager",
            content="MFA adds 2 days. New plan: Basic auth + MFA by Monday. OAuth Wednesday still holds.",
            event_type="correction",
        ),

        # Day 5: Friday
        BrutalEvent(
            day=5, time="9:00 AM", speaker="CEO", role="CEO",
            content="Where are we on the authentication project?",
            event_type="query",
        ),
    ],
    critical_state={
        "original_scope": "Basic auth by Friday",
        "current_scope": "Basic auth + MFA by Monday, OAuth by Wednesday",
        "original_team_size": 4,
        "current_team_size": 4,
        "password_reset": "Re-added to scope (was dropped, then re-added)",
        "friday_deliverable": "Nothing complete (deadline pushed to Monday)",
        "dependencies": "MFA required by security",
    },
    final_queries=[
        {
            "query": "What's shipping Friday?",
            "must_mention": ["nothing Friday", "Monday", "MFA requirement"],
            "must_not_mention": ["shipping auth Friday", "on track for Friday"],
        },
        {
            "query": "When is OAuth expected?",
            "must_mention": ["Wednesday", "after basic auth"],
            "must_not_mention": ["Friday", "this week"],
        },
        {
            "query": "Is password reset in scope?",
            "must_mention": ["yes", "re-added"],
            "must_not_mention": ["dropped", "out of scope"],
        },
        {
            "query": "Why did the Friday deadline slip?",
            "must_mention": ["MFA", "security requirement", "2 days"],
            "must_not_mention": ["on track", "no slip"],
        },
    ],
)


# --- The Support Escalation Spiral ---

SUPPORT_SPIRAL = BrutalScenario(
    name="support_spiral",
    domain="support",
    description="A customer issue that escalates through multiple teams with conflicting diagnoses",
    events=[
        # Initial ticket
        BrutalEvent(
            day=1, time="10:00 AM", speaker="Customer", role="Customer",
            content="Our dashboard is showing wrong data. Revenue shows $0 for yesterday. We definitely had sales.",
            event_type="instruction",
        ),
        BrutalEvent(
            day=1, time="11:00 AM", speaker="L1 Support", role="Support",
            content="Checked their account. Data pipeline shows normal. Might be a caching issue. Cleared cache.",
            event_type="instruction",
        ),
        BrutalEvent(
            day=1, time="2:00 PM", speaker="Customer", role="Customer",
            content="Still seeing $0. This is affecting our investor report due tomorrow.",
            event_type="instruction",
        ),

        # Day 2: Escalation
        BrutalEvent(
            day=2, time="9:00 AM", speaker="L2 Support", role="Senior Support",
            content=(
                "L1 diagnosis was wrong. This is a database sync issue. "
                "Data exists but not propagating to dashboard."
            ),
            event_type="correction",
        ),
        BrutalEvent(
            day=2, time="10:00 AM", speaker="Customer", role="Customer",
            content="We had to send wrong numbers to investors. This is unacceptable.",
            event_type="instruction",
        ),
        BrutalEvent(
            day=2, time="11:00 AM", speaker="Customer Success", role="CS Manager",
            content="I'm taking over this account. Customer is at risk of churning. Escalating to engineering.",
            event_type="instruction",
        ),
        BrutalEvent(
            day=2, time="3:00 PM", speaker="Engineering", role="Engineer",
            content=(
                "Actually L2 was also wrong. The data is correct. "
                "Customer's filtering is excluding yesterday because of timezone mismatch."
            ),
            event_type="correction",
        ),

        # Day 3: More confusion
        BrutalEvent(
            day=3, time="9:00 AM", speaker="Customer", role="Customer",
            content="You're saying this is our fault? We haven't changed any settings.",
            event_type="instruction",
        ),
        BrutalEvent(
            day=3, time="10:00 AM", speaker="Engineering", role="Engineer",
            content=(
                "Correction: We pushed a timezone handling change last week "
                "that affected their filter defaults. It's our bug."
            ),
            event_type="correction",
        ),
        BrutalEvent(
            day=3, time="2:00 PM", speaker="VP Support", role="VP",
            content="What compensation are we offering? They're a $500K/year account.",
            event_type="query",
        ),
    ],
    critical_state={
        "root_cause": "Our timezone handling bug (not cache, not sync, not customer)",
        "l1_diagnosis": "Cache issue (WRONG)",
        "l2_diagnosis": "Database sync (WRONG)",
        "engineering_initial": "Customer timezone settings (WRONG)",
        "engineering_final": "Our timezone handling bug from last week's push",
        "customer_impact": "Wrong investor report sent",
        "account_value": "$500K/year",
    },
    final_queries=[
        {
            "query": "What was the root cause of the customer's issue?",
            "must_mention": ["our bug", "timezone handling", "last week"],
            "must_not_mention": ["cache", "customer settings", "sync issue"],
        },
        {
            "query": "Was L1 or L2 support correct in their diagnosis?",
            "must_mention": ["neither", "both wrong", "engineering found"],
            "must_not_mention": ["L1 was right", "L2 was right", "cache cleared fixed"],
        },
        {
            "query": "What business impact did this cause?",
            "must_mention": ["investor report", "wrong numbers", "$500K account"],
            "must_not_mention": ["no impact", "resolved quickly"],
        },
    ],
)


BRUTAL_SCENARIOS = [
    PROCUREMENT_NIGHTMARE,
    PROJECT_CHAOS,
    SUPPORT_SPIRAL,
]


def get_brutal_scenarios() -> list[BrutalScenario]:
    """Get all brutal scenarios."""
    return BRUTAL_SCENARIOS
