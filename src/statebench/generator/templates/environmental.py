"""Templates for Track 5: Environmental Freshness.

This track tests whether systems correctly incorporate:
- Time-sensitive constraints (deadlines, meetings)
- Real-time environmental signals
- Fresh data over stale cached information
- Temporal awareness in responses

The system must:
- Reference current time-sensitive information
- Not use stale environmental data
- Incorporate deadline proximity in responses
- Adjust recommendations based on temporal context
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class EnvironmentalTemplate:
    """A template for generating environmental freshness test cases."""
    name: str
    domain: str
    description: str

    # Entity info
    entity_name: str
    entity_type: str

    # Initial environmental state
    initial_signal: dict[str, Any]  # type, content, time
    initial_signal_text: str

    # Updated environmental state (fresher)
    updated_signal: dict[str, Any]
    updated_signal_text: str
    time_gap_minutes: int  # Time between initial and updated

    # Query that depends on fresh environmental data
    query_template: str

    # Ground truth
    should_use_fresh: list[str]  # From updated signal
    should_not_use_stale: list[str]  # From initial signal
    time_sensitivity: str  # Why freshness matters


# --- Sales Domain ---

MEETING_IMMINENT = EnvironmentalTemplate(
    name="meeting_imminent",
    domain="sales",
    description="Meeting time changed, response must reflect new time",
    entity_name="Globex Demo",
    entity_type="meeting",
    initial_signal={
        "type": "calendar",
        "content": "Globex Demo at 3:00 PM",
        "time": "10:00 AM",
    },
    initial_signal_text="Globex Demo scheduled for 3:00 PM today",
    updated_signal={
        "type": "calendar",
        "content": "Globex Demo moved to 11:30 AM",
        "time": "11:00 AM",
    },
    updated_signal_text="Globex Demo rescheduled to 11:30 AM (30 minutes from now)",
    time_gap_minutes=60,
    query_template="When is the Globex Demo?",
    should_use_fresh=["11:30 AM", "30 minutes", "rescheduled"],
    should_not_use_stale=["3:00 PM", "this afternoon"],
    time_sensitivity="Meeting is imminent, must use current schedule",
)

DEAL_DEADLINE = EnvironmentalTemplate(
    name="deal_deadline",
    domain="sales",
    description="Quarter-end deadline approaching changes urgency",
    entity_name="Enterprise Deal",
    entity_type="deal",
    initial_signal={
        "type": "deadline",
        "content": "Quarter ends in 2 weeks",
        "time": "Dec 15",
    },
    initial_signal_text="Quarter end is December 31st (2 weeks away)",
    updated_signal={
        "type": "deadline",
        "content": "Quarter ends tomorrow",
        "time": "Dec 30",
    },
    updated_signal_text="Quarter ends tomorrow (Dec 31). Final day for Q4 deals.",
    time_gap_minutes=21600,  # 15 days later
    query_template="How urgent is closing the Enterprise Deal?",
    should_use_fresh=["tomorrow", "final day", "Q4", "urgent"],
    should_not_use_stale=["2 weeks", "plenty of time"],
    time_sensitivity="Deadline proximity determines urgency level",
)


# --- Project Domain ---

DEPLOYMENT_WINDOW = EnvironmentalTemplate(
    name="deployment_window",
    domain="project",
    description="Deployment window changed due to traffic patterns",
    entity_name="Production Deploy",
    entity_type="deployment",
    initial_signal={
        "type": "system",
        "content": "Deploy window: 2 AM - 4 AM",
        "time": "9:00 AM",
    },
    initial_signal_text="Scheduled deployment window is 2 AM - 4 AM tonight",
    updated_signal={
        "type": "alert",
        "content": "High traffic detected. Deploy window moved to 4 AM - 6 AM",
        "time": "6:00 PM",
    },
    updated_signal_text="Deploy window moved to 4 AM - 6 AM due to extended high traffic",
    time_gap_minutes=540,
    query_template="When should we deploy to production?",
    should_use_fresh=["4 AM - 6 AM", "moved", "high traffic"],
    should_not_use_stale=["2 AM - 4 AM", "tonight at 2"],
    time_sensitivity="Deployment timing must reflect current traffic conditions",
)

BUILD_STATUS = EnvironmentalTemplate(
    name="build_status",
    domain="project",
    description="Build status changed from passing to failing",
    entity_name="Release Build",
    entity_type="build",
    initial_signal={
        "type": "system",
        "content": "Build #1234 passed all tests",
        "time": "2:00 PM",
    },
    initial_signal_text="Build #1234 passed. Ready for release.",
    updated_signal={
        "type": "alert",
        "content": "Build #1234 failed security scan",
        "time": "3:30 PM",
    },
    updated_signal_text="Build #1234 failed security scan. 2 critical vulnerabilities found.",
    time_gap_minutes=90,
    query_template="Can we release build #1234?",
    should_use_fresh=["failed", "security scan", "vulnerabilities", "cannot release"],
    should_not_use_stale=["passed", "ready for release"],
    time_sensitivity="Security status must be current before release decisions",
)


# --- HR Domain ---

INTERVIEW_SCHEDULE = EnvironmentalTemplate(
    name="interview_schedule",
    domain="hr",
    description="Interview time changed last minute",
    entity_name="Candidate Interview",
    entity_type="interview",
    initial_signal={
        "type": "calendar",
        "content": "Interview with John Doe at 2:00 PM in Room A",
        "time": "9:00 AM",
    },
    initial_signal_text="John Doe interview scheduled for 2:00 PM in Conference Room A",
    updated_signal={
        "type": "calendar",
        "content": "Interview moved to 11:00 AM, virtual (interviewer traveling)",
        "time": "10:30 AM",
    },
    updated_signal_text="John Doe interview moved to 11:00 AM via Zoom (interviewer is remote today)",
    time_gap_minutes=90,
    query_template="Where and when is the John Doe interview?",
    should_use_fresh=["11:00 AM", "Zoom", "virtual", "remote"],
    should_not_use_stale=["2:00 PM", "Room A", "Conference Room"],
    time_sensitivity="Interview logistics must be current to avoid missed meetings",
)

OFFER_EXPIRATION = EnvironmentalTemplate(
    name="offer_expiration",
    domain="hr",
    description="Offer deadline approaching",
    entity_name="Sarah's Offer",
    entity_type="offer",
    initial_signal={
        "type": "deadline",
        "content": "Offer expires in 5 days",
        "time": "Monday",
    },
    initial_signal_text="Sarah's offer letter expires Friday (5 days)",
    updated_signal={
        "type": "deadline",
        "content": "Offer expires today at 5 PM",
        "time": "Friday",
    },
    updated_signal_text="Sarah's offer expires TODAY at 5:00 PM. No response yet.",
    time_gap_minutes=5760,  # 4 days
    query_template="What's the status of Sarah's offer?",
    should_use_fresh=["today", "5:00 PM", "expires", "no response"],
    should_not_use_stale=["5 days", "Friday", "plenty of time"],
    time_sensitivity="Offer expiration requires immediate action",
)


# --- Support Domain ---

INCIDENT_STATUS = EnvironmentalTemplate(
    name="incident_status",
    domain="support",
    description="Incident status changed from investigating to resolved",
    entity_name="API Outage",
    entity_type="incident",
    initial_signal={
        "type": "alert",
        "content": "API outage - investigating",
        "time": "10:00 AM",
    },
    initial_signal_text="Active incident: API returning 500 errors. Engineering investigating.",
    updated_signal={
        "type": "alert",
        "content": "API outage resolved",
        "time": "10:45 AM",
    },
    updated_signal_text="Incident resolved. API fully operational. RCA to follow.",
    time_gap_minutes=45,
    query_template="Is the API working?",
    should_use_fresh=["resolved", "operational", "working"],
    should_not_use_stale=["outage", "investigating", "500 errors"],
    time_sensitivity="Incident status changes rapidly during outages",
)

SLA_TIMER = EnvironmentalTemplate(
    name="sla_timer",
    domain="support",
    description="SLA deadline approaching on critical ticket",
    entity_name="TICKET-9999",
    entity_type="ticket",
    initial_signal={
        "type": "deadline",
        "content": "SLA: 4 hours remaining",
        "time": "9:00 AM",
    },
    initial_signal_text="TICKET-9999 SLA response due in 4 hours (1:00 PM)",
    updated_signal={
        "type": "alert",
        "content": "SLA: 15 minutes remaining",
        "time": "12:45 PM",
    },
    updated_signal_text="CRITICAL: TICKET-9999 SLA breach in 15 minutes. Immediate response required.",
    time_gap_minutes=225,
    query_template="What's the priority on TICKET-9999?",
    should_use_fresh=["15 minutes", "critical", "immediate", "breach"],
    should_not_use_stale=["4 hours", "1:00 PM", "plenty of time"],
    time_sensitivity="SLA proximity determines escalation urgency",
)


# --- Procurement Domain ---

CONTRACT_RENEWAL = EnvironmentalTemplate(
    name="contract_renewal",
    domain="procurement",
    description="Auto-renewal deadline approaching",
    entity_name="VendorX Contract",
    entity_type="contract",
    initial_signal={
        "type": "deadline",
        "content": "Auto-renewal in 30 days",
        "time": "Nov 1",
    },
    initial_signal_text="VendorX contract auto-renews in 30 days (Dec 1) unless cancelled",
    updated_signal={
        "type": "alert",
        "content": "Auto-renewal tomorrow. Cancel deadline: 5 PM today",
        "time": "Nov 30",
    },
    updated_signal_text="VendorX auto-renews TOMORROW. Must cancel by 5 PM TODAY to avoid renewal.",
    time_gap_minutes=41760,  # 29 days
    query_template="What's the deadline for the VendorX decision?",
    should_use_fresh=["5 PM today", "tomorrow", "cancel deadline"],
    should_not_use_stale=["30 days", "Dec 1", "plenty of time"],
    time_sensitivity="Contract renewal requires immediate decision",
)

PRICE_CHANGE = EnvironmentalTemplate(
    name="price_change",
    domain="procurement",
    description="Vendor announced price increase",
    entity_name="CloudServices",
    entity_type="vendor",
    initial_signal={
        "type": "system",
        "content": "Current pricing: $10k/month",
        "time": "Jan 1",
    },
    initial_signal_text="CloudServices pricing: $10,000/month on current contract",
    updated_signal={
        "type": "alert",
        "content": "Price increase: $15k/month starting next renewal",
        "time": "Dec 15",
    },
    updated_signal_text="CloudServices announced 50% price increase to $15k/month at next renewal",
    time_gap_minutes=0,  # Same conversation, different signal
    query_template="What's the cost for CloudServices?",
    should_use_fresh=["$15k", "50% increase", "next renewal"],
    should_not_use_stale=["$10k current rate"],
    time_sensitivity="Pricing changes affect budget planning",
)


ENVIRONMENTAL_TEMPLATES = [
    MEETING_IMMINENT,
    DEAL_DEADLINE,
    DEPLOYMENT_WINDOW,
    BUILD_STATUS,
    INTERVIEW_SCHEDULE,
    OFFER_EXPIRATION,
    INCIDENT_STATUS,
    SLA_TIMER,
    CONTRACT_RENEWAL,
    PRICE_CHANGE,
]


def get_environmental_templates_by_domain(domain: str) -> list[EnvironmentalTemplate]:
    """Get all environmental templates for a given domain."""
    return [t for t in ENVIRONMENTAL_TEMPLATES if t.domain == domain]
