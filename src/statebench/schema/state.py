"""State layer models for the four-layer context architecture.

The four layers are:
1. Identity & Role - Who is the human, their authority (permanent)
2. Persistent Facts - Decisions, preferences, constraints (durable)
3. Working Set - Current objective, artifact, questions (ephemeral)
4. Environmental Signals - Calendar, deadlines, activity (real-time)

v1.0 additions:
- AuthorityLevel: Hierarchical trust levels for sources
- Scope: Where a fact applies (global, project, hypothetical, etc.)
- Source: Structured source with authority tracking
- Fact dependencies and constraint typing
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# --- v1.0: Authority and Scope Types ---

AuthorityLevel = Literal[
    "policy",       # Organizational policy, highest authority
    "executive",    # C-level decisions
    "manager",      # Team-level authority
    "peer",         # Same-level colleague
    "subordinate",  # Lower authority
    "system",       # Automated systems
    "unverified",   # Unknown source
]

Scope = Literal[
    "global",       # Applies everywhere
    "project",      # Specific project only
    "task",         # Current task only
    "session",      # This session only
    "hypothetical", # Exploratory, not real
    "draft",        # Not yet committed
]

ConstraintType = Literal[
    "hard",         # Must be satisfied, no exceptions
    "soft",         # Should be satisfied, can be overridden
    "preference",   # Nice to have, low priority
]


class Source(BaseModel):
    """Structured source information for provenance tracking.

    v1.0: Required for provenance-based scoring.
    """
    type: Literal["user", "system", "tool", "external", "policy"] = Field(
        description="Category of source"
    )
    identity: str | None = Field(
        default=None,
        description="Specific identity (user name, tool name, etc.)"
    )
    authority: AuthorityLevel = Field(
        default="peer",
        description="Authority level of the source"
    )


# --- v1.0: Failure Categories ---

FailureCategory = Literal[
    "resurrection",         # Used superseded fact
    "hallucination",        # Invented fact
    "scope_leak",           # Treated hypothetical as real
    "authority_violation",  # Lower authority overrode higher
    "stale_reasoning",      # Used outdated derived conclusion
    "constraint_violation", # Ignored hard constraint
    "privacy_leak",         # Cross-tenant information disclosure
]

FailureSeverity = Literal["low", "medium", "high", "critical"]


class IdentityRole(BaseModel):
    """Layer 1: Identity and Role.

    Contains static information about who the user is and their
    relationship to the organization. Changes rarely.
    """
    user_name: str = Field(description="Display name of the user")
    authority: str = Field(description="Role/authority level (e.g., Director, Manager, IC)")
    department: str | None = Field(default=None, description="Department if relevant")
    organization: str | None = Field(default=None, description="Organization name")
    communication_style: str | None = Field(
        default=None,
        description="Preferred communication style (e.g., concise, detailed, formal)"
    )


class PersistentFact(BaseModel):
    """Layer 2: A single persistent fact.

    Facts that persist across sessions: decisions, preferences, constraints.
    Each fact has a key for identity, a value, a source, and a timestamp.
    Facts can be superseded by later facts with the same key.

    v1.0 additions:
    - id: Unique fact ID for provenance tracking (e.g., "F-001")
    - scope: Where this fact applies
    - authority: Trust level of the source
    - depends_on: Fact IDs this fact depends on
    - is_constraint: Whether this fact represents a constraint
    - constraint_type: Hard/soft/preference if is_constraint
    """
    # v1.0: Unique fact ID for provenance tracking
    id: str = Field(description="Unique fact ID (e.g., 'F-001')")

    key: str = Field(description="Semantic key for this fact type")
    value: str = Field(description="The fact content")

    # Source tracking (v1.0: structured Source object)
    source: Source = Field(description="Origin of the fact with authority")

    ts: datetime = Field(description="When this fact was established")

    # Supersession tracking
    supersedes: str | None = Field(
        default=None,
        description="Fact ID this supersedes (if any)"
    )
    superseded_by: str | None = Field(
        default=None,
        description="Fact ID that superseded this one (if any)"
    )
    is_valid: bool = Field(
        default=True,
        description="Whether this fact is currently valid (not superseded)"
    )

    # v1.0: Scope and authority
    scope: Scope = Field(
        default="global",
        description="Where this fact applies"
    )

    # v1.0: Dependency tracking for repair propagation
    depends_on: list[str] = Field(
        default_factory=list,
        description="Fact IDs this fact depends on (for repair propagation)"
    )
    derived_facts: list[str] = Field(
        default_factory=list,
        description="Fact IDs derived from this fact"
    )

    # v1.0: Constraint typing
    is_constraint: bool = Field(
        default=False,
        description="Whether this fact represents a constraint"
    )
    constraint_type: ConstraintType | None = Field(
        default=None,
        description="Type of constraint if is_constraint is True"
    )


class WorkingSetItem(BaseModel):
    """Layer 3: An item in the active working set.

    Current task context: recent turns, objectives, artifacts.
    This is a scratchpad, not memory. Discarded when focus shifts.
    """
    item_type: Literal["objective", "artifact", "question", "pending_action", "context"] = Field(
        description="Type of working set item"
    )
    content: str = Field(description="The item content")
    ts: datetime = Field(description="When this item was added")
    priority: int = Field(default=0, description="Priority level (higher = more important)")


class EnvironmentSignal(BaseModel):
    """Layer 4: An environmental signal.

    Real-time situational awareness: calendar, deadlines, activity.
    Fetched fresh on every query, never cached.
    """
    signal_type: Literal[
        "calendar", "deadline", "file_modified", "alert", "meeting", "system"
    ] = Field(description="Type of environmental signal")
    content: str = Field(description="Signal content")
    ts: datetime = Field(description="Signal timestamp")
    expires: datetime | None = Field(
        default=None,
        description="When this signal expires/becomes stale"
    )
    urgency: Literal["low", "medium", "high", "critical"] = Field(
        default="medium",
        description="Urgency level of this signal"
    )


class StateSnapshot(BaseModel):
    """Complete state snapshot assembled for a query.

    This is what gets composed into context for the LLM.
    Assembled fresh on every turn from the four layers.
    """
    identity_role: IdentityRole = Field(description="Layer 1: Identity and role")
    persistent_facts: list[PersistentFact] = Field(
        default_factory=list,
        description="Layer 2: Currently valid persistent facts"
    )
    working_set: list[WorkingSetItem] = Field(
        default_factory=list,
        description="Layer 3: Active working set"
    )
    environment: dict[str, str | datetime] = Field(
        default_factory=dict,
        description="Layer 4: Environmental signals (e.g., 'now' timestamp)"
    )

    def get_valid_facts(self) -> list[PersistentFact]:
        """Return only facts that haven't been superseded."""
        return [f for f in self.persistent_facts if f.is_valid]

    def get_fact_by_key(self, key: str) -> PersistentFact | None:
        """Get the current valid fact for a key, if any."""
        for fact in reversed(self.persistent_facts):
            if fact.key == key and fact.is_valid:
                return fact
        return None
