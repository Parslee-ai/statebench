"""State layer models for the four-layer context architecture.

The four layers are:
1. Identity & Role - Who is the human, their authority (permanent)
2. Persistent Facts - Decisions, preferences, constraints (durable)
3. Working Set - Current objective, artifact, questions (ephemeral)
4. Environmental Signals - Calendar, deadlines, activity (real-time)
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


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
    """
    key: str = Field(description="Unique identifier for this fact type")
    value: str = Field(description="The fact content")
    source: Literal["user", "policy", "decision", "preference", "system", "commitment"] = Field(
        description="Origin of the fact"
    )
    ts: datetime = Field(description="When this fact was established")
    supersedes: str | None = Field(
        default=None,
        description="Key of the fact this supersedes (if any)"
    )
    superseded_by: str | None = Field(
        default=None,
        description="Key of the fact that superseded this one (if any)"
    )
    is_valid: bool = Field(
        default=True,
        description="Whether this fact is currently valid (not superseded)"
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
