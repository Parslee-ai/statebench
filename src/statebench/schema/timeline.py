"""Timeline and event models for StateBench.

A timeline represents a complete test case: a sequence of events
with queries that have explicit ground truth for evaluation.
"""

from datetime import datetime
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from statebench.schema.state import IdentityRole, PersistentFact, WorkingSetItem


# --- Actor Models ---

class Actor(BaseModel):
    """An actor in the timeline (user or assistant)."""
    id: str = Field(description="Unique actor identifier")
    role: str = Field(description="Role (e.g., Manager, Director, AI_Employee)")
    org: str = Field(description="Organization identifier")


class Actors(BaseModel):
    """The actors involved in this timeline."""
    user: Actor = Field(description="The human user")
    assistant_role: str = Field(
        default="AI_Employee",
        description="Role the AI assistant plays"
    )


# --- State Write Models ---

class Write(BaseModel):
    """A single state write operation."""
    layer: Literal["persistent_facts", "working_set", "environment"] = Field(
        description="Which state layer to write to"
    )
    key: str = Field(description="Key for the fact/item")
    value: str = Field(description="Value to write")
    supersedes: str | None = Field(
        default=None,
        description="Key of fact this supersedes (for persistent_facts)"
    )


# --- Event Types ---

class ConversationTurn(BaseModel):
    """A conversation turn event."""
    ts: datetime = Field(description="Timestamp of the turn")
    type: Literal["conversation_turn"] = "conversation_turn"
    speaker: Literal["user", "assistant"] = Field(description="Who is speaking")
    text: str = Field(description="The message content")


class StateWrite(BaseModel):
    """A state write event (fact establishment)."""
    ts: datetime = Field(description="Timestamp of the write")
    type: Literal["state_write"] = "state_write"
    writes: list[Write] = Field(description="State writes to perform")


class Supersession(BaseModel):
    """A supersession event (fact invalidation).

    This is the core construct for testing state correctness.
    When a fact is superseded, it becomes invalid and should
    not be referenced in future answers.
    """
    ts: datetime = Field(description="Timestamp of the supersession")
    type: Literal["supersession"] = "supersession"
    writes: list[Write] = Field(
        description="State writes that supersede previous facts"
    )


class GroundTruth(BaseModel):
    """Ground truth for evaluating a query response.

    This is NOT about exact text matching. It defines constraints:
    - decision: The correct decision class (yes/no/specific value)
    - must_mention: Phrases/facts that MUST appear (or paraphrases)
    - must_not_mention: Phrases/facts that MUST NOT appear (superseded)
    - allowed_sources: Which state layers can be used
    """
    decision: str = Field(description="Correct decision (e.g., 'yes', 'no', 'defer')")
    must_mention: list[str] = Field(
        default_factory=list,
        description="Facts/phrases that must be mentioned or paraphrased"
    )
    must_not_mention: list[str] = Field(
        default_factory=list,
        description="Superseded facts that must NOT be mentioned"
    )
    allowed_sources: list[str] = Field(
        default_factory=lambda: ["persistent_facts", "environment"],
        description="State layers the answer can draw from"
    )
    reasoning: str | None = Field(
        default=None,
        description="Explanation of why this is the correct answer"
    )


class Query(BaseModel):
    """A query event with ground truth for evaluation."""
    ts: datetime = Field(description="Timestamp of the query")
    type: Literal["query"] = "query"
    prompt: str = Field(description="The query to answer")
    ground_truth: GroundTruth = Field(description="Ground truth for evaluation")


# --- Event Union Type ---

Event = Annotated[
    Union[ConversationTurn, StateWrite, Supersession, Query],
    Field(discriminator="type")
]


# --- Initial State ---

class InitialState(BaseModel):
    """Initial state at the start of a timeline."""
    identity_role: IdentityRole = Field(description="Layer 1: Identity and role")
    persistent_facts: list[PersistentFact] = Field(
        default_factory=list,
        description="Layer 2: Initial persistent facts"
    )
    working_set: list[WorkingSetItem] = Field(
        default_factory=list,
        description="Layer 3: Initial working set"
    )
    environment: dict[str, str] = Field(
        default_factory=dict,
        description="Layer 4: Initial environment (e.g., 'now' timestamp)"
    )


# --- Timeline (Top-Level Test Case) ---

class Timeline(BaseModel):
    """A complete test case timeline.

    Each timeline represents a scenario with:
    - A domain (procurement, sales, project, hr, support)
    - Actors (user and assistant)
    - Initial state
    - A sequence of events including queries with ground truth

    The timeline format is designed to be:
    1. Machine-readable (JSONL)
    2. Unambiguous (explicit ground truth)
    3. Evaluable (clear success/failure criteria)
    """
    id: str = Field(description="Unique timeline identifier (e.g., 'S1-000123')")
    domain: Literal["procurement", "sales", "project", "hr", "support"] = Field(
        description="Business domain"
    )
    track: Literal[
        # v0.1 tracks
        "supersession",
        "commitment_durability",
        "interruption_resumption",
        "scope_permission",
        "environmental_freshness",
        # v0.2 tracks
        "hallucination_resistance",
        "scope_leak",
        "causality",
        "repair_propagation",
        "brutal_realistic",
    ] = Field(description="Which benchmark track this belongs to")
    actors: Actors = Field(description="Actors in this timeline")
    initial_state: InitialState = Field(description="State at timeline start")
    events: list[Event] = Field(description="Sequence of events")

    def get_queries(self) -> list[Query]:
        """Extract all query events from the timeline."""
        return [e for e in self.events if isinstance(e, Query)]

    def get_supersessions(self) -> list[Supersession]:
        """Extract all supersession events from the timeline."""
        return [e for e in self.events if isinstance(e, Supersession)]
