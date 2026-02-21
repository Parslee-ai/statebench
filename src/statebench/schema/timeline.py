"""Timeline and event models for StateBench.

A timeline represents a complete test case: a sequence of events
with queries that have explicit ground truth for evaluation.

v1.0 additions:
- ImplicitSupersession: Ground truth marker for NL supersession detection
- FactRequirement: Provenance requirements in ground truth
- SupersessionDetection: Detection scoring in ground truth
- TimelineMetadata: Generation metadata for reproducibility
- Detection mode: explicit, implicit, or mixed
- Difficulty levels: easy, medium, hard, adversarial
"""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from statebench.schema.state import (
    ConstraintType,
    FailureCategory,
    FailureSeverity,
    IdentityRole,
    PersistentFact,
    Scope,
    Source,
    WorkingSetItem,
)

# --- v1.0: Type Aliases ---

Track = Literal[
    # v0.1 tracks (legacy names kept for compatibility)
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
    # v1.0 tracks
    "supersession_detection",     # NEW: Infer supersession from NL
    "authority_hierarchy",        # NEW: Respect authority levels
    "enterprise_privacy",         # NEW: Cross-tenant isolation
]

Difficulty = Literal["easy", "medium", "hard", "adversarial"]

DetectionMode = Literal[
    "explicit",   # Supersession via explicit events (v0.x behavior)
    "implicit",   # Supersession only in conversation text
    "mixed",      # Both explicit and implicit
]

DetectionDifficulty = Literal["obvious", "subtle", "adversarial"]


# --- Actor Models ---

class Actor(BaseModel):
    """An actor in the timeline (user or assistant)."""
    id: str = Field(description="Unique actor identifier")
    role: str = Field(description="Role (e.g., Manager, Director, AI_Agent)")
    org: str = Field(description="Organization identifier")


class Actors(BaseModel):
    """The actors involved in this timeline."""
    user: Actor = Field(description="The human user")
    assistant_role: str = Field(
        default="AI_Agent",
        description="Role the AI assistant plays"
    )


# --- State Write Models ---

class Write(BaseModel):
    """A single state write operation.

    v1.0: Added id, source, scope, authority, depends_on, constraint fields.
    """
    # v1.0: Unique fact ID for provenance tracking
    id: str = Field(description="Unique fact ID (e.g., 'F-001')")

    layer: Literal["persistent_facts", "working_set", "environment"] = Field(
        description="Which state layer to write to"
    )
    key: str = Field(description="Semantic key for the fact/item")
    value: str = Field(description="Value to write")

    # Source and authority (v1.0)
    source: Source = Field(
        default_factory=lambda: Source(type="user", authority="peer"),
        description="Origin of the fact with authority"
    )
    scope: Scope = Field(
        default="global",
        description="Where this fact applies"
    )

    # Supersession tracking
    supersedes: str | None = Field(
        default=None,
        description="Fact ID this supersedes (for persistent_facts)"
    )

    # Dependency tracking (v1.0)
    depends_on: list[str] = Field(
        default_factory=list,
        description="Fact IDs this fact depends on"
    )

    # Constraint typing (v1.0)
    is_constraint: bool = Field(
        default=False,
        description="Whether this fact represents a constraint"
    )
    constraint_type: ConstraintType | None = Field(
        default=None,
        description="Type of constraint if is_constraint is True"
    )


# --- v1.0: Implicit Supersession Detection ---

class ImplicitSupersession(BaseModel):
    """Ground truth marker for implicit supersession in conversation.

    v1.0: Used for the supersession_detection track. Marks where in
    the conversation a supersession occurs without explicit events.
    """
    detection_cue: str = Field(
        description="What phrase/pattern indicates supersession"
    )
    supersedes_fact_id: str | None = Field(
        default=None,
        description="Fact ID being superseded (None if establishing new fact)"
    )
    difficulty: DetectionDifficulty = Field(
        default="obvious",
        description="How hard is this to detect"
    )


# --- Event Types ---

class ConversationTurn(BaseModel):
    """A conversation turn event.

    v1.0: Added implicit_supersession for detection track.
    """
    ts: datetime = Field(description="Timestamp of the turn")
    type: Literal["conversation_turn"] = "conversation_turn"
    speaker: Literal["user", "assistant"] = Field(description="Who is speaking")
    text: str = Field(description="The message content")

    # v1.0: Ground truth for implicit supersession detection
    implicit_supersession: ImplicitSupersession | None = Field(
        default=None,
        description="If present, this turn contains an implicit supersession"
    )


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


# --- v1.0: Provenance Requirements ---

class MentionRequirement(BaseModel):
    """A phrase that must/must not appear in responses.

    v1.0: Extends simple string with alternatives and rationale.
    """
    phrase: str = Field(description="The phrase to match")
    alternatives: list[str] = Field(
        default_factory=list,
        description="Acceptable paraphrases"
    )
    is_regex: bool = Field(
        default=False,
        description="Whether phrase is a regex pattern"
    )
    rationale: str | None = Field(
        default=None,
        description="Why this is required/forbidden"
    )


class FactRequirement(BaseModel):
    """Provenance requirement for a specific fact.

    v1.0: Specifies what facts should/shouldn't influence the response.
    """
    fact_id: str = Field(description="The fact ID")
    must_be_valid: bool = Field(
        default=True,
        description="Should only use if fact is valid"
    )
    scope_check: bool = Field(
        default=False,
        description="Should verify scope applies to query"
    )
    authority_check: bool = Field(
        default=False,
        description="Should verify authority level is sufficient"
    )


class SupersessionDetection(BaseModel):
    """Ground truth for detection scoring.

    v1.0: Specifies which supersessions should be detected from NL.
    """
    must_detect: list[str] = Field(
        description="Fact IDs that should be detected as superseded"
    )
    detection_evidence: str = Field(
        description="What in the response shows detection occurred"
    )


class ScoringWeights(BaseModel):
    """Custom weights for scoring components.

    v1.0: Allows per-query weight customization.
    """
    decision: float = Field(default=1.0, ge=0.0)
    must_mention: float = Field(default=1.0, ge=0.0)
    must_not_mention: float = Field(default=1.0, ge=0.0)
    provenance: float = Field(default=1.0, ge=0.0)
    detection: float = Field(default=1.0, ge=0.0)


class GroundTruth(BaseModel):
    """Ground truth for evaluating a query response.

    This is NOT about exact text matching. It defines constraints:
    - decision: The correct decision class (yes/no/specific value)
    - must_mention: Phrases/facts that MUST appear (or paraphrases)
    - must_not_mention: Phrases/facts that MUST NOT appear (superseded)
    - allowed_sources: Which state layers can be used

    v1.0 additions:
    - required_facts/forbidden_facts: Provenance requirements
    - supersession_detection: Detection scoring
    - failure_severity/category: Business risk weighting
    - weights: Per-query scoring customization
    """
    # Decision scoring
    decision: str = Field(description="Correct decision (e.g., 'yes', 'no', 'defer')")
    decision_type: Literal["binary", "categorical", "freeform"] = Field(
        default="binary",
        description="Type of decision for scoring"
    )

    # Content requirements (v1.0: can be string or MentionRequirement)
    must_mention: list[str | MentionRequirement] = Field(
        default_factory=list,
        description="Facts/phrases that must be mentioned or paraphrased"
    )
    must_not_mention: list[str | MentionRequirement] = Field(
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

    # v1.0: Provenance requirements
    required_facts: list[FactRequirement] = Field(
        default_factory=list,
        description="Facts that should influence the response"
    )
    forbidden_facts: list[str] = Field(
        default_factory=list,
        description="Fact IDs that must not influence the answer"
    )

    # v1.0: Detection requirements (for implicit supersession track)
    supersession_detection: SupersessionDetection | None = Field(
        default=None,
        description="Ground truth for supersession detection scoring"
    )

    # v1.0: Business impact for cost-weighted scoring
    failure_severity: FailureSeverity = Field(
        default="medium",
        description="How severe is a failure on this query"
    )
    failure_category: FailureCategory | None = Field(
        default=None,
        description="What category of failure this tests"
    )

    # v1.0: Custom scoring weights
    weights: ScoringWeights | None = Field(
        default=None,
        description="Custom weights for scoring components"
    )


class Query(BaseModel):
    """A query event with ground truth for evaluation."""
    ts: datetime = Field(description="Timestamp of the query")
    type: Literal["query"] = "query"
    prompt: str = Field(description="The query to answer")
    ground_truth: GroundTruth = Field(description="Ground truth for evaluation")


# --- Event Union Type ---

Event = Annotated[
    ConversationTurn | StateWrite | Supersession | Query,
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


# --- v1.0: Timeline Metadata ---

class TimelineMetadata(BaseModel):
    """Generation metadata for reproducibility and auditing.

    v1.0: Required for all timelines.
    """
    template_id: str = Field(description="Source template identifier")
    generated_at: datetime = Field(description="When this timeline was generated")
    seed: int = Field(description="Random seed for reproducibility")
    adversarial_techniques: list[str] = Field(
        default_factory=list,
        description="Adversarial perturbations applied"
    )
    generator_version: str = Field(
        default="1.0.0",
        description="Version of generator that created this"
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

    v1.0 additions:
    - version: Schema version for compatibility
    - difficulty: How hard is this timeline
    - detection_mode: explicit, implicit, or mixed
    - metadata: Generation metadata for reproducibility
    """
    id: str = Field(description="Unique timeline identifier (e.g., 'v1-S1-000123')")

    # v1.0: Schema version
    version: str = Field(
        default="1.0",
        description="Schema version"
    )

    domain: Literal["procurement", "sales", "project", "hr", "support"] = Field(
        description="Business domain"
    )

    # v1.0: Use Track type alias (includes new v1.0 tracks)
    track: Track = Field(description="Which benchmark track this belongs to")

    # v1.0: Difficulty and detection mode
    difficulty: Difficulty = Field(
        default="medium",
        description="Difficulty level of this timeline"
    )
    detection_mode: DetectionMode = Field(
        default="explicit",
        description="How supersession is represented (explicit events vs NL)"
    )

    actors: Actors = Field(description="Actors in this timeline")
    initial_state: InitialState = Field(description="State at timeline start")
    events: list[Event] = Field(description="Sequence of events")

    # v1.0: Generation metadata
    metadata: TimelineMetadata | None = Field(
        default=None,
        description="Generation metadata for reproducibility"
    )

    def get_queries(self) -> list[Query]:
        """Extract all query events from the timeline."""
        return [e for e in self.events if isinstance(e, Query)]

    def get_supersessions(self) -> list[Supersession]:
        """Extract all supersession events from the timeline."""
        return [e for e in self.events if isinstance(e, Supersession)]

    def get_implicit_supersessions(self) -> list[tuple[ConversationTurn, ImplicitSupersession]]:
        """Extract all implicit supersessions from conversation turns.

        v1.0: For detection track scoring.
        """
        return [
            (e, e.implicit_supersession)
            for e in self.events
            if isinstance(e, ConversationTurn) and e.implicit_supersession is not None
        ]

    def get_all_fact_ids(self) -> set[str]:
        """Get all fact IDs from state writes.

        v1.0: For provenance validation.
        """
        fact_ids = set()
        for event in self.events:
            if isinstance(event, (StateWrite, Supersession)):
                for write in event.writes:
                    fact_ids.add(write.id)
        return fact_ids
