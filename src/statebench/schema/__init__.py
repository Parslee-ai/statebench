"""Schema models for StateBench timelines and state.

v1.0 additions:
- Source, AuthorityLevel, Scope: Provenance types
- ImplicitSupersession: NL detection markers
- FactRequirement, SupersessionDetection: Ground truth extensions
- TimelineMetadata: Generation metadata
- Track, Difficulty, DetectionMode: Type aliases
"""

from statebench.schema.state import (
    # v1.0: Provenance types
    AuthorityLevel,
    ConstraintType,
    # Core state types
    EnvironmentSignal,
    FailureCategory,
    FailureSeverity,
    IdentityRole,
    PersistentFact,
    Scope,
    Source,
    StateSnapshot,
    WorkingSetItem,
)
from statebench.schema.timeline import (
    # Core timeline types
    Actor,
    ConversationTurn,
    # v1.0: Detection types
    DetectionDifficulty,
    DetectionMode,
    Difficulty,
    Event,
    # v1.0: Provenance types
    FactRequirement,
    GroundTruth,
    ImplicitSupersession,
    MentionRequirement,
    Query,
    ScoringWeights,
    StateWrite,
    Supersession,
    SupersessionDetection,
    Timeline,
    TimelineMetadata,
    Track,
    Write,
)

__all__ = [
    # Core timeline types
    "Timeline",
    "Event",
    "ConversationTurn",
    "StateWrite",
    "Supersession",
    "Query",
    "GroundTruth",
    "Actor",
    "Write",
    # Core state types
    "IdentityRole",
    "PersistentFact",
    "WorkingSetItem",
    "EnvironmentSignal",
    "StateSnapshot",
    # v1.0: Type aliases
    "Track",
    "Difficulty",
    "DetectionMode",
    "DetectionDifficulty",
    "AuthorityLevel",
    "Scope",
    "ConstraintType",
    "FailureCategory",
    "FailureSeverity",
    # v1.0: Provenance types
    "Source",
    "ImplicitSupersession",
    "MentionRequirement",
    "FactRequirement",
    "SupersessionDetection",
    "ScoringWeights",
    "TimelineMetadata",
]
