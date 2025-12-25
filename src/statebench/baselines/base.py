"""Abstract base class for memory strategies.

Each baseline implements this interface to provide a consistent way
to process events and build context for LLM queries.

v1.0 additions:
- FactMetadata: Metadata about facts for provenance tracking
- ContextResult: Context with provenance (replaces str return)
- Updated build_context() signature for provenance support
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from statebench.schema.state import AuthorityLevel, Scope
from statebench.schema.timeline import Event

# --- v1.0: Provenance Tracking Types ---

@dataclass
class FactMetadata:
    """Metadata about a fact for provenance tracking.

    v1.0: Every fact in context must have this metadata for scoring.
    """
    fact_id: str
    key: str
    value: str
    layer: int  # 1=identity, 2=persistent, 3=working, 4=environment

    # Validity tracking
    is_valid: bool
    superseded_by: str | None = None

    # Scope and authority
    scope: Scope = "global"
    authority: AuthorityLevel = "peer"
    source: str = "user"

    # Dependencies (for repair propagation)
    depends_on: list[str] = field(default_factory=list)
    derived_facts: list[str] = field(default_factory=list)

    # Constraint info
    is_constraint: bool = False
    constraint_type: str | None = None

    def to_context_line(self) -> str:
        """Format this fact for inclusion in context."""
        validity = "" if self.is_valid else " [SUPERSEDED]"
        scope_str = f" [{self.scope}]" if self.scope != "global" else ""
        return f"[{self.fact_id}] {self.key}: {self.value}{validity}{scope_str}"


@dataclass
class ContextResult:
    """Context with provenance tracking.

    v1.0: Replaces str return from build_context().
    Provides full provenance for scoring.
    """
    # The context string for the LLM
    context: str

    # Provenance: which facts are in context
    facts_included: list[FactMetadata] = field(default_factory=list)

    # Which facts exist but were excluded (e.g., superseded, out of scope)
    facts_excluded: list[FactMetadata] = field(default_factory=list)

    # Why each fact was included/excluded
    inclusion_reasons: dict[str, str] = field(default_factory=dict)

    # Token usage
    token_count: int = 0

    def get_included_fact_ids(self) -> set[str]:
        """Get IDs of all included facts."""
        return {f.fact_id for f in self.facts_included}

    def get_valid_included_facts(self) -> list[FactMetadata]:
        """Get only valid facts from included set."""
        return [f for f in self.facts_included if f.is_valid]

    def get_superseded_included_facts(self) -> list[FactMetadata]:
        """Get superseded facts that were incorrectly included."""
        return [f for f in self.facts_included if not f.is_valid]


class MemoryStrategy(ABC):
    """Abstract base class for memory strategies.

    A memory strategy defines how context is accumulated from events
    and assembled into a prompt for the LLM.

    v1.0: build_context() now returns ContextResult with provenance.
    Legacy strategies can use build_context_legacy() and wrap result.
    """

    def __init__(self, token_budget: int = 8000):
        """Initialize the strategy with a token budget.

        Args:
            token_budget: Maximum tokens for context assembly
        """
        self.token_budget = token_budget

    @abstractmethod
    def process_event(self, event: Event) -> None:
        """Process an event and update internal state.

        Called for each event in the timeline (except queries).

        Args:
            event: The event to process
        """
        pass

    @abstractmethod
    def build_context(self, query: str) -> ContextResult:
        """Build context with provenance for an LLM query.

        v1.0: Returns ContextResult with full provenance tracking.

        Called when a Query event is encountered.

        Args:
            query: The query prompt

        Returns:
            ContextResult with context string and provenance metadata
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for a new timeline."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this strategy."""
        pass

    @property
    def expects_initial_state(self) -> bool:
        """Whether this strategy requires the initial state snapshot."""
        return False

    def get_system_prompt(self) -> str:
        """Return an optional system prompt for the LLM.

        Subclasses can override this to provide role/instruction context.
        """
        return (
            "You are an AI assistant helping with business tasks. "
            "Answer questions based on the context provided. "
            "Be concise and accurate."
        )

    def format_prompt(self, query: str) -> str:
        """Format the final prompt for the LLM.

        Args:
            query: The query to answer

        Returns:
            Complete prompt including context
        """
        result = self.build_context(query)
        if result.context:
            return f"{result.context}\n\n---\n\nUser question: {query}"
        else:
            return query

    def format_prompt_with_provenance(self, query: str) -> tuple[str, ContextResult]:
        """Format prompt and return provenance.

        v1.0: Use this when you need both the prompt and provenance.

        Args:
            query: The query to answer

        Returns:
            Tuple of (formatted prompt, context result with provenance)
        """
        result = self.build_context(query)
        if result.context:
            prompt = f"{result.context}\n\n---\n\nUser question: {query}"
        else:
            prompt = query
        return prompt, result


# --- v1.0: Helper for migrating legacy strategies ---

def wrap_legacy_context(context: str) -> ContextResult:
    """Wrap a legacy context string in ContextResult.

    Use this to migrate v0.x strategies that return str.
    Provenance will be empty (not tracked).

    Args:
        context: Legacy context string

    Returns:
        ContextResult with empty provenance
    """
    return ContextResult(
        context=context,
        facts_included=[],
        facts_excluded=[],
        inclusion_reasons={},
        token_count=len(context) // 4,  # Rough estimate
    )
