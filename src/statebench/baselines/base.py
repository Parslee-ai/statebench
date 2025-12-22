"""Abstract base class for memory strategies.

Each baseline implements this interface to provide a consistent way
to process events and build context for LLM queries.
"""

from abc import ABC, abstractmethod
from typing import Any

from statebench.schema.timeline import Event, ConversationTurn, StateWrite, Supersession


class MemoryStrategy(ABC):
    """Abstract base class for memory strategies.

    A memory strategy defines how context is accumulated from events
    and assembled into a prompt for the LLM.
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
    def build_context(self, query: str) -> str:
        """Build context string for an LLM query.

        Called when a Query event is encountered.

        Args:
            query: The query prompt

        Returns:
            Context string to prepend to the query
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
        context = self.build_context(query)
        if context:
            return f"{context}\n\n---\n\nUser question: {query}"
        else:
            return query
