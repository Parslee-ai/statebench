"""Tests for MemgineStrategy (StateBench integration)."""

from datetime import datetime

from statebench.baselines import BASELINE_REGISTRY, get_baseline
from statebench.memgine.strategy import MemgineStrategy
from statebench.schema.state import (
    IdentityRole,
    PersistentFact,
    Source,
    WorkingSetItem,
)
from statebench.schema.timeline import (
    ConversationTurn,
    InitialState,
    StateWrite,
    Supersession,
    Write,
)


def test_registered_in_baseline_registry() -> None:
    assert "memgine" in BASELINE_REGISTRY
    # Registry uses lazy proxy to avoid circular imports
    strategy = BASELINE_REGISTRY["memgine"]()
    assert isinstance(strategy, MemgineStrategy)


def test_get_baseline() -> None:
    strategy = get_baseline("memgine")
    assert isinstance(strategy, MemgineStrategy)
    assert strategy.name == "memgine"
    assert strategy.expects_initial_state is True


def test_full_timeline_integration() -> None:
    """Run a full timeline through the strategy, mimicking harness behavior."""
    strategy = MemgineStrategy(token_budget=8000)
    ts = datetime(2024, 1, 1)

    # Step 1: Initialize from state
    initial = InitialState(
        identity_role=IdentityRole(user_name="Alice", authority="Director"),
        persistent_facts=[
            PersistentFact(
                id="F-001",
                key="budget",
                value="Project budget is $50k",
                source=Source(type="user", authority="peer"),
                ts=ts,
            ),
            PersistentFact(
                id="F-002",
                key="team_size",
                value="Team has 5 engineers",
                source=Source(type="system", authority="system"),
                ts=ts,
            ),
        ],
        working_set=[],
        environment={"now": "2024-01-01T10:00:00"},
    )
    strategy.initialize_from_state(initial)

    # Step 2: Process events
    strategy.process_event(
        ConversationTurn(
            ts=ts, speaker="user", text="Let's discuss the budget."
        )
    )
    strategy.process_event(
        ConversationTurn(
            ts=ts, speaker="assistant", text="The current budget is $50k."
        )
    )

    # Step 3: Supersede a fact
    strategy.process_event(
        Supersession(
            ts=ts,
            writes=[
                Write(
                    id="F-003",
                    layer="persistent_facts",
                    key="budget",
                    value="Budget increased to $75k",
                    supersedes="F-001",
                )
            ],
        )
    )

    # Step 4: Build context for query
    result = strategy.build_context("What's the current budget?")

    # Verify context contains current info
    assert "$75k" in result.context
    assert "Alice" in result.context

    # Verify provenance
    included_ids = {f.fact_id for f in result.facts_included}
    excluded_ids = {f.fact_id for f in result.facts_excluded}
    assert "F-003" in included_ids  # New budget
    assert "F-002" in included_ids  # Team size (unchanged)
    assert "F-001" in excluded_ids  # Old budget (superseded)

    # Verify system prompt
    prompt = strategy.get_system_prompt()
    assert "CONSTRAINT" in prompt


def test_reset() -> None:
    strategy = MemgineStrategy()
    ts = datetime(2024, 1, 1)

    initial = InitialState(
        identity_role=IdentityRole(user_name="Alice", authority="Director"),
        persistent_facts=[
            PersistentFact(
                id="F-001",
                key="budget",
                value="$50k",
                source=Source(type="user", authority="peer"),
                ts=ts,
            ),
        ],
    )
    strategy.initialize_from_state(initial)

    # Verify data exists
    result1 = strategy.build_context("test")
    assert "$50k" in result1.context

    # Reset
    strategy.reset()
    result2 = strategy.build_context("test")
    assert result2.context == ""


def test_format_prompt() -> None:
    strategy = MemgineStrategy()
    ts = datetime(2024, 1, 1)

    initial = InitialState(
        identity_role=IdentityRole(user_name="Alice", authority="Manager"),
        persistent_facts=[],
    )
    strategy.initialize_from_state(initial)

    prompt = strategy.format_prompt("What's the status?")
    assert "Alice" in prompt
    assert "What's the status?" in prompt
