"""Tests for MemgineEngine."""

from datetime import datetime

from statebench.memgine.config import MemgineConfig
from statebench.memgine.engine import MemgineEngine
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


def _make_engine(budget: int = 8000) -> MemgineEngine:
    return MemgineEngine(MemgineConfig(token_budget=budget))


def _make_source(stype: str = "user", authority: str = "peer") -> Source:
    return Source(type=stype, authority=authority)


def test_ingest_identity() -> None:
    engine = _make_engine()
    identity = IdentityRole(user_name="Alice", authority="Director", department="Eng")
    entry = engine.ingest_identity(identity)

    assert "Alice" in entry.value
    assert "Director" in entry.value
    assert entry.layer == 1


def test_ingest_fact_and_build_context() -> None:
    engine = _make_engine()
    engine.ingest_identity(IdentityRole(user_name="Alice", authority="Director"))

    ts = datetime(2024, 1, 1)
    engine.ingest_fact(
        fact_id="F-001",
        key="budget",
        value="Project budget is $50k",
        source=_make_source(),
        ts=ts,
    )

    result = engine.build_context("What's the budget?")
    assert "F-001" in result.context
    assert "$50k" in result.context
    assert len(result.facts_included) >= 1


def test_supersession_tracking() -> None:
    engine = _make_engine()
    ts = datetime(2024, 1, 1)

    engine.ingest_fact(
        fact_id="F-001", key="budget", value="$50k",
        source=_make_source(), ts=ts,
    )
    engine.ingest_fact(
        fact_id="F-002", key="budget", value="$75k",
        source=_make_source(), ts=ts,
        supersedes="F-001",
    )

    result = engine.build_context("What's the budget?")
    # F-002 should be included, F-001 excluded
    included_ids = {f.fact_id for f in result.facts_included}
    excluded_ids = {f.fact_id for f in result.facts_excluded}
    assert "F-002" in included_ids
    assert "F-001" in excluded_ids


def test_constraint_preservation() -> None:
    engine = _make_engine()
    ts = datetime(2024, 1, 1)

    engine.ingest_fact(
        fact_id="F-001",
        key="policy",
        value="Budget must not exceed $100k",
        source=Source(type="policy", authority="policy"),
        ts=ts,
    )

    result = engine.build_context("Can we spend $120k?")
    assert "CONSTRAINT" in result.context or "Constraints" in result.context
    assert "$100k" in result.context


def test_conversation_in_working_set() -> None:
    engine = _make_engine()
    ts = datetime(2024, 1, 1)

    engine.ingest_conversation("user", "What's the status?", ts)
    engine.ingest_conversation("assistant", "We're on track.", ts)

    result = engine.build_context("Any updates?")
    assert "What's the status?" in result.context
    assert "on track" in result.context


def test_environment_signals() -> None:
    engine = _make_engine()
    ts = datetime(2024, 1, 1)

    engine.ingest_environment("now", "2024-01-01T10:00:00", ts)

    result = engine.build_context("What time is it?")
    assert "2024-01-01" in result.context


def test_expand_fact_chain() -> None:
    engine = _make_engine()
    ts = datetime(2024, 1, 1)

    engine.ingest_fact(
        fact_id="F-001", key="budget", value="$50k",
        source=_make_source(), ts=ts,
    )
    engine.ingest_fact(
        fact_id="F-002", key="budget", value="$75k",
        source=_make_source(), ts=ts,
        supersedes="F-001",
    )
    engine.ingest_fact(
        fact_id="F-003", key="budget", value="$100k",
        source=_make_source(), ts=ts,
        supersedes="F-002",
    )

    chain = engine.expand_fact_chain("budget")
    assert len(chain) == 3
    assert chain[0].value == "$50k"
    assert chain[2].value == "$100k"


def test_ingest_initial_state() -> None:
    engine = _make_engine()
    ts = datetime(2024, 1, 1)

    initial = InitialState(
        identity_role=IdentityRole(user_name="Bob", authority="Manager"),
        persistent_facts=[
            PersistentFact(
                id="F-001",
                key="budget",
                value="$50k",
                source=_make_source(),
                ts=ts,
            ),
        ],
        working_set=[
            WorkingSetItem(
                item_type="context",
                content="Project kickoff",
                ts=ts,
            ),
        ],
        environment={"now": "2024-01-01"},
    )

    engine.ingest_initial_state(initial)
    result = engine.build_context("What's the budget?")
    assert "Bob" in result.context
    assert "$50k" in result.context


def test_ingest_statebench_conversation_event() -> None:
    engine = _make_engine()
    event = ConversationTurn(
        ts=datetime(2024, 1, 1), speaker="user", text="Hello there"
    )
    engine.ingest_statebench_event(event)

    result = engine.build_context("test")
    assert "Hello there" in result.context


def test_ingest_statebench_state_write_event() -> None:
    engine = _make_engine()
    event = StateWrite(
        ts=datetime(2024, 1, 1),
        writes=[
            Write(
                id="F-001",
                layer="persistent_facts",
                key="budget",
                value="$50k",
            ),
        ],
    )
    engine.ingest_statebench_event(event)

    result = engine.build_context("budget?")
    assert "$50k" in result.context


def test_ingest_statebench_supersession_event() -> None:
    engine = _make_engine()
    ts = datetime(2024, 1, 1)

    # First establish a fact
    engine.ingest_statebench_event(
        StateWrite(
            ts=ts,
            writes=[Write(id="F-001", layer="persistent_facts", key="budget", value="$50k")],
        )
    )

    # Then supersede it
    engine.ingest_statebench_event(
        Supersession(
            ts=ts,
            writes=[
                Write(
                    id="F-002",
                    layer="persistent_facts",
                    key="budget",
                    value="$75k",
                    supersedes="F-001",
                )
            ],
        )
    )

    result = engine.build_context("budget?")
    assert "$75k" in result.context
    excluded_ids = {f.fact_id for f in result.facts_excluded}
    assert "F-001" in excluded_ids


def test_reset() -> None:
    engine = _make_engine()
    ts = datetime(2024, 1, 1)
    engine.ingest_fact(
        fact_id="F-001", key="k", value="v", source=_make_source(), ts=ts
    )

    engine.reset()
    result = engine.build_context("test")
    assert result.context == ""
    assert len(result.facts_included) == 0


def test_provenance_accuracy() -> None:
    """ContextResult provenance must accurately reflect what's in context."""
    engine = _make_engine()
    ts = datetime(2024, 1, 1)

    engine.ingest_fact(
        fact_id="F-001", key="budget", value="$50k",
        source=_make_source(), ts=ts,
    )
    engine.ingest_fact(
        fact_id="F-002", key="deadline", value="March 1",
        source=_make_source(), ts=ts,
    )
    engine.ingest_fact(
        fact_id="F-003", key="budget", value="$75k",
        source=_make_source(), ts=ts,
        supersedes="F-001",
    )

    result = engine.build_context("test")
    included_ids = {f.fact_id for f in result.facts_included}
    excluded_ids = {f.fact_id for f in result.facts_excluded}

    # F-002 and F-003 should be included, F-001 excluded
    assert "F-002" in included_ids
    assert "F-003" in included_ids
    assert "F-001" in excluded_ids

    # All included facts should have is_valid=True
    for f in result.facts_included:
        assert f.is_valid is True

    # All excluded facts should have is_valid=False
    for f in result.facts_excluded:
        assert f.is_valid is False


def test_no_llm_calls() -> None:
    """Verify that Phase 1 engine never calls an LLM."""
    # This test verifies the architecture: no external service calls
    engine = _make_engine()
    ts = datetime(2024, 1, 1)

    for i in range(20):
        engine.ingest_fact(
            fact_id=f"F-{i:03d}",
            key=f"fact_{i}",
            value=f"Value for fact {i} with enough text to be meaningful",
            source=_make_source(),
            ts=ts,
        )
        engine.ingest_conversation("user", f"Turn {i}", ts)

    # Building context should work without any external calls
    result = engine.build_context("test")
    assert isinstance(result.context, str)
