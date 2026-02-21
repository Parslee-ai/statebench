"""Tests for LayerState."""

from statebench.memgine.layers import LayerState, infer_scope, is_constraint
from statebench.schema.state import Source


def test_register_and_validity() -> None:
    state = LayerState()
    state.register_fact(entry_id=0, fact_id="F-001", fact_key="budget")

    assert state.is_fact_valid("F-001")
    assert "F-001" in state.get_valid_fact_ids()
    assert len(state.get_superseded_fact_ids()) == 0


def test_invalidate_fact() -> None:
    state = LayerState()
    state.register_fact(entry_id=0, fact_id="F-001", fact_key="budget")
    state.register_fact(entry_id=1, fact_id="F-002", fact_key="budget")

    invalidated = state.invalidate_fact("F-001", "F-002")
    assert "F-001" in invalidated
    assert not state.is_fact_valid("F-001")
    assert state.is_fact_valid("F-002")
    assert "F-001" in state.get_superseded_fact_ids()


def test_supersession_chain() -> None:
    state = LayerState()
    state.register_fact(entry_id=0, fact_id="F-001", fact_key="budget")
    state.register_fact(entry_id=1, fact_id="F-002", fact_key="budget")
    state.register_fact(entry_id=2, fact_id="F-003", fact_key="budget")

    chain = state.get_chain_for_key("budget")
    assert chain == [0, 1, 2]


def test_constraints_tracking() -> None:
    state = LayerState()
    state.register_fact(entry_id=0, fact_id="F-001", fact_key="policy", is_constraint=True)
    state.register_fact(entry_id=1, fact_id="F-002", fact_key="budget")

    assert "F-001" in state.constraints
    assert "F-002" not in state.constraints


def test_dependency_tracking() -> None:
    state = LayerState()
    state.register_fact(entry_id=0, fact_id="F-001", fact_key="budget")
    state.register_fact(
        entry_id=1, fact_id="F-002", fact_key="total_cost", depends_on=("F-001",)
    )

    assert "F-001" in state.dependencies["F-002"]
    assert "F-002" in state.dependents["F-001"]


def test_cascade_invalidation() -> None:
    state = LayerState()
    state.register_fact(entry_id=0, fact_id="F-001", fact_key="budget")
    state.register_fact(
        entry_id=1, fact_id="F-002", fact_key="total", depends_on=("F-001",)
    )

    invalidated = state.invalidate_fact("F-001", "F-003")
    # F-001 is invalidated, F-002 should be flagged too (cascade)
    assert "F-001" in invalidated
    assert "F-002" in invalidated


def test_is_constraint_policy_source() -> None:
    source = Source(type="policy", authority="policy")
    assert is_constraint("Any value here", source) is True


def test_is_constraint_formal_language() -> None:
    source = Source(type="user", authority="peer")
    assert is_constraint("Budget must not exceed $50k", source) is True
    assert is_constraint("I prefer coffee", source) is False


def test_is_constraint_correction_excluded() -> None:
    source = Source(type="policy", authority="policy")
    # Even from policy source, corrections are not constraints
    assert is_constraint("CORRECTION: budget changed", Source(type="user", authority="peer")) is False


def test_infer_scope() -> None:
    assert infer_scope("What if we increased the budget?") == "hypothetical"
    assert infer_scope("This is a draft proposal") == "draft"
    assert infer_scope("Only for this task, use Python") == "task"
    assert infer_scope("The budget is $50k") == "global"


def test_clear() -> None:
    state = LayerState()
    state.register_fact(entry_id=0, fact_id="F-001", fact_key="budget")
    state.register_working_set(entry_id=1)
    state.register_environment("now", entry_id=2)

    state.clear()
    assert len(state.get_valid_fact_ids()) == 0
    assert len(state.working_set_entries) == 0
    assert len(state.environment_entries) == 0
