"""Tests for working-set repetition dedup (Whose-Facts-Win repetition bias)."""

from datetime import datetime, timedelta

from statebench.memgine.config import MemgineConfig
from statebench.memgine.engine import MemgineEngine

T0 = datetime(2026, 1, 1, 12, 0)


def _engine(dedupe: bool = True) -> MemgineEngine:
    return MemgineEngine(MemgineConfig(token_budget=8000, dedupe_repeated_turns=dedupe))


def _turns(eng, texts):
    for i, t in enumerate(texts):
        eng.ingest_conversation("user", t, T0 + timedelta(minutes=i))


def _recent_context(eng) -> str:
    ctx = eng.build_context("what is the status?")
    if "## Recent Context" not in ctx.context:
        return ""
    return ctx.context.split("## Recent Context", 1)[1]


def test_repeated_turn_appears_once() -> None:
    eng = _engine()
    _turns(eng, ["The price is $10.", "ok", "The price is $10!", "The price is $10"])
    rc = _recent_context(eng)
    assert rc.count("price is $10") == 1  # 3 restatements collapse to 1


def test_keeps_most_recent_instance_and_order() -> None:
    eng = _engine()
    _turns(eng, ["A", "B", "A", "C"])  # A repeated
    rc = _recent_context(eng).lower()
    # order of survivors is chronological: B, A(most-recent), C
    assert rc.index("user: b") < rc.index("user: a") < rc.index("user: c")


def test_value_symbols_not_over_merged() -> None:
    # "$10" and "10%" in otherwise-identical sentences must stay distinct.
    eng = _engine()
    _turns(eng, ["the rate is $10", "the rate is 10%"])
    rc = _recent_context(eng)
    assert "the rate is $10" in rc and "the rate is 10%" in rc


def test_distinct_turns_not_collapsed() -> None:
    eng = _engine()
    _turns(eng, ["price is $10", "price is $12", "deadline friday"])
    rc = _recent_context(eng)
    assert "price is $10" in rc and "price is $12" in rc and "deadline friday" in rc


def test_dedupe_disabled_keeps_repeats() -> None:
    eng = _engine(dedupe=False)
    _turns(eng, ["same thing", "same thing", "same thing"])
    rc = _recent_context(eng)
    assert rc.count("same thing") == 3


def test_punctuation_only_turns_not_collapsed() -> None:
    eng = _engine()
    _turns(eng, ["...", "...", "real content"])
    rc = _recent_context(eng)
    assert rc.count("...") == 2  # empty-content turns are never merged
