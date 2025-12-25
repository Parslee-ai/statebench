from datetime import datetime, timedelta

from statebench.baselines.ablations import (
    FactExtractionWithSupersessionStrategy,
    StateBasedNoSupersessionStrategy,
    TranscriptLatestWinsStrategy,
)
from statebench.baselines.base import MemoryStrategy
from statebench.baselines.fact_extract import FactExtractionStrategy
from statebench.baselines.no_memory import NoMemoryStrategy
from statebench.baselines.rag import RAGTranscriptStrategy
from statebench.baselines.state_based import StateBasedStrategy
from statebench.baselines.summary import RollingSummaryStrategy
from statebench.baselines.transcript import TranscriptReplayStrategy
from statebench.runner.harness import EvaluationHarness
from statebench.schema.state import IdentityRole
from statebench.schema.timeline import (
    Actor,
    Actors,
    ConversationTurn,
    InitialState,
    StateWrite,
    Supersession,
    Timeline,
    Write,
)


class DummyStatefulStrategy(MemoryStrategy):
    """Simple strategy to verify initialize_from_state is invoked."""

    def __init__(self):
        super().__init__(token_budget=1000)
        self.initialized_with = None

    def initialize_from_state(self, initial_state: InitialState) -> None:  # type: ignore[override]
        self.initialized_with = initial_state

    @property
    def name(self) -> str:
        return "dummy_stateful"

    @property
    def expects_initial_state(self) -> bool:
        return True

    def process_event(self, event):  # type: ignore[override]
        pass

    def build_context(self, query: str) -> str:  # type: ignore[override]
        return ""

    def reset(self) -> None:  # type: ignore[override]
        pass


class DummyFactExtractionStrategy(FactExtractionWithSupersessionStrategy):
    """Fact-extraction strategy with deterministic embeddings for testing."""

    def __init__(self):
        super().__init__(token_budget=1024, top_k=5)

    def _embed(self, texts):  # type: ignore[override]
        return [[float(len(text))] for text in texts]

    def _get_client(self):  # type: ignore[override]
        raise AssertionError("Network client should not be used in tests")


class StubFactExtractionStrategy(FactExtractionStrategy):
    """Deterministic fact extraction for testing."""

    def __init__(self):
        super().__init__(token_budget=1024, top_k=1)

    def _extract_facts(self, text, speaker):  # type: ignore[override]
        return [f"{speaker}:{text}"]

    def _embed(self, texts):  # type: ignore[override]
        vectors = []
        for text in texts:
            text_lower = text.lower()
            vectors.append([
                1.0 if "alpha" in text_lower else 0.0,
                1.0 if "beta" in text_lower else 0.0,
                1.0 if "prefers" in text_lower else 0.0,
            ])
        return vectors

    def _get_client(self):  # type: ignore[override]
        raise AssertionError("Network client should not be used in tests")


class StubRollingSummaryStrategy(RollingSummaryStrategy):
    """Rolling summary that avoids external LLM calls."""

    def __init__(self, recent_turns: int = 1):
        super().__init__(token_budget=2048, recent_turns=recent_turns, provider="openai")
        self.summaries: list[str] = []

    def _summarize(self, text: str) -> str:  # type: ignore[override]
        summary = f"SUMMARY:{text.splitlines()[0]}"
        self.summaries.append(text)
        return summary

    def _get_client(self):  # type: ignore[override]
        raise AssertionError("LLM client should not be used in tests")


class StubRAGTranscriptStrategy(RAGTranscriptStrategy):
    """RAG strategy with deterministic embeddings."""

    def __init__(self):
        super().__init__(token_budget=2048, chunk_size=2, top_k=1)

    def _embed(self, texts):  # type: ignore[override]
        vectors = []
        for text in texts:
            vectors.append([
                1.0 if "alpha" in text.lower() else 0.0,
                1.0 if "beta" in text.lower() else 0.0,
            ])
        return vectors

    def _get_client(self):  # type: ignore[override]
        raise AssertionError("Network client should not be used in tests")


_DEF_TS = datetime(2025, 1, 1, 12, 0, 0)


def _make_initial_state():
    identity = IdentityRole(
        user_name="Tester",
        authority="Manager",
        department="QA",
        organization="Acme",
    )
    return InitialState(identity_role=identity)


def _make_timeline(events):
    actors = Actors(
        user=Actor(id="u1", role="Manager", org="acme"),
        assistant_role="AI_Agent",
    )
    return Timeline(
        id="T1",
        domain="sales",
        track="supersession",
        actors=actors,
        initial_state=_make_initial_state(),
        events=events,
    )


def test_state_based_strategy_updates_environment_layer():
    strategy = StateBasedStrategy(token_budget=2048)
    initial_state = _make_initial_state()
    strategy.initialize_from_state(initial_state)

    # Inject an environment signal via state write
    env_write = StateWrite(
        ts=_DEF_TS,
        writes=[
            Write(id="W-001", layer="environment", key="alert", value="Outage ongoing", supersedes=None)
        ],
    )
    strategy.process_event(env_write)
    assert strategy.environment["alert"] == "Outage ongoing"

    # Supersession should replace the environment entry
    env_supersession = Supersession(
        ts=_DEF_TS,
        writes=[
            Write(
                id="W-002",
                layer="environment",
                key="alert_resolved",
                value="Incident resolved",
                supersedes="alert",
            )
        ],
    )
    strategy.process_event(env_supersession)
    assert "alert" not in strategy.environment
    assert strategy.environment["alert_resolved"] == "Incident resolved"


def test_state_based_no_supersession_updates_environment_layer():
    strategy = StateBasedNoSupersessionStrategy(token_budget=2048)
    initial_state = _make_initial_state()
    strategy.initialize_from_state(initial_state)

    env_write = StateWrite(
        ts=_DEF_TS,
        writes=[
            Write(id="W-003", layer="environment", key="calendar", value="Meeting at 2pm", supersedes=None)
        ],
    )
    strategy.process_event(env_write)
    assert strategy.environment["calendar"] == "Meeting at 2pm"

    env_supersession = Supersession(
        ts=_DEF_TS,
        writes=[
            Write(
                id="W-004",
                layer="environment",
                key="calendar",
                value="Meeting moved to 3pm",
                supersedes="calendar",
            )
        ],
    )
    strategy.process_event(env_supersession)
    assert strategy.environment["calendar"] == "Meeting moved to 3pm"


def test_fact_extraction_with_supersession_tracks_keys():
    strategy = DummyFactExtractionStrategy()

    state_write = StateWrite(
        ts=_DEF_TS,
        writes=[
            Write(
                id="W-005",
                layer="persistent_facts",
                key="purchase_training_program",
                value="approved purchase for $50000",
                supersedes=None,
            )
        ],
    )
    strategy.process_event(state_write)
    assert len(strategy.facts) == 1
    assert strategy.facts[0]["source_key"] == "purchase_training_program"
    assert strategy.facts[0]["valid"]

    supersession = Supersession(
        ts=_DEF_TS,
        writes=[
            Write(
                id="W-006",
                layer="persistent_facts",
                key="purchase_training_program_v2",
                value="purchase cancelled",
                supersedes="purchase_training_program",
            )
        ],
    )
    strategy.process_event(supersession)

    old_fact = strategy.facts[0]
    assert old_fact["valid"] is False
    assert any(f["valid"] and f["source_key"] == "purchase_training_program_v2" for f in strategy.facts)

    result = strategy.build_context("purchase status")
    context = result.context if hasattr(result, 'context') else result
    assert "cancelled" in context
    assert "approved" not in context


def test_harness_initializes_stateful_strategies():
    harness = EvaluationHarness(use_llm_judge=False)
    strategy = DummyStatefulStrategy()
    timeline = _make_timeline([])

    harness.run_timeline(timeline, strategy)

    assert strategy.initialized_with is timeline.initial_state
def _conversation_turn(offset_minutes: int, speaker: str, text: str) -> ConversationTurn:
    return ConversationTurn(
        ts=_DEF_TS + timedelta(minutes=offset_minutes),
        speaker=speaker,
        text=text,
    )


def test_no_memory_strategy_returns_empty_context():
    strategy = NoMemoryStrategy()
    strategy.process_event(_conversation_turn(0, "user", "Need a budget"))

    result = strategy.build_context("question")
    context = result.context if hasattr(result, 'context') else result
    assert "No memory" in context


def test_transcript_replay_accumulates_conversation():
    strategy = TranscriptReplayStrategy(token_budget=2048)
    strategy.process_event(_conversation_turn(0, "user", "Hello"))
    strategy.process_event(_conversation_turn(1, "assistant", "Hi there"))

    result = strategy.build_context("What's next?")
    context = result.context if hasattr(result, 'context') else result
    assert "User: Hello" in context
    assert "Assistant" not in context


def test_transcript_latest_wins_produces_history():
    strategy = TranscriptLatestWinsStrategy(token_budget=2048)
    strategy.process_event(_conversation_turn(0, "user", "Do X"))
    strategy.process_event(_conversation_turn(1, "user", "Actually do Y"))

    result = strategy.build_context("What should we do?")
    context = result.context if hasattr(result, 'context') else result
    assert "Actually do Y" in context
    assert "Latest commitments" in context


def test_rolling_summary_includes_summary_and_recent_turns():
    strategy = StubRollingSummaryStrategy(recent_turns=1)
    strategy.process_event(_conversation_turn(0, "user", "Turn one"))
    strategy.process_event(_conversation_turn(1, "user", "Turn two"))
    strategy.process_event(_conversation_turn(2, "user", "Turn three"))

    result = strategy.build_context("Status?")
    context = result.context if hasattr(result, 'context') else result
    assert "SUMMARY:" in context
    assert "User: Turn three" in context


def test_rag_transcript_strategy_retrieves_relevant_chunk():
    strategy = StubRAGTranscriptStrategy()
    strategy.process_event(_conversation_turn(0, "user", "Alpha details"))
    strategy.process_event(_conversation_turn(1, "assistant", "Acknowledged Alpha"))
    strategy.process_event(_conversation_turn(2, "user", "Beta follow-up"))

    result = strategy.build_context("Alpha status")
    context = result.context if hasattr(result, 'context') else result
    assert "Alpha" in context
    assert "Beta" not in context


def test_fact_extraction_strategy_retrieves_top_facts():
    strategy = StubFactExtractionStrategy()
    strategy.process_event(_conversation_turn(0, "user", "Prefers Alpha tea"))
    strategy.process_event(_conversation_turn(1, "user", "Ordering Beta supplies"))

    result = strategy.build_context("Alpha preferences")
    context = result.context if hasattr(result, 'context') else result
    assert "Prefers" in context
    assert "Beta" not in context
