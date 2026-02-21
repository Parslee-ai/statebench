"""Memory strategy baselines for StateBench."""

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

__all__ = [
    "MemoryStrategy",
    "NoMemoryStrategy",
    "TranscriptReplayStrategy",
    "RollingSummaryStrategy",
    "RAGTranscriptStrategy",
    "FactExtractionStrategy",
    "StateBasedStrategy",
    "StateBasedNoSupersessionStrategy",
    "FactExtractionWithSupersessionStrategy",
    "TranscriptLatestWinsStrategy",
]

BASELINE_REGISTRY = {
    # Core baselines
    "no_memory": NoMemoryStrategy,
    "transcript_replay": TranscriptReplayStrategy,
    "rolling_summary": RollingSummaryStrategy,
    "rag_transcript": RAGTranscriptStrategy,
    "fact_extraction": FactExtractionStrategy,
    "state_based": StateBasedStrategy,
    # Ablation baselines
    "state_based_no_supersession": StateBasedNoSupersessionStrategy,
    "fact_extraction_with_supersession": FactExtractionWithSupersessionStrategy,
    "transcript_latest_wins": TranscriptLatestWinsStrategy,
}


def get_baseline(name: str, **kwargs: object) -> MemoryStrategy:
    """Get a baseline strategy by name.

    Args:
        name: Strategy name
        **kwargs: Arguments to pass to the strategy constructor

    Returns:
        Instantiated strategy

    Raises:
        ValueError: If strategy name is unknown
    """
    if name not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline: {name}. "
            f"Available: {list(BASELINE_REGISTRY.keys())}"
        )
    strategy_class = BASELINE_REGISTRY[name]
    return strategy_class(**kwargs)  # type: ignore[no-any-return]
