"""Memory strategy baselines for StateBench."""

from statebench.baselines.ablations import (
    FactExtractionWithSupersessionStrategy,
    StateBasedNoSupersessionStrategy,
    TranscriptLatestWinsStrategy,
)
from statebench.baselines.base import MemoryStrategy
from statebench.baselines.dialectic import DialecticStrategy
from statebench.baselines.fact_extract import FactExtractionStrategy
from statebench.baselines.flat_file import FlatFileStrategy
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
    "MemgineStrategy",
    "DialecticStrategy",
    "FlatFileStrategy",
]


def _get_memgine_strategy() -> type:
    """Lazy import to avoid circular dependency."""
    from statebench.memgine.strategy import MemgineStrategy
    return MemgineStrategy


class _LazyMemgine:
    """Lazy proxy that defers import of MemgineStrategy until instantiation."""

    def __call__(self, **kwargs: object) -> MemoryStrategy:
        cls = _get_memgine_strategy()
        return cls(**kwargs)  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return "<MemgineStrategy (lazy)>"


class _LazyMemgineDistill:
    """Memgine + CAR NL-distillation front-end for conversational supersession."""

    def __call__(self, **kwargs: object) -> MemoryStrategy:
        from statebench.baselines.memgine_distill import MemgineDistillStrategy

        return MemgineDistillStrategy(**kwargs)  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return "<MemgineDistillStrategy (lazy)>"


class _LazyMemgineSafeSupersession:
    """Memgine variant that hides superseded *values* (anti-resurrection)."""

    def __call__(self, **kwargs: object) -> MemoryStrategy:
        cls = _get_memgine_strategy()
        kwargs.setdefault("show_superseded_values", False)
        kwargs.setdefault("cascade_supersession_to_transcript", True)
        return cls(**kwargs)  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return "<MemgineStrategy supersession-safe (lazy)>"


class _LazyCarMemgine:
    """Lazy proxy for Rust car-memgine strategy."""

    def __call__(self, **kwargs: object) -> MemoryStrategy:
        from statebench.baselines.car_memgine_strategy import CarMemgineStrategy
        return CarMemgineStrategy(**kwargs)  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return "<CarMemgineStrategy (lazy)>"


BASELINE_REGISTRY: dict[str, type | _LazyMemgine | _LazyCarMemgine] = {
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
    # Engine baselines
    "memgine": _LazyMemgine(),
    "memgine_safe_supersession": _LazyMemgineSafeSupersession(),
    "memgine_distill": _LazyMemgineDistill(),
    # Dialectic memory engine (reasoning-first, inspired by Honcho)
    "dialectic": DialecticStrategy,
    # Rust car-memgine (graph-based)
    "car_memgine": _LazyCarMemgine(),
    # OpenClaw flat-file baseline
    "flat_file": FlatFileStrategy,
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


# Lazy accessor for MemgineStrategy for tests that import it directly
def __getattr__(name: str) -> type:
    if name == "MemgineStrategy":
        return _get_memgine_strategy()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
