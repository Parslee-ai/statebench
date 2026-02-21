"""Hugging Face Hub integration for StateBench.

Provides functions to prepare and publish StateBench datasets to HuggingFace Hub,
enabling easy access via `load_dataset("parslee/statebench")`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from statebench.runner.harness import load_timelines
from statebench.schema.timeline import Query, Timeline

if TYPE_CHECKING:
    from datasets import DatasetDict  # type: ignore[import-untyped]


def timeline_to_hf_row(timeline: Timeline) -> dict[str, Any]:
    """Convert a Timeline to a HuggingFace-compatible flat dictionary.

    Direct fields are kept as-is. Complex objects (events, initial_state, actors, metadata)
    are JSON-serialized for storage. Convenience fields are computed for quick filtering.
    """
    # Count queries in events
    num_queries = sum(1 for e in timeline.events if isinstance(e, Query))

    return {
        # Direct metadata fields
        "id": timeline.id,
        "version": timeline.version,
        "domain": timeline.domain,
        "track": timeline.track,
        "difficulty": timeline.difficulty,
        "detection_mode": timeline.detection_mode,
        # JSON-serialized complex objects (preserves full structure)
        "actors": timeline.actors.model_dump_json(),
        "initial_state": timeline.initial_state.model_dump_json(),
        "events": json.dumps([e.model_dump(mode="json") for e in timeline.events]),
        "metadata": timeline.metadata.model_dump_json() if timeline.metadata else "null",
        # Convenience fields for filtering/analysis
        "num_events": len(timeline.events),
        "num_queries": num_queries,
        # User identity (flattened for easy access)
        "user_name": timeline.initial_state.identity_role.user_name,
        "user_authority": timeline.initial_state.identity_role.authority,
        "user_department": timeline.initial_state.identity_role.department,
        "user_organization": timeline.initial_state.identity_role.organization,
    }


def load_split_as_rows(split_path: Path) -> list[dict[str, Any]]:
    """Load a JSONL split file and convert all timelines to HF rows."""
    rows = []
    for timeline in load_timelines(split_path):
        rows.append(timeline_to_hf_row(timeline))
    return rows


def prepare_hf_dataset(release_dir: Path) -> DatasetDict:
    """Convert a StateBench release to HuggingFace DatasetDict format.

    Loads train/dev/test splits from the release directory and creates
    a DatasetDict suitable for pushing to HuggingFace Hub.

    Args:
        release_dir: Path to release directory (e.g., data/releases/v1.0)

    Returns:
        DatasetDict with train, validation (dev), and test splits.

    Raises:
        FileNotFoundError: If required split files are missing.
        ImportError: If datasets library is not installed.
    """
    try:
        from datasets import Dataset, DatasetDict
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required for HuggingFace integration. "
            "Install with: pip install datasets"
        ) from e

    release_path = Path(release_dir)

    # Validate release directory
    required_splits = ["train.jsonl", "dev.jsonl", "test.jsonl"]
    for split_file in required_splits:
        if not (release_path / split_file).exists():
            raise FileNotFoundError(
                f"Missing required split file: {release_path / split_file}"
            )

    # Load each split
    train_rows = load_split_as_rows(release_path / "train.jsonl")
    dev_rows = load_split_as_rows(release_path / "dev.jsonl")
    test_rows = load_split_as_rows(release_path / "test.jsonl")

    # Create DatasetDict (HF convention: dev -> validation)
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_rows),
        "validation": Dataset.from_list(dev_rows),
        "test": Dataset.from_list(test_rows),
    })

    return dataset_dict


def push_to_hub(
    release_dir: Path,
    repo_id: str = "parslee/statebench",
    private: bool = False,
    token: str | None = None,
) -> str:
    """Push a StateBench release to HuggingFace Hub.

    Args:
        release_dir: Path to release directory (e.g., data/releases/v1.0)
        repo_id: HuggingFace repository ID (e.g., "parslee/statebench")
        private: Whether the repository should be private
        token: HuggingFace API token (uses cached token if not provided)

    Returns:
        URL of the published dataset.

    Raises:
        ImportError: If huggingface_hub is not installed.
    """
    try:
        import huggingface_hub  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The 'huggingface_hub' package is required. "
            "Install with: pip install huggingface_hub"
        ) from e

    # Prepare the dataset
    dataset_dict = prepare_hf_dataset(release_dir)

    # Push to hub
    dataset_dict.push_to_hub(
        repo_id,
        private=private,
        token=token,
    )

    return f"https://huggingface.co/datasets/{repo_id}"


def prepare_hf_staging(release_dir: Path, output_dir: Path) -> dict[str, int]:
    """Prepare HF dataset files locally for inspection before pushing.

    Creates Parquet files in the output directory matching HF Hub structure.

    Args:
        release_dir: Path to release directory
        output_dir: Path to output staging directory

    Returns:
        Dictionary with counts per split.
    """
    dataset_dict = prepare_hf_dataset(release_dir)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split_name, dataset in dataset_dict.items():
        # Save as parquet (HF preferred format)
        parquet_path = output_path / f"{split_name}.parquet"
        dataset.to_parquet(str(parquet_path))
        counts[split_name] = len(dataset)

    # Also save a sample as JSON for easy inspection
    sample_path = output_path / "sample.json"
    sample = dataset_dict["test"][0]
    with open(sample_path, "w") as f:
        json.dump(sample, f, indent=2)

    return counts


def load_from_hub(repo_id: str = "parslee/statebench", split: str | None = None) -> Any:
    """Load StateBench dataset from HuggingFace Hub.

    Convenience wrapper around datasets.load_dataset.

    Args:
        repo_id: HuggingFace repository ID
        split: Optional split to load (train, validation, test)

    Returns:
        Dataset or DatasetDict depending on split parameter.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required. "
            "Install with: pip install datasets"
        ) from e

    return load_dataset(repo_id, split=split)


def hf_row_to_timeline(row: dict[str, Any]) -> Timeline:
    """Convert a HuggingFace row back to a Timeline object.

    Useful for evaluation after loading from HuggingFace Hub.
    """
    from statebench.schema.timeline import Actors, InitialState, TimelineMetadata

    # Parse JSON-serialized fields
    actors = Actors.model_validate_json(row["actors"])
    initial_state = InitialState.model_validate_json(row["initial_state"])
    events_raw = json.loads(row["events"])
    metadata_raw = row.get("metadata", "null")
    metadata = (
        TimelineMetadata.model_validate_json(metadata_raw)
        if metadata_raw and metadata_raw != "null"
        else None
    )

    # Parse events with discriminated union
    from statebench.schema.timeline import (
        ConversationTurn,
        Query,
        StateWrite,
        Supersession,
    )

    events: list[ConversationTurn | StateWrite | Supersession | Query] = []
    for event_data in events_raw:
        event_type = event_data.get("type")
        if event_type == "conversation_turn":
            events.append(ConversationTurn.model_validate(event_data))
        elif event_type == "state_write":
            events.append(StateWrite.model_validate(event_data))
        elif event_type == "supersession":
            events.append(Supersession.model_validate(event_data))
        elif event_type == "query":
            events.append(Query.model_validate(event_data))
        else:
            raise ValueError(f"Unknown event type: {event_type}")

    return Timeline(
        id=row["id"],
        version=row["version"],
        domain=row["domain"],
        track=row["track"],
        difficulty=row["difficulty"],
        detection_mode=row["detection_mode"],
        actors=actors,
        initial_state=initial_state,
        events=events,
        metadata=metadata,
    )
