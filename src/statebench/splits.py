"""Split management for StateBench v1.0.

This module provides functionality for:
1. Creating reproducible train/dev/test/hidden splits
2. Quarterly refresh of hidden splits
3. Canary items for contamination detection
4. Split verification and statistics
"""

import hashlib
import json
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from statebench.schema.timeline import Timeline

T = TypeVar("T")


# =============================================================================
# Split Configuration
# =============================================================================

@dataclass
class SplitConfig:
    """Configuration for dataset splits."""
    train: float = 0.60
    dev: float = 0.15
    test: float = 0.15
    hidden: float = 0.10

    def __post_init__(self) -> None:
        total = self.train + self.dev + self.test + self.hidden
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    def as_dict(self) -> dict[str, float]:
        return {
            "train": self.train,
            "dev": self.dev,
            "test": self.test,
            "hidden": self.hidden,
        }


@dataclass
class SplitMetadata:
    """Metadata about a split."""
    version: str
    created_at: str
    seed: int
    config: dict[str, float]
    counts: dict[str, int]
    canary_count: int = 0
    canary_ids: list[str] = field(default_factory=list)


# =============================================================================
# Canary Generation
# =============================================================================

@dataclass
class CanaryTimeline:
    """A canary timeline for contamination detection.

    Canaries are synthetic timelines with unique, memorable patterns
    that would only appear in responses if a model was trained on
    the hidden split.
    """
    id: str
    canary_phrase: str  # Unique phrase to detect
    timeline: Timeline


def generate_canary_phrase() -> str:
    """Generate a unique canary phrase.

    These phrases are designed to be:
    - Unique (won't appear in normal training data)
    - Memorable (would likely be reproduced if seen)
    - Detectable (easy to grep for)
    """
    # Generate unique identifier
    unique_id = uuid.uuid4().hex[:8].upper()

    # Canary patterns that would stand out
    patterns = [
        f"CANARY-{unique_id}-STATEBENCH",
        f"Project code: ZEPHYR-{unique_id}",
        f"Reference number: SB-{unique_id}-V1",
        f"Internal memo: CANARY-{unique_id}",
        f"Case ID: DETECT-{unique_id}",
    ]

    return random.choice(patterns)


def create_canary_timeline(base_timeline: Timeline) -> CanaryTimeline:
    """Create a canary version of a timeline.

    Injects a unique canary phrase into the timeline that would
    be detectable if a model was trained on this data.
    """
    canary_phrase = generate_canary_phrase()

    # Create a modified copy
    canary_id = f"CANARY-{base_timeline.id}"

    # Inject canary into first conversation turn
    modified_events = list(base_timeline.events)
    for i, event in enumerate(modified_events):
        if hasattr(event, "text"):
            # Add canary to conversation turn
            from statebench.schema.timeline import ConversationTurn
            if isinstance(event, ConversationTurn):
                modified_events[i] = ConversationTurn(
                    ts=event.ts,
                    speaker=event.speaker,
                    text=f"{event.text} [{canary_phrase}]",
                )
                break

    # Create modified timeline
    canary_timeline = Timeline(
        id=canary_id,
        version=base_timeline.version,
        domain=base_timeline.domain,
        track=base_timeline.track,
        difficulty=base_timeline.difficulty,
        detection_mode=base_timeline.detection_mode,
        actors=base_timeline.actors,
        initial_state=base_timeline.initial_state,
        events=modified_events,
    )

    return CanaryTimeline(
        id=canary_id,
        canary_phrase=canary_phrase,
        timeline=canary_timeline,
    )


# =============================================================================
# Split Manager
# =============================================================================

class SplitManager:
    """Manage train/dev/test/hidden splits for StateBench.

    Provides reproducible splitting with:
    - Stratified splits by track
    - Hidden split with canaries for contamination detection
    - Quarterly refresh mechanism
    - Split verification
    """

    def __init__(
        self,
        version: str = "1.0",
        config: SplitConfig | None = None,
    ):
        self.version = version
        self.config = config or SplitConfig()
        self._rng: random.Random | None = None

    def _get_rng(self, seed: int) -> random.Random:
        """Get a seeded random generator."""
        if self._rng is None or True:  # Always create fresh for reproducibility
            self._rng = random.Random(seed)
        return self._rng

    def create_splits(
        self,
        timelines: list[Timeline],
        seed: int = 42,
        stratify_by_track: bool = True,
    ) -> dict[str, list[Timeline]]:
        """Create reproducible train/dev/test/hidden splits.

        Args:
            timelines: List of timelines to split
            seed: Random seed for reproducibility
            stratify_by_track: If True, maintain track proportions in each split

        Returns:
            Dict mapping split name to list of timelines
        """
        rng = self._get_rng(seed)

        if stratify_by_track:
            return self._stratified_split(timelines, rng)
        else:
            return self._random_split(timelines, rng)

    def _random_split(
        self,
        timelines: list[Timeline],
        rng: random.Random,
    ) -> dict[str, list[Timeline]]:
        """Simple random split."""
        shuffled = list(timelines)
        rng.shuffle(shuffled)

        n = len(shuffled)
        splits: dict[str, list[Timeline]] = {}

        idx = 0
        for split_name in ["train", "dev", "test", "hidden"]:
            ratio = getattr(self.config, split_name)
            count = int(n * ratio)
            # Ensure hidden gets at least what's left
            if split_name == "hidden":
                count = n - idx
            splits[split_name] = shuffled[idx:idx + count]
            idx += count

        return splits

    def _stratified_split(
        self,
        timelines: list[Timeline],
        rng: random.Random,
    ) -> dict[str, list[Timeline]]:
        """Stratified split maintaining track proportions."""
        # Group by track
        by_track: dict[str, list[Timeline]] = {}
        for tl in timelines:
            tl_track = tl.track
            if tl_track not in by_track:
                by_track[tl_track] = []
            by_track[tl_track].append(tl)

        # Split each track
        splits: dict[str, list[Timeline]] = {
            "train": [],
            "dev": [],
            "test": [],
            "hidden": [],
        }

        for track, track_timelines in by_track.items():
            track_shuffled = list(track_timelines)
            rng.shuffle(track_shuffled)

            n = len(track_shuffled)
            idx = 0

            for split_name in ["train", "dev", "test", "hidden"]:
                ratio = getattr(self.config, split_name)
                count = max(1, int(n * ratio)) if n > 4 else 1
                if split_name == "hidden":
                    count = n - idx  # Rest goes to hidden
                if idx + count > n:
                    count = n - idx

                splits[split_name].extend(track_shuffled[idx:idx + count])
                idx += count

        # Shuffle final splits
        for split_name in splits:
            rng.shuffle(splits[split_name])

        return splits

    def add_canaries(
        self,
        hidden_split: list[Timeline],
        n_canaries: int = 10,
        seed: int = 42,
    ) -> tuple[list[Timeline], list[CanaryTimeline]]:
        """Add canary items to hidden split for contamination detection.

        Args:
            hidden_split: The hidden split timelines
            n_canaries: Number of canaries to add
            seed: Random seed for reproducibility

        Returns:
            Tuple of (augmented split, list of canary objects)
        """
        rng = self._get_rng(seed)

        # Select random timelines to convert to canaries
        n_canaries = min(n_canaries, len(hidden_split))
        canary_bases = rng.sample(hidden_split, n_canaries)

        canaries: list[CanaryTimeline] = []
        augmented = list(hidden_split)

        for base in canary_bases:
            canary = create_canary_timeline(base)
            canaries.append(canary)
            augmented.append(canary.timeline)

        rng.shuffle(augmented)
        return augmented, canaries

    def refresh_hidden_split(
        self,
        current_hidden: list[Timeline],
        new_timelines: list[Timeline],
        refresh_ratio: float = 0.25,
        seed: int = 42,
    ) -> list[Timeline]:
        """Quarterly refresh of hidden split.

        Replaces a portion of the hidden split with new timelines
        to prevent overfitting to a static hidden set.

        Args:
            current_hidden: Current hidden split
            new_timelines: Pool of new timelines to sample from
            refresh_ratio: Fraction of hidden split to refresh (default 25%)
            seed: Random seed

        Returns:
            Updated hidden split
        """
        rng = self._get_rng(seed)

        n_to_replace = int(len(current_hidden) * refresh_ratio)

        # Remove random items from current
        keep_indices = set(range(len(current_hidden)))
        remove_indices = set(rng.sample(list(keep_indices), n_to_replace))
        keep_indices -= remove_indices

        refreshed = [current_hidden[i] for i in sorted(keep_indices)]

        # Add new items
        new_samples = rng.sample(new_timelines, min(n_to_replace, len(new_timelines)))
        refreshed.extend(new_samples)

        rng.shuffle(refreshed)
        return refreshed

    def compute_split_hash(self, split: list[Timeline]) -> str:
        """Compute a hash of a split for verification.

        Used to verify split integrity and detect modifications.
        """
        ids = sorted([tl.id for tl in split])
        content = json.dumps(ids, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def save_splits(
        self,
        splits: dict[str, list[Timeline]],
        output_dir: Path,
        seed: int,
        canaries: list[CanaryTimeline] | None = None,
    ) -> SplitMetadata:
        """Save splits to disk.

        Args:
            splits: Dict of split name to timelines
            output_dir: Directory to save to
            seed: Seed used to create splits
            canaries: Optional list of canary objects

        Returns:
            SplitMetadata object
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        counts: dict[str, int] = {}

        for split_name, timelines in splits.items():
            output_path = output_dir / f"{split_name}.jsonl"
            with open(output_path, "w") as f:
                for tl in timelines:
                    f.write(tl.model_dump_json() + "\n")
            counts[split_name] = len(timelines)

        # Save metadata
        canary_ids = [c.id for c in canaries] if canaries else []
        metadata = SplitMetadata(
            version=self.version,
            created_at=datetime.now().isoformat(),
            seed=seed,
            config=self.config.as_dict(),
            counts=counts,
            canary_count=len(canary_ids),
            canary_ids=canary_ids,
        )

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({
                "version": metadata.version,
                "created_at": metadata.created_at,
                "seed": metadata.seed,
                "config": metadata.config,
                "counts": metadata.counts,
                "canary_count": metadata.canary_count,
                # Don't save canary IDs in public metadata!
            }, f, indent=2)

        # Save canary registry separately (private)
        if canaries:
            canary_registry_path = output_dir / ".canary_registry.json"
            canary_data = [
                {"id": c.id, "phrase": c.canary_phrase}
                for c in canaries
            ]
            with open(canary_registry_path, "w") as f:
                json.dump(canary_data, f, indent=2)

        return metadata

    def load_split(self, split_path: Path) -> list[Timeline]:
        """Load a split from disk."""
        timelines: list[Timeline] = []
        with open(split_path) as f:
            for line in f:
                if line.strip():
                    timelines.append(Timeline.model_validate_json(line))
        return timelines

    def verify_split(
        self,
        split: list[Timeline],
        expected_hash: str,
    ) -> bool:
        """Verify a split's integrity."""
        actual_hash = self.compute_split_hash(split)
        return actual_hash == expected_hash


# =============================================================================
# Contamination Detection
# =============================================================================

def check_contamination(
    responses: list[str],
    canary_registry_path: Path,
) -> dict[str, list[str]]:
    """Check responses for canary contamination.

    Args:
        responses: List of model responses to check
        canary_registry_path: Path to canary registry JSON

    Returns:
        Dict mapping canary phrase to list of responses containing it
    """
    with open(canary_registry_path) as f:
        canaries = json.load(f)

    contamination: dict[str, list[str]] = {}

    for canary in canaries:
        phrase = canary["phrase"]
        matches = [r for r in responses if phrase.lower() in r.lower()]
        if matches:
            contamination[phrase] = matches

    return contamination


def format_contamination_report(
    contamination: dict[str, list[str]],
) -> str:
    """Format contamination check results as markdown."""
    if not contamination:
        return "## Contamination Check: PASSED\n\nNo canary phrases detected."

    lines = [
        "## Contamination Check: FAILED",
        "",
        f"**{len(contamination)} canary phrases detected in responses!**",
        "",
        "This indicates the model may have been trained on the hidden split.",
        "",
        "### Detected Canaries",
        "",
    ]

    for phrase, matches in contamination.items():
        lines.append(f"- `{phrase}`: {len(matches)} occurrence(s)")

    return "\n".join(lines)
