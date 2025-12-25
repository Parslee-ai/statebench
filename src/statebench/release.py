"""Canonical benchmark release generation.

Creates versioned, reproducible benchmark datasets with train/dev/test splits
and cryptographic hashes for verification.
"""

import hashlib
import json
import random
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from statebench.generator.engine import TimelineGenerator
from statebench.schema.timeline import Timeline


class SplitConfig(TypedDict):
    train: float
    dev: float
    test: float


class ReleaseVersionConfig(TypedDict):
    seed: int
    tracks: list[str]
    count_per_track: int
    splits: SplitConfig
    adversarial_ratio: float
    include_adversarial_slice: bool


# Canonical release configuration
RELEASE_CONFIG: dict[str, ReleaseVersionConfig] = {
    "v0.1": {
        "seed": 20251221,  # Fixed seed for reproducibility
        "tracks": [
            "supersession",
            "commitment_durability",
            "interruption_resumption",
            "scope_permission",
            "environmental_freshness",
        ],
        "count_per_track": 100,  # 500 total
        "splits": {
            "train": 0.6,  # 60% = 300 timelines
            "dev": 0.2,    # 20% = 100 timelines
            "test": 0.2,   # 20% = 100 timelines
        },
        "adversarial_ratio": 0.3,
        "include_adversarial_slice": True,  # Add dedicated adversarial cases
    },
    "v0.2": {
        "seed": 20251221,  # Fixed seed for reproducibility
        "tracks": [
            "causality",
            "hallucination_resistance",
            "repair_propagation",
            "scope_leak",
            "brutal_realistic",
        ],
        "count_per_track": 100,  # 500 total
        "splits": {
            "train": 0.6,  # 60% = 300 timelines
            "dev": 0.2,    # 20% = 100 timelines
            "test": 0.2,   # 20% = 100 timelines
        },
        "adversarial_ratio": 0.3,
        "include_adversarial_slice": True,
    },
}


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def generate_release(
    version: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate a canonical benchmark release.

    Args:
        version: Release version (e.g., "v0.1")
        output_dir: Directory to write release files

    Returns:
        Release manifest with hashes and metadata
    """
    if version not in RELEASE_CONFIG:
        raise ValueError(f"Unknown version: {version}. Available: {list(RELEASE_CONFIG.keys())}")

    config = RELEASE_CONFIG[version]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all timelines
    generator = TimelineGenerator(seed=config["seed"])
    all_timelines: list[Timeline] = []

    for track in config["tracks"]:
        for timeline in generator.generate_track(
            track,
            count=config["count_per_track"],
            adversarial_ratio=config["adversarial_ratio"],
        ):
            all_timelines.append(timeline)

    # Shuffle with fixed seed for split assignment
    rng = random.Random(config["seed"])
    rng.shuffle(all_timelines)

    # Split into train/dev/test
    total = len(all_timelines)
    train_end = int(total * config["splits"]["train"])
    dev_end = train_end + int(total * config["splits"]["dev"])

    splits = {
        "train": all_timelines[:train_end],
        "dev": all_timelines[train_end:dev_end],
        "test": all_timelines[dev_end:],
    }

    # Write split files and compute hashes
    splits_manifest: dict[str, dict[str, Any]] = {}
    manifest: dict[str, Any] = {
        "version": version,
        "created": datetime.utcnow().isoformat() + "Z",
        "seed": config["seed"],
        "total_timelines": total,
        "tracks": config["tracks"],
        "count_per_track": config["count_per_track"],
        "adversarial_ratio": config["adversarial_ratio"],
        "splits": splits_manifest,
    }

    for split_name, timelines in splits.items():
        # Write JSONL file
        file_path = output_dir / f"{split_name}.jsonl"
        with open(file_path, "w") as f:
            for timeline in timelines:
                f.write(timeline.model_dump_json() + "\n")

        # Compute hash
        file_hash = compute_file_hash(file_path)

        # Track distribution
        track_counts: dict[str, int] = {}
        for t in timelines:
            track_counts[t.track] = track_counts.get(t.track, 0) + 1

        splits_manifest[split_name] = {
            "file": f"{split_name}.jsonl",
            "count": len(timelines),
            "sha256": file_hash,
            "tracks": track_counts,
        }

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def verify_release(release_dir: Path) -> tuple[bool, list[str]]:
    """Verify a benchmark release against its manifest.

    Args:
        release_dir: Directory containing release files

    Returns:
        Tuple of (all_valid, list of error messages)
    """
    manifest_path = release_dir / "manifest.json"
    if not manifest_path.exists():
        return False, ["manifest.json not found"]

    with open(manifest_path) as f:
        manifest = json.load(f)

    errors = []
    for split_name, split_info in manifest["splits"].items():
        file_path = release_dir / split_info["file"]
        if not file_path.exists():
            errors.append(f"{split_info['file']} not found")
            continue

        actual_hash = compute_file_hash(file_path)
        if actual_hash != split_info["sha256"]:
            errors.append(
                f"{split_info['file']}: hash mismatch. "
                f"Expected {split_info['sha256'][:16]}..., got {actual_hash[:16]}..."
            )

        # Verify line count
        with open(file_path) as f:
            line_count = sum(1 for line in f if line.strip())
        if line_count != split_info["count"]:
            errors.append(
                f"{split_info['file']}: count mismatch. "
                f"Expected {split_info['count']}, got {line_count}"
            )

    return len(errors) == 0, errors


def load_split(release_dir: Path, split: str) -> Iterator[Timeline]:
    """Load a specific split from a release.

    Args:
        release_dir: Directory containing release files
        split: Split name ("train", "dev", or "test")

    Yields:
        Timeline objects
    """
    file_path = release_dir / f"{split}.jsonl"
    if not file_path.exists():
        raise FileNotFoundError(f"Split file not found: {file_path}")

    with open(file_path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                yield Timeline.model_validate(data)
