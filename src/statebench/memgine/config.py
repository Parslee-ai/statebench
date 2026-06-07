"""Configuration for the Memgine engine.

MemgineConfig: Per-layer thresholds, token budgets, and compaction parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LayerBudget:
    """Token budget allocation for a single layer."""

    fraction: float  # Fraction of total budget (0.0 - 1.0)

    def tokens(self, total: int) -> int:
        return int(total * self.fraction)


@dataclass(frozen=True)
class CompactionThresholds:
    """Compaction trigger thresholds (fraction of budget)."""

    soft: float = 0.70  # Start compaction
    hard: float = 0.95  # Blocking compaction


@dataclass
class MemgineConfig:
    """Configuration for the Memgine engine."""

    # Total token budget
    token_budget: int = 8000

    # Per-layer budget fractions (must sum to 1.0)
    layer1_budget: LayerBudget = field(default_factory=lambda: LayerBudget(fraction=0.05))
    layer2_budget: LayerBudget = field(default_factory=lambda: LayerBudget(fraction=0.50))
    layer3_budget: LayerBudget = field(default_factory=lambda: LayerBudget(fraction=0.30))
    layer4_budget: LayerBudget = field(default_factory=lambda: LayerBudget(fraction=0.15))

    # Compaction thresholds
    thresholds: CompactionThresholds = field(default_factory=CompactionThresholds)

    # Working set limits
    working_set_max: int = 10  # Max items before compaction triggers
    working_set_keep_recent: int = 3  # Items to keep at Level 2

    # Environment limits
    environment_max: int = 5

    # Supersession rendering: when True, show "(changed from: <old value>)"
    # annotations. These aid strong models but inject the superseded value into
    # context, which weaker models tend to resurrect (raising SFRR). Set False
    # to replace the old value with a value-free "(value updated)" marker.
    show_superseded_values: bool = True

    def layer_tokens(self, layer: int) -> int:
        """Get token budget for a layer."""
        budgets = {
            1: self.layer1_budget,
            2: self.layer2_budget,
            3: self.layer3_budget,
            4: self.layer4_budget,
        }
        return budgets[layer].tokens(self.token_budget)

    def with_layer_weights(self, weights: dict[int, float]) -> "MemgineConfig":
        """Return a new config with adjusted layer budget fractions.

        Used by query-complexity routing to adapt budgets per query type.
        The weights dict maps layer (1-4) to fraction (must sum to 1.0).
        """
        return MemgineConfig(
            token_budget=self.token_budget,
            layer1_budget=LayerBudget(fraction=weights.get(1, self.layer1_budget.fraction)),
            layer2_budget=LayerBudget(fraction=weights.get(2, self.layer2_budget.fraction)),
            layer3_budget=LayerBudget(fraction=weights.get(3, self.layer3_budget.fraction)),
            layer4_budget=LayerBudget(fraction=weights.get(4, self.layer4_budget.fraction)),
            thresholds=self.thresholds,
            working_set_max=self.working_set_max,
            working_set_keep_recent=self.working_set_keep_recent,
            environment_max=self.environment_max,
            show_superseded_values=self.show_superseded_values,
        )

    def validate(self) -> None:
        """Check configuration is valid."""
        total = (
            self.layer1_budget.fraction
            + self.layer2_budget.fraction
            + self.layer3_budget.fraction
            + self.layer4_budget.fraction
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Layer budget fractions must sum to 1.0, got {total}")
        if self.thresholds.soft >= self.thresholds.hard:
            raise ValueError("Soft threshold must be less than hard threshold")
