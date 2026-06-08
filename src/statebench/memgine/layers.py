"""Layer State: semantic index over the immutable store.

Tracks validity, supersession chains, dependencies, and constraints.
Does NOT own raw data — only maintains pointers (EntryIds) back to the store.

Key insight: StoreEntry is frozen. Mutable state (validity, superseded_by)
lives here in indices, preserving immutability of the store.
"""

from __future__ import annotations

from statebench.memgine.types import EntryId
from statebench.schema.state import Source


class LayerState:
    """Semantic index over the store for all four layers."""

    def __init__(self) -> None:
        # Layer 1: Identity entry (at most one)
        self.identity_entry: EntryId | None = None

        # Layer 2: Fact tracking
        # fact_id -> current valid entry id
        self.valid_facts: dict[str, EntryId] = {}
        # fact_key -> ordered list of entry ids (supersession chain)
        self.supersession_chains: dict[str, list[EntryId]] = {}
        # entry_id -> validity
        self.validity: dict[EntryId, bool] = {}
        # fact_id -> set of fact_ids that depend on it
        self.dependents: dict[str, set[str]] = {}
        # fact_id -> set of fact_ids it depends on
        self.dependencies: dict[str, set[str]] = {}
        # fact_ids that are constraints (never compacted)
        self.constraints: set[str] = set()
        # fact_id -> fact_id that superseded it
        self.superseded_by: dict[str, str] = {}

        # Layer 3: Working set entry ids (ordered by insertion)
        self.working_set_entries: list[EntryId] = []

        # Layer 4: Environment entry ids, keyed by signal key
        self.environment_entries: dict[str, EntryId] = {}

    def register_identity(self, entry_id: EntryId) -> None:
        """Register the identity entry."""
        self.identity_entry = entry_id

    def register_fact(
        self,
        entry_id: EntryId,
        fact_id: str,
        fact_key: str,
        *,
        is_constraint: bool = False,
        depends_on: tuple[str, ...] = (),
    ) -> None:
        """Register a new fact entry."""
        self.valid_facts[fact_id] = entry_id
        self.validity[entry_id] = True

        # Add to supersession chain
        if fact_key not in self.supersession_chains:
            self.supersession_chains[fact_key] = []
        self.supersession_chains[fact_key].append(entry_id)

        if is_constraint:
            self.constraints.add(fact_id)

        # Track dependencies
        self.dependencies[fact_id] = set(depends_on)
        if fact_id not in self.dependents:
            self.dependents[fact_id] = set()
        for dep_id in depends_on:
            if dep_id not in self.dependents:
                self.dependents[dep_id] = set()
            self.dependents[dep_id].add(fact_id)

    def invalidate_fact(self, fact_id: str, superseded_by_id: str) -> set[str]:
        """Invalidate a fact and cascade to dependents.

        Returns the set of all fact_ids that were invalidated (including cascaded).
        """
        invalidated: set[str] = set()
        self._invalidate_recursive(fact_id, superseded_by_id, invalidated)
        return invalidated

    def _invalidate_recursive(
        self, fact_id: str, superseded_by_id: str, invalidated: set[str]
    ) -> None:
        """Recursively invalidate a fact and its dependents."""
        if fact_id in invalidated:
            return
        if fact_id not in self.valid_facts:
            return

        entry_id = self.valid_facts.pop(fact_id)
        self.validity[entry_id] = False
        self.superseded_by[fact_id] = superseded_by_id
        invalidated.add(fact_id)

        # Type II propagation (CUPMem): a fact whose premise just died is stale —
        # and so is anything derived from IT, transitively. Dependents are NOT
        # hard-invalidated (their value needs RECALCULATION from the new base, not
        # deletion); they're flagged (returned -> engine needs_review). We walk the
        # full transitive dependent closure so multi-hop derived chains
        # (D2 <- D1 <- F) all get flagged, not just the direct dependent D1.
        #
        # Direct (1-hop) dependents whose base was replaced render inline as
        # "RECALCULATE" under the correcting parent; deeper hops (whose immediate
        # premise was flagged, not superseded) surface in the orphan-invalidated
        # section instead. Either way they are visible, not silently dropped.
        self._propagate_stale(fact_id, invalidated)

    def _propagate_stale(self, fact_id: str, invalidated: set[str]) -> None:
        """Flag the transitive dependent closure of ``fact_id`` as recalc-pending.

        Iterative (worklist) rather than recursive: ``invalidate_fact`` is the
        shared cascade for ALL supersession, and a pathologically deep derived
        chain must not blow the Python recursion limit.
        """
        stack = [fact_id]
        while stack:
            for dependent_id in self.dependents.get(stack.pop(), set()):
                if dependent_id in self.valid_facts and dependent_id not in invalidated:
                    invalidated.add(dependent_id)
                    stack.append(dependent_id)

    def register_working_set(self, entry_id: EntryId) -> None:
        """Register a working set entry."""
        self.working_set_entries.append(entry_id)

    def register_environment(self, key: str, entry_id: EntryId) -> None:
        """Register an environment entry, replacing any previous entry for the same key."""
        self.environment_entries[key] = entry_id

    def get_valid_fact_ids(self) -> set[str]:
        """Get all currently valid fact IDs."""
        return set(self.valid_facts.keys())

    def get_superseded_fact_ids(self) -> set[str]:
        """Get all superseded fact IDs."""
        return set(self.superseded_by.keys())

    def is_fact_valid(self, fact_id: str) -> bool:
        """Check if a fact is currently valid."""
        return fact_id in self.valid_facts

    def get_chain_for_key(self, fact_key: str) -> list[EntryId]:
        """Get the full supersession chain for a fact key."""
        return list(self.supersession_chains.get(fact_key, []))

    def clear(self) -> None:
        """Reset all indices."""
        self.identity_entry = None
        self.valid_facts.clear()
        self.supersession_chains.clear()
        self.validity.clear()
        self.dependents.clear()
        self.dependencies.clear()
        self.constraints.clear()
        self.superseded_by.clear()
        self.working_set_entries.clear()
        self.environment_entries.clear()


def is_constraint(value: str, source: Source) -> bool:
    """Detect if a fact is a formal constraint.

    Ported from StateBasedStrategy._is_constraint.
    """
    # Don't mark corrections, updates, or conversation snippets as constraints
    if any(
        marker in value
        for marker in [
            "INVALIDATED",
            "CORRECTION",
            "delayed",
            "changed",
            "let's go with",
            "just fyi",
            "btw",
            "hold on",
            "wait",
            "best we can do",
            "fine,",
            "bad news",
            "good news",
        ]
    ):
        return False

    value_lower = value.lower()
    formal_indicators = [
        "must",
        "require",
        "policy",
        "limit is",
        "maximum is",
        "minimum is",
        "cannot exceed",
        "not allowed",
        "prohibited",
        "mandatory",
        "approval required",
        "needs approval",
        "authority to",
    ]
    has_formal = any(ind in value_lower for ind in formal_indicators)

    if source.type == "policy":
        return True
    if source.authority in ("policy", "executive") and has_formal:
        return True
    return has_formal


def infer_scope(text: str) -> str:
    """Infer scope from conversation text.

    Ported from StateBasedStrategy._infer_scope.
    """
    text_lower = text.lower()

    if any(
        phrase in text_lower
        for phrase in [
            "what if",
            "hypothetically",
            "suppose",
            "imagine",
            "let's say",
            "in theory",
            "potentially",
        ]
    ):
        return "hypothetical"
    if any(
        phrase in text_lower
        for phrase in [
            "draft",
            "preliminary",
            "not final",
            "pending",
            "proposal",
            "tentative",
        ]
    ):
        return "draft"
    if any(
        phrase in text_lower
        for phrase in ["for this task", "for this project", "just for this", "only for"]
    ):
        return "task"
    return "global"


def infer_constraint_type(value: str) -> str | None:
    """Infer the type of constraint.

    Ported from StateBasedStrategy._infer_constraint_type.
    """
    value_lower = value.lower()
    if any(kw in value_lower for kw in ["budget", "$", "cost", "price", "spend"]):
        return "budget"
    if any(kw in value_lower for kw in ["deadline", "due", "by", "before", "until"]):
        return "deadline"
    if any(
        kw in value_lower
        for kw in ["capacity", "available", "team", "resource", "hours"]
    ):
        return "capacity"
    if any(
        kw in value_lower
        for kw in ["policy", "require", "must", "approval", "authority"]
    ):
        return "policy"
    return None
