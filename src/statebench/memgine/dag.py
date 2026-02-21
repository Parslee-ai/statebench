"""Summary DAG: hierarchical summaries with provenance pointers.

Per-layer subtrees. Leaf nodes compress spans of StoreEntries.
Condensed nodes compress other SummaryNodes.
Lossless expansion via recursive pointer following.
"""

from __future__ import annotations

from statebench.memgine.store import ImmutableStore
from statebench.memgine.types import EntryId, Layer, NodeId, StoreEntry, SummaryNode


class SummaryDAG:
    """Hierarchical summary DAG with per-layer subtrees."""

    def __init__(self, store: ImmutableStore) -> None:
        self._store = store
        self._nodes: list[SummaryNode] = []
        self._by_layer: dict[Layer, list[NodeId]] = {1: [], 2: [], 3: [], 4: []}

    def create_leaf(
        self,
        layer: Layer,
        summary_text: str,
        entry_ids: list[EntryId],
        *,
        covered_fact_ids: list[str] | None = None,
        all_superseded: bool = False,
    ) -> SummaryNode:
        """Create a leaf summary node from store entries."""
        node_id = len(self._nodes)
        node = SummaryNode(
            id=node_id,
            layer=layer,
            summary_text=summary_text,
            entry_ids=tuple(entry_ids),
            covered_fact_ids=tuple(covered_fact_ids or []),
            all_superseded=all_superseded,
        )
        self._nodes.append(node)
        self._by_layer[layer].append(node_id)
        return node

    def create_condensed(
        self,
        layer: Layer,
        summary_text: str,
        child_node_ids: list[NodeId],
        *,
        covered_fact_ids: list[str] | None = None,
        all_superseded: bool = False,
    ) -> SummaryNode:
        """Create a condensed summary node from child summary nodes."""
        node_id = len(self._nodes)
        node = SummaryNode(
            id=node_id,
            layer=layer,
            summary_text=summary_text,
            child_node_ids=tuple(child_node_ids),
            covered_fact_ids=tuple(covered_fact_ids or []),
            all_superseded=all_superseded,
        )
        self._nodes.append(node)
        self._by_layer[layer].append(node_id)
        return node

    def get(self, node_id: NodeId) -> SummaryNode:
        """Get a summary node by ID."""
        return self._nodes[node_id]

    def get_by_layer(self, layer: Layer) -> list[SummaryNode]:
        """Get all summary nodes for a layer."""
        return [self._nodes[nid] for nid in self._by_layer[layer]]

    def expand(self, node_id: NodeId) -> list[EntryId]:
        """Lossless expansion: recursively resolve to original entry IDs."""
        node = self._nodes[node_id]
        if node.is_leaf:
            return list(node.entry_ids)

        # Recursive expansion of child nodes
        result: list[EntryId] = []
        for child_id in node.child_node_ids:
            result.extend(self.expand(child_id))
        return result

    def expand_to_entries(self, node_id: NodeId) -> list[StoreEntry]:
        """Expand and resolve to actual StoreEntry objects."""
        entry_ids = self.expand(node_id)
        return [self._store.get(eid) for eid in entry_ids]

    def remove_nodes(self, node_ids: set[NodeId]) -> None:
        """Mark nodes as removed (replaced by None placeholder to preserve IDs)."""
        for nid in node_ids:
            layer = self._nodes[nid].layer
            if nid in self._by_layer[layer]:
                self._by_layer[layer].remove(nid)

    def clear(self) -> None:
        """Reset the DAG."""
        self._nodes.clear()
        for layer in self._by_layer:
            self._by_layer[layer].clear()
