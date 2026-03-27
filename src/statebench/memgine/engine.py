"""Memgine Engine: main orchestrator.

Ties all components together: ImmutableStore, LayerState, SummaryDAG,
CompactionEngine. Provides typed ingest methods and context assembly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import tiktoken

from statebench.baselines.base import ContextResult, FactMetadata
from statebench.confidence import confidence_annotation, infer_confidence
from statebench.constraint_checker import (
    FactValue as ConstraintFactValue,
    check_constraints,
    format_constraint_checklist,
)
from statebench.query_classifier import QueryComplexity, classify_query
from statebench.memgine.compaction import CompactionEngine
from statebench.memgine.config import MemgineConfig
from statebench.memgine.dag import SummaryDAG
from statebench.memgine.layers import (
    LayerState,
    infer_constraint_type,
    infer_scope,
    is_constraint,
)
from statebench.memgine.store import ImmutableStore
from statebench.memgine.types import StoreEntry
from statebench.schema.state import (
    IdentityRole,
    Scope,
    Source,
)
from statebench.schema.timeline import (
    ConversationTurn,
    Event,
    InitialState,
    StateWrite,
    Supersession,
)

# Type alias for memory types (from state_based.py)
MemoryType = Literal["user", "capability", "organizational"]


class MemgineEngine:
    """Main orchestrator for the Memgine memory engine.

    Combines LCM's architectural patterns with StateBench's semantic layer awareness.
    """

    def __init__(self, config: MemgineConfig | None = None) -> None:
        self._config = config or MemgineConfig()
        self._config.validate()

        self._store = ImmutableStore()
        self._layers = LayerState()
        self._dag = SummaryDAG(self._store)
        self._compactor = CompactionEngine(
            self._config, self._store, self._layers, self._dag
        )

        # Mutable metadata tracked alongside immutable store
        self._needs_review: set[str] = set()  # fact_ids needing review
        self._corrections: list[dict[str, str | datetime]] = []
        self._known_unknowns: dict[str, datetime] = {}
        self._fact_id_counts: dict[str, int] = {}  # for deduplicating collisions
        self._last_event_ts: datetime | None = None
        self._user_role: str | None = None  # from identity, for access control

        self._encoder = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------ #
    # Typed Ingest API
    # ------------------------------------------------------------------ #

    def ingest_identity(self, identity: IdentityRole) -> StoreEntry:
        """Ingest a Layer 1 identity entry."""
        self._user_role = identity.authority
        lines = [f"User: {identity.user_name}", f"Role: {identity.authority}"]
        if identity.department:
            lines.append(f"Department: {identity.department}")
        if identity.organization:
            lines.append(f"Organization: {identity.organization}")
        text = "\n".join(lines)

        entry = self._store.append(
            layer=1,
            kind="identity",
            key="identity",
            value=text,
            ts=datetime.min,
        )
        self._layers.register_identity(entry.id)
        return entry

    def ingest_fact(
        self,
        fact_id: str,
        key: str,
        value: str,
        source: Source,
        ts: datetime,
        *,
        scope: Scope = "global",
        supersedes: str | None = None,
        depends_on: list[str] | None = None,
        is_constraint_flag: bool = False,
        constraint_type: str | None = None,
    ) -> StoreEntry:
        """Ingest a Layer 2 persistent fact."""
        deps = tuple(depends_on or [])

        # Deduplicate fact_id collisions (e.g. multiple W-AUTO writes)
        if fact_id in self._fact_id_counts:
            self._fact_id_counts[fact_id] += 1
            fact_id = f"{fact_id}-{self._fact_id_counts[fact_id]}"
        else:
            self._fact_id_counts[fact_id] = 1

        # Detect constraint
        effective_constraint = is_constraint_flag or is_constraint(value, source)
        effective_constraint_type = constraint_type or (
            infer_constraint_type(value) if effective_constraint else None
        )

        # Handle supersession
        if supersedes:
            # Resolve key-based supersession: if supersedes is a key name
            # (not a fact_id), look up the current fact_id for that key
            resolved_supersedes = supersedes
            if supersedes not in self._layers.valid_facts:
                # Try to resolve as a key via supersession chains
                if supersedes in self._layers.supersession_chains:
                    chain = self._layers.supersession_chains[supersedes]
                    if chain:
                        # Find the latest valid entry in the chain
                        for eid in reversed(chain):
                            entry_check = self._store.get(eid)
                            if entry_check.fact_id and self._layers.is_fact_valid(
                                entry_check.fact_id
                            ):
                                resolved_supersedes = entry_check.fact_id
                                break
                else:
                    # Search by key across all valid facts
                    for fid in list(self._layers.get_valid_fact_ids()):
                        fentry = self._store.get_by_fact_id(fid)
                        if fentry and fentry.key == supersedes:
                            resolved_supersedes = fid
                            break

            invalidated = self._layers.invalidate_fact(resolved_supersedes, fact_id)
            # Track cascaded invalidations for review
            for inv_id in invalidated:
                if inv_id != resolved_supersedes:
                    self._needs_review.add(inv_id)

            # Record correction
            old_entry = self._store.get_by_fact_id(resolved_supersedes)
            if old_entry:
                self._corrections.append({
                    "old_fact_id": resolved_supersedes,
                    "old_value": old_entry.value,
                    "new_value": value,
                    "ts": ts,
                })

        entry = self._store.append(
            layer=2,
            kind="fact",
            key=key,
            value=value,
            ts=ts,
            fact_id=fact_id,
            source_type=source.type,
            source_identity=source.identity,
            authority=source.authority,
            scope=scope,
            supersedes_fact_id=supersedes,
            depends_on=deps,
            is_constraint=effective_constraint,
            constraint_type=effective_constraint_type,
        )

        self._layers.register_fact(
            entry.id,
            fact_id,
            key,
            is_constraint=effective_constraint,
            depends_on=deps,
        )

        return entry

    def ingest_conversation(
        self, speaker: str, text: str, ts: datetime
    ) -> StoreEntry:
        """Ingest a Layer 3 conversation turn."""
        entry = self._store.append(
            layer=3,
            kind="conversation",
            key=speaker,
            value=f"{speaker.title()}: {text}",
            ts=ts,
        )
        self._layers.register_working_set(entry.id)

        # Track known unknowns — but skip scoped entries (brainstorm, what-if,
        # draft, sandbox) to avoid leaking hypothetical terms into context
        text_lower = text.lower()
        is_scoped = "[scope:" in text_lower or any(
            phrase in text_lower
            for phrase in [
                "hypothetically", "what if", "brainstorm", "draft",
                "sandbox", "just exploratory", "what about",
            ]
        )
        if not is_scoped and (
            "?" in text
            or any(
                phrase in text_lower
                for phrase in ["need info", "don't know", "not sure", "find out"]
            )
        ):
            self._known_unknowns[text] = ts

        return entry

    def ingest_environment(
        self,
        key: str,
        value: str,
        ts: datetime,
        *,
        expires: datetime | None = None,
        urgency: Literal["low", "medium", "high", "critical"] = "medium",
    ) -> StoreEntry:
        """Ingest a Layer 4 environment signal."""
        entry = self._store.append(
            layer=4,
            kind="environment",
            key=key,
            value=value,
            ts=ts,
            expires=expires,
            urgency=urgency,
        )
        self._layers.register_environment(key, entry.id)
        return entry

    # ------------------------------------------------------------------ #
    # StateBench Bridge
    # ------------------------------------------------------------------ #

    def ingest_initial_state(self, initial_state: InitialState) -> None:
        """Bridge: ingest StateBench InitialState."""
        self.ingest_identity(initial_state.identity_role)

        for fact in initial_state.persistent_facts:
            entry = self.ingest_fact(
                fact_id=fact.id,
                key=fact.key,
                value=fact.value,
                source=fact.source,
                ts=fact.ts,
                scope=fact.scope,
                depends_on=list(fact.depends_on),
                is_constraint_flag=fact.is_constraint,
                constraint_type=fact.constraint_type,
            )
            # Handle pre-superseded facts
            if not fact.is_valid:
                self._layers.validity[entry.id] = False
                if fact.id in self._layers.valid_facts:
                    del self._layers.valid_facts[fact.id]
                if fact.superseded_by:
                    self._layers.superseded_by[fact.id] = fact.superseded_by

        for item in initial_state.working_set:
            self.ingest_conversation("context", item.content, item.ts)

        for key, value in initial_state.environment.items():
            self.ingest_environment(key, str(value), datetime.min)

    def ingest_statebench_event(self, event: Event) -> None:
        """Bridge: ingest a StateBench Event."""
        # Track latest event timestamp for environment freshness
        if hasattr(event, "ts"):
            if self._last_event_ts is None or event.ts > self._last_event_ts:
                self._last_event_ts = event.ts

        if isinstance(event, ConversationTurn):
            self.ingest_conversation(event.speaker, event.text, event.ts)

        elif isinstance(event, StateWrite):
            for write in event.writes:
                if write.layer == "persistent_facts":
                    self.ingest_fact(
                        fact_id=write.id,
                        key=write.key,
                        value=write.value,
                        source=write.source,
                        ts=event.ts,
                        scope=write.scope,
                        depends_on=list(write.depends_on),
                        is_constraint_flag=write.is_constraint,
                        constraint_type=write.constraint_type,
                    )
                elif write.layer == "environment":
                    self.ingest_environment(write.key, write.value, event.ts)

        elif isinstance(event, Supersession):
            for write in event.writes:
                if write.layer == "persistent_facts":
                    self.ingest_fact(
                        fact_id=write.id,
                        key=write.key,
                        value=write.value,
                        source=write.source,
                        ts=event.ts,
                        scope=write.scope,
                        supersedes=write.supersedes,
                        depends_on=list(write.depends_on),
                        is_constraint_flag=write.is_constraint,
                        constraint_type=write.constraint_type,
                    )
                elif write.layer == "environment":
                    self.ingest_environment(write.key, write.value, event.ts)

    # ------------------------------------------------------------------ #
    # Query API
    # ------------------------------------------------------------------ #

    def build_context(self, query: str) -> ContextResult:
        """Assemble context from all layers with full provenance.

        Section order:
        1. Identity (static)
        2. Constraints (framing — LLM reads these before evaluating facts)
        3. Facts (query-relevance sorted, with dependency annotations)
        4. Invalidated facts (repair propagation)
        5. Working Set (chronological — temporal order matters)
        6. Environment (freshness sorted)
        7. Known Unknowns (prevent hallucination, nearest to query)

        v2.0: Query-complexity routing adapts context assembly per query type.
        """
        classification = classify_query(query)

        # Adaptive token budgets: adjust layer fractions based on query type
        query_config = self._config.with_layer_weights(
            classification.suggested_layer_weights
        )
        # Use adapted config for compaction decisions (exception-safe)
        original_config = self._compactor._config
        self._compactor._config = query_config
        try:
            compaction_result = self._compactor.maybe_compact()
        finally:
            self._compactor._config = original_config

        parts: list[str] = []
        facts_included: list[FactMetadata] = []
        facts_excluded: list[FactMetadata] = []
        inclusion_reasons: dict[str, str] = {}
        query_tokens = self._extract_keywords(query)

        # --- Layer 1: Identity (always verbatim) ---
        if self._layers.identity_entry is not None:
            entry = self._store.get(self._layers.identity_entry)
            parts.append(f"## Identity\n{entry.value}")

        # --- Layer 2a: Constraints (before facts — framing for evaluation) ---
        constraint_entries = self._get_constraint_entries()
        if constraint_entries:
            lines = []
            for entry in sorted(constraint_entries, key=lambda e: e.ts):
                # Access control: exclude [RESTRICTED:] constraints
                if "[RESTRICTED:" in entry.value:
                    if entry.fact_id:
                        fm = self._entry_to_fact_metadata(entry, valid=True)
                        facts_excluded.append(fm)
                        inclusion_reasons[entry.fact_id] = "excluded: restricted (access control)"
                    continue
                ct = entry.constraint_type or "CONSTRAINT"
                display_key = self._display_key(entry.key) if entry.key else ""
                key_prefix = f"{display_key}: " if display_key else ""
                lines.append(f"⚠️ [{entry.fact_id}] [{ct}] {key_prefix}{entry.value}")
                fm = self._entry_to_fact_metadata(entry, valid=True)
                facts_included.append(fm)
                if entry.fact_id:
                    inclusion_reasons[entry.fact_id] = "active constraint"
            if lines:
                parts.append("## ⚠️ Active Constraints (CHECK ALL)\n" + "\n".join(lines))

        # --- Constraint pre-computation (for constraint-check queries) ---
        if constraint_entries and classification.complexity in (
            QueryComplexity.CONSTRAINT_CHECK,
            QueryComplexity.AGGREGATION,
        ):
            valid_non_constraints = self._get_valid_non_constraint_entries()
            c_fvs = [
                ConstraintFactValue(
                    fact_id=e.fact_id or "", key=e.key or "", value=e.value
                )
                for e in constraint_entries
                if "[RESTRICTED:" not in e.value
            ]
            f_fvs = [
                ConstraintFactValue(
                    fact_id=e.fact_id or "", key=e.key or "", value=e.value
                )
                for e in valid_non_constraints
            ]
            check_results = check_constraints(c_fvs, f_fvs)
            checklist = format_constraint_checklist(check_results)
            if checklist:
                parts.append(checklist)

        # --- Layer 2b: Valid non-constraint facts ---
        valid_entries = self._get_valid_non_constraint_entries()
        dag_layer2 = self._dag.get_by_layer(2)
        compacted_entry_ids = self._compactor.compacted_entries

        verbatim_entries = [e for e in valid_entries if e.id not in compacted_entry_ids]
        has_dag = len(dag_layer2) > 0

        fact_lines: list[str] = []
        if has_dag:
            for node in dag_layer2:
                if not node.all_superseded:
                    fact_lines.append(node.summary_text)
                    for fid in node.covered_fact_ids:
                        dag_entry = self._store.get_by_fact_id(fid)
                        if dag_entry:
                            fm = self._entry_to_fact_metadata(dag_entry, valid=True)
                            facts_included.append(fm)
                            inclusion_reasons[fid] = "valid fact (compacted)"

        # Verbatim entries — sorted by query relevance (most relevant last,
        # closest to the query in the prompt, exploiting LLM recency attention)
        invalidated_entries: list[StoreEntry] = []
        scorable_entries: list[tuple[float, StoreEntry]] = []
        for entry in verbatim_entries:
            is_invalidated = (
                (entry.fact_id and entry.fact_id in self._needs_review)
                or "[INVALIDATED" in entry.value
            )
            if is_invalidated:
                invalidated_entries.append(entry)
                continue
            # Access control: exclude [RESTRICTED:] facts the user can't see
            if "[RESTRICTED:" in entry.value and entry.fact_id:
                fm = self._entry_to_fact_metadata(entry, valid=True)
                facts_excluded.append(fm)
                inclusion_reasons[entry.fact_id] = "excluded: restricted (access control)"
                continue
            if entry.fact_id:
                score = self._query_relevance(entry, query_tokens)
                scorable_entries.append((score, entry))

        scorable_entries.sort(key=lambda pair: (pair[0], pair[1].ts))

        # Classify invalidated entries: inline near correcting parent or orphan
        inline_map: dict[str, list[StoreEntry]] = {}
        orphan_invalidated: list[StoreEntry] = []
        for inv_entry in invalidated_entries:
            parent_fid = self._find_correcting_parent(inv_entry)
            if parent_fid:
                inline_map.setdefault(parent_fid, []).append(inv_entry)
            else:
                orphan_invalidated.append(inv_entry)

        # Build reverse supersession map: new_fact_id -> old_fact_ids it replaced
        supersedes_map: dict[str, list[str]] = {}
        for old_id, new_id in self._layers.superseded_by.items():
            supersedes_map.setdefault(new_id, []).append(old_id)

        for _score, entry in scorable_entries:
            mtype = self._infer_memory_type(entry)
            dep_note = self._format_dependency_note(entry)
            display_key = self._display_key(entry.key) if entry.key else ""
            key_prefix = f"{display_key}: " if display_key else ""
            conf = infer_confidence(
                authority=entry.authority,
                source_type=entry.source_type or "user",
                scope=entry.scope,
                is_constraint=entry.is_constraint,
                ts=entry.ts,
                now=self._last_event_ts,
            )
            conf_note = confidence_annotation(conf)
            fact_lines.append(
                f"- [{entry.fact_id}] [{mtype}]{conf_note} {key_prefix}{entry.value}{dep_note}"
            )
            # Show what this fact replaced — only when old/new share semantic
            # overlap (true value change). Skip when they're unrelated concepts
            # (diagnostic chain evolution) to avoid misleading annotations.
            if entry.fact_id and entry.fact_id in supersedes_map:
                for old_fid in supersedes_map[entry.fact_id]:
                    old_entry = self._store.get_by_fact_id(old_fid)
                    if old_entry and self._has_semantic_overlap(
                        old_entry.value, entry.value
                    ):
                        fact_lines.append(f"  (changed from: {old_entry.value})")
            # Inline invalidated dependents near their correcting parent
            if entry.fact_id and entry.fact_id in inline_map:
                for inv_entry in inline_map[entry.fact_id]:
                    fact_lines.append(
                        f"  \u26a0\ufe0f RECALCULATE [{inv_entry.fact_id}]: {inv_entry.value}"
                    )
                    if inv_entry.fact_id:
                        inv_fm = self._entry_to_fact_metadata(inv_entry, valid=True)
                        facts_included.append(inv_fm)
                        inclusion_reasons[inv_entry.fact_id] = (
                            "inlined for recalculation (near correcting parent)"
                        )
            fm = self._entry_to_fact_metadata(entry, valid=True)
            facts_included.append(fm)
            if entry.fact_id:
                inclusion_reasons[entry.fact_id] = f"valid {mtype} fact"

        if fact_lines:
            parts.append("## Current Facts\n" + "\n".join(fact_lines))

        # --- Argument chains (for constraint/repair queries) ---
        if classification.complexity in (
            QueryComplexity.CONSTRAINT_CHECK,
            QueryComplexity.REPAIR,
        ):
            all_verbatim = [e for _, e in scorable_entries]
            chains_text = self._build_argument_chains(all_verbatim, constraint_entries)
            if chains_text:
                parts.append(chains_text)

        # --- Layer 2c: Orphan invalidated facts (no matching parent found) ---
        # Most invalidated entries are inlined near their correcting parent above.
        # Only orphans (where no parent could be identified) go in a separate section.
        if orphan_invalidated:
            inv_lines = []
            for e in sorted(orphan_invalidated, key=lambda e: e.ts):
                dep_note = self._format_dependency_note(e)
                corrected_note = self._format_correction_link(e)
                inv_lines.append(
                    f"❌ [{e.fact_id}] {e.value}{dep_note}{corrected_note}"
                )
            parts.append("## ⚠️ OUTDATED — Must Recalculate\n" + "\n".join(inv_lines))
            for e in orphan_invalidated:
                if e.fact_id:
                    fm = self._entry_to_fact_metadata(e, valid=True)
                    facts_included.append(fm)
                    inclusion_reasons[e.fact_id] = "included for recalculation (orphan)"

        # --- Active Commitments (prominent, between facts and working set) ---
        # Surface commitment-type facts prominently so the LLM doesn't lose
        # them when preferences change. Scans valid facts for commitment
        # language and echoes them in a dedicated section.
        commitment_lines = self._extract_fact_commitments(verbatim_entries)
        if commitment_lines:
            parts.append(
                "## Active Commitments (still in effect)\n"
                + "\n".join(f"- {c}" for c in commitment_lines)
            )

        # --- Superseded facts (excluded from context, tracked in provenance) ---
        for fact_id in self._layers.get_superseded_fact_ids():
            sup_entry = self._store.get_by_fact_id(fact_id)
            if sup_entry:
                fm = self._entry_to_fact_metadata(sup_entry, valid=False)
                facts_excluded.append(fm)
                sup_by = self._layers.superseded_by.get(fact_id, "unknown")
                inclusion_reasons[fact_id] = f"excluded: superseded by {sup_by}"

        # --- Layer 3: Working Set (chronological — temporal order matters) ---
        ws_entries = self._get_working_set_entries()
        if ws_entries:
            # Detect and suppress interruption turns
            ws_entries = self._filter_interruptions(ws_entries)
            # Suppress completed scoped sections (brainstorm, what-if, etc.)
            ws_entries = self._filter_scoped_sections(ws_entries)

            # Architectural enforcement: remove ALL scoped working set entries
            # entirely so the LLM cannot leak hypothetical/draft content it never sees.
            ws_entries = [
                e for e in ws_entries
                if "[SCOPE:" not in e.value.upper() and "[scope:" not in e.value
            ]

            ws_lines: list[str] = []
            for entry in ws_entries:
                scope = infer_scope(entry.value)
                if scope == "hypothetical":
                    ws_lines.append(f"[HYPOTHETICAL] {entry.value}")
                elif scope == "draft":
                    ws_lines.append(f"[DRAFT] {entry.value}")
                else:
                    ws_lines.append(entry.value)
            if ws_lines:
                parts.append("## Recent Context\n" + "\n".join(ws_lines))

        # --- Layer 4: Environment (freshness sorted) ---
        env_entries = self._get_environment_entries()
        if env_entries:
            env_lines = []
            for e in env_entries:
                if e.key == "now" and self._last_event_ts:
                    env_lines.append(f"- now: {self._last_event_ts.isoformat()}")
                else:
                    env_lines.append(f"- {e.key}: {e.value}")
            parts.append("## Environment\n" + "\n".join(env_lines))

        # --- Known Unknowns (prevent hallucination, nearest to query) ---
        # For simple lookups, limit to 3 most recent to save tokens
        if self._known_unknowns:
            sorted_unknowns = sorted(
                self._known_unknowns.items(), key=lambda kv: kv[1], reverse=True
            )
            if classification.complexity == QueryComplexity.SIMPLE_LOOKUP:
                sorted_unknowns = sorted_unknowns[:3]
            unk_lines = [f"- {text}" for text, _ in sorted_unknowns]
            parts.append("## Known Unknowns\n" + "\n".join(unk_lines))

        context = "\n\n".join(parts)
        result = ContextResult(
            context=context,
            facts_included=facts_included,
            facts_excluded=facts_excluded,
            inclusion_reasons=inclusion_reasons,
            token_count=len(self._encoder.encode(context)),
            query_complexity=classification.complexity.value,
        )

        # Attach compaction metrics if compaction was triggered
        if compaction_result is not None:
            result.compaction_triggered = True
            result.compaction_tokens_before = compaction_result.tokens_before
            result.compaction_tokens_after = compaction_result.tokens_after
            result.compaction_entries_compacted = compaction_result.entries_compacted

        return result

    def expand(self, node_id: int) -> list[StoreEntry]:
        """Lossless expansion of a summary node."""
        return self._dag.expand_to_entries(node_id)

    def expand_fact_chain(self, fact_key: str) -> list[StoreEntry]:
        """Get full supersession history for a fact key."""
        entry_ids = self._layers.get_chain_for_key(fact_key)
        return [self._store.get(eid) for eid in entry_ids]

    # ------------------------------------------------------------------ #
    # Dreaming / Consolidation
    # ------------------------------------------------------------------ #

    def consolidate(self) -> dict[str, int]:
        """Run a deterministic consolidation pass (inspired by Honcho's 'dreaming').

        No LLM calls — uses logical rules only. Operations:
        1. Detect latent contradictions between valid facts
        2. Resolve dangling dependency references
        3. Merge logically equivalent facts (same key, same value)

        Returns a summary of operations performed.
        """
        ops = {"contradictions_found": 0, "dangling_resolved": 0, "duplicates_merged": 0}

        # 1. Detect contradictions: two valid facts with the same key but different values
        key_to_facts: dict[str, list[str]] = {}
        for fact_id in self._layers.get_valid_fact_ids():
            entry = self._store.get_by_fact_id(fact_id)
            if entry and entry.key:
                key_to_facts.setdefault(entry.key, []).append(fact_id)

        for key, fids in key_to_facts.items():
            if len(fids) <= 1:
                continue
            # Multiple valid facts for same key — keep the newest, flag others
            entries = []
            for fid in fids:
                e = self._store.get_by_fact_id(fid)
                if e:
                    entries.append((fid, e))
            entries.sort(key=lambda pair: pair[1].ts)

            # If values are identical, merge (keep newest, invalidate older)
            if len(set(e.value for _, e in entries)) == 1:
                # All same value — keep the latest
                for fid, _ in entries[:-1]:
                    self._layers.invalidate_fact(fid, entries[-1][0])
                    ops["duplicates_merged"] += 1
            else:
                # Different values for same key — the latest should win,
                # but only mark as contradiction (don't auto-resolve — that's
                # the LLM's job, we just detect)
                for fid, e in entries[:-1]:
                    self._needs_review.add(fid)
                    ops["contradictions_found"] += 1

        # 2. Resolve dangling dependencies: if a fact depends on a
        #    fact_id that doesn't exist, clean up the reference
        for fact_id in list(self._layers.get_valid_fact_ids()):
            deps = self._layers.dependencies.get(fact_id, set())
            for dep_id in list(deps):
                if dep_id not in self._layers.valid_facts and dep_id not in self._layers.superseded_by:
                    deps.discard(dep_id)
                    ops["dangling_resolved"] += 1

        return ops

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _get_constraint_entries(self) -> list[StoreEntry]:
        """Get store entries for all active constraints."""
        entries = []
        for fact_id in self._layers.constraints:
            if self._layers.is_fact_valid(fact_id):
                entry = self._store.get_by_fact_id(fact_id)
                if entry:
                    entries.append(entry)
        return entries

    def _get_valid_non_constraint_entries(self) -> list[StoreEntry]:
        """Get store entries for valid non-constraint facts."""
        entries = []
        for fact_id in self._layers.get_valid_fact_ids():
            if fact_id not in self._layers.constraints:
                entry = self._store.get_by_fact_id(fact_id)
                if entry:
                    # Architectural enforcement: exclude hypothetical/draft facts
                    # so the LLM cannot leak scoped content it never sees.
                    if entry.scope in ("hypothetical", "draft"):
                        continue
                    entries.append(entry)
        return entries

    def _get_working_set_entries(self) -> list[StoreEntry]:
        """Get active working set entries (not discarded), capped at max."""
        entries = []
        for eid in self._layers.working_set_entries:
            if eid not in self._compactor.discarded_entries:
                entries.append(self._store.get(eid))
        # Cap to working_set_max (keep most recent)
        if len(entries) > self._config.working_set_max:
            entries = entries[-self._config.working_set_max:]
        return entries

    def _filter_interruptions(self, entries: list[StoreEntry]) -> list[StoreEntry]:
        """Detect and suppress interruption turns from working set.

        Recognizes "Hold on"/"Back to" patterns and excludes the
        interruption content. This prevents the LLM from mentioning
        off-topic interruption terms in its response.
        """
        start_idx = None
        end_idx = None

        for i, entry in enumerate(entries):
            text_lower = entry.value.lower()
            if start_idx is None:
                if any(phrase in text_lower for phrase in [
                    "hold on", "pause this", "one moment", "quick interruption",
                    "sorry, need to", "let me handle", "stepping away",
                ]):
                    start_idx = i
            elif any(phrase in text_lower for phrase in [
                "back to", "resume", "where were we", "anyway,",
                "returning to", "back on track", "incident resolved",
            ]):
                end_idx = i
                break

        if start_idx is not None and end_idx is not None:
            # Exclude interruption turns (keep the "back to" marker)
            return entries[:start_idx] + entries[end_idx:]

        return entries

    def _filter_scoped_sections(self, entries: list[StoreEntry]) -> list[StoreEntry]:
        """Suppress completed scoped sections from working set.

        Detects "[SCOPE: X]" entry markers paired with exit phrases like
        "that's enough" / "real commitments" and strips the entire scoped
        block. This prevents hypothetical/draft/brainstorm/sandbox data
        from leaking into the LLM's response.
        """
        start_idx = None
        end_idx = None

        for i, entry in enumerate(entries):
            text_lower = entry.value.lower()
            if start_idx is None:
                if "[scope:" in text_lower or "just exploratory" in text_lower:
                    start_idx = i
            elif any(phrase in text_lower for phrase in [
                "that's enough", "real commitments", "back to real",
                "let's talk about real",
            ]):
                end_idx = i + 1  # Include the exit turn in the removed block
                break

        if start_idx is not None and end_idx is not None:
            return entries[:start_idx] + entries[end_idx:]

        return entries

    def _extract_commitments(self, entries: list[StoreEntry]) -> list[str]:
        """Extract commitment statements from conversation turns.

        Recognizes phrases like 'committed to', 'decided to', 'agreed to'
        and extracts the commitment for prominent display.
        """
        commitments = []
        for entry in entries:
            text_lower = entry.value.lower()
            for phrase in self._COMMITMENT_PHRASES:
                if phrase in text_lower:
                    commitments.append(entry.value)
                    break
        return commitments

    _COMMITMENT_PHRASES = [
        "committed to", "committed for", "commitment for",
        "decided to", "agreed to", "will approve",
        "set sprint goal", "allocated",
    ]

    def _extract_fact_commitments(self, entries: list[StoreEntry]) -> list[str]:
        """Extract commitment-type facts from the persistent facts layer.

        Scans valid facts for commitment language and returns their values
        for prominent display in the Active Commitments section.
        """
        commitments = []
        for entry in entries:
            if entry.fact_id and entry.fact_id in self._needs_review:
                continue  # Skip invalidated entries
            if "[INVALIDATED" in entry.value:
                continue
            text_lower = entry.value.lower()
            for phrase in self._COMMITMENT_PHRASES:
                if phrase in text_lower:
                    commitments.append(entry.value)
                    break
        return commitments

    def _get_recalled_entries(
        self, query_tokens: set[str], max_recall: int = 3
    ) -> list[StoreEntry]:
        """Retrieve relevant older working set entries that were capped out.

        This leverages memgine's immutable store to recall important context
        that state_based permanently drops after its working set cap.
        """
        all_ws = []
        for eid in self._layers.working_set_entries:
            if eid not in self._compactor.discarded_entries:
                all_ws.append(eid)

        if len(all_ws) <= self._config.working_set_max:
            return []  # Nothing was capped, no recall needed

        # IDs that are in the current (capped) working set
        current_ids = set(all_ws[-self._config.working_set_max:])
        older = [self._store.get(eid) for eid in all_ws if eid not in current_ids]

        if not older or not query_tokens:
            return []

        # Score by query relevance and return top matches
        scored = [(self._query_relevance(e, query_tokens), e) for e in older]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for score, entry in scored[:max_recall] if score > 0]

    def _get_environment_entries(self) -> list[StoreEntry]:
        """Get active environment entries, sorted by recency (freshest first)."""
        entries = []
        for eid in self._layers.environment_entries.values():
            if eid not in self._compactor.discarded_entries:
                entries.append(self._store.get(eid))
        return sorted(entries, key=lambda e: e.ts, reverse=True)

    def _query_relevance(self, entry: StoreEntry, query_tokens: set[str]) -> float:
        """Score fact relevance to query (0.0 = irrelevant, 1.0 = highly relevant)."""
        if not query_tokens:
            return 0.0
        fact_tokens = self._extract_keywords(f"{entry.key} {entry.value}")
        overlap = query_tokens & fact_tokens
        if not overlap:
            return 0.0
        return len(overlap) / len(query_tokens)

    # Common stop words to exclude from semantic overlap checks
    _STOP_WORDS = frozenset({
        "the", "and", "for", "are", "but", "not", "you", "all",
        "can", "her", "was", "one", "our", "out", "has", "had",
        "this", "that", "with", "have", "from", "they", "been",
        "said", "each", "she", "which", "their", "will", "other",
        "about", "many", "then", "them", "these", "some", "would",
        "into", "more", "also", "its", "just", "over", "such",
    })

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        """Extract lowercase keyword tokens from text."""
        tokens = set()
        for raw in text.replace("_", " ").split():
            cleaned = "".join(ch for ch in raw.lower() if ch.isalnum())
            if len(cleaned) >= 3:
                tokens.add(cleaned)
        return tokens

    @classmethod
    def _has_semantic_overlap(cls, old_value: str, new_value: str) -> bool:
        """Check if two values share enough semantic overlap for a 'changed from' annotation.

        Returns True for value replacements (same concept, different value)
        and False for unrelated concept progressions (e.g., customer complaint
        superseded by a diagnostic finding about a different topic).
        """
        old_tokens = cls._extract_keywords(old_value) - cls._STOP_WORDS
        new_tokens = cls._extract_keywords(new_value) - cls._STOP_WORDS
        if not old_tokens or not new_tokens:
            return False
        overlap = old_tokens & new_tokens
        # Any shared content word indicates the facts are about the same
        # concept (e.g., "prefer" in billing preferences, "budget" in budget
        # changes). Zero overlap means unrelated concepts (e.g., a customer
        # complaint superseded by a diagnostic finding).
        return len(overlap) >= 1

    @staticmethod
    def _display_key(key: str) -> str:
        """Clean up fact keys for human-readable display.

        Strips scope prefixes like 'global_' and converts underscores to spaces.
        """
        # Strip common scope prefixes
        for prefix in ("global_", "task_", "local_"):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        return key.replace("_", " ")

    def _format_dependency_note(self, entry: StoreEntry) -> str:
        """Format dependency annotation for a fact line.

        Shows which facts this entry depends on, helping the LLM
        understand causal chains for repair propagation.
        """
        if not entry.depends_on:
            return ""
        # Only show dependencies that are still valid facts
        valid_deps = [
            dep_id for dep_id in entry.depends_on
            if self._layers.is_fact_valid(dep_id)
        ]
        if not valid_deps:
            return ""
        return f" (depends on: {', '.join(valid_deps)})"

    def _build_argument_chains(
        self, entries: list[StoreEntry], constraint_entries: list[StoreEntry],
    ) -> str:
        """Build conclusion-aware argument chains from dependency data.

        Groups facts into premise→conclusion structures so the LLM can
        verify inference chains rather than reasoning from scratch.

        Returns formatted text for inclusion in context.
        """
        # Build a lookup: fact_id -> entry
        by_fid: dict[str, StoreEntry] = {}
        for e in entries:
            if e.fact_id:
                by_fid[e.fact_id] = e
        for e in constraint_entries:
            if e.fact_id:
                by_fid[e.fact_id] = e

        # Find entries with dependencies (these form conclusions)
        conclusions: list[tuple[StoreEntry, list[StoreEntry]]] = []
        used_as_premise: set[str] = set()

        for entry in entries:
            if not entry.depends_on:
                continue
            premises = []
            for dep_id in entry.depends_on:
                dep_entry = by_fid.get(dep_id)
                if dep_entry and self._layers.is_fact_valid(dep_id):
                    premises.append(dep_entry)
                    used_as_premise.add(dep_id)
            if premises:
                conclusions.append((entry, premises))

        if not conclusions:
            return ""

        lines = ["## Argument Chains"]
        for conclusion, premises in conclusions:
            fid = conclusion.fact_id or "?"
            lines.append(f"CONCLUSION [{fid}]: {conclusion.value}")
            for p in premises:
                pfid = p.fact_id or "?"
                ptype = "constraint" if p.is_constraint else self._infer_memory_type(p)
                lines.append(f"  PREMISE [{pfid}] ({ptype}): {p.value}")

        return "\n".join(lines)

    @staticmethod
    def _clean_invalidated_text(value: str) -> str:
        """Strip verbose [INVALIDATED] markers, keeping only the original conclusion.

        Input:  "[INVALIDATED - was based on wrong data: $200K] Original conclusion: ..."
        Output: "..."  (just the original conclusion text)

        This prevents the LLM from seeing (and parroting) the wrong base values
        that are embedded in the marker text.
        """
        marker = "[INVALIDATED - was based on wrong data: "
        if marker in value:
            # Find end of marker bracket
            start = value.index(marker)
            try:
                bracket_end = value.index("]", start + len(marker))
            except ValueError:
                return value  # Malformed marker, return as-is

            after_bracket = value[bracket_end + 1:].strip()
            # Strip "Original conclusion: " prefix if present
            prefix = "Original conclusion: "
            if after_bracket.startswith(prefix):
                return after_bracket[len(prefix):]
            return after_bracket if after_bracket else value

        # Generic [INVALIDATED] without the specific format
        if value.startswith("[INVALIDATED"):
            try:
                bracket_end = value.index("]")
                return value[bracket_end + 1:].strip() or value
            except ValueError:
                pass

        return value

    def _format_correction_link(self, entry: StoreEntry) -> str:
        """Link an INVALIDATED fact to its corrected base value.

        Parses '[INVALIDATED - was based on wrong data: X]' and finds
        the correction that replaced X, showing the corrected value.
        """
        marker = "[INVALIDATED - was based on wrong data: "
        text = entry.value
        if marker not in text:
            return ""
        start = text.index(marker) + len(marker)
        end = text.index("]", start)
        wrong_value = text[start:end]

        for corr in self._corrections:
            if corr["old_value"] == wrong_value:
                return f"\n  → Corrected value: {corr['new_value']} (recalculate with this)"
        return ""

    def _find_correcting_parent(self, entry: StoreEntry) -> str | None:
        """Find the fact_id of the currently-valid fact that caused this invalidation.

        Three strategies, tried in order:
        1. Direct supersession: this entry's fact_id was directly superseded
        2. Dependency chain: one of this entry's dependencies was superseded
        3. Marker parsing: extract old value from [INVALIDATED] marker,
           match against corrections to find the valid replacement
        """
        fact_id = entry.fact_id
        if not fact_id:
            return None

        # Strategy 1: Direct supersession
        superseder = self._layers.superseded_by.get(fact_id)
        if superseder and self._layers.is_fact_valid(superseder):
            return superseder

        # Strategy 2: depends_on → superseded_by chain
        for dep_id in entry.depends_on:
            dep_superseder = self._layers.superseded_by.get(dep_id)
            if dep_superseder and self._layers.is_fact_valid(dep_superseder):
                return dep_superseder

        # Strategy 3: Parse [INVALIDATED] marker, match against corrections
        marker = "[INVALIDATED - was based on wrong data: "
        if marker in entry.value:
            start = entry.value.index(marker) + len(marker)
            try:
                end = entry.value.index("]", start)
                wrong_value = entry.value[start:end]
                for corr in self._corrections:
                    if corr["old_value"] == wrong_value:
                        new_value = str(corr["new_value"])
                        for valid_fid in self._layers.get_valid_fact_ids():
                            valid_entry = self._store.get_by_fact_id(valid_fid)
                            if valid_entry and new_value in valid_entry.value:
                                return valid_fid
            except ValueError:
                pass

        return None

    def _infer_memory_type(self, entry: StoreEntry) -> str:
        """Classify fact into tri-partite memory type (from paper).

        Returns 'org' for organizational knowledge or 'usr' for user memory.
        """
        org_source_types = {"policy", "system", "external"}
        if entry.source_type and entry.source_type in org_source_types:
            return "org"
        if entry.authority in ("policy", "executive"):
            return "org"
        if entry.source_identity:
            org_identifiers = {
                "finance_system", "hr_system", "calendar_system",
                "inventory_system", "crm_system", "erp_system",
                "sharepoint", "confluence", "database",
            }
            if entry.source_identity in org_identifiers:
                return "org"
        return "usr"

    def _entry_to_fact_metadata(self, entry: StoreEntry, *, valid: bool) -> FactMetadata:
        """Convert a store entry to FactMetadata for provenance."""
        superseded_by = None
        if entry.fact_id and not valid:
            superseded_by = self._layers.superseded_by.get(entry.fact_id)

        return FactMetadata(
            fact_id=entry.fact_id or "",
            key=entry.key,
            value=entry.value,
            layer=2,
            is_valid=valid,
            superseded_by=superseded_by,
            scope=entry.scope,
            authority=entry.authority,
            source=entry.source_type or "user",
            depends_on=list(entry.depends_on),
            is_constraint=entry.is_constraint,
            constraint_type=entry.constraint_type,
        )

    def reset(self) -> None:
        """Reset all state."""
        self._store.clear()
        self._layers.clear()
        self._dag.clear()
        self._compactor.clear()
        self._needs_review.clear()
        self._corrections.clear()
        self._known_unknowns.clear()
        self._fact_id_counts.clear()
        self._last_event_ts = None
        self._user_role = None
