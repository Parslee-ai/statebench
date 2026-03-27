"""Dialectic Memory Engine — a reasoning-first baseline.

Inspired by Honcho's memory-as-reasoning approach, but fully deterministic
(no LLM calls at ingestion or retrieval).

Architecture:
1. Fact Store (immutable): Raw events with provenance
2. Conclusion Store (mutable): Derived knowledge with cited premises
3. Dialectic Processor: Lightweight logical rules at ingestion time:
   - Detect contradictions between facts
   - Extend inference chains when new facts arrive
   - Cascade supersession to all conclusions citing a premise

Key insight from Honcho: Memory isn't storage, it's prediction.
Instead of storing raw facts, store *conclusions* that cite premises.
When a premise changes, all conclusions citing it are automatically suspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import tiktoken

from statebench.baselines.base import ContextResult, FactMetadata, MemoryStrategy
from statebench.confidence import (
    Confidence,
    confidence_annotation,
    infer_confidence,
)
from statebench.constraint_checker import (
    FactValue,
    check_constraints,
    format_constraint_checklist,
)
from statebench.query_classifier import QueryComplexity, classify_query
from statebench.schema.state import IdentityRole, Scope, Source
from statebench.schema.timeline import (
    ConversationTurn,
    Event,
    InitialState,
    StateWrite,
    Supersession,
    Write,
)


@dataclass
class Premise:
    """A raw fact used as evidence for conclusions."""

    fact_id: str
    key: str
    value: str
    source: Source
    ts: datetime
    scope: Scope = "global"
    is_valid: bool = True
    superseded_by: str | None = None
    is_constraint: bool = False
    constraint_type: str | None = None
    confidence: Confidence = "likely"


@dataclass
class Conclusion:
    """A derived piece of knowledge citing its premises.

    Unlike raw facts, conclusions carry reasoning traces:
    which premises support them, and what logical operation produced them.
    """

    conclusion_id: str
    text: str
    reasoning_type: Literal["deductive", "inductive", "abductive"]
    premise_ids: list[str]  # fact_ids this conclusion depends on
    ts: datetime
    is_valid: bool = True
    invalidation_reason: str | None = None


@dataclass
class Contradiction:
    """A detected contradiction between two facts."""

    fact_id_a: str
    fact_id_b: str
    description: str
    detected_at: datetime


class DialecticStrategy(MemoryStrategy):
    """Reasoning-first memory engine.

    Stores both raw premises and derived conclusions. When premises
    change, conclusions are automatically invalidated. The LLM sees
    structured argument chains, not raw fact lists.
    """

    def __init__(self, token_budget: int = 8000, working_set_size: int = 10):
        super().__init__(token_budget)
        self.working_set_size = working_set_size

        # Stores
        self.identity: IdentityRole | None = None
        self.premises: dict[str, Premise] = {}  # fact_id -> Premise
        self.premises_by_key: dict[str, str] = {}  # key -> fact_id
        self.conclusions: dict[str, Conclusion] = {}
        self.contradictions: list[Contradiction] = []
        self.working_set: list[dict[str, str | datetime | Scope]] = []
        self.environment: dict[str, str | datetime] = {}
        self._environment_ts: dict[str, datetime] = {}
        self.known_unknowns: dict[str, datetime] = {}

        self._conclusion_counter = 0
        self._encoder = tiktoken.get_encoding("cl100k_base")

    @property
    def name(self) -> str:
        return "dialectic"

    @property
    def expects_initial_state(self) -> bool:
        return True

    def _next_conclusion_id(self) -> str:
        self._conclusion_counter += 1
        return f"C-{self._conclusion_counter:04d}"

    def _count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    # ── Ingestion ──────────────────────────────────────────────────────

    def initialize_from_state(self, initial_state: InitialState) -> None:
        self.identity = initial_state.identity_role

        for fact in initial_state.persistent_facts:
            premise = Premise(
                fact_id=fact.id,
                key=fact.key,
                value=fact.value,
                source=fact.source,
                ts=fact.ts,
                scope=fact.scope,
                is_valid=fact.is_valid,
                superseded_by=fact.superseded_by,
                is_constraint=fact.is_constraint,
                confidence=infer_confidence(
                    authority=fact.source.authority,
                    source_type=fact.source.type,
                    scope=fact.scope,
                    is_constraint=fact.is_constraint,
                    ts=fact.ts,
                ),
            )
            self.premises[fact.id] = premise
            self.premises_by_key[fact.key] = fact.id

        # Auto-derive conclusions from dependency relationships
        for fact in initial_state.persistent_facts:
            if fact.depends_on:
                valid_deps = [d for d in fact.depends_on if d in self.premises]
                if valid_deps:
                    self._derive_conclusion(fact.id, valid_deps, fact.ts)

        # Detect initial contradictions (full scan, one-time cost)
        self._detect_contradictions()

        self.working_set = [
            {"content": item.content, "ts": item.ts, "scope": "global"}
            for item in initial_state.working_set
        ]
        self.environment = dict(initial_state.environment)
        self._environment_ts = {k: datetime.min for k in self.environment}

    def process_event(self, event: Event) -> None:
        if isinstance(event, ConversationTurn):
            self.working_set.append({
                "content": f"{event.speaker.title()}: {event.text}",
                "ts": event.ts,
                "scope": "global",
            })
            if len(self.working_set) > self.working_set_size:
                self.working_set = self.working_set[-self.working_set_size:]

            if "?" in event.text or any(
                p in event.text.lower()
                for p in ["need info", "don't know", "not sure"]
            ):
                self.known_unknowns[event.text] = event.ts

        elif isinstance(event, StateWrite):
            for write in event.writes:
                if write.layer == "persistent_facts":
                    self._ingest_fact(write, event.ts)
                elif write.layer == "environment":
                    self._update_environment(write.key, write.value, event.ts)

        elif isinstance(event, Supersession):
            for write in event.writes:
                if write.layer == "persistent_facts":
                    self._ingest_supersession(write, event.ts)
                elif write.layer == "environment":
                    self._update_environment(write.key, write.value, event.ts)

    def _ingest_fact(self, write: Write, ts: datetime) -> None:
        """Ingest a new fact and run dialectic processing."""
        premise = Premise(
            fact_id=write.id,
            key=write.key,
            value=write.value,
            source=write.source,
            ts=ts,
            scope=write.scope,
            is_constraint=write.is_constraint,
            confidence=infer_confidence(
                authority=write.source.authority,
                source_type=write.source.type,
                scope=write.scope,
                is_constraint=write.is_constraint,
                ts=ts,
            ),
        )
        self.premises[write.id] = premise
        self.premises_by_key[write.key] = write.id

        # Auto-derive conclusions from dependencies
        deps = list(write.depends_on) if write.depends_on else []
        valid_deps = [d for d in deps if d in self.premises]
        if valid_deps:
            self._derive_conclusion(write.id, valid_deps, ts)

        # Check for contradictions on this key only (O(k), not O(n))
        self._detect_contradictions_for_key(write.key)

    def _ingest_supersession(self, write: Write, ts: datetime) -> None:
        """Process a supersession and cascade to conclusions."""
        # Invalidate the old premise
        if write.supersedes and write.supersedes in self.premises:
            old = self.premises[write.supersedes]
            old.is_valid = False
            old.superseded_by = write.id

            # CASCADE: invalidate all conclusions citing this premise
            self._cascade_invalidation(write.supersedes, ts)

        # Ingest the new fact
        self._ingest_fact(write, ts)

    def _derive_conclusion(
        self, fact_id: str, premise_ids: list[str], ts: datetime
    ) -> None:
        """Derive a conclusion from a fact and its premises.

        This is a deductive conclusion: "Given premises A, B, C,
        fact X follows."
        """
        premise = self.premises.get(fact_id)
        if not premise:
            return

        premise_texts = []
        for pid in premise_ids:
            p = self.premises.get(pid)
            if p and p.is_valid:
                premise_texts.append(f"{p.key}: {p.value}")

        if not premise_texts:
            return

        conclusion = Conclusion(
            conclusion_id=self._next_conclusion_id(),
            text=premise.value,
            reasoning_type="deductive",
            premise_ids=premise_ids,
            ts=ts,
        )
        self.conclusions[conclusion.conclusion_id] = conclusion

    def _cascade_invalidation(self, invalidated_fact_id: str, ts: datetime) -> None:
        """Invalidate all conclusions that cite a given premise."""
        for cid, conclusion in self.conclusions.items():
            if invalidated_fact_id in conclusion.premise_ids and conclusion.is_valid:
                conclusion.is_valid = False
                conclusion.invalidation_reason = (
                    f"Premise {invalidated_fact_id} was superseded"
                )

    def _detect_contradictions_for_key(self, key: str) -> None:
        """Detect contradictions for a single key (O(k) where k = facts with this key)."""
        valid_for_key = [
            p for p in self.premises.values()
            if p.is_valid and p.key == key
        ]
        if len(valid_for_key) <= 1:
            return

        valid_for_key.sort(key=lambda p: p.ts)
        newest = valid_for_key[-1]
        for older in valid_for_key[:-1]:
            already_found = any(
                c.fact_id_a == older.fact_id and c.fact_id_b == newest.fact_id
                for c in self.contradictions
            )
            if not already_found:
                self.contradictions.append(Contradiction(
                    fact_id_a=older.fact_id,
                    fact_id_b=newest.fact_id,
                    description=f"Key '{key}': '{older.value}' vs '{newest.value}'",
                    detected_at=newest.ts,
                ))

    def _detect_contradictions(self) -> None:
        """Detect contradictions across all keys (used for initial state only)."""
        seen_keys: set[str] = set()
        for p in self.premises.values():
            if p.is_valid and p.key not in seen_keys:
                seen_keys.add(p.key)
                self._detect_contradictions_for_key(p.key)

    def _update_environment(
        self, key: str, value: str | datetime, ts: datetime
    ) -> None:
        prev_ts = self._environment_ts.get(key)
        if prev_ts and prev_ts > ts:
            return
        self.environment[key] = value
        self._environment_ts[key] = ts
        if len(self.environment) > 5:
            oldest = min(self._environment_ts.items(), key=lambda x: x[1])[0]
            if oldest != key:
                self.environment.pop(oldest, None)
                self._environment_ts.pop(oldest, None)

    # ── Context Assembly ───────────────────────────────────────────────

    def build_context(self, query: str) -> ContextResult:
        classification = classify_query(query)
        parts: list[str] = []
        facts_included: list[FactMetadata] = []
        facts_excluded: list[FactMetadata] = []
        inclusion_reasons: dict[str, str] = {}

        # Identity
        if self.identity:
            lines = [f"User: {self.identity.user_name}", f"Role: {self.identity.authority}"]
            if self.identity.department:
                lines.append(f"Department: {self.identity.department}")
            if self.identity.organization:
                lines.append(f"Organization: {self.identity.organization}")
            parts.append("## Identity\n" + "\n".join(lines))

        # Constraints
        constraints = [p for p in self.premises.values() if p.is_valid and p.is_constraint]
        if constraints:
            lines = []
            for c in sorted(constraints, key=lambda x: x.ts):
                ct = c.constraint_type or "CONSTRAINT"
                lines.append(f"⚠️ [{c.fact_id}] [{ct}] {c.value}")
                facts_included.append(self._to_metadata(c))
                inclusion_reasons[c.fact_id] = "active constraint"
            parts.append("## ⚠️ Active Constraints (CHECK ALL)\n" + "\n".join(lines))

        # Constraint pre-computation
        if constraints and classification.complexity in (
            QueryComplexity.CONSTRAINT_CHECK,
            QueryComplexity.AGGREGATION,
        ):
            c_fvs = [FactValue(c.fact_id, c.key, c.value) for c in constraints]
            all_valid = [p for p in self.premises.values() if p.is_valid and not p.is_constraint]
            f_fvs = [FactValue(f.fact_id, f.key, f.value) for f in all_valid]
            results = check_constraints(c_fvs, f_fvs)
            checklist = format_constraint_checklist(results)
            if checklist:
                parts.append(checklist)

        # Argument chains (the dialectic differentiator)
        valid_conclusions = [c for c in self.conclusions.values() if c.is_valid]
        if valid_conclusions:
            chain_lines = ["## Argument Chains"]
            for conc in sorted(valid_conclusions, key=lambda x: x.ts):
                chain_lines.append(f"CONCLUSION [{conc.conclusion_id}] ({conc.reasoning_type}): {conc.text}")
                for pid in conc.premise_ids:
                    p = self.premises.get(pid)
                    if p:
                        ptype = "constraint" if p.is_constraint else "fact"
                        chain_lines.append(f"  PREMISE [{pid}] ({ptype}): {p.value}")
            parts.append("\n".join(chain_lines))

        # Invalidated conclusions (for repair queries)
        invalid_conclusions = [c for c in self.conclusions.values() if not c.is_valid]
        if invalid_conclusions and classification.complexity in (
            QueryComplexity.REPAIR,
            QueryComplexity.CONSTRAINT_CHECK,
        ):
            inv_lines = []
            for conc in invalid_conclusions:
                reason = conc.invalidation_reason or "premise changed"
                inv_lines.append(f"⚠️ STALE [{conc.conclusion_id}]: {conc.text} — {reason}")
            parts.append("## ⚠️ Stale Conclusions (RECALCULATE)\n" + "\n".join(inv_lines))

        # Valid facts (non-constraint)
        valid_facts = [
            p for p in self.premises.values()
            if p.is_valid and not p.is_constraint
            and p.scope not in ("hypothetical", "draft")
        ]
        if valid_facts:
            type_labels = {"organizational": "org", "user": "usr"}
            fact_lines = []
            for f in sorted(valid_facts, key=lambda x: x.ts):
                src_type = f.source.type if f.source.type in type_labels else "usr"
                label = type_labels.get(src_type, "usr")
                conf_note = confidence_annotation(f.confidence)
                fact_lines.append(f"- [{f.fact_id}] [{label}]{conf_note} {f.value}")
                facts_included.append(self._to_metadata(f))
                inclusion_reasons[f.fact_id] = f"valid {label} fact"
            parts.append("## Current Facts\n" + "\n".join(fact_lines))

        # Contradictions (if any detected)
        active_contradictions = []
        for c in self.contradictions:
            a = self.premises.get(c.fact_id_a)
            b = self.premises.get(c.fact_id_b)
            if a and a.is_valid and b and b.is_valid:
                active_contradictions.append(c)
        if active_contradictions:
            contra_lines = []
            for c in active_contradictions:
                contra_lines.append(f"⚠️ CONTRADICTION: {c.description}")
            parts.append("## ⚠️ Detected Contradictions\n" + "\n".join(contra_lines))

        # Superseded facts (excluded)
        for p in self.premises.values():
            if not p.is_valid:
                facts_excluded.append(self._to_metadata(p))
                inclusion_reasons[p.fact_id] = f"excluded: superseded by {p.superseded_by}"

        # Working set
        if self.working_set:
            ws_lines = [str(item["content"]) for item in self.working_set]
            parts.append("## Recent Context\n" + "\n".join(ws_lines))

        # Environment
        if self.environment:
            sorted_env = sorted(
                self.environment.items(),
                key=lambda kv: self._environment_ts.get(kv[0], datetime.min),
                reverse=True,
            )[:5]
            env_lines = [f"- {k}: {v}" for k, v in sorted_env]
            parts.append("## Environment\n" + "\n".join(env_lines))

        # Known unknowns
        if self.known_unknowns:
            unk_lines = [
                f"- {text}" for text, _ in sorted(
                    self.known_unknowns.items(), key=lambda kv: kv[1], reverse=True
                )
            ]
            parts.append("## Known Unknowns\n" + "\n".join(unk_lines))

        context = "\n\n".join(parts)

        return ContextResult(
            context=context,
            facts_included=facts_included,
            facts_excluded=facts_excluded,
            inclusion_reasons=inclusion_reasons,
            token_count=self._count_tokens(context),
            query_complexity=classification.complexity.value,
        )

    def reset(self) -> None:
        self.identity = None
        self.premises = {}
        self.premises_by_key = {}
        self.conclusions = {}
        self.contradictions = []
        self.working_set = []
        self.environment = {}
        self._environment_ts = {}
        self.known_unknowns = {}
        self._conclusion_counter = 0

    def get_system_prompt(self) -> str:
        return (
            "You are an AI agent. Answer based ONLY on the structured context.\n\n"
            "CRITICAL RULES:\n"
            "1. For YES/NO decisions, CHECK ALL CONSTRAINTS - if ANY blocks, answer NO\n"
            "2. Multiple constraints must ALL be satisfied simultaneously\n"
            "3. NEVER invent details not explicitly stated — but DO perform "
            "arithmetic on stated values and DO recalculate when values change\n"
            "4. If info wasn't provided, say 'not specified' - don't assume\n"
            "5. Items marked [HYPOTHETICAL] are what-if scenarios - not real\n"
            "6. Items marked [DRAFT] are tentative - not finalized\n\n"
            "ARGUMENT CHAINS:\n"
            "7. 'Argument Chains' show CONCLUSION ← PREMISE relationships\n"
            "8. If a premise has changed, the conclusion is INVALID — recalculate\n"
            "9. 'Stale Conclusions' are conclusions whose premises changed\n\n"
            "CONTRADICTIONS:\n"
            "10. If contradictions are listed, acknowledge them explicitly\n"
            "11. Use the most recent/authoritative fact to resolve contradictions\n\n"
            "Be accurate, concise, and explicit about what you know vs. don't know."
        )

    def _to_metadata(self, premise: Premise) -> FactMetadata:
        return FactMetadata(
            fact_id=premise.fact_id,
            key=premise.key,
            value=premise.value,
            layer=2,
            is_valid=premise.is_valid,
            superseded_by=premise.superseded_by,
            scope=premise.scope,
            authority=premise.source.authority,
            source=premise.source.type,
            is_constraint=premise.is_constraint,
            constraint_type=premise.constraint_type,
        )
