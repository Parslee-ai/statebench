"""State-Based Context Snapshot Baseline.

This is the architecture proposed in the paper. Context is assembled
from four layers with explicit supersession tracking:
1. Identity & Role
2. Persistent Facts (tri-partite: User Memory, Capability Memory, Org Knowledge)
3. Working Set
4. Environmental Signals

Enhanced with:
- Tri-partite fact decomposition (user/capability/organizational)
- Scope metadata (global/task/hypothetical/draft)
- Dependency tracking for repair propagation
- Constraint enforcement in prompts
- Known-unknowns tracking to prevent hallucination

v1.0: Full provenance tracking with ContextResult and FactMetadata.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import tiktoken

from statebench.baselines.base import ContextResult, FactMetadata, MemoryStrategy
from statebench.schema.state import IdentityRole, Scope, Source
from statebench.schema.timeline import (
    ConversationTurn,
    Event,
    InitialState,
    StateWrite,
    Supersession,
)

# Type alias for memory types
MemoryType = Literal["user", "capability", "organizational"]


@dataclass
class EnhancedFact:
    """A fact with memory type, scope, dependencies, and constraint metadata.

    v1.0: Added fact_id for provenance tracking.
    """
    # v1.0: Unique fact ID for provenance
    fact_id: str

    key: str
    value: str
    source: Source  # v1.0: Now structured Source object
    ts: datetime
    is_valid: bool = True
    superseded_by: str | None = None

    # Tri-partite memory classification (from paper)
    # - user: Private preferences, corrections, decisions (User Memory)
    # - capability: Learned patterns, heuristics (Capability Memory)
    # - organizational: Policies, system data, documents (Organizational Knowledge)
    memory_type: MemoryType = "user"

    # Scope tracking (fixes scope_leak)
    scope: Scope = "global"
    scope_id: str | None = None

    # Dependency tracking (fixes repair_propagation)
    depends_on: list[str] = field(default_factory=list)
    derived_facts: list[str] = field(default_factory=list)
    needs_review: bool = False

    # Constraint metadata (fixes causality)
    is_constraint: bool = False
    constraint_type: str | None = None

    def to_fact_metadata(self) -> FactMetadata:
        """Convert to FactMetadata for provenance tracking."""
        return FactMetadata(
            fact_id=self.fact_id,
            key=self.key,
            value=self.value,
            layer=2,  # Persistent facts layer
            is_valid=self.is_valid,
            superseded_by=self.superseded_by,
            scope=self.scope,
            authority=self.source.authority,
            source=self.source.type,
            depends_on=list(self.depends_on),
            derived_facts=list(self.derived_facts),
            is_constraint=self.is_constraint,
            constraint_type=self.constraint_type,
        )


class StateBasedStrategy(MemoryStrategy):
    """State-based context with explicit supersession tracking.

    This strategy maintains the four-layer state architecture:
    - Identity & Role: Static user/role information
    - Persistent Facts: Durable facts with supersession, scope, and dependency tracking
    - Working Set: Recent conversation context
    - Environment: Real-time signals (timestamps, etc.)

    v1.0: Full provenance tracking with ContextResult.
    """

    def __init__(
        self,
        token_budget: int = 8000,
        working_set_size: int = 10,  # Increased from 5 for complex scenarios
    ):
        super().__init__(token_budget)
        self.working_set_size = working_set_size

        # State layers
        self.identity: IdentityRole | None = None
        self.facts: dict[str, EnhancedFact] = {}  # Keyed by fact_id
        self.facts_by_key: dict[str, str] = {}  # key -> fact_id mapping
        self.superseded: set[str] = set()  # Set of superseded fact_ids
        self.working_set: list[dict[str, str | datetime | Scope]] = []
        self.environment: dict[str, str | datetime] = {}
        self._environment_ts: dict[str, datetime] = {}

        # Known unknowns (prevents hallucination)
        self.known_unknowns: dict[str, datetime] = {}

        # Correction history (for repair propagation)
        self.corrections: list[dict[str, str | datetime]] = []

        # v1.0: Fact ID counter for auto-generation
        self._fact_id_counter = 0

        self._encoder = tiktoken.get_encoding("cl100k_base")

    def _generate_fact_id(self) -> str:
        """Generate a unique fact ID."""
        self._fact_id_counter += 1
        return f"F-{self._fact_id_counter:04d}"

    @property
    def name(self) -> str:
        return "state_based"

    @property
    def expects_initial_state(self) -> bool:
        return True

    def _count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def _is_constraint(self, value: str, source: Source) -> bool:
        """Detect if a fact is a formal constraint (not casual mentions).

        v1.0: source is now a Source object.
        """
        # Don't mark corrections, updates, or conversation snippets as constraints
        if any(marker in value for marker in [
            "INVALIDATED", "CORRECTION", "delayed", "changed",
            "let's go with", "just fyi", "btw", "hold on", "wait",
            "best we can do", "fine,", "bad news", "good news"
        ]):
            return False

        # Only mark as constraint if it's a formal policy statement
        # Must have BOTH a constraint keyword AND a formal structure indicator
        value_lower = value.lower()

        # Formal constraint indicators (policy-like language)
        formal_indicators = [
            "must", "require", "policy", "limit is", "maximum is", "minimum is",
            "cannot exceed", "not allowed", "prohibited", "mandatory",
            "approval required", "needs approval", "authority to"
        ]

        has_formal = any(ind in value_lower for ind in formal_indicators)

        # Source from policy is always a constraint
        if source.type == "policy":
            return True

        # High authority sources are more likely to be constraints
        if source.authority in ("policy", "executive") and has_formal:
            return True

        return has_formal

    def _infer_constraint_type(self, value: str) -> str | None:
        """Infer the type of constraint."""
        value_lower = value.lower()
        if any(kw in value_lower for kw in ["budget", "$", "cost", "price", "spend"]):
            return "budget"
        if any(kw in value_lower for kw in ["deadline", "due", "by", "before", "until"]):
            return "deadline"
        if any(kw in value_lower for kw in ["capacity", "available", "team", "resource", "hours"]):
            return "capacity"
        if any(kw in value_lower for kw in ["policy", "require", "must", "approval", "authority"]):
            return "policy"
        return None

    def _infer_scope(self, text: str) -> str:
        """Infer scope from conversation text."""
        text_lower = text.lower()

        if any(phrase in text_lower for phrase in [
            "what if", "hypothetically", "suppose", "imagine", "let's say",
            "in theory", "potentially"
        ]):
            return "hypothetical"

        if any(phrase in text_lower for phrase in [
            "draft", "preliminary", "not final", "pending", "proposal", "tentative"
        ]):
            return "draft"

        if any(phrase in text_lower for phrase in [
            "for this task", "for this project", "just for this", "only for"
        ]):
            return "task"

        return "global"

    def _infer_memory_type(self, source: Source) -> MemoryType:
        """Classify fact into tri-partite memory structure.

        From paper:
        - User Memory: Private preferences, corrections, decisions
        - Capability Memory: Learned patterns, heuristics, strategies
        - Organizational Knowledge: Policies, system data, documents

        v1.0: source is now a Source object.
        """
        # Organizational sources - policies, systems, documents
        org_source_types = {"policy", "system", "external"}
        if source.type in org_source_types:
            return "organizational"

        # High authority sources are typically organizational
        if source.authority in ("policy", "executive"):
            return "organizational"

        # Check identity for system indicators
        if source.identity:
            org_identifiers = {
                "finance_system", "hr_system", "calendar_system",
                "inventory_system", "crm_system", "erp_system",
                "sharepoint", "confluence", "database"
            }
            if source.identity in org_identifiers:
                return "organizational"

        # Capability memory - learned patterns (not yet used in benchmark)
        # Would check for observation/pattern/heuristic sources

        # User memory - preferences, decisions, corrections
        # Default: user type sources
        return "user"

    def initialize_from_state(self, initial_state: InitialState) -> None:
        """Initialize state from timeline's initial state.

        v1.0: Handles PersistentFact with id and Source object.
        """
        self.identity = initial_state.identity_role

        for fact in initial_state.persistent_facts:
            # v1.0: PersistentFact now has id field
            fact_id = fact.id

            enhanced = EnhancedFact(
                fact_id=fact_id,
                key=fact.key,
                value=fact.value,
                source=fact.source,  # Now a Source object
                ts=fact.ts,
                is_valid=fact.is_valid,
                superseded_by=fact.superseded_by,
                memory_type=self._infer_memory_type(fact.source),
                scope=fact.scope,
                depends_on=list(fact.depends_on),
                derived_facts=list(fact.derived_facts),
                is_constraint=self._is_constraint(fact.value, fact.source),
                constraint_type=self._infer_constraint_type(fact.value),
            )
            self.facts[fact_id] = enhanced
            self.facts_by_key[fact.key] = fact_id

            # Track superseded facts
            if not fact.is_valid:
                self.superseded.add(fact_id)

        self.working_set = [
            {"content": item.content, "ts": item.ts, "scope": "global"}
            for item in initial_state.working_set
        ]

        self.environment = dict(initial_state.environment)
        # Initialize environment timestamps so later writes can compare freshness
        now = datetime.min
        self._environment_ts = {k: now for k in self.environment.keys()}

    def process_event(self, event: Event) -> None:
        """Process an event and update state layers.

        v1.0: Handles Write with id and Source object.
        """
        if isinstance(event, ConversationTurn):
            scope = self._infer_scope(event.text)

            self.working_set.append({
                "content": f"{event.speaker.title()}: {event.text}",
                "ts": event.ts,
                "scope": scope,
            })

            if len(self.working_set) > self.working_set_size:
                self.working_set = self.working_set[-self.working_set_size:]

            # Track known-unknowns so the model can explicitly say "not provided"
            if "?" in event.text or any(phrase in event.text.lower() for phrase in [
                "need info", "don't know", "not sure", "find out"
            ]):
                self.known_unknowns[event.text] = event.ts

        elif isinstance(event, StateWrite):
            for write in event.writes:
                if write.layer == "persistent_facts":
                    # v1.0: Use fact ID from write, or generate one
                    fact_id = write.id

                    # Infer dependencies from other facts
                    deps = list(write.depends_on) if write.depends_on else []
                    deps.extend(self._infer_dependencies(write.value))

                    # Determine if this is a constraint
                    is_const = write.is_constraint or self._is_constraint(
                        write.value, write.source
                    )
                    const_type = write.constraint_type or self._infer_constraint_type(
                        write.value
                    )

                    fact = EnhancedFact(
                        fact_id=fact_id,
                        key=write.key,
                        value=write.value,
                        source=write.source,  # Now a Source object
                        ts=event.ts,
                        memory_type=self._infer_memory_type(write.source),
                        scope=write.scope,
                        depends_on=deps,
                        is_constraint=is_const,
                        constraint_type=const_type,
                    )
                    self.facts[fact_id] = fact
                    self.facts_by_key[write.key] = fact_id

                    # Update parent facts with derived references
                    for dep_id in deps:
                        if dep_id in self.facts:
                            self.facts[dep_id].derived_facts.append(fact_id)

                elif write.layer == "environment":
                    self._update_environment(
                        write.key, write.value, event.ts, write.supersedes
                    )

        elif isinstance(event, Supersession):
            for write in event.writes:
                if write.layer == "persistent_facts":
                    # v1.0: Use fact ID from write
                    fact_id = write.id

                    old_value = None
                    old_memory_type: MemoryType = "user"

                    # write.supersedes now contains fact_id, not key
                    if write.supersedes and write.supersedes in self.facts:
                        old_fact = self.facts[write.supersedes]
                        old_value = old_fact.value
                        old_memory_type = old_fact.memory_type
                        old_fact.is_valid = False
                        old_fact.superseded_by = fact_id
                        self.superseded.add(write.supersedes)

                        # Propagate invalidation to derived facts
                        self._propagate_invalidation(write.supersedes)

                        # Record the correction for context
                        self.corrections.append({
                            "old_key": write.supersedes,
                            "old_value": old_value,
                            "new_value": write.value,
                            "ts": event.ts,
                        })

                    deps = list(write.depends_on) if write.depends_on else []
                    deps.extend(self._infer_dependencies(write.value))

                    # Determine if this is a constraint
                    is_const = write.is_constraint or self._is_constraint(
                        write.value, write.source
                    )
                    const_type = write.constraint_type or self._infer_constraint_type(
                        write.value
                    )

                    fact = EnhancedFact(
                        fact_id=fact_id,
                        key=write.key,
                        value=write.value,
                        source=write.source,
                        ts=event.ts,
                        memory_type=old_memory_type,  # Inherit from superseded fact
                        scope=write.scope,
                        depends_on=deps,
                        is_constraint=is_const,
                        constraint_type=const_type,
                    )
                    self.facts[fact_id] = fact
                    self.facts_by_key[write.key] = fact_id

                elif write.layer == "environment":
                    self._update_environment(
                        write.key, write.value, event.ts, write.supersedes
                    )

    def _update_environment(
        self,
        key: str,
        value: str | datetime,
        ts: datetime,
        supersedes: str | None,
    ) -> None:
        """Update environment entries, respecting freshness and supersession."""
        if supersedes:
            self.environment.pop(supersedes, None)
            self._environment_ts.pop(supersedes, None)

        prev_ts = self._environment_ts.get(key)
        if prev_ts and prev_ts > ts:
            # Ignore stale updates
            return
        self.environment[key] = value
        self._environment_ts[key] = ts

        # Keep the environment bounded
        if len(self.environment) > 5:
            oldest_key = min(self._environment_ts.items(), key=lambda item: item[1])[0]
            if oldest_key != key:
                self.environment.pop(oldest_key, None)
                self._environment_ts.pop(oldest_key, None)

    def _extract_keywords(self, text: str) -> set[str]:
        tokens = set()
        for raw in text.replace("_", " ").split():
            cleaned = "".join(ch for ch in raw.lower() if ch.isalnum())
            if len(cleaned) >= 3:
                tokens.add(cleaned)
        return tokens

    def _infer_dependencies(self, value: str) -> list[str]:
        """Infer which existing facts this value depends on.

        Returns fact_ids of facts that this value likely depends on.
        """
        deps: list[str] = []
        value_tokens = self._extract_keywords(value)

        for fact_id, fact in self.facts.items():
            if fact.is_valid:
                keywords = self._extract_keywords(fact.value)
                keywords.update(self._extract_keywords(fact.key))
                if value_tokens & keywords:
                    deps.append(fact_id)

        return deps

    def _propagate_invalidation(self, superseded_fact_id: str) -> None:
        """Mark all facts derived from a superseded fact as needing review.

        v1.0: Uses fact_ids instead of keys.
        """
        if superseded_fact_id not in self.facts:
            return

        old_fact = self.facts[superseded_fact_id]
        for derived_fact_id in old_fact.derived_facts:
            if derived_fact_id in self.facts:
                self.facts[derived_fact_id].needs_review = True
                self._propagate_invalidation(derived_fact_id)

    def _get_valid_facts(self) -> list[EnhancedFact]:
        """Return only currently valid facts."""
        return [f for f in self.facts.values() if f.is_valid]

    def _get_constraints(self) -> list[EnhancedFact]:
        """Return all active constraints."""
        return [f for f in self.facts.values() if f.is_valid and f.is_constraint]

    def build_context(self, query: str) -> ContextResult:
        """Build structured context from state layers with provenance.

        v1.0: Returns ContextResult with full provenance tracking.
        """
        parts: list[str] = []
        facts_included: list[FactMetadata] = []
        facts_excluded: list[FactMetadata] = []
        inclusion_reasons: dict[str, str] = {}

        # Layer 1: Identity & Role
        if self.identity:
            identity_text = (
                f"User: {self.identity.user_name}\n"
                f"Role: {self.identity.authority}"
            )
            if self.identity.department:
                identity_text += f"\nDepartment: {self.identity.department}"
            if self.identity.organization:
                identity_text += f"\nOrganization: {self.identity.organization}"
            parts.append(f"## Identity\n{identity_text}")

        # Layer 2a: CONSTRAINTS (emphasized, from any memory type)
        constraints = self._get_constraints()
        if constraints:
            constraint_text = "\n".join(
                f"⚠️ [{c.fact_id}] [{c.constraint_type or 'CONSTRAINT'}] {c.value}"
                for c in sorted(constraints, key=lambda x: x.ts)
            )
            parts.append(f"## ⚠️ Active Constraints (CHECK ALL)\n{constraint_text}")

            # Track provenance for constraints
            for c in constraints:
                facts_included.append(c.to_fact_metadata())
                inclusion_reasons[c.fact_id] = "active constraint"

        # Layer 2b: Facts by memory type (tri-partite structure from paper)
        other_facts = [f for f in self._get_valid_facts() if not f.is_constraint]

        # Check for invalidated facts that need recalculation
        invalidated = [
            f for f in other_facts
            if "[INVALIDATED" in f.value or f.needs_review
        ]
        valid_other = [
            f for f in other_facts
            if "[INVALIDATED" not in f.value and not f.needs_review
        ]

        if valid_other:
            # Memory type abbreviations for compact display
            type_labels = {"organizational": "org", "user": "usr", "capability": "cap"}
            facts_text = "\n".join(
                f"- [{f.fact_id}] [{type_labels.get(f.memory_type, 'usr')}] {f.value}"
                for f in sorted(valid_other, key=lambda x: x.ts)
            )
            parts.append(f"## Current Facts\n{facts_text}")

            # Track provenance for included facts
            for f in valid_other:
                facts_included.append(f.to_fact_metadata())
                inclusion_reasons[f.fact_id] = f"valid {f.memory_type} fact"

        if invalidated:
            invalid_text = "\n".join(
                f"❌ [{f.fact_id}] {f.value}"
                for f in sorted(invalidated, key=lambda x: x.ts)
            )
            parts.append(f"## ⚠️ INVALIDATED - Must Recalculate\n{invalid_text}")

            # Track these as included but marked for review
            for f in invalidated:
                facts_included.append(f.to_fact_metadata())
                inclusion_reasons[f.fact_id] = "included for recalculation (needs_review)"

        # Track superseded facts as excluded
        for fact_id in self.superseded:
            if fact_id in self.facts:
                f = self.facts[fact_id]
                facts_excluded.append(f.to_fact_metadata())
                inclusion_reasons[fact_id] = f"excluded: superseded by {f.superseded_by}"

        # Layer 2c: Corrected values (only show if there are meaningful corrections)
        if self.corrections:
            # Filter to show only significant corrections
            significant_corrections = [
                c for c in self.corrections
                if isinstance(c.get('new_value'), str)
                and isinstance(c.get('old_value'), str)
                and "INVALIDATED" not in str(c['new_value'])
                and "INVALIDATED" not in str(c['old_value'])
                and len(str(c['new_value'])) > 10
                and c['old_value'] != c['new_value']
            ]
            if 0 < len(significant_corrections) <= 3:
                correction_lines: list[str] = []
                for corr in significant_corrections[-3:]:
                    new_val = str(corr['new_value'])[:60]
                    new_ellip = "..." if len(str(corr['new_value'])) > 60 else ""
                    old_val = str(corr['old_value'])[:40]
                    old_ellip = "..." if len(str(corr['old_value'])) > 40 else ""
                    correction_lines.append(
                        f"🔄 {new_val}{new_ellip}\n   (was: {old_val}{old_ellip})"
                    )
                corrected_text = "\n".join(correction_lines)
                parts.append(f"## 🔄 Recent Corrections\n{corrected_text}")

        # Layer 2d: Superseded facts overview
        if self.superseded:
            superseded_text = "\n".join(
                f"- {fact_id}: superseded" for fact_id in sorted(self.superseded)
            )
            parts.append(f"## ⚠️ Superseded Facts\n{superseded_text}")

        # Layer 3: Working Set (include all but mark hypothetical/draft)
        if self.working_set:
            working_lines: list[str] = []
            for item in self.working_set:
                scope = item.get("scope", "global")
                content = str(item["content"])
                if scope == "hypothetical":
                    working_lines.append(f"[HYPOTHETICAL] {content}")
                elif scope == "draft":
                    working_lines.append(f"[DRAFT] {content}")
                else:
                    working_lines.append(content)
            if working_lines:
                working_text = "\n".join(working_lines)
                parts.append(f"## Recent Context\n{working_text}")

        # Layer 4: Environment
        if self.environment:
            sorted_env = sorted(
                self.environment.items(),
                key=lambda kv: self._environment_ts.get(kv[0], datetime.min),
                reverse=True,
            )[:5]
            env_text = "\n".join(
                f"- {k}: {v}" for k, v in sorted_env
            )
            parts.append(f"## Environment\n{env_text}")

        if self.known_unknowns:
            unknown_text = "\n".join(
                f"- {text}" for text, _ in sorted(
                    self.known_unknowns.items(), key=lambda kv: kv[1], reverse=True
                )
            )
            parts.append(f"## Known Unknowns\n{unknown_text}")

        context = "\n\n".join(parts)

        return ContextResult(
            context=context,
            facts_included=facts_included,
            facts_excluded=facts_excluded,
            inclusion_reasons=inclusion_reasons,
            token_count=self._count_tokens(context),
        )

    def reset(self) -> None:
        """Reset all state layers."""
        self.identity = None
        self.facts = {}
        self.facts_by_key = {}
        self.superseded = set()
        self.working_set = []
        self.environment = {}
        self._environment_ts = {}
        self.known_unknowns = {}
        self.corrections = []
        self._fact_id_counter = 0

    def get_system_prompt(self) -> str:
        return (
            "You are an AI agent. Answer based ONLY on the structured context.\n\n"
            "CRITICAL RULES:\n"
            "1. CHECK ALL CONSTRAINTS before deciding - if ANY blocks, answer NO\n"
            "2. Multiple constraints must ALL be satisfied simultaneously\n"
            "3. NEVER invent details not explicitly stated (budgets, timelines)\n"
            "4. If info wasn't provided, say 'not specified' - don't assume\n"
            "5. Items marked [HYPOTHETICAL] are what-if scenarios - not real\n"
            "6. Items marked [DRAFT] are tentative - not finalized\n\n"
            "⚠️ REPAIR/CORRECTION RULES:\n"
            "7. If facts are marked [INVALIDATED], their conclusions are WRONG\n"
            "8. You MUST recalculate using the CORRECTED values\n"
            "9. When base data changes, derived conclusions change too\n\n"
            "Be accurate, concise, and explicit about what you know vs. don't know."
        )
