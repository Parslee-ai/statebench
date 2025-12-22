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
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
import tiktoken

from statebench.baselines.base import MemoryStrategy
from statebench.schema.timeline import (
    Event,
    ConversationTurn,
    StateWrite,
    Supersession,
    InitialState,
)
from statebench.schema.state import IdentityRole


@dataclass
class EnhancedFact:
    """A fact with memory type, scope, dependencies, and constraint metadata."""
    key: str
    value: str
    source: str
    ts: datetime
    is_valid: bool = True
    superseded_by: str | None = None

    # Tri-partite memory classification (from paper)
    # - user: Private preferences, corrections, decisions (User Memory)
    # - capability: Learned patterns, heuristics (Capability Memory)
    # - organizational: Policies, system data, documents (Organizational Knowledge)
    memory_type: Literal["user", "capability", "organizational"] = "user"

    # Scope tracking (fixes scope_leak)
    scope: Literal["global", "task", "hypothetical", "draft", "session"] = "global"
    scope_id: str | None = None

    # Dependency tracking (fixes repair_propagation)
    depends_on: list[str] = field(default_factory=list)
    derived_facts: list[str] = field(default_factory=list)
    needs_review: bool = False

    # Constraint metadata (fixes causality)
    is_constraint: bool = False
    constraint_type: str | None = None


class StateBasedStrategy(MemoryStrategy):
    """State-based context with explicit supersession tracking.

    This strategy maintains the four-layer state architecture:
    - Identity & Role: Static user/role information
    - Persistent Facts: Durable facts with supersession, scope, and dependency tracking
    - Working Set: Recent conversation context
    - Environment: Real-time signals (timestamps, etc.)
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
        self.facts: dict[str, EnhancedFact] = {}
        self.superseded: set[str] = set()
        self.working_set: list[dict] = []
        self.environment: dict[str, str | datetime] = {}
        self._environment_ts: dict[str, datetime] = {}

        # Known unknowns (prevents hallucination)
        self.known_unknowns: dict[str, datetime] = {}

        # Correction history (for repair propagation)
        self.corrections: list[dict] = []

        self._encoder = tiktoken.get_encoding("cl100k_base")

    @property
    def name(self) -> str:
        return "state_based"

    @property
    def expects_initial_state(self) -> bool:
        return True

    def _count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def _is_constraint(self, value: str, source: str) -> bool:
        """Detect if a fact is a formal constraint (not casual mentions)."""
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
        if source == "policy":
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

    def _infer_memory_type(self, source: str, value: str) -> str:
        """Classify fact into tri-partite memory structure.

        From paper:
        - User Memory: Private preferences, corrections, decisions
        - Capability Memory: Learned patterns, heuristics, strategies
        - Organizational Knowledge: Policies, system data, documents
        """
        # Organizational sources - policies, systems, documents
        org_sources = {
            "policy", "finance_system", "hr_system", "calendar_system",
            "inventory_system", "crm_system", "erp_system", "document",
            "sharepoint", "confluence", "database"
        }
        if source in org_sources:
            return "organizational"

        # Capability memory - learned patterns (not yet used in benchmark)
        capability_sources = {"observation", "pattern", "heuristic", "strategy"}
        if source in capability_sources:
            return "capability"

        # User memory - preferences, decisions, corrections
        # Default: user, decision, preference, correction, etc.
        return "user"

    def initialize_from_state(self, initial_state: InitialState) -> None:
        """Initialize state from timeline's initial state."""
        self.identity = initial_state.identity_role

        for fact in initial_state.persistent_facts:
            enhanced = EnhancedFact(
                key=fact.key,
                value=fact.value,
                source=fact.source,
                ts=fact.ts,
                is_valid=fact.is_valid,
                memory_type=self._infer_memory_type(fact.source, fact.value),
                is_constraint=self._is_constraint(fact.value, fact.source),
                constraint_type=self._infer_constraint_type(fact.value),
            )
            self.facts[fact.key] = enhanced

        self.working_set = [
            {"content": item.content, "ts": item.ts, "scope": "global"}
            for item in initial_state.working_set
        ]

        self.environment = dict(initial_state.environment)
        # Initialize environment timestamps so later writes can compare freshness
        now = datetime.min
        self._environment_ts = {k: now for k in self.environment.keys()}

    def process_event(self, event: Event) -> None:
        """Process an event and update state layers."""
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
                    # Infer dependencies
                    deps = self._infer_dependencies(write.value)

                    fact = EnhancedFact(
                        key=write.key,
                        value=write.value,
                        source="decision",
                        ts=event.ts,
                        memory_type=self._infer_memory_type("decision", write.value),
                        depends_on=deps,
                        is_constraint=self._is_constraint(write.value, "decision"),
                        constraint_type=self._infer_constraint_type(write.value),
                    )
                    self.facts[write.key] = fact

                    # Update parent facts
                    for dep_key in deps:
                        if dep_key in self.facts:
                            self.facts[dep_key].derived_facts.append(write.key)
                elif write.layer == "environment":
                    self._update_environment(write.key, write.value, event.ts, write.supersedes)

        elif isinstance(event, Supersession):
            for write in event.writes:
                if write.layer == "persistent_facts":
                    old_value = None
                    old_memory_type = "user"
                    if write.supersedes and write.supersedes in self.facts:
                        old_fact = self.facts[write.supersedes]
                        old_value = old_fact.value
                        old_memory_type = old_fact.memory_type  # Preserve memory type
                        old_fact.is_valid = False
                        old_fact.superseded_by = write.key
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

                    deps = self._infer_dependencies(write.value)
                    fact = EnhancedFact(
                        key=write.key,
                        value=write.value,
                        source="decision",
                        ts=event.ts,
                        memory_type=old_memory_type,  # Inherit from superseded fact
                        depends_on=deps,
                        is_constraint=self._is_constraint(write.value, "decision"),
                        constraint_type=self._infer_constraint_type(write.value),
                    )
                    self.facts[write.key] = fact
                elif write.layer == "environment":
                    self._update_environment(write.key, write.value, event.ts, write.supersedes)

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
        """Infer which existing facts this value depends on."""
        deps = []
        value_tokens = self._extract_keywords(value)

        for key, fact in self.facts.items():
            if fact.is_valid:
                keywords = self._extract_keywords(fact.value)
                keywords.update(self._extract_keywords(fact.key))
                if value_tokens & keywords:
                    deps.append(key)

        return deps

    def _propagate_invalidation(self, superseded_key: str) -> None:
        """Mark all facts derived from a superseded fact as needing review."""
        if superseded_key not in self.facts:
            return

        old_fact = self.facts[superseded_key]
        for derived_key in old_fact.derived_facts:
            if derived_key in self.facts:
                self.facts[derived_key].needs_review = True
                self._propagate_invalidation(derived_key)

    def _get_valid_facts(self) -> list[EnhancedFact]:
        """Return only currently valid facts."""
        return [f for f in self.facts.values() if f.is_valid]

    def _get_constraints(self) -> list[EnhancedFact]:
        """Return all active constraints."""
        return [f for f in self.facts.values() if f.is_valid and f.is_constraint]

    def build_context(self, query: str) -> str:
        """Build structured context from state layers."""
        parts = []

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
                f"⚠️ [{c.constraint_type or 'CONSTRAINT'}] {c.value}"
                for c in sorted(constraints, key=lambda x: x.ts)
            )
            parts.append(f"## ⚠️ Active Constraints (CHECK ALL)\n{constraint_text}")

        # Layer 2b: Facts by memory type (tri-partite structure from paper)
        # Display together but with memory type labels for clarity
        other_facts = [f for f in self._get_valid_facts() if not f.is_constraint]

        # Check for invalidated facts that need recalculation
        invalidated = [f for f in other_facts if "[INVALIDATED" in f.value or f.needs_review]
        valid_other = [f for f in other_facts if "[INVALIDATED" not in f.value and not f.needs_review]

        if valid_other:
            # Memory type abbreviations for compact display
            type_labels = {"organizational": "org", "user": "usr", "capability": "cap"}
            facts_text = "\n".join(
                f"- [{type_labels.get(f.memory_type, 'usr')}] {f.value}"
                for f in sorted(valid_other, key=lambda x: x.ts)
            )
            parts.append(f"## Current Facts\n{facts_text}")

        if invalidated:
            invalid_text = "\n".join(
                f"❌ {f.value}"
                for f in sorted(invalidated, key=lambda x: x.ts)
            )
            parts.append(f"## ⚠️ INVALIDATED - Must Recalculate\n{invalid_text}")

        # Layer 2c: Corrected values (only show if there are meaningful corrections)
        if self.corrections:
            # Filter to show only significant corrections (not invalidated or trivial)
            significant_corrections = [
                c for c in self.corrections
                if "INVALIDATED" not in c['new_value']
                and "INVALIDATED" not in c['old_value']
                and len(c['new_value']) > 10  # Skip very short corrections
                and c['old_value'] != c['new_value']  # Actual change
            ]
            # Only show if we have 1-3 significant corrections (not cluttered)
            if 0 < len(significant_corrections) <= 3:
                corrected_text = "\n".join(
                    f"🔄 {c['new_value'][:60]}{'...' if len(c['new_value']) > 60 else ''}"
                    f"\n   (was: {c['old_value'][:40]}{'...' if len(c['old_value']) > 40 else ''})"
                    for c in significant_corrections[-3:]
                )
                parts.append(f"## 🔄 Recent Corrections\n{corrected_text}")

        # Layer 2d: Superseded facts overview
        if self.superseded:
            superseded_text = "\n".join(
                f"- {key}: superseded" for key in sorted(self.superseded)
            )
            parts.append(f"## ⚠️ Superseded Facts\n{superseded_text}")

        # Layer 3: Working Set (include all but mark hypothetical/draft)
        if self.working_set:
            working_lines = []
            for item in self.working_set:
                scope = item.get("scope", "global")
                content = item["content"]
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
                key=lambda item: self._environment_ts.get(item[0], datetime.min),
                reverse=True,
            )[:5]
            env_text = "\n".join(
                f"- {k}: {v}" for k, v in sorted_env
            )
            parts.append(f"## Environment\n{env_text}")

        if self.known_unknowns:
            unknown_text = "\n".join(
                f"- {text}" for text, _ in sorted(
                    self.known_unknowns.items(), key=lambda item: item[1], reverse=True
                )
            )
            parts.append(f"## Known Unknowns\n{unknown_text}")

        return "\n\n".join(parts)

    def reset(self) -> None:
        """Reset all state layers."""
        self.identity = None
        self.facts = {}
        self.superseded = set()
        self.working_set = []
        self.environment = {}
        self._environment_ts = {}
        self.known_unknowns = {}
        self.corrections = []

    def get_system_prompt(self) -> str:
        return (
            "You are an AI agent. Answer based ONLY on the structured context provided.\n\n"
            "CRITICAL RULES:\n"
            "1. CHECK ALL CONSTRAINTS before deciding - if ANY constraint blocks, the answer is NO\n"
            "2. Multiple constraints must ALL be satisfied simultaneously\n"
            "3. NEVER invent details not explicitly stated (budgets, timelines, approvals)\n"
            "4. If information wasn't provided, say 'not specified' - don't assume or guess\n"
            "5. Items marked [HYPOTHETICAL] are what-if scenarios - don't treat as real\n"
            "6. Items marked [DRAFT] are tentative - not finalized\n\n"
            "⚠️ REPAIR/CORRECTION RULES:\n"
            "7. If facts are marked [INVALIDATED], their conclusions are WRONG\n"
            "8. You MUST recalculate using the CORRECTED values, not the old conclusions\n"
            "9. When base data changes, derived conclusions change too\n\n"
            "Be accurate, concise, and explicit about what you know vs. don't know."
        )
