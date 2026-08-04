"""Skill-augmented memory baselines (Paper 4).

``skill_augmented`` is Memgine plus retrieved frontier-distilled skills. Two
controls ship alongside it, because each is a live deflationary explanation:

``skill_augmented_handwritten``
    The same pipeline with a hand-written procedure instead of a distilled one.
    If this matches, the distillation is doing nothing and the result is prompt
    engineering.

``skill_augmented_offfamily``
    Skills from the wrong task family. Measures over-generalization: a skill that
    fires here is worse than no skill at all.

The skill store is injected rather than constructed, so the same strategy class
serves every arm and the arms cannot differ by accident.
"""

from __future__ import annotations

from typing import Any

from statebench.baselines.base import ContextResult, MemoryStrategy
from statebench.schema.timeline import Event, InitialState
from statebench.skills.store import Skill, SkillStore

# A competent engineer's hand-written procedure for state-tracking questions,
# written WITHOUT looking at any distilled skill. This is the control: if it
# performs as well as distilled artifacts, artifact distillation adds nothing
# over ordinary prompt engineering.
HANDWRITTEN = Skill(
    skill_id="SK-HANDWRITTEN",
    task_family="*",
    principle=(
        "answer only from the currently valid state, and prefer the most recent "
        "authoritative value for the attribute the question asks about"
    ),
    procedure=[
        "identify which attribute the question is asking about",
        "find every value that has been asserted for that attribute",
        "discard any value that a later event replaced or invalidated",
        "if a policy or higher-authority source constrains the value, apply it",
        "answer from what remains; if nothing remains, say the information is unavailable",
    ],
    preconditions=["the question asks about a specific attribute of state"],
    counter_indications=[
        "the question asks about history rather than the current value",
    ],
    derived_by="hand-written control",
    confidence=0.5,
)


class SkillAugmentedStrategy(MemoryStrategy):
    """Memgine context plus retrieved skills."""

    def __init__(
        self,
        token_budget: int = 8000,
        model: str | None = None,
        skills: SkillStore | None = None,
        task_family: str | None = None,
        top_k: int = 2,
        **_: Any,
    ) -> None:
        super().__init__(token_budget)
        from statebench.memgine.engine import MemgineEngine

        self._engine = MemgineEngine()
        self._skills = skills or SkillStore()
        self._task_family = task_family
        self._top_k = top_k
        self._actor_scope: dict[str, str] = {}
        self.last_retrieved: list[Skill] = []

    @property
    def name(self) -> str:
        return "skill_augmented"

    @property
    def expects_initial_state(self) -> bool:
        return True

    def initialize_from_state(self, initial_state: InitialState) -> None:
        self._engine.ingest_initial_state(initial_state)
        ident = initial_state.identity_role
        # The actor's governance scope, used to withhold skills distilled under a
        # different scope (Paper 4 RQ5).
        self._actor_scope = {
            k: str(v)
            for k, v in (
                ("org", ident.organization),
                ("actor_authority", ident.authority),
            )
            if v
        }

    def process_event(self, event: Event) -> None:
        self._engine.ingest_statebench_event(event)

    def build_context(self, query: str) -> ContextResult:
        result = self._engine.build_context(query)
        retrieved = self._skills.retrieve(
            query,
            task_family=self._task_family,
            top_k=self._top_k,
            actor_scope=self._actor_scope or None,
        )
        self.last_retrieved = retrieved
        if retrieved:
            block = "\n".join(s.render() for s in retrieved)
            result.context = (
                f"{result.context}\n\n## Applicable Procedure\n{block}"
                if result.context
                else f"## Applicable Procedure\n{block}"
            )
        return result

    def get_system_prompt(self) -> str:
        base = self._engine_prompt()
        if self._skills:
            base += (
                " An 'Applicable Procedure' section may be present. It describes a "
                "general method learned from similar past problems; follow it, but "
                "take every concrete value from the state above, never from the "
                "procedure. If the procedure's stated non-applicability conditions "
                "hold, ignore it."
            )
        return base

    @staticmethod
    def _engine_prompt() -> str:
        from statebench.baselines.state_based import StateBasedStrategy

        return StateBasedStrategy().get_system_prompt()

    def reset(self) -> None:
        self._engine.reset()
        self._actor_scope = {}
        self.last_retrieved = []


def make_handwritten_store() -> SkillStore:
    """Control arm: one hand-written procedure, no distillation."""
    store = SkillStore()
    store.add(HANDWRITTEN)
    return store
