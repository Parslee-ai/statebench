"""Skill artifacts and their store.

A skill carries its own applicability conditions. That is the property that makes
it a governed object rather than a prompt fragment: it states when it applies,
when it does *not*, which episodes produced it, and under whose scope — so it can
be audited, attributed, and revoked.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Skill:
    """A transferable procedure with explicit applicability conditions."""

    skill_id: str
    task_family: str
    principle: str
    procedure: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    # When NOT to fire. Paper 3 §7.2 found a reconstruction stage that never
    # declines; carrying negative conditions in the artifact moves that judgment
    # out of the model. Whether it helps is an open question (Paper 4 RQ4).
    counter_indications: list[str] = field(default_factory=list)
    source_episodes: list[str] = field(default_factory=list)
    # Governance context inherited from the distilling episodes. A skill derived
    # from one org's episodes must not silently serve another's (RQ5).
    source_scope: dict[str, str] = field(default_factory=dict)
    derived_by: str = ""
    confidence: float = 0.5

    def render(self) -> str:
        lines = [f"[{self.skill_id}] {self.principle}"]
        for step in self.procedure:
            lines.append(f"    - {step}")
        if self.counter_indications:
            lines.append(f"    does NOT apply when: {'; '.join(self.counter_indications)}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)


_TOKEN = re.compile(r"\W+")


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.split(text.lower()) if len(t) >= 3}


class SkillStore:
    """Holds skills and retrieves them by task family and query relevance.

    Retrieval deliberately uses the same keyword-overlap primitive as the memory
    engine's relevance scorer. A stronger retriever here would confound the
    question under test, which is whether the *artifact* helps — not whether a
    better retriever does.
    """

    def __init__(self) -> None:
        self._skills: list[Skill] = []

    def __len__(self) -> int:
        return len(self._skills)

    def add(self, skill: Skill) -> None:
        self._skills.append(skill)

    def all(self) -> list[Skill]:
        return list(self._skills)

    def retrieve(
        self,
        query: str,
        task_family: str | None = None,
        top_k: int = 2,
        actor_scope: dict[str, str] | None = None,
    ) -> list[Skill]:
        """Retrieve applicable skills.

        ``actor_scope`` enforces the governance condition: a skill whose
        ``source_scope`` conflicts with the requesting actor's scope is withheld.
        This is `Γ` applied to the experience plane (Paper 3 §3) — the same
        restriction that our first reconstruction implementation violated.
        """
        q = _tokens(query)
        scored: list[tuple[float, Skill]] = []
        for skill in self._skills:
            if task_family and skill.task_family != task_family:
                continue
            if actor_scope and self._scope_conflict(skill, actor_scope):
                continue
            text = f"{skill.principle} {' '.join(skill.procedure)} {' '.join(skill.preconditions)}"
            overlap = q & _tokens(text)
            score = len(overlap) / len(q) if q else 0.0
            scored.append((score, skill))
        scored.sort(key=lambda pair: (-pair[0], pair[1].skill_id))
        return [s for score, s in scored[:top_k] if score > 0]

    @staticmethod
    def _scope_conflict(skill: Skill, actor_scope: dict[str, str]) -> bool:
        """True when a skill's origin scope excludes this actor."""
        for key, value in skill.source_scope.items():
            actor_value = actor_scope.get(key)
            if actor_value is not None and actor_value != value:
                return True
        return False

    # -- persistence -------------------------------------------------------- #

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([s.to_dict() for s in self._skills], indent=1))

    @classmethod
    def load(cls, path: Path) -> SkillStore:
        store = cls()
        for raw in json.loads(Path(path).read_text()):
            store.add(Skill(**raw))
        return store

    def clear(self) -> None:
        self._skills.clear()
