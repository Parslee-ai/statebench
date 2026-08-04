"""Frontier-distilled skill artifacts (Paper 4).

A *skill* is an experience-plane object in the sense of Paper 3 §3: a transferable
procedure together with the conditions under which it applies. It is produced by a
frontier model observing episodes it solved correctly, and consumed by a weaker
model on later instances of the same abstract task.

The distinction from per-episode distillation (``baselines.memgine_distill``) is
amortization: the frontier model runs N episodes once and then leaves the loop,
while M >> N later episodes are served locally.
"""

from statebench.skills.distill import distill_skills, scrub_check
from statebench.skills.store import Skill, SkillStore

__all__ = ["Skill", "SkillStore", "distill_skills", "scrub_check"]
