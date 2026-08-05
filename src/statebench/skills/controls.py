"""Deflationary controls for the skill-transfer experiment.

The claim under test is that skills distilled from *worked episodes* lift a weak
model. Two cheaper things could produce the same lift, and if either does, the
claim is about prompt engineering rather than about experience transfer.

``targeted``
    A procedure hand-written for this specific task family by someone who knows
    the task. Tests whether domain expertise alone suffices.

``description_only``
    A procedure written by the same frontier model that does the distillation,
    but shown *only the one-line task description* -- no episodes. This is the
    sharper control: it holds the author constant and varies only whether
    worked examples were available, isolating what the episodes contribute.

The targeted control carries a contamination caveat that the description-only
one does not: it was written after seeing the distilled skills, so it cannot be
fully independent of them. The description-only control is generated fresh and
has no such exposure, which is why it is the one to weight.
"""

from __future__ import annotations

from typing import Any

from statebench.skills.store import Skill, SkillStore

# One-line task descriptions, taken verbatim from the public track table in the
# repository README. This is what the description-only control is allowed to see.
TRACK_DESCRIPTIONS = {
    "repair_propagation": "Corrections cascade to derived conclusions",
    "supersession": "Facts invalidated by newer facts stay dead",
    "supersession_detection": "Implicit supersession without explicit markers",
    "commitment_durability": "Confirmed commitments persist across interruptions",
    "interruption_resumption": "Context survives topic switches",
    "scope_permission": "Role-based access control enforcement",
    "environmental_freshness": "Time-sensitive state expiration",
    "authority_hierarchy": "Higher-authority sources override lower",
    "enterprise_privacy": "Confidential information stays restricted",
    "hallucination_resistance": "System refuses to invent unstated facts",
    "scope_leak": "Hypothetical/draft content stays contained",
    "causality": "Causal chain and dependency reasoning",
    "brutal_realistic": "Multi-failure compound scenarios",
}

# Hand-written targeted procedures. Written from the task description and an
# understanding of what the track requires -- NOT copied from any distilled
# artifact -- but authored after the distilled skills had been seen, so treat
# this control as contaminated in the direction of *favouring* it.
TARGETED: dict[str, Skill] = {
    "repair_propagation": Skill(
        skill_id="SK-TARGETED-repair",
        task_family="repair_propagation",
        principle=(
            "when a value is corrected, any conclusion computed from the old "
            "value is stale and must be recomputed rather than reused"
        ),
        procedure=[
            "identify what the question asks for",
            "if it is a derived figure, find the inputs it was computed from",
            "check whether any of those inputs was later corrected",
            "if so, treat the previously stated conclusion as void",
            "recompute from the corrected inputs and give the new result",
            "do not repeat the superseded conclusion as if it still held",
        ],
        preconditions=["the answer depends on a value that was later corrected"],
        counter_indications=[
            "the question asks only for a directly stated fact with no derivation",
        ],
        derived_by="hand-written targeted control",
        confidence=0.5,
    ),
    "supersession": Skill(
        skill_id="SK-TARGETED-supersession",
        task_family="supersession",
        principle=(
            "when a later authoritative statement replaces an earlier one for the "
            "same attribute, only the later value is current"
        ),
        procedure=[
            "identify the attribute the question asks about",
            "collect every value asserted for that attribute, in order",
            "keep the latest one that was not itself replaced",
            "answer from that value alone",
        ],
        preconditions=["two or more values were asserted for one attribute"],
        counter_indications=[
            "the later statement concerns a different attribute",
            "the later statement adds detail without replacing the value",
        ],
        derived_by="hand-written targeted control",
        confidence=0.5,
    ),
}

DESCRIPTION_ONLY_PROMPT = """\
You are writing a reusable procedure for a smaller language model to follow when \
it answers questions in a particular task family.

All you are told about the task family is its name and one-line description:

  name: {family}
  description: {description}

Write the procedure you would give a smaller model to help it answer these \
questions correctly. You have not seen any examples, so write what the \
description implies.

Output ONLY a JSON object:
{{
  "principle": "<one sentence>",
  "procedure": ["<step>", "<step>", "..."],
  "preconditions": ["<when this applies>", "..."],
  "counter_indications": ["<when this must NOT be applied>", "..."]
}}
"""


def make_targeted_store(task_family: str) -> SkillStore:
    """Hand-written procedure targeted at one task family."""
    store = SkillStore()
    skill = TARGETED.get(task_family)
    if skill:
        store.add(skill)
    return store


# Deliberately impoverished descriptions: true of the track, but giving no hint
# of the method. The rich description for repair_propagation ("corrections
# cascade to derived conclusions") nearly *is* the procedure, which is the
# leading explanation for why distilled episodes added nothing over it. Holding
# the task fixed and varying only what the description reveals isolates that.
IMPOVERISHED_DESCRIPTIONS = {
    "repair_propagation": (
        "Business scenarios in which information changes over the course of a "
        "conversation"
    ),
    "supersession": (
        "Business scenarios in which information changes over the course of a "
        "conversation"
    ),
    "brutal_realistic": "Multi-failure compound scenarios",
}


def make_description_only_store(
    task_family: str, complete: Any, model: str, impoverished: bool = False
) -> SkillStore:
    """Frontier-written procedure from the task description alone, no episodes.

    Holds the author constant against the distillation arm and varies only
    whether worked examples were available.
    """
    from statebench.skills.distill import _extract_json_object

    if impoverished:
        description = IMPOVERISHED_DESCRIPTIONS.get(task_family, task_family)
    else:
        description = TRACK_DESCRIPTIONS.get(task_family, task_family)
    raw = complete(
        DESCRIPTION_ONLY_PROMPT.format(family=task_family, description=description),
        model=model,
        max_tokens=600,
    )
    parsed = _extract_json_object(raw) or {}
    store = SkillStore()
    if parsed.get("principle"):
        store.add(
            Skill(
                skill_id=(
                    f"SK-DESCPOOR-{task_family[:12]}" if impoverished
                    else f"SK-DESCONLY-{task_family[:12]}"
                ),
                task_family=task_family,
                principle=str(parsed["principle"]).strip(),
                procedure=[str(x) for x in parsed.get("procedure", [])][:6],
                preconditions=[str(x) for x in parsed.get("preconditions", [])][:4],
                counter_indications=[
                    str(x) for x in parsed.get("counter_indications", [])
                ][:4],
                derived_by=(
                    f"{model} (impoverished description, no episodes)"
                    if impoverished
                    else f"{model} (description only, no episodes)"
                ),
                confidence=0.5,
            )
        )
    return store


# Track pairs whose correct behavior actively conflicts, for the
# over-generalization control. The first run used scope_permission as the
# off-family track and skills *helped* there, which is uninformative: nothing
# about access control contradicts "recompute from corrected inputs".
#
# A real over-generalization test needs a family where following the skill leads
# you wrong. supersession_maintain is that: it is built so the later event looks
# like an update but does NOT invalidate anything, so a procedure that says
# "treat the earlier conclusion as void" produces the wrong answer.
CONFLICTING_OFF_FAMILY = {
    "repair_propagation": "supersession_maintain",
    "supersession": "supersession_maintain",
    "authority_hierarchy": "authority_maintain",
}
