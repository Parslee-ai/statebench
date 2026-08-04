"""Distill reusable skills from solved episodes.

The frontier model is shown several episodes of one task family — the events, the
question, and the correct answer — and asked for the *transferable procedure*,
explicitly not the answers.

**Contamination is the central risk.** If a distilled skill contains an evaluation
instance's entity, amount, or date, then any measured "transfer" is answer
leakage and the result is worthless. Two guards:

1. distillation and evaluation draw from disjoint splits (train vs dev);
2. ``scrub_check`` rejects any skill containing a literal from the evaluation set.

The second is not redundant with the first: templates reuse entity pools across
splits, so a skill can name an entity that also appears in evaluation even when
the *episodes* are disjoint.
"""

from __future__ import annotations

import json
import re
from typing import Any

from statebench.skills.store import Skill

DISTILL_PROMPT = """\
You are extracting a reusable SKILL from worked examples, for a smaller model to \
use later on new problems of the same kind.

Below are {n} solved episodes from the task family "{family}". Each shows what \
happened, what was asked, and the correct answer.

{episodes}

Write the transferable procedure. Rules:
- State the general principle, not the answers. A later problem will have \
different names, amounts, and dates.
- NEVER include any specific entity name, amount, date, or value from these \
episodes. If you name a specific value the skill is useless and will be discarded.
- Say when the procedure does NOT apply — that matters as much as when it does.

Output ONLY a JSON object:
{{
  "principle": "<one sentence, general>",
  "procedure": ["<step>", "<step>", "..."],
  "preconditions": ["<structural condition that must hold>", "..."],
  "counter_indications": ["<when this must NOT be applied>", "..."]
}}
"""


def _episode_text(timeline: Any, ground_truth_answer: str) -> str:
    """Render one episode compactly for the distiller."""
    parts = []
    for event in timeline.events:
        kind = getattr(event, "type", "")
        if kind == "conversation_turn":
            parts.append(f"  {event.speaker}: {event.text}")
        elif kind in ("state_write", "supersession"):
            for w in event.writes:
                marker = "SUPERSEDES " + str(w.supersedes) if w.supersedes else "writes"
                parts.append(f"  [{kind}] {marker} {w.key} = {w.value}")
        elif kind == "query":
            parts.append(f"  QUESTION: {event.prompt}")
    parts.append(f"  CORRECT ANSWER: {ground_truth_answer}")
    return "\n".join(parts)


def scrub_check(skill: Skill, forbidden_literals: set[str]) -> list[str]:
    """Return evaluation-set literals leaking into a skill (empty = clean).

    Case-insensitive, word-boundary matched, so "Acme" does not fire on "acme-like"
    but does on "Acme Corp".
    """
    blob = " ".join(
        [skill.principle, *skill.procedure, *skill.preconditions, *skill.counter_indications]
    ).lower()
    hits = []
    for lit in forbidden_literals:
        lit = lit.strip()
        if len(lit) < 3:
            continue
        if re.search(rf"(?<!\w){re.escape(lit.lower())}(?!\w)", blob):
            hits.append(lit)
    return sorted(set(hits))


def evaluation_literals(timelines: list[Any]) -> set[str]:
    """Every specific value appearing in the evaluation set.

    Deliberately over-inclusive: a false rejection costs one skill, while a false
    acceptance invalidates the experiment.
    """
    lits: set[str] = set()
    for tl in timelines:
        ident = tl.initial_state.identity_role
        for v in (ident.user_name, ident.organization, ident.department):
            if v:
                lits.add(str(v))
        for fact in tl.initial_state.persistent_facts:
            lits.update(_value_literals(fact.value))
        for event in tl.events:
            for w in getattr(event, "writes", None) or []:
                lits.update(_value_literals(w.value))
            prompt = getattr(event, "prompt", None)
            if prompt:
                lits.update(_value_literals(prompt))
    return {x for x in lits if len(x) >= 3}


_MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")
_PCT = re.compile(r"\d+(?:\.\d+)?\s?%")
_PROPER = re.compile(r"\b[A-Z][a-z]{2,}\b")
_ID = re.compile(r"\b[A-Za-z]*\d[A-Za-z0-9-]*\b")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?:\n])\s+")

# Capitalized tokens that carry no scenario identity. Without this the deny-list
# picks up ordinary words and rejects clean skills: a dry run discarded a correct
# skill because the word "Can" appeared sentence-initially in a scenario prompt.
_COMMON = frozenset(
    {
        "The", "This", "That", "There", "Their", "They", "These", "Those",
        "User", "Please", "Note", "Based", "When", "What", "Which", "Should",
        "Total", "Project", "Team", "Current", "New", "Can", "Could", "Would",
        "Will", "Shall", "Does", "Did", "Are", "Was", "Were", "Have", "Has",
        "How", "Why", "Where", "Who", "And", "But", "For", "With", "From",
        "Given", "Since", "After", "Before", "Now", "Then", "Also", "Both",
        "Confirmed", "Approved", "Updated", "Original", "Final", "Yes", "Not",
    }
)


def _value_literals(text: str) -> set[str]:
    """Specific values in ``text``. Sentence-initial capitals are excluded:
    a leading capital marks position, not identity."""
    out: set[str] = set()
    for pat in (_MONEY, _PCT, _ID):
        out |= {m.group(0).strip() for m in pat.finditer(text)}
    leading = set()
    for sentence in _SENTENCE_SPLIT.split(text):
        head = sentence.strip().split(" ", 1)[0].strip(".,;:?!") if sentence.strip() else ""
        if head:
            leading.add(head)
    out |= {
        m.group(0)
        for m in _PROPER.finditer(text)
        if m.group(0) not in _COMMON and m.group(0) not in leading
    }
    return out


def distill_skills(
    timelines: list[Any],
    answers: list[str],
    task_family: str,
    complete: Any,
    model: str,
    forbidden_literals: set[str] | None = None,
    episodes_per_skill: int = 3,
    max_skills: int = 4,
    source_scope: dict[str, str] | None = None,
) -> tuple[list[Skill], list[dict]]:
    """Distill skills from solved episodes.

    Returns ``(accepted, rejected)`` where each rejection records why, so the
    scrub rate is reportable rather than silent.
    """
    accepted: list[Skill] = []
    rejected: list[dict] = []
    forbidden = forbidden_literals or set()

    batches = [
        (timelines[i : i + episodes_per_skill], answers[i : i + episodes_per_skill])
        for i in range(0, len(timelines), episodes_per_skill)
    ][:max_skills]

    for idx, (batch_tl, batch_ans) in enumerate(batches, 1):
        if not batch_tl:
            continue
        rendered = "\n\n".join(
            f"EPISODE {i+1}:\n{_episode_text(tl, ans)}"
            for i, (tl, ans) in enumerate(zip(batch_tl, batch_ans))
        )
        prompt = DISTILL_PROMPT.format(
            n=len(batch_tl), family=task_family, episodes=rendered
        )
        raw = complete(prompt, model=model, max_tokens=700)
        parsed = _extract_json_object(raw)
        if not parsed or not parsed.get("principle"):
            rejected.append({"batch": idx, "reason": "unparseable distiller output"})
            continue

        skill = Skill(
            skill_id=f"SK-{task_family[:12]}-{idx:03d}",
            task_family=task_family,
            principle=str(parsed.get("principle", "")).strip(),
            procedure=[str(x) for x in parsed.get("procedure", [])][:6],
            preconditions=[str(x) for x in parsed.get("preconditions", [])][:4],
            counter_indications=[str(x) for x in parsed.get("counter_indications", [])][:4],
            source_episodes=[tl.id for tl in batch_tl],
            source_scope=dict(source_scope or {}),
            derived_by=model,
            confidence=0.8,
        )
        leaks = scrub_check(skill, forbidden)
        if leaks:
            rejected.append(
                {"batch": idx, "reason": "contamination", "literals": leaks[:8]}
            )
            continue
        accepted.append(skill)

    return accepted, rejected


def _extract_json_object(text: str) -> dict | None:
    if not text:
        return None
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None
