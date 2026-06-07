"""Memgine + NL-distillation front-end (CAR-backed).

Memgine derives its supersession/constraint intelligence from *structured*
StateWrite/Supersession events. On conversational input (LoCoMo-style, and the
StateBench `supersession_detection` track) there are no such events -- state, and
its corrections, live only in dialogue text -- so Memgine never marks anything
superseded and the old value leaks via the raw transcript.

This strategy attempts to close that gap: it uses a local CAR model to distill
durable facts from each conversation turn (with explicit, reasoning-based
supersession detection -- "which existing fact does this invalidate?"), and feeds
them into a MemgineEngine as structured facts. The engine's normal machinery
(with the anti-resurrection `show_superseded_values` fix) then handles the rest.

EMPIRICAL RESULT (StateBench supersession_detection, dev n=15, CAR-local):
this UNDERPERFORMS plain memgine. With a qwen3-4b extractor + same-key matching:
SFRR 0.47 -> 1.00. With a qwen3-8b extractor + reasoning-based supersession
detection: acc 0.93 -> 0.60, SFRR 0.47 -> 0.87. The cause is information loss:
replacing the raw transcript (which a capable reader parses correctly) with lossy
local-model extraction discards the correct value and presents mis-extracted or
un-superseded facts as current. Implicit supersessions here are often semantic /
authority-based across *different* attributes (a CEO directive overriding a team
decision), which neither key-matching nor a weak local reasoner reliably catches.

Conclusion: eager structured distillation needs a frontier-grade extractor to beat
simply preserving the transcript and reasoning over it at query time. Kept as a
documented baseline + a harness for future frontier-extractor experiments; NOT a
recommended default. See [[memgine-supersession-annotation-leak]] notes.

All inference routes through CAR (local models).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from statebench.baselines.base import ContextResult, MemoryStrategy
from statebench.memgine.config import MemgineConfig
from statebench.memgine.engine import MemgineEngine
from statebench.runner.car_provider import car_extract_json
from statebench.schema.state import Source
from statebench.schema.timeline import ConversationTurn, Event, InitialState

_EXTRACT_PROMPT = """You maintain a memory of durable state from a conversation. Given the CURRENT known facts and a NEW message, output the facts the message asserts or updates, and mark which existing fact (if any) each one SUPERSEDES.

CURRENT FACTS (key: value):
{current}

NEW MESSAGE:
{speaker}: {text}

Output ONLY a JSON array. Each item:
  {{"key": "<short snake_case attribute>", "value": "<current value, short>", "supersedes": "<existing key this invalidates, or null>"}}

Rules:
- Emit a fact only for durable state: decisions, policies, locations, amounts, deadlines, assignments, statuses, preferences, commitments. Skip questions/acks/chit-chat.
- A fact SUPERSEDES an existing one when it replaces it — the SAME attribute with a new value, OR a higher-authority/newer decision that overrides an earlier one (e.g. a CEO directive overriding a team decision, even on a differently-named attribute). Set "supersedes" to that existing key.
- Reuse an existing key when it's the same attribute. Never emit the old value.
- Return [] if nothing durable.

JSON:"""


def _norm_key(key: str) -> str:
    return "".join(c for c in key.lower().strip() if c.isalnum() or c == "_").strip("_")


class MemgineDistillStrategy(MemoryStrategy):
    """Memgine with a CAR-distillation ingestion front-end for conversational state."""

    def __init__(
        self,
        token_budget: int = 8000,
        extract_model: str = "mlx/qwen3-8b:4bit",
        show_superseded_values: bool = False,
        keep_transcript: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(token_budget)
        config = MemgineConfig(
            token_budget=token_budget,
            show_superseded_values=show_superseded_values,
        )
        self._engine = MemgineEngine(config)
        self._extract_model = extract_model
        # Keep the raw turn in the working set too? Off by default so the test
        # isolates whether structured distillation (not the transcript) carries
        # the corrected value.
        self._keep_transcript = keep_transcript
        # norm_key -> (orig_key, value, fact_id) for the current valid facts.
        self._facts: dict[str, tuple[str, str, str]] = {}
        self._key_to_factid: dict[str, str] = {}
        self._counter = 0

    @property
    def name(self) -> str:
        return "memgine_distill"

    @property
    def expects_initial_state(self) -> bool:
        return True

    def initialize_from_state(self, initial_state: InitialState) -> None:
        self._engine.ingest_initial_state(initial_state)
        for fact in initial_state.persistent_facts:
            nk = _norm_key(fact.key)
            self._facts[nk] = (fact.key, fact.value, fact.id)
            self._key_to_factid[nk] = fact.id

    def process_event(self, event: Event) -> None:
        if isinstance(event, ConversationTurn):
            self._distill_turn(event)
        else:
            # Explicit StateWrite / Supersession events pass straight through.
            self._engine.ingest_statebench_event(event)
            for write in getattr(event, "writes", []) or []:
                if getattr(write, "layer", None) == "persistent_facts":
                    nk = _norm_key(write.key)
                    self._facts[nk] = (write.key, write.value, write.id)
                    self._key_to_factid[nk] = write.id

    def _distill_turn(self, event: ConversationTurn) -> None:
        facts = self._extract(event.text, event.speaker)
        for fact in facts:
            key = str(fact.get("key", "")).strip()
            value = str(fact.get("value", "")).strip()
            if not key or not value:
                continue
            nk = _norm_key(key)
            # Supersession: the model's explicit "supersedes" key, else same key.
            sup_key = fact.get("supersedes")
            sup_nk = _norm_key(str(sup_key)) if sup_key else nk
            supersedes_fid = self._facts.get(sup_nk, (None, None, None))[2]
            self._counter += 1
            fid = f"distill-{self._counter}"
            self._engine.ingest_fact(
                fact_id=fid,
                key=key,
                value=value,
                source=Source(type="user", identity=event.speaker, authority="peer"),
                ts=event.ts,
                supersedes=supersedes_fid,
            )
            # Retire the superseded key, register the new fact under its key.
            if supersedes_fid and sup_nk in self._facts and sup_nk != nk:
                del self._facts[sup_nk]
            self._facts[nk] = (key, value, fid)
            self._key_to_factid[nk] = fid
        if self._keep_transcript:
            self._engine.ingest_conversation(event.speaker, event.text, event.ts)

    def _extract(self, text: str, speaker: str) -> list[dict[str, Any]]:
        current = (
            "\n".join(f"- {k}: {v}" for (k, v, _) in self._facts.values())
            or "(none yet)"
        )
        prompt = _EXTRACT_PROMPT.format(current=current, speaker=speaker, text=text)
        result = car_extract_json(prompt, model=self._extract_model)
        if isinstance(result, list):
            return [f for f in result if isinstance(f, dict)]
        return []

    def build_context(self, query: str) -> ContextResult:
        return self._engine.build_context(query)

    def consolidate(self) -> dict[str, int]:
        return self._engine.consolidate()

    def reset(self) -> None:
        self._engine.reset()
        self._facts = {}
        self._key_to_factid = {}
        self._counter = 0

    def get_system_prompt(self) -> str:
        # Reuse Memgine's system prompt for parity.
        from statebench.memgine.strategy import MemgineStrategy

        return MemgineStrategy.get_system_prompt(self)  # type: ignore[arg-type]
