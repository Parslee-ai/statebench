"""StateBench strategy wrapper for the Rust car-memgine engine.

Calls the car-memgine-eval binary via subprocess, sending events as JSONL
and receiving context responses. This tests the actual Rust code path.
"""

from __future__ import annotations

import json
import subprocess
import os
from datetime import datetime
from typing import Any

from statebench.baselines.base import ContextResult, FactMetadata, MemoryStrategy
from statebench.schema.state import IdentityRole, Source
from statebench.schema.timeline import (
    ConversationTurn,
    Event,
    InitialState,
    StateWrite,
    Supersession,
)

# Path to the Rust eval binary
EVAL_BINARY = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..",
    "car-rs", "target", "release", "car-memgine-eval"
)

# Try to find it relative to the statebench package
if not os.path.exists(EVAL_BINARY):
    # Try from the car repo root
    _car_root = os.environ.get("CAR_ROOT", os.path.expanduser("~/git/car"))
    EVAL_BINARY = os.path.join(_car_root, "car-rs", "target", "release", "car-memgine-eval")


class CarMemgineStrategy(MemoryStrategy):
    """StateBench strategy backed by the Rust car-memgine graph engine."""

    def __init__(self, token_budget: int = 8000, **kwargs: Any) -> None:
        super().__init__(token_budget)
        self._proc: subprocess.Popen | None = None
        self._start_process()
        self._send({"cmd": "reset", "token_budget": token_budget})

    @property
    def name(self) -> str:
        return "car_memgine"

    @property
    def expects_initial_state(self) -> bool:
        return True

    def _start_process(self) -> None:
        if not os.path.exists(EVAL_BINARY):
            raise FileNotFoundError(
                f"car-memgine-eval binary not found at {EVAL_BINARY}. "
                f"Build it: cd car-rs && cargo build -p car-memgine-eval --release"
            )
        self._proc = subprocess.Popen(
            [EVAL_BINARY],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def _send(self, cmd: dict) -> dict:
        assert self._proc and self._proc.stdin and self._proc.stdout
        line = json.dumps(cmd, default=str) + "\n"
        self._proc.stdin.write(line)
        self._proc.stdin.flush()
        resp_line = self._proc.stdout.readline().strip()
        if not resp_line:
            return {"error": "no response"}
        return json.loads(resp_line)

    def initialize_from_state(self, initial_state: InitialState) -> None:
        if initial_state.identity_role:
            identity = initial_state.identity_role
            self._send({
                "cmd": "identity",
                "user_name": identity.user_name,
                "authority": identity.authority,
                "department": getattr(identity, "department", None),
                "organization": getattr(identity, "organization", None),
            })

        # Ingest initial facts
        for fact in getattr(initial_state, "persistent_facts", []):
            self._ingest_write(fact)

        # Working set items
        for item in getattr(initial_state, "working_set", []):
            content = getattr(item, "content", None) or str(item)
            self._send({
                "cmd": "conversation",
                "speaker": "context",
                "text": content,
                "ts": datetime.now().isoformat(),
            })

        # Environment — can be dict[str, str] or list of signals
        env = getattr(initial_state, "environment", {})
        if isinstance(env, dict):
            for key, value in env.items():
                self._send({
                    "cmd": "environment",
                    "key": key,
                    "value": str(value),
                    "ts": datetime.now().isoformat(),
                })
        elif isinstance(env, list):
            for signal in env:
                self._send({
                    "cmd": "environment",
                    "key": getattr(signal, "key", "signal"),
                    "value": getattr(signal, "value", str(signal)),
                    "ts": datetime.now().isoformat(),
                })

    def process_event(self, event: Event) -> None:
        if isinstance(event, ConversationTurn):
            self._send({
                "cmd": "conversation",
                "speaker": event.speaker,
                "text": event.text,
                "ts": event.ts.isoformat() if hasattr(event, "ts") else None,
            })

        elif isinstance(event, (StateWrite, Supersession)):
            for write in event.writes:
                self._ingest_write(write, is_supersession=isinstance(event, Supersession))

    def _ingest_write(self, write: Any, is_supersession: bool = False) -> None:
        source = getattr(write, "source", None)
        cmd: dict[str, Any] = {
            "cmd": "fact",
            "fact_id": write.id,
            "key": write.key,
            "value": write.value,
            "source_type": source.type if source else "user",
            "authority": source.authority if source else "peer",
            "scope": getattr(write, "scope", "global"),
            "depends_on": list(getattr(write, "depends_on", [])),
            "is_constraint": getattr(write, "is_constraint", False),
        }

        # Handle supersession
        supersedes = getattr(write, "supersedes", None)
        if supersedes:
            cmd["supersedes"] = supersedes

        if hasattr(write, "ts"):
            cmd["ts"] = write.ts.isoformat() if write.ts else None

        self._send(cmd)

    def build_context(self, query: str) -> ContextResult:
        resp = self._send({"cmd": "context", "query": query})

        context = resp.get("context", "")
        token_count = resp.get("token_count", len(context) // 4)

        return ContextResult(
            context=context,
            facts_included=[],  # TODO: extend Rust engine to return provenance
            facts_excluded=[],
            inclusion_reasons={},
            token_count=token_count,
        )

    def reset(self) -> None:
        if self._proc:
            self._send({"cmd": "reset", "token_budget": self.token_budget})

    def get_system_prompt(self) -> str:
        return (
            "You are an AI agent. Answer based ONLY on the structured context.\n\n"
            "CRITICAL RULES:\n"
            "1. For YES/NO decisions, CHECK ALL CONSTRAINTS - if ANY blocks, answer NO\n"
            "2. Multiple constraints must ALL be satisfied simultaneously\n"
            "3. NEVER invent details not explicitly stated — but DO perform "
            "arithmetic on stated values (e.g. $120K - $150K = -$30K) and DO "
            "recalculate schedules when dates shift (preserve the same durations)\n"
            "4. If info wasn't provided, say 'not specified' - don't assume\n"
            "5. Items marked [HYPOTHETICAL] are what-if scenarios - not real\n"
            "6. Items marked [DRAFT] are tentative - not finalized\n\n"
            "ANSWER GUIDANCE:\n"
            "7. If Recent Context CONTRADICTS or UPDATES a Current Fact, the "
            "conversation correction takes precedence over the stored fact\n"
            "8. ALWAYS answer using information from Current Facts when relevant data exists. "
            "Only say 'That information is not available.' when Current Facts contains "
            "absolutely no relevant data — do NOT name entities or terms from the question\n\n"
            "⚠️ REPAIR/CORRECTION RULES:\n"
            "9. Lines marked '⚠️ RECALCULATE' under a fact show stale conclusions — "
            "the base value they used has changed. Recalculate using the current value above, "
            "preserving the same durations and proportions from the original\n"
            "10. Show your arithmetic explicitly when recalculating\n"
            "11. Facts with (depends on: X) are derived from X — if X changed, "
            "recalculate this fact too\n\n"
            "CONTEXT FORMAT:\n"
            "- If info is marked '(changed from: ...)' only the NEW value applies\n"
            "- [org] = organizational/policy data, [usr] = user-provided info\n"
            "- 'Known Unknowns' = things NOT in the data — say 'not specified'\n\n"
            "Be accurate, concise, and explicit about what you know vs. don't know."
        )

    def __del__(self) -> None:
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass
