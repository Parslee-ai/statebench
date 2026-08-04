"""CAR (Common Agent Runtime) inference provider for StateBench.

Routes generation and judging through the local ``car-server`` daemon via the
``car_runtime`` PyO3 client, so evaluation uses CAR-managed local models
(MLX Qwen3, Apple Foundation, ...) instead of remote OpenAI / Anthropic APIs.

The daemon self-authenticates from the on-disk auth-token (or ``$CAR_AUTH_TOKEN``),
so constructing ``CarRuntime()`` is all that is required to connect.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

# A reasonable, fast local default. Override per-run with --model.
DEFAULT_MODEL = "mlx/qwen3-4b:4bit"

# The PyO3 client's per-call read timeout defaults to 30s, which large padded
# contexts can blow through. Give local inference more headroom.
os.environ.setdefault("CAR_DAEMON_TIMEOUT", "180")

# StateBench answers are short (a decision + a sentence). Capping output keeps
# every local-model call fast; the harness otherwise allows up to 1500 tokens.
MAX_OUTPUT_TOKENS = int(os.environ.get("CAR_MAX_TOKENS", "384"))

_runtime: Any = None


def get_runtime() -> Any:
    """Lazily connect to the running car-server daemon."""
    global _runtime
    if _runtime is None:
        from car_runtime import CarRuntime  # imported lazily so the dep is optional

        _runtime = CarRuntime()
    return _runtime


def _reset_runtime() -> None:
    """Force a fresh connection (e.g. after the daemon dropped the channel)."""
    global _runtime
    _runtime = None


# Gateway-side failures that are worth retrying. Hosted (parslee/*) routes
# return "managed inference failed" intermittently under sustained load — a
# single call succeeds while a long evaluation run dies partway. The harness
# already backs off on Anthropic 529s; local inference had no equivalent, so a
# transient blip killed a whole baseline.
_TRANSIENT = (
    "managed inference failed",
    "503",
    "service unavailable",
    "timed out",
    "temporarily",
    "rate limit",
    "too many requests",
)
_CHANNEL_LOST = ("channel closed", "closed connection", "timed out", "rpc")

MAX_RETRIES = int(os.environ.get("CAR_MAX_RETRIES", "5"))
RETRY_BASE_DELAY = float(os.environ.get("CAR_RETRY_BASE_DELAY", "3.0"))


def _infer_with_reconnect(prompt: str, model: str, max_tokens: int) -> Any:
    """Call infer_tracked, reconnecting and backing off on transient failures."""
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return get_runtime().infer_tracked(
                prompt, model=model, max_tokens=max_tokens
            )
        except RuntimeError as e:
            last = e
            msg = str(e).lower()
            if any(s in msg for s in _CHANNEL_LOST):
                _reset_runtime()
            if not any(s in msg for s in _TRANSIENT) and not any(
                s in msg for s in _CHANNEL_LOST
            ):
                raise  # a real error (bad model, no credential) — fail fast
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_DELAY * (2**attempt))
    assert last is not None
    raise last


def _strip_think(text: str) -> str:
    """Drop any ``<think>...</think>`` block a reasoning model may still emit."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for CAR's local path, which
    currently reports ``usage: null`` instead of real counts."""
    return max(1, len(text) // 4)


def _parse_infer(raw: Any) -> tuple[str, int | None, int | None]:
    """Parse an ``infer_tracked`` response into (text, tokens, latency_ms)."""
    if not isinstance(raw, str):
        # Some bindings may already hand back a dict.
        data = raw if isinstance(raw, dict) else {}
    else:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return _strip_think(raw), None, None
    text = _strip_think(data.get("text", "") or "")
    tokens: int | None = None
    usage = data.get("usage")
    if isinstance(usage, dict):
        tokens = int(usage.get("input_tokens", 0) or 0) + int(
            usage.get("output_tokens", 0) or 0
        )
    latency = data.get("latency_ms")
    latency_ms = int(latency) if isinstance(latency, (int, float)) else None
    return text, tokens, latency_ms


def car_generate(
    system_prompt: str,
    user_prompt: str,
    model: str | None,
    max_tokens: int,
) -> tuple[str, int, int]:
    """Generate a response through CAR. Returns (text, tokens, latency_ms).

    The system prompt is folded into a single prompt and Qwen "thinking" mode
    is disabled (``/no_think``) — it otherwise spends the whole token budget on a
    ``<think>`` block and stalls generation.
    """
    model = model or DEFAULT_MODEL
    max_tokens = min(max_tokens, MAX_OUTPUT_TOKENS)
    prompt = f"{system_prompt}\n\nUser: {user_prompt} /no_think\nAssistant:"

    start = time.time()
    raw = _infer_with_reconnect(prompt, model, max_tokens)
    wall_ms = int((time.time() - start) * 1000)

    text, tokens, reported_ms = _parse_infer(raw)
    if tokens is None:
        tokens = _estimate_tokens(prompt) + _estimate_tokens(text)
    return text, tokens, reported_ms if reported_ms is not None else wall_ms


def car_complete(prompt: str, model: str | None = None, max_tokens: int = 64) -> str:
    """One-shot completion used by the judge (paraphrase / decision extraction)."""
    raw = _infer_with_reconnect(f"{prompt} /no_think", model or DEFAULT_MODEL, max_tokens)
    text, _, _ = _parse_infer(raw)
    return text


def car_extract_json(prompt: str, model: str | None = None, max_tokens: int = 320) -> Any:
    """Run a CAR completion expected to return a JSON array/object; parse defensively.

    Local models often wrap JSON in prose or code fences; we extract the first
    balanced array/object. Returns None on parse failure.
    """
    text = car_complete(prompt, model=model, max_tokens=max_tokens)
    if not text:
        return None
    # Strip code fences.
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("[") or p.startswith("{"):
                text = p
                break
    # Find the first balanced [...] or {...}.
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None
