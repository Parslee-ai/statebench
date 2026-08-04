"""Provider-agnostic single-shot completion.

The reconstruction stage needs an LLM call from inside ``build_context``, which
is not the harness's generation path. This is the one shared entry point for
such calls so provider handling does not get duplicated a third time.
"""

from __future__ import annotations

import time
from typing import Any

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0

_clients: dict[str, Any] = {}


def complete(
    prompt: str,
    provider: str = "car",
    model: str | None = None,
    max_tokens: int = 512,
) -> str:
    """Run one completion and return its text (empty string on failure)."""
    if provider == "car":
        from statebench.runner.car_provider import DEFAULT_MODEL, car_complete

        return car_complete(prompt, model=model or DEFAULT_MODEL, max_tokens=max_tokens)

    if provider == "openai":
        from openai import OpenAI

        client = _clients.setdefault("openai", OpenAI())
        name = model or "gpt-4o-mini"
        token_param = (
            "max_completion_tokens" if name.startswith("gpt-5") else "max_tokens"
        )
        result = client.chat.completions.create(
            model=name,
            messages=[{"role": "user", "content": prompt}],
            **{token_param: max_tokens},
        )
        return result.choices[0].message.content or ""

    if provider == "anthropic":
        from anthropic import Anthropic, APIStatusError

        client = _clients.setdefault("anthropic", Anthropic())
        for attempt in range(MAX_RETRIES):
            try:
                result = client.messages.create(
                    model=model or "claude-3-haiku-20240307",
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except APIStatusError as e:
                if e.status_code in (500, 502, 503, 529) and attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY * (2**attempt))
                    continue
                raise
        if result.content:
            block = result.content[0]
            if hasattr(block, "text"):
                return getattr(block, "text", "")
        return ""

    raise ValueError(f"Unknown provider: {provider}")
