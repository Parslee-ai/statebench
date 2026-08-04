"""Run a command with an OpenAI key pulled from CAR's secret store.

Several baselines (``rolling_summary``, ``rag_transcript``, ``fact_extraction``)
construct an OpenAI client directly for summarization and embeddings, so they
need ``OPENAI_API_KEY`` in the environment regardless of which provider the
harness itself is pointed at.

The key is read from the CAR keychain and injected into the child process
environment only. It is never written to disk, never logged, and never printed.

Usage::

    uv run python experiments/with_car_secret.py -- python -m something --flag
"""

from __future__ import annotations

import os
import subprocess
import sys

CAR_SECRET_KEY = "OPENAI_KEY"
TARGET_ENV_VAR = "OPENAI_API_KEY"


def fetch_secret(key: str = CAR_SECRET_KEY) -> str:
    from car_runtime import CarRuntime

    raw = CarRuntime().secret_get(key)
    if isinstance(raw, str):
        stripped = raw.strip()
        # secret_get may hand back either the bare value or a JSON envelope.
        if stripped.startswith("{"):
            import json

            data = json.loads(stripped)
            for field in ("value", "secret", "data"):
                if data.get(field):
                    return str(data[field])
            raise SystemExit(f"no value field in secret envelope: {sorted(data)}")
        return stripped
    if isinstance(raw, dict):
        for field in ("value", "secret", "data"):
            if raw.get(field):
                return str(raw[field])
    raise SystemExit(f"could not read secret {key!r}")


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        raise SystemExit("usage: with_car_secret.py -- <command> [args...]")

    env = dict(os.environ)
    env[TARGET_ENV_VAR] = fetch_secret()
    # Confirm presence without revealing any part of the value.
    print(f"{TARGET_ENV_VAR} injected from CAR secret {CAR_SECRET_KEY!r}", flush=True)
    sys.exit(subprocess.call(argv, env=env))


if __name__ == "__main__":
    main()
