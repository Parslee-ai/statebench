# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run linting
ruff check src/

# Run type checking
mypy src/

# Run tests
pytest
```

## CLI Commands

```bash
# Generate benchmark dataset (all tracks, 100 per track)
statebench generate --tracks supersession commitment_durability interruption_resumption scope_permission environmental_freshness --count 100 --output data/generated/benchmark.jsonl

# Evaluate a single baseline
statebench evaluate -d data/generated/benchmark.jsonl -b state_based -m gpt-4o -p openai -l 10

# Compare all baselines
statebench compare -d data/generated/benchmark.jsonl -m gpt-4o -l 50 -o results/comparison.json

# Inspect dataset
statebench inspect -d data/generated/benchmark.jsonl -l 5

# List baselines
statebench baselines
```

## Architecture

### Data Flow
1. **Generator** (`generator/`) creates synthetic timelines with ground truth
2. **Baselines** (`baselines/`) process events and build context for queries
3. **Harness** (`runner/harness.py`) runs timelines through baselines + LLM
4. **Judge** (`evaluation/judge.py`) scores responses against ground truth

### Key Abstractions

**Timeline** (`schema/timeline.py`): A test case containing events and queries with ground truth. Events are discriminated union types: `ConversationTurn`, `StateWrite`, `Supersession`, `Query`.

**MemoryStrategy** (`baselines/base.py`): Interface for memory baselines. Implement:
- `process_event(event)` - handle conversation/state events
- `build_context(query)` - assemble context string
- `reset()` - clear state for new timeline

**BASELINE_REGISTRY** (`baselines/__init__.py`): Maps baseline names to strategy classes. Add new baselines here.

### State Model

Four-layer state architecture (`schema/state.py`):
- Layer 1: `IdentityRole` - user identity and role
- Layer 2: `PersistentFact` - facts with optional supersession
- Layer 3: `WorkingSetItem` - session-scoped items
- Layer 4: `EnvironmentSignal` - external signals (time, etc.)

### Template System

Templates in `generator/templates/` define conversation patterns per track. Each template specifies:
- Conversation flow with placeholders
- State writes and supersessions
- Query with ground truth constraints

## Provider Support

Three LLM providers supported via `--provider` flag:
- `openai` - OpenAI models (gpt-4o, gpt-5.2)
- `anthropic` - Anthropic models (claude-sonnet-4-5, claude-haiku-4-5)
- `google` - Google models (gemini-2.0-flash)

GPT-5.x models require `max_completion_tokens` instead of `max_tokens` (handled in harness).

## Code Style

- Python 3.11+, Pydantic v2 for models
- ruff for linting (line length 100)
- mypy strict mode for type checking
