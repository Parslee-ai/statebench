# StateBench Tasks for lm-evaluation-harness

This package provides [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) compatible tasks for [StateBench](https://github.com/parslee-ai/statebench).

## Installation

```bash
pip install statebench-lm-eval
```

Or install from source:

```bash
cd statebench-lm-eval
pip install -e .
```

## Usage

After installation, the `statebench` task will be available in lm-evaluation-harness:

```bash
# Run with a HuggingFace model
lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks statebench \
    --batch_size auto

# Run with OpenAI API
lm_eval --model openai-completions \
    --model_args model=gpt-4o \
    --tasks statebench
```

## Available Tasks

| Task | Description |
|------|-------------|
| `statebench` | Full benchmark (all tracks) |
| `statebench_supersession` | Supersession track only |
| `statebench_scope` | Scope/permission tracks |

## Limitations

The lm-evaluation-harness integration uses a **simplified mode** where:

1. Context is pre-computed using the `transcript_replay` baseline
2. Each query is flattened to an independent document
3. Only decision accuracy is measured (not full SFRR metrics)

For proper baseline comparison and full metrics (SFRR, must-mention rates, provenance), use the native StateBench harness:

```bash
pip install statebench
statebench evaluate -d test.jsonl -b state_based -m gpt-4o
```

## How It Works

1. Timelines are loaded from `parslee/statebench` on HuggingFace Hub
2. For each timeline, events are processed sequentially using `transcript_replay`
3. Context is built before each query
4. The model is prompted with context + query
5. The response is compared against the expected decision
