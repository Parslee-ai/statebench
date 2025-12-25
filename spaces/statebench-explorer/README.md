---
title: StateBench Explorer
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
datasets:
  - parslee/statebench
---

# StateBench Explorer

Interactive inspection tool for the [StateBench](https://huggingface.co/datasets/parslee/statebench) benchmark.

## Features

- **Browse timelines** from train/validation/test splits
- **Filter by track** (13 evaluation tracks)
- **View events**: conversation turns, state writes, supersessions, queries
- **Inspect ground truth**: expected decisions, must mention, must not mention
- **Compare baselines**: see context built by different memory strategies

## Usage

1. Select a **split** (test, validation, train)
2. Optionally filter by **track**
3. Choose a **timeline ID**
4. Select a **baseline** to see how it builds context
5. Click **Inspect Timeline**

## Tracks

| Track | Description |
|-------|-------------|
| supersession | Facts invalidated by newer information |
| commitment_durability | Commitments survive interruptions |
| scope_permission | Role-based access control |
| causality | Multi-constraint dependencies |
| ... | See dataset card for full list |

## Links

- [Dataset](https://huggingface.co/datasets/parslee/statebench)
- [GitHub](https://github.com/Parslee-ai/statebench)
- [Evaluation Guide](https://github.com/Parslee-ai/statebench#evaluation)
