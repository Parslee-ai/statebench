---
layout: default
title: StateBench
---

# StateBench

**Your AI has memory problems. StateBench proves it.**

---

## The Dirty Secret of AI Memory

Most AI systems claim to have "memory." They do not.

They have chat history. They replay transcripts. They stuff old messages into context windows and hope for the best.

Then they forget decisions. Contradict themselves. Reference facts that were explicitly corrected three turns ago.

This is not a model intelligence problem. It is an architecture problem.

StateBench exposes it.

---

## What Actually Goes Wrong

We tested six different memory strategies across multiple frontier models. The failures are systematic.

**Resurrection.** User says "I moved to Seattle." System acknowledges it. Later, system ships the order to Portland—the old address that was explicitly invalidated.

**Stale Reasoning.** User corrects a date. System says "Got it, Thursday not Tuesday." Then schedules the meeting for Tuesday anyway.

**Authority Violation.** CFO sets policy: max 15% discount. Intern suggests 25%. System approves the 25%.

**Scope Leak.** User explores a hypothetical. "What if we had $50k budget?" System treats it as the actual budget for the rest of the conversation.

These are not edge cases. They happen constantly in production. They are why enterprise AI deployments require constant human oversight.

---

## The Insight

Humans do not converse the way chat interfaces assume.

When you text a competent colleague, they do not reread your entire message history before responding. They remember outcomes, not transcripts. Decisions, not verbatim wording. Constraints, not old reasoning paths.

What feels like conversation is actually stateful coordination over time.

**Context is not text. Context is state.**

The moment you ask the model to manage its own memory, you have already failed the architecture.

---

## What StateBench Measures

StateBench is not a retrieval benchmark. It does not ask "can you find this fact?"

It asks: **does your system maintain correct state over time?**

Three metrics that matter:

| Metric | What It Tests |
|--------|---------------|
| **Decision Accuracy** | When facts change, do decisions change? |
| **SFRR** | How often do dead facts resurrect? |
| **Must Mention** | Does relevant information actually appear? |

---

## The Results

We compared transcript replay, rolling summaries, RAG, fact extraction, and state-based context assembly.

State-based wins on accuracy. But there is a tradeoff.

### GPT-5.2 (OpenAI)

| Baseline | Decision Acc | SFRR ↓ | Must Mention |
|----------|--------------|--------|--------------|
| **state_based** | **80.3%** | 34.4% | **79.8%** |
| rolling_summary | 72.1% | **21.3%** | 66.4% |
| transcript_replay | 60.7% | 24.6% | 67.2% |
| fact_extraction | 63.9% | 27.9% | 56.3% |

### Claude Opus 4.5 (Anthropic)

| Baseline | Decision Acc | SFRR ↓ | Must Mention |
|----------|--------------|--------|--------------|
| state_based_no_supersession | **62.9%** | 38.2% | 86.0% |
| **state_based** | 58.2% | 41.0% | **87.4%** |
| transcript_replay | 53.0% | **33.5%** | 74.8% |
| rolling_summary | 51.4% | 45.8% | 73.6% |

GPT-5.2 achieves significantly higher decision accuracy. Opus 4.5 shows stronger fact grounding (higher must-mention rates) but weaker decision reasoning.

---

## The Tradeoff No One Talks About

More context means higher accuracy. It also means more opportunities for resurrection.

Transcript replay has lower SFRR not because it handles supersession well, but because it includes less information. It misses relevant facts entirely.

State-based approaches surface more facts—including ones the model should ignore.

This is not a context management problem alone. It is also a model reasoning problem. Even with explicit "this fact is superseded" markers, models still reference dead information.

The architecture helps. It does not solve everything.

---

## The State-Based Approach

Four layers. Assembled fresh every turn. No transcript replay.

**Layer 1: Identity.** Who is the human. What authority they have. Static.

**Layer 2: Persistent Facts.** Decisions, constraints, commitments. Durable. Explicitly superseded when they change.

**Layer 3: Working Set.** Current objective, recent turns, open questions. Ephemeral. Discarded when focus shifts.

**Layer 4: Environment.** Calendar, deadlines, external signals. Fetched fresh.

Superseded facts are marked invalid but not deleted. Only valid facts enter context. The model reasons. The architecture remembers.

[Full algorithm specification →](ALGORITHM)

---

## Try It

```bash
pip install statebench

# Generate test suite
statebench generate --tracks all --count 100 --output benchmark.jsonl

# Test your system
statebench evaluate -d benchmark.jsonl -b state_based -m gpt-4o

# Compare approaches
statebench compare -d benchmark.jsonl -m gpt-4o
```

[GitHub repository →](https://github.com/Parslee-ai/statebench)

---

## The Paper

For the formal treatment—architecture specification, enterprise requirements, full experimental methodology:

[**Beyond Conversation: A State-Based Context Architecture for Enterprise AI Agents**](state-based-context-architecture.pdf)

Liotta, 2025.

---

## The Bottom Line

AI agents do not chat. They maintain state and respond in language.

The conversation is an illusion layer for humans. Internally, there is only state reconciliation, intent detection, context composition, and reasoning.

Externally, it feels like texting someone who just remembers things correctly.

That is the goal. StateBench measures whether you are there yet.

---

MIT License · [Parslee](https://github.com/Parslee-ai)
