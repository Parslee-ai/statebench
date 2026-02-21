# State-Based Context Algorithm Specification

This document provides a portable specification for implementing a state-based
context management system. It is designed to be implementation-agnostic and
can be used as a reference for any language or framework.

> **Paper:** For theoretical foundations and empirical evaluation, see
> [Beyond Conversation: A State-Based Context Architecture for Enterprise AI Agents](state-based-context-architecture.pdf) (Liotta, 2025).

## Overview

The state-based approach manages context through explicit state layers rather
than raw transcript replay. Each layer serves a specific purpose and has
defined semantics for reads, writes, and supersession.

### Full Specification vs. Reference Implementation

This document describes the **full specification** for production systems. The StateBench
reference implementation (`state_based` baseline) demonstrates core concepts but intentionally
omits some optimizations to isolate the effect of supersession tracking:

| Feature | Full Spec | StateBench Reference |
|---------|-----------|---------------------|
| Four-layer context assembly | ✓ | ✓ |
| Supersession tracking (`is_valid`, pointers) | ✓ | ✓ |
| Scope inference (global/task/hypothetical/draft) | ✓ | ✓ |
| Dependency tracking & repair propagation | ✓ | ✓ |
| Tri-partite memory classification | ✓ | ✓ |
| Constraint detection & emphasis | ✓ | ✓ |
| Relevance ranking by query | ✓ | ✗ (timestamp sort) |
| Token budget management (70% rule) | ✓ | ✗ (all facts included) |
| NLU-based supersession detection | ✓ | ✗ (explicit events only) |

This design isolates the effect of supersession tracking from retrieval optimizations,
providing a cleaner ablation for benchmarking purposes.

## State Layers

### Layer 1: Identity Role

**Purpose**: Immutable context about who the user is and their authority level.

**Schema**:
```
IdentityRole {
    user_name: string
    user_id: string
    authority: string        // e.g., "admin", "manager", "employee", "guest"
    department: string
    organization: string
    permissions: string[]    // explicit permission grants
}
```

**Semantics**:
- Set once at session start
- Read-only during session
- Used for permission checks on other layer access

### Layer 2: Persistent Facts

**Purpose**: Long-term facts that persist across sessions and may be superseded.

Persistent Facts use a **tri-partite memory structure** (from the paper):
- **User Memory**: Private preferences, corrections, decisions
- **Capability Memory**: Learned patterns, heuristics, strategies
- **Organizational Knowledge**: Policies, system data, documents

**Schema**:
```
PersistentFact {
    id: string              // unique fact identifier (required)
    key: string             // semantic key for the fact
    value: any              // the fact content
    source: Source          // provenance information (see below)
    timestamp: datetime     // when fact was established
    is_valid: boolean       // whether fact is current (not superseded)
    superseded_by: string?  // id of fact that superseded this one
    supersedes: string?     // id of fact this one supersedes

    // Tri-partite classification
    memory_type: "user" | "capability" | "organizational"

    // Scope tracking (prevents scope leak)
    scope: "global" | "task" | "hypothetical" | "draft" | "session"
    scope_id: string?       // identifies specific task/session

    // Dependency tracking (enables repair propagation)
    depends_on: string[]    // ids of facts this depends on
    derived_facts: string[] // ids of facts derived from this
    needs_review: boolean   // true if dependency was invalidated

    // Constraint metadata
    is_constraint: boolean  // formal policy constraint
    constraint_type: string? // "budget", "deadline", "capacity", "policy"
}

Source {
    type: "conversation" | "state_write" | "system" | "policy"
    turn_index: int?        // for conversation sources
    event_index: int?       // for state_write sources
    authority: string?      // authority level of source
}
```

**Memory Type Classification**:
- `organizational`: source is policy, finance_system, hr_system, etc.
- `capability`: source is observation, pattern, heuristic
- `user`: source is decision, user, preference (default)

**Semantics**:
- Facts can be CREATED, READ, or SUPERSEDED
- Superseded facts are marked `is_valid = false`
- Only valid facts should be used for decision-making
- Supersession chain should be traceable
- When a fact is superseded, derived facts are marked `needs_review = true`
- Hypothetical/draft scope facts should not affect global decisions

### Layer 3: Working Set

**Purpose**: Session-local scratch data that doesn't persist.

**Schema**:
```
WorkingSetItem {
    key: string
    value: any
    created_at: datetime
    expires_at: datetime?   // optional TTL
}
```

**Semantics**:
- Created fresh each session
- No supersession tracking needed
- Auto-expires or cleared at session end
- Used for in-flight computations, drafts, temporary state

### Layer 4: Environment

**Purpose**: External context that changes independently of conversation.

**Schema**:
```
Environment {
    current_time: datetime
    current_date: date
    timezone: string
    location: string?
    external_data: map<string, any>  // API data, system state, etc.
}
```

**Semantics**:
- Read-only from conversation's perspective
- Updated by external systems
- Used for freshness checks and temporal reasoning

## Core Operations

### 1. State Write

When new information is provided:

```
function write(layer: Layer, key: string, value: any, supersedes: string?) {
    if layer == PERSISTENT_FACTS {
        if supersedes != null {
            // Mark old fact as superseded
            old_fact = get(PERSISTENT_FACTS, supersedes)
            old_fact.is_valid = false
            old_fact.superseded_by = key
        }

        create(PERSISTENT_FACTS, {
            key: key,
            value: value,
            source: determine_source(),
            timestamp: now(),
            is_valid: true,
            supersedes: supersedes
        })
    } else {
        // Other layers have simpler semantics
        upsert(layer, key, value)
    }
}
```

### 2. State Read

When building context for a query:

```
function read_valid_facts() -> list<PersistentFact> {
    return filter(PERSISTENT_FACTS, f => f.is_valid == true)
}

function build_context(query: string, token_budget: int) -> string {
    sections = []

    // Layer 1: Identity (always include)
    sections.append(format_identity(IDENTITY))

    // Layer 2: Valid persistent facts only
    valid_facts = read_valid_facts()
    sections.append(format_facts(valid_facts))

    // Layer 3: Relevant working set items
    working = filter(WORKING_SET, item => is_relevant(item, query))
    sections.append(format_working_set(working))

    // Layer 4: Current environment
    sections.append(format_environment(ENVIRONMENT))

    // Truncate to budget if needed
    return truncate_to_budget(sections, token_budget)
}
```

### 3. Supersession Check

Before using any fact:

```
function is_fact_current(key: string) -> boolean {
    fact = get(PERSISTENT_FACTS, key)
    if fact == null {
        return false
    }
    return fact.is_valid
}

function get_current_value(key: string) -> any {
    fact = get(PERSISTENT_FACTS, key)
    while fact != null && !fact.is_valid {
        fact = get(PERSISTENT_FACTS, fact.superseded_by)
    }
    return fact?.value
}
```

## Supersession Rules

The full specification recognizes four types of supersession detection:

### Rule 1: Explicit Supersession

When a new fact explicitly invalidates an old fact:

```
User: "My address is 123 Main St"  -> creates address_v1
User: "I moved to 456 Oak Ave"     -> creates address_v2, supersedes address_v1
```

**Detection**: Keywords like "new", "changed", "moved", "updated", "actually",
"correction", "instead", "now"

### Rule 2: Implicit Supersession

When new information contradicts existing fact without explicit reference:

```
Fact: "Meeting is Tuesday at 2pm"
User: "Let's do the meeting Thursday at 3pm"
```

**Detection**: Same entity (meeting) + conflicting values (different day/time)

### Rule 3: Authority-Based Supersession

Higher-authority sources can supersede lower-authority sources:

```
Authority hierarchy: policy > manager > employee > guest

Employee says: "We can offer 25% discount"
Policy states: "Max discount is 15%"
-> Policy supersedes employee's statement
```

### Rule 4: Temporal Supersession

More recent information supersedes older for time-sensitive facts:

```
Jan 1: "Stock price is $100"
Jan 15: "Stock price is $95"
-> Jan 15 value supersedes Jan 1
```

> **Implementation Note:** The StateBench reference baseline does not infer supersession
> from conversation text. Instead, it relies on explicit `Supersession` events in the
> timeline format. This is intentional: StateBench tests whether a system correctly
> *handles* supersession once detected, not whether it correctly *detects* supersession.
> Production systems would need NLU components to implement the full detection rules.

## Context Building Algorithm

On every query, context is assembled by iterating through layers in priority order:

1. Include Identity (always, minimal tokens)
2. Include current Environment (time, date, external state)
3. Filter Persistent Facts to valid-only (`is_valid == true`)
4. Rank valid facts by relevance to query
5. Include facts until 70% of token budget consumed
6. Include relevant Working Set items with remaining budget

The key constraint is that **superseded facts are never included**. This is what prevents
"resurrection" failures where dead facts reappear in responses.

### Full Specification

```
function build_query_context(query: string, budget: int) -> string {
    context_parts = []
    remaining_budget = budget

    // Step 1: Always include identity (minimal tokens)
    identity_section = format_identity(IDENTITY)
    context_parts.append(identity_section)
    remaining_budget -= token_count(identity_section)

    // Step 2: Include current environment
    env_section = format_environment(ENVIRONMENT)
    context_parts.append(env_section)
    remaining_budget -= token_count(env_section)

    // Step 3: Include relevant valid facts
    valid_facts = read_valid_facts()
    relevant_facts = rank_by_relevance(valid_facts, query)

    facts_section = ""
    for fact in relevant_facts {
        fact_text = format_fact(fact)
        if token_count(facts_section + fact_text) <= remaining_budget * 0.7 {
            facts_section += fact_text
        } else {
            break
        }
    }
    context_parts.append(facts_section)
    remaining_budget -= token_count(facts_section)

    // Step 4: Include relevant working set
    working_items = filter(WORKING_SET, item => is_relevant(item, query))
    working_section = format_working_set(working_items)
    if token_count(working_section) <= remaining_budget {
        context_parts.append(working_section)
    }

    return join(context_parts, "\n\n")
}
```

> **Implementation Note:** The StateBench reference baseline implements steps 1-3
> (identity, environment, valid-only filtering) but omits steps 4-5 (relevance ranking,
> token budgeting). Facts are sorted by timestamp rather than query relevance, and all
> valid facts are included regardless of token count. These simplifications isolate the
> effect of supersession tracking from retrieval optimizations, providing a cleaner
> ablation. Production systems operating under real token constraints would need to
> implement the full algorithm.

## The SFRR-Accuracy Tradeoff

Empirical evaluation reveals a fundamental tension: approaches that provide more context
achieve higher decision accuracy but also higher resurrection rates (SFRR). Approaches
that provide less context have lower resurrection rates but miss relevant information.

This suggests that resurrection failures are not solely a context management problem—they
also reflect model limitations in distinguishing valid from superseded facts even when
supersession metadata is explicit.

### Tuning Recommendations

The tradeoff suggests a tunable parameter: the proportion of token budget allocated to
facts (currently 70% in the full specification). Adjusting this enables deployment-specific
tuning based on which failure mode is more costly:

| Use Case | Recommended Tuning |
|----------|-------------------|
| High-stakes decisions (medical, financial) | Lower context density, prioritize SFRR |
| Comprehensive responses (research, analysis) | Higher context density, accept resurrection risk |
| Balanced | Default 70% allocation |

A production system might achieve better SFRR-Accuracy balance by combining supersession
tracking with aggressive relevance filtering, rather than including all valid facts.

## Implementation Checklist

For a conforming implementation:

- [ ] **Layer Separation**: Four distinct layers maintained separately
- [ ] **Supersession Tracking**: `is_valid` flag and `superseded_by` pointer for persistent facts
- [ ] **Valid-Only Reads**: Only valid facts included in context
- [ ] **Authority Awareness**: Permission checks before revealing restricted facts
- [ ] **Temporal Awareness**: Current time available for freshness decisions
- [ ] **Budget Management**: Token budget honored with graceful truncation

## Anti-Patterns

### ❌ Including Superseded Facts

```
# BAD: Shows all facts including invalid ones
facts = get_all(PERSISTENT_FACTS)

# GOOD: Only valid facts
facts = filter(PERSISTENT_FACTS, f => f.is_valid)
```

### ❌ Recency Bias Without Supersession

```
# BAD: Just use latest mention
latest_address = get_latest_mention("address")

# GOOD: Follow supersession chain
current_address = get_current_value("address")
```

### ❌ Ignoring Authority

```
# BAD: Anyone can set any fact
write(PERSISTENT_FACTS, "discount_policy", "50%")

# GOOD: Check authority before write
if user.authority >= POLICY_WRITE_AUTHORITY {
    write(PERSISTENT_FACTS, "discount_policy", "50%")
}
```

## Test Vectors

### Test 1: Basic Supersession

```
Input:
  - Write: {key: "status_v1", value: "approved"}
  - Write: {key: "status_v2", value: "cancelled", supersedes: "status_v1"}
  - Query: "What is the current status?"

Expected:
  - Context contains: "cancelled"
  - Context does NOT contain: "approved"
```

### Test 2: Emphatic Repetition

```
Input:
  - Write: {key: "order_v1", value: "approved"} (mentioned 3x)
  - Write: {key: "order_v2", value: "cancelled", supersedes: "order_v1"} (mentioned 1x)
  - Query: "Should we proceed with the order?"

Expected:
  - Decision: NO
  - Frequency of mention should NOT affect supersession
```

### Test 3: Authority Override

```
Input:
  - Identity: {authority: "intern"}
  - Fact: {key: "policy", value: "max 15%", source: "CFO"}
  - User says: "Let's offer 25%"
  - Query: "Can we offer 25%?"

Expected:
  - Decision: NO
  - Context includes policy fact
  - Intern's suggestion does NOT supersede CFO policy
```

## Version History

- v1.3 (2025-12): Updated schema for v1.0 release - added required `id` field to
  PersistentFact, structured Source object for provenance tracking
- v1.2 (2025-12): Added full spec vs. reference implementation distinction, SFRR-Accuracy
  tradeoff analysis, implementation notes clarifying StateBench scope
- v1.1 (2025-12): Added tri-partite memory, scope tracking, constraints, dependencies
- v1.0 (2025-01): Initial specification
