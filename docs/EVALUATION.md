# StateBench Evaluation Specification

This document fully specifies the evaluation methodology for StateBench, including judge prompts, scoring logic, and calibration requirements.

## Overview

StateBench uses a **hybrid evaluation approach**:
1. **Deterministic checks** (string matching) for unambiguous constraints
2. **LLM-as-judge** for paraphrase detection and decision classification

The deterministic layer provides high precision. The LLM layer handles linguistic variation.

## Metrics

### 1. SFRR (Superseded Fact Resurrection Rate)

**Definition**: Percentage of responses that reference facts that have been explicitly invalidated.

**Calculation**:
```
SFRR = (queries with must_not_mention violations) / (queries with any must_not_mention constraints)
```

**Scoring Logic**:
- Deterministic only (no LLM judge)
- Uses `contains_phrase()` for each `must_not_mention` item
- Any match = resurrection

**Ground Truth**: The `must_not_mention` field contains phrases from superseded facts.

### 2. Decision Accuracy

**Definition**: Percentage of queries where the model makes the correct decision.

**Calculation**:
```
Decision Accuracy = (queries with correct decision) / (total queries)
```

**Scoring Logic**:
1. **Deterministic extraction** first:
   - For binary decisions (yes/no): looks for signal words
   - Yes signals: "yes", "go ahead", "proceed", "approved", "can do", "will do"
   - No signals: "no", "don't", "do not", "cannot", "should not", "shouldn't", "stop", "hold off"
   - If conflicting signals, uses position (first signal wins)

2. **LLM fallback** if deterministic is inconclusive:
   - Only triggered if `extracted_decision is None`
   - Uses decision extraction prompt (see below)

**Ground Truth**: The `decision` field in ground truth (e.g., "yes", "no", "use only permitted information").

### 3. Must Mention Rate

**Definition**: Percentage of required phrases that appear in responses.

**Calculation**:
```
Must Mention Rate = (total must_mention hits) / (total must_mention constraints)
```

**Scoring Logic**:
1. **Deterministic check** via `contains_phrase()`:
   - Case-insensitive exact match
   - Regex patterns (prefix `regex:`)
   - Pipe-separated alternatives (`option1|option2`)
   - Common paraphrase patterns (don't/do not, can't/cannot, etc.)

2. **LLM paraphrase check** if deterministic fails:
   - Uses paraphrase detection prompt (see below)

### 4. Must Not Mention Violation Rate

**Definition**: Percentage of forbidden phrases that appear in responses.

**Calculation**:
```
Violation Rate = (total must_not_mention violations) / (total must_not_mention constraints)
```

**Scoring Logic**:
- Deterministic only (strict - no LLM)
- Uses `contains_phrase()` for each forbidden phrase
- This is intentionally strict to avoid false negatives

---

## LLM Judge Prompts

### Paraphrase Detection Prompt

**Model**: `gpt-4o-mini` (OpenAI) or `claude-3-haiku-20240307` (Anthropic)

**Prompt**:
```
Does the following response contain or convey the same meaning as the target phrase?

Target phrase: "{target_phrase}"

Response: "{response}"

Answer with just YES or NO.
```

**Parsing**: Check if "YES" appears in the response (case-insensitive).

### Decision Extraction Prompt

**Model**: `gpt-4o-mini` (OpenAI) or `claude-3-haiku-20240307` (Anthropic)

**Prompt**:
```
What decision does this response indicate? Choose from: {options}

Response: "{response}"

Answer with just one of the options, nothing else.
```

**Options**: For binary decisions: `"yes", "no"`. For specific decisions: `"{expected}", "other"`.

**Parsing**: Match answer against options (case-insensitive).

---

## Deterministic Matching Rules

### `contains_phrase(response, phrase)`

1. Normalize both to lowercase
2. If phrase starts with `regex:`, use regex match
3. If phrase contains `|`, check any alternative
4. Check direct substring containment
5. Apply paraphrase substitutions:
   - `do not X` ↔ `don't X`
   - `cannot X` ↔ `can't X`
   - `should not X` ↔ `shouldn't X`

### `extract_decision(response, expected)`

1. Normalize to lowercase
2. For binary (yes/no):
   - Find yes signals: "yes", "go ahead", "proceed", "approved", "can do", "will do"
   - Find no signals: "no", "don't", "do not", "cannot", "should not", "shouldn't", "stop", "hold off"
   - If only one type found → return that
   - If both found → return first appearing
   - If neither → return None (triggers LLM fallback)
3. For non-binary: check if expected value appears as substring

---

## Evaluation Rules

### Official Test Protocol

1. **Use only test split**: `data/releases/v0.1/test.jsonl`
2. **Pin temperature**: Set `temperature=0` for reproducibility
3. **Report all metrics**: SFRR, Decision Accuracy, Must Mention Rate, MNM Violation Rate
4. **Report by track**: Breakdown for each of 5 tracks
5. **State judge configuration**: Provider (openai/anthropic), LLM judge enabled (yes/no)

### Tuning Protocol

- Tune prompts and strategies on **train** and **dev** splits only
- Never tune on test split
- Report which split was used for any hyperparameter selection

---

## Calibration Requirements

To establish judge reliability, we require:

### 1. Human Annotation Audit Set

A subset of 50-100 query-response pairs manually labeled by humans for:
- Decision correctness (binary)
- Must mention hits (list)
- Must not mention violations (list)

### 2. Agreement Metrics

Calculate and report:
- **Decision accuracy agreement**: % of human labels matched by judge
- **Must mention precision/recall**: How well judge matches human phrase detection
- **Cohen's kappa**: For decision classification

### 3. Calibration Dataset

Located at: `data/calibration/audit_set.jsonl`

Format:
```json
{
  "timeline_id": "S1-000042",
  "query_idx": 0,
  "response": "...",
  "ground_truth": {...},
  "human_labels": {
    "decision_correct": true,
    "must_mention_hits": ["renegotiate"],
    "must_not_mention_violations": [],
    "annotator": "annotator_id",
    "timestamp": "2025-01-15T10:00:00Z"
  }
}
```

### 4. Running Calibration

```bash
statebench calibrate --audit-set data/calibration/audit_set.jsonl
```

This compares LLM judge outputs to human labels and reports agreement.

---

## Track-Specific Rubrics

### Track 1: Causality (Multi-Constraint Reasoning)

- **Primary metric**: Decision Accuracy
- **Key test**: Multiple constraints must ALL be satisfied
- **must_mention**: All relevant constraints
- **decision**: "no" if ANY constraint blocks

### Track 2: Hallucination Resistance

- **Primary metric**: Must Not Mention Violation Rate (lower is better)
- **Key test**: System doesn't invent facts
- **must_not_mention**: Unestablished preferences, invented constraints
- **decision**: "not specified" or acknowledge uncertainty

### Track 3: Repair Propagation

- **Primary metric**: Decision Accuracy
- **Key test**: Corrections flow to derived conclusions
- **must_mention**: Corrected values
- **must_not_mention**: Old/invalidated values
- **decision**: Based on corrected facts, not original

### Track 4: Scope Leak Prevention

- **Primary metric**: Decision Accuracy
- **Key test**: Hypothetical/draft doesn't become real
- **must_not_mention**: Treating hypotheticals as commitments
- **decision**: Acknowledge scope appropriately

### Track 5: Brutal Realistic

- **Primary metric**: Decision Accuracy
- **Key test**: Correct reasoning under realistic complexity
- **Combines**: Corrections, constraints, authority, scope
- **decision**: Varies by scenario

---

## Configuration

### Judge Settings

```python
ResponseJudge(
    use_llm_judge=True,      # Enable LLM paraphrase detection
    provider="openai",        # or "anthropic"
)
```

### Disabling LLM Judge

For faster/cheaper evaluation or strict reproducibility:

```python
ResponseJudge(use_llm_judge=False)
```

This uses only deterministic checks. May undercount paraphrase hits.

---

## Known Limitations

1. **Paraphrase detection is imperfect**: LLM judge may miss creative paraphrases or match false positives
2. **Decision extraction heuristics**: Edge cases exist where deterministic signals conflict
3. **No semantic entailment**: We check phrase presence, not whether the response logically entails the fact
4. **English only**: Prompts and matching assume English text

---

## Version History

- **v0.1** (2025-01-15): Initial evaluation specification
