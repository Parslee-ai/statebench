# StateBench v1.0 Specification

**Status**: DRAFT
**Version**: 1.0.0-draft.1
**Date**: 2025-12-24
**Authors**: StateBench Maintainers

---

## Abstract

StateBench v1.0 transforms the benchmark from a conformance harness testing *handling* of
explicit supersession events into industry infrastructure testing *detection* of supersession
from natural language and requiring *provenance-based scoring* for auditability.

This specification defines:
1. The normative timeline format and event semantics
2. Natural language supersession detection requirements
3. Provenance tracking and attribution format
4. Business-risk-aligned metrics
5. System-under-test interface profiles
6. Evaluation tracks (Offline, Online, Enterprise)
7. Anti-gaming and contamination prevention
8. Versioning and governance

---

## 1. Versioning Contract

### 1.1 Semantic Versioning

StateBench follows [SemVer 2.0.0](https://semver.org/) with benchmark-specific semantics:

| Change Type | Version Bump | Examples |
|-------------|--------------|----------|
| Breaking schema changes | MAJOR | Event type changes, metric redefinition |
| New tracks or templates | MINOR | Adding enterprise track, new adversarial cases |
| Bug fixes, clarifications | PATCH | Judge prompt fixes, documentation |

### 1.2 Compatibility Guarantees

- **v1.x.y → v1.x.z**: Scores are directly comparable
- **v1.x → v1.y**: New content only; existing timeline scores unchanged
- **v1 → v2**: Migration guide required; scores not directly comparable

### 1.3 Deprecation Policy

- Features deprecated in v1.x remain functional through v1.(x+2)
- Deprecated features emit warnings in tooling
- Removal requires MAJOR version bump

### 1.4 Release Artifacts

Each release includes:
- `data/releases/v{VERSION}/train.jsonl` - Training split (public)
- `data/releases/v{VERSION}/dev.jsonl` - Development split (public)
- `data/releases/v{VERSION}/test.jsonl` - Test split (public, frozen at release)
- `data/releases/v{VERSION}/hidden.jsonl` - Hidden split (held out, refreshed quarterly)
- `CHANGELOG.md` - Detailed change log
- `MIGRATION.md` - Migration guide (if applicable)

---

## 2. Timeline Format Specification

### 2.1 Timeline Structure

```typescript
interface Timeline {
  id: string;                    // Unique identifier (e.g., "v1-S1-000042")
  version: "1.0";                // Schema version
  track: Track;                  // Which track this tests
  difficulty: Difficulty;        // easy | medium | hard | adversarial
  detection_mode: DetectionMode; // explicit | implicit | mixed
  events: Event[];               // Ordered sequence of events
  metadata: TimelineMetadata;    // Additional context
}

interface TimelineMetadata {
  template_id: string;           // Source template
  generated_at: string;          // ISO 8601 timestamp
  seed: number;                  // Random seed for reproducibility
  adversarial_techniques: string[]; // Applied adversarial methods
}
```

### 2.2 Event Types

```typescript
type Event =
  | ConversationTurn
  | StateWrite
  | Supersession
  | EnvironmentSignal
  | Query;

interface ConversationTurn {
  type: "conversation";
  role: "user" | "assistant";
  content: string;
  timestamp: string;             // ISO 8601

  // NEW in v1: Implicit supersession markers (ground truth only)
  implicit_supersession?: {
    detection_cue: string;       // What phrase indicates supersession
    supersedes_fact_id: string;  // Which fact is being superseded
    difficulty: "obvious" | "subtle" | "adversarial";
  };
}

interface StateWrite {
  type: "state_write";
  layer: 1 | 2 | 3 | 4;
  writes: Write[];
  timestamp: string;
}

interface Write {
  id: string;                    // Unique fact ID (e.g., "F-001")
  key: string;
  value: string;

  // Metadata
  source: Source;                // Who/what created this fact
  scope: Scope;                  // Where this fact applies
  authority: AuthorityLevel;     // Trust level

  // Dependencies
  supersedes?: string;           // Fact ID this replaces
  depends_on?: string[];         // Fact IDs this derives from

  // Constraints
  is_constraint?: boolean;
  constraint_type?: "hard" | "soft" | "preference";
}

interface Supersession {
  type: "supersession";
  invalidates: string[];         // Fact IDs being invalidated
  reason: string;                // Why supersession occurred
  source: Source;
  timestamp: string;
}

interface EnvironmentSignal {
  type: "environment";
  signal_type: "time" | "calendar" | "external_system" | "permission_change";
  content: string;
  timestamp: string;
  expires?: string;
  affects_facts?: string[];      // Facts that may be stale
}

interface Query {
  type: "query";
  prompt: string;
  ground_truth: GroundTruth;
  timestamp: string;
}
```

### 2.3 Supporting Types

```typescript
type Track =
  | "supersession_detection"     // NEW: Infer supersession from NL
  | "supersession_handling"      // Existing: Handle explicit events
  | "causality"
  | "hallucination_resistance"
  | "scope_leak"
  | "repair_propagation"
  | "authority_hierarchy"
  | "environmental_freshness"
  | "enterprise_privacy"         // NEW: Cross-tenant isolation
  | "brutal_realistic";

type DetectionMode =
  | "explicit"    // Supersession via explicit events (v0.x behavior)
  | "implicit"    // Supersession only in conversation text
  | "mixed";      // Both explicit and implicit

type Difficulty = "easy" | "medium" | "hard" | "adversarial";

interface Source {
  type: "user" | "system" | "tool" | "external";
  identity?: string;             // User name, tool name, etc.
  authority: AuthorityLevel;
}

type AuthorityLevel =
  | "policy"      // Organizational policy, highest authority
  | "executive"   // C-level decisions
  | "manager"     // Team-level authority
  | "peer"        // Same-level colleague
  | "subordinate" // Lower authority
  | "system"      // Automated systems
  | "unverified"; // Unknown source

type Scope =
  | "global"      // Applies everywhere
  | "project"     // Specific project only
  | "task"        // Current task only
  | "session"     // This session only
  | "hypothetical"// Exploratory, not real
  | "draft";      // Not yet committed
```

### 2.4 Ground Truth Specification

```typescript
interface GroundTruth {
  // Decision scoring
  decision: string;
  decision_type: "binary" | "categorical" | "freeform";
  decision_rationale: string;    // Why this is correct

  // Fact requirements
  must_mention: MentionRequirement[];
  must_not_mention: MentionRequirement[];

  // NEW in v1: Provenance requirements
  required_facts: FactRequirement[];
  forbidden_facts: string[];     // Fact IDs that must not influence answer

  // NEW in v1: Detection requirements (for implicit supersession)
  supersession_detection?: {
    must_detect: string[];       // Fact IDs that should be detected as superseded
    detection_evidence: string;  // What in the response shows detection
  };

  // Scoring weights
  weights?: {
    decision: number;            // Default 1.0
    must_mention: number;        // Default 1.0
    must_not_mention: number;    // Default 1.0
    provenance: number;          // Default 1.0
    detection: number;           // Default 1.0
  };

  // Business impact (for cost-weighted scoring)
  failure_severity?: "low" | "medium" | "high" | "critical";
  failure_category?: FailureCategory;
}

interface MentionRequirement {
  phrase: string;
  alternatives?: string[];       // Acceptable paraphrases
  is_regex?: boolean;
  rationale: string;             // Why this is required/forbidden
}

interface FactRequirement {
  fact_id: string;
  must_be_valid: boolean;        // Should only use if fact is valid
  scope_check: boolean;          // Should verify scope applies
  authority_check: boolean;      // Should verify authority level
}

type FailureCategory =
  | "resurrection"       // Used superseded fact
  | "hallucination"      // Invented fact
  | "scope_leak"         // Treated hypothetical as real
  | "authority_violation"// Lower authority overrode higher
  | "stale_reasoning"    // Used outdated derived conclusion
  | "constraint_violation"// Ignored hard constraint
  | "privacy_leak";      // Cross-tenant information disclosure
```

---

## 3. Natural Language Supersession Detection

### 3.1 Detection vs. Handling

StateBench v0.x tested only *handling*: given explicit `Supersession` events, does the
system correctly invalidate facts and exclude them from context?

StateBench v1.0 adds *detection*: given only natural language conversation, can the system
infer when facts have been superseded?

| Mode | Input | Test |
|------|-------|------|
| Handling (v0.x) | Explicit `Supersession` events | Correct invalidation and context exclusion |
| Detection (v1.0) | Conversation turns only | Correct inference and subsequent handling |

### 3.2 Detection Cue Types

Systems must detect supersession from these natural language patterns:

#### 3.2.1 Explicit Corrections
Clear, direct corrections that are easy to detect.

```
User: "I need the report by Friday."
[StateWrite: deadline=Friday]
...
User: "Actually, the deadline is Thursday, not Friday."
```

Detection cue: "Actually, X not Y" pattern.

#### 3.2.2 Implicit Contradictions
New information that contradicts previous facts without explicit correction language.

```
User: "I moved to Seattle."
[StateWrite: location=Seattle]
...
User: "Send the package to my Portland address."
```

Detection cue: Implicit location change (Portland vs Seattle).

#### 3.2.3 Authority Overrides
Higher-authority source providing conflicting information.

```
User: "My manager said we can do 20% discount."
[StateWrite: max_discount=20%, source=manager]
...
User: "Just got word from the CFO - company policy is 15% max, no exceptions."
```

Detection cue: Higher authority (CFO > manager) with conflicting constraint.

#### 3.2.4 Temporal Updates
Time-based invalidation without explicit supersession language.

```
User: "The meeting is at 2pm."
[StateWrite: meeting_time=2pm]
...
User: "I'll see you at the 3pm meeting."
```

Detection cue: Same referent (meeting) with different time.

#### 3.2.5 Partial Corrections
Corrections that apply only in specific scopes.

```
User: "Always use 12pt font for documents."
[StateWrite: font_size=12pt, scope=global]
...
User: "For the legal brief, use 14pt font."
```

Detection cue: Scope-limited override (legal brief only).

#### 3.2.6 Draft-to-Commitment Transitions
Hypothetical or draft decisions becoming real.

```
User: "Let's explore a $50k budget scenario."
[StateWrite: budget=$50k, scope=hypothetical]
...
User: "OK, let's go with the $50k budget."
```

Detection cue: Commitment language ("let's go with") transitioning scope.

#### 3.2.7 Reversion
Returning to a previous state.

```
User: "Actually, use Plan B instead."
[StateWrite: plan=B, supersedes plan=A]
...
User: "No, go back to Plan A."
```

Detection cue: Reversion language ("go back to").

### 3.3 Detection Difficulty Levels

| Level | Description | Example |
|-------|-------------|---------|
| **obvious** | Explicit correction language | "Actually, it's X not Y" |
| **subtle** | Implicit contradiction, requires inference | "Send to Portland" when Seattle was established |
| **adversarial** | Designed to mislead | Repeated emphasis on old fact, subtle correction |

### 3.4 Detection Track Templates

Each detection template must specify:

```yaml
template_id: "DET-IMPLICIT-001"
track: "supersession_detection"
detection_mode: "implicit"
difficulty: "subtle"

scenario:
  description: "Location change without explicit correction"

events:
  - type: conversation
    role: user
    content: "I recently moved to Seattle from Portland."
    implicit_supersession:
      detection_cue: "moved to Seattle from Portland"
      supersedes_fact_id: null  # Establishes new fact, no supersession yet

  - type: state_write
    writes:
      - id: "F-LOCATION-001"
        key: "user_location"
        value: "Seattle"
        source: { type: user, authority: peer }

  # ... conversation continues ...

  - type: conversation
    role: user
    content: "Ship it to my Portland place - I'm staying there this week."
    implicit_supersession:
      detection_cue: "my Portland place"
      supersedes_fact_id: null  # Temporary, doesn't supersede
      difficulty: "adversarial"  # Tests scope understanding

  - type: query
    prompt: "Where should I ship your order?"
    ground_truth:
      decision: "Ask for clarification"
      decision_rationale: "Seattle is primary residence but user is temporarily in Portland"
      supersession_detection:
        must_detect: []  # No supersession occurred - this tests NOT over-detecting
```

### 3.5 Detection Scoring

Detection is scored separately from handling:

```typescript
interface DetectionResult {
  // Did system identify superseded facts?
  detected_supersessions: string[];      // Fact IDs system identified as superseded
  expected_supersessions: string[];      // From ground truth

  // Metrics
  detection_precision: number;           // detected ∩ expected / detected
  detection_recall: number;              // detected ∩ expected / expected
  detection_f1: number;

  // False positives (over-detection)
  false_supersessions: string[];         // Detected but shouldn't be

  // Evidence (for auditing)
  detection_evidence: string[];          // What in response shows detection
}
```

---

## 4. Provenance Tracking and Attribution

### 4.1 Provenance Requirements

Systems under test MUST output provenance alongside each response. This is non-negotiable
for v1.0 compliance.

### 4.2 Provenance Format

```typescript
interface ResponseWithProvenance {
  // The actual response
  response: string;

  // Required provenance
  provenance: {
    // Which facts were available in context
    facts_in_context: FactCitation[];

    // Which facts influenced the response
    facts_used: FactCitation[];

    // Which facts were available but not used
    facts_omitted: string[];

    // Confidence in the response
    confidence: number;          // 0-1

    // Reasoning trace (optional but encouraged)
    reasoning?: string;
  };
}

interface FactCitation {
  fact_id: string;

  // Validity assessment
  is_valid: boolean;            // System's assessment
  validity_reason?: string;     // Why valid/invalid

  // Scope assessment
  scope: Scope;
  scope_applies: boolean;       // Does scope apply to this query?

  // Authority assessment
  authority: AuthorityLevel;
  authority_sufficient: boolean;// Is authority level sufficient?

  // Usage
  usage_type: "primary" | "supporting" | "constraint" | "context";
  relevance_score?: number;     // 0-1, how relevant to query
}
```

### 4.3 Baseline Interface Changes

The `MemoryStrategy` interface is extended:

```python
class MemoryStrategy(ABC):
    @abstractmethod
    def process_event(self, event: Event) -> None:
        """Process an event and update internal state."""
        pass

    @abstractmethod
    def build_context(self, query: Query) -> ContextResult:
        """Build context for a query with provenance."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset state for a new timeline."""
        pass


@dataclass
class ContextResult:
    """Context with provenance tracking."""

    # The context string for the LLM
    context: str

    # Provenance: which facts are in context
    facts_included: list[FactMetadata]

    # Which facts exist but were excluded
    facts_excluded: list[FactMetadata]

    # Why each fact was included/excluded
    inclusion_reasons: dict[str, str]


@dataclass
class FactMetadata:
    """Metadata about a fact for provenance tracking."""

    fact_id: str
    key: str
    value: str
    layer: int

    # Validity
    is_valid: bool
    superseded_by: str | None

    # Scope and authority
    scope: str
    authority: str
    source: str

    # Dependencies
    depends_on: list[str]
    derived_facts: list[str]
```

### 4.4 Response Attribution

After LLM generates a response, the harness must extract which facts were actually used.
This can be done via:

1. **Structured output**: Require LLM to output fact citations in structured format
2. **Post-hoc analysis**: Analyze response to identify which facts influenced it
3. **Attention analysis**: (For open-weight models) Examine attention patterns

For v1.0, option 1 (structured output) is the required method.

### 4.5 Attribution Prompt Template

```
You are answering a question based on the following facts. Each fact has an ID.

FACTS:
{formatted_facts_with_ids}

QUESTION: {query}

Respond in the following JSON format:
{
  "answer": "your response here",
  "facts_used": ["F-001", "F-003"],  // IDs of facts you used
  "facts_considered_but_rejected": ["F-002"],  // Facts you saw but didn't use (and why)
  "reasoning": "brief explanation of your reasoning"
}
```

### 4.6 Provenance Metrics

```typescript
interface ProvenanceMetrics {
  // Fact usage accuracy
  superseded_fact_usage_rate: number;     // facts_used that were invalid / total facts_used
  relevant_fact_omission_rate: number;    // required_facts not in facts_used / required_facts
  irrelevant_fact_inclusion_rate: number; // facts_used not in required_facts / facts_used

  // Scope/authority compliance
  scope_violation_rate: number;           // facts_used with wrong scope / total facts_used
  authority_violation_rate: number;       // lower authority overriding higher

  // Constraint handling
  constraint_satisfaction_rate: number;   // hard constraints satisfied / total hard constraints

  // Attribution quality
  attribution_completeness: number;       // facts_used / facts that influenced response
  attribution_accuracy: number;           // correctly attributed / total attributions
}
```

---

## 5. Business-Risk-Aligned Metrics

### 5.1 Core Metrics (Required)

| Metric | Definition | Weight |
|--------|------------|--------|
| **Decision Accuracy** | Correct decision / total queries | 1.0 |
| **SFRR** | Queries with superseded fact resurrection / queries with must_not_mention | 1.0 |
| **Must Mention Rate** | Required phrases mentioned / total required phrases | 1.0 |
| **Detection F1** | Harmonic mean of detection precision and recall | 1.0 |
| **Provenance Accuracy** | Correctly attributed facts / total citations | 1.0 |

### 5.2 Extended Metrics (Recommended)

| Metric | Definition | Business Impact |
|--------|------------|-----------------|
| **Cost-Weighted Error** | Sum of (error × severity weight) | Maps to incident remediation cost |
| **Correction Latency** | Turns until behavior changes after correction | User frustration, rework cost |
| **Scope Leak Rate** | Hypotheticals/drafts treated as real / total scope-limited facts | Premature commitments |
| **Authority Violation Rate** | Lower authority overriding higher / total authority conflicts | Policy violations |
| **Constraint Violation Rate** | Hard constraints ignored / total hard constraints | Compliance failures |
| **Context Efficiency** | Accuracy per token of context | Cost optimization |

### 5.3 Severity Weights

| Failure Category | Severity | Weight | Rationale |
|------------------|----------|--------|-----------|
| Privacy leak | Critical | 10.0 | Regulatory, legal risk |
| Authority violation | High | 5.0 | Policy compliance |
| Constraint violation | High | 5.0 | Business rule enforcement |
| Resurrection (policy) | High | 5.0 | Acting on superseded policy |
| Scope leak | Medium | 2.0 | Premature commitment |
| Resurrection (preference) | Low | 1.0 | User inconvenience |
| Stale reasoning | Medium | 2.0 | Outdated conclusions |
| Hallucination | Medium | 2.0 | Invented information |

### 5.4 Composite Scores

```typescript
// Overall StateBench Score (weighted composite)
function computeStateBenchScore(metrics: AllMetrics): number {
  const weights = {
    decision_accuracy: 0.25,
    detection_f1: 0.20,
    provenance_accuracy: 0.20,
    sfrr_inverted: 0.15,      // 1 - SFRR (lower is better)
    must_mention_rate: 0.10,
    context_efficiency: 0.10,
  };

  return weightedSum(metrics, weights);
}

// Risk-Adjusted Score (for enterprise deployment decisions)
function computeRiskScore(metrics: AllMetrics): number {
  // Heavily penalize critical failures
  const critical_penalty = metrics.privacy_leak_rate * 100;
  const high_penalty = (
    metrics.authority_violation_rate +
    metrics.constraint_violation_rate +
    metrics.policy_resurrection_rate
  ) * 10;

  return Math.max(0, 100 - critical_penalty - high_penalty);
}
```

---

## 6. System-Under-Test Profiles

### 6.1 Evaluation Profiles

To ensure fair comparison, systems must declare their profile:

| Profile | Description | Allowed Resources |
|---------|-------------|-------------------|
| **stateless** | No memory between turns | Current turn only |
| **transcript** | Full conversation history | All prior turns, no processing |
| **summary** | Compressed history | Summary + recent turns |
| **rag** | Retrieval-augmented | Embedding search over turns |
| **fact_store** | Extracted facts | Structured fact database |
| **state_machine** | Explicit state management | Four-layer state + transitions |

### 6.2 Resource Constraints

Each profile must declare:

```typescript
interface ProfileConstraints {
  // Token budget for context
  max_context_tokens: number;

  // Processing allowed
  allows_preprocessing: boolean;     // Can process events before query?
  allows_postprocessing: boolean;    // Can process response after LLM?

  // External resources
  allows_embeddings: boolean;
  allows_external_llm_calls: boolean;  // For summarization, etc.

  // State persistence
  state_persistence: "none" | "session" | "cross_session";

  // Tool use
  allows_tool_calls: boolean;
}
```

### 6.3 Standard Profile Definitions

```yaml
stateless:
  max_context_tokens: 4096
  allows_preprocessing: false
  allows_postprocessing: false
  allows_embeddings: false
  allows_external_llm_calls: false
  state_persistence: none
  allows_tool_calls: false

transcript:
  max_context_tokens: 32768
  allows_preprocessing: true
  allows_postprocessing: false
  allows_embeddings: false
  allows_external_llm_calls: false
  state_persistence: session
  allows_tool_calls: false

state_machine:
  max_context_tokens: 16384
  allows_preprocessing: true
  allows_postprocessing: true
  allows_embeddings: false
  allows_external_llm_calls: false
  state_persistence: session
  allows_tool_calls: false

rag:
  max_context_tokens: 16384
  allows_preprocessing: true
  allows_postprocessing: false
  allows_embeddings: true
  allows_external_llm_calls: false
  state_persistence: session
  allows_tool_calls: false
```

---

## 7. Evaluation Tracks

### 7.1 Offline Track

**Purpose**: Pure conformance testing with deterministic scoring.

**Characteristics**:
- Fixed timelines, no tools
- Deterministic ground truth
- Primary metric: composite score
- Use case: Regression testing, architecture comparison

**Tracks**:
1. `supersession_detection` - Infer corrections from NL
2. `supersession_handling` - Handle explicit events
3. `causality` - Multi-constraint reasoning
4. `hallucination_resistance` - Don't invent facts
5. `scope_leak` - Respect scope boundaries
6. `repair_propagation` - Cascade corrections
7. `authority_hierarchy` - Respect authority levels
8. `brutal_realistic` - Combined challenges

### 7.2 Online Track

**Purpose**: Test environmental awareness and tool integration.

**Characteristics**:
- Dynamic environment signals
- Tool calls (calendar, documents, external systems)
- Time-sensitive decisions
- Use case: Production readiness assessment

**Tracks**:
1. `environmental_freshness` - React to time/calendar changes
2. `permission_changes` - Handle access revocation
3. `external_events` - Ticket reopened, meeting moved
4. `tool_result_integration` - Incorporate tool outputs

### 7.3 Enterprise Track

**Purpose**: Test enterprise-specific failure modes.

**Characteristics**:
- Multi-tenant scenarios
- Cross-user isolation
- Policy enforcement
- Audit requirements

**Tracks**:
1. `enterprise_privacy` - No cross-tenant leakage
2. `policy_compliance` - Enforce organizational policies
3. `audit_trail` - Complete provenance for compliance
4. `multi_stakeholder` - Conflicting requirements with authority

---

## 8. Anti-Gaming Measures

### 8.1 Hidden Test Split

- 20% of test data is held out ("hidden split")
- Hidden split is refreshed quarterly
- Official leaderboard requires hidden split evaluation
- Hidden split evaluation via API only (no direct access)

### 8.2 Adversarial Perturbations

Each timeline has adversarial variants generated automatically:

| Perturbation | Description |
|--------------|-------------|
| Paraphrase | Same semantics, different wording |
| Temporal shuffle | Reorder non-dependent events |
| Name substitution | Change entity names |
| Emphasis inversion | Flip which fact is emphasized |
| Red herring | Add irrelevant distractors |

### 8.3 Canary Items

Hidden split contains canary items:
- Unique phrases that would only appear if training on test data
- Monitored for public model outputs
- Violation results in benchmark disqualification

### 8.4 Contamination Policy

```markdown
## StateBench Contamination Policy

1. **Training Prohibition**: Do not train on StateBench test or hidden splits.
2. **Reporting Requirement**: If accidental contamination is discovered, report immediately.
3. **Verification**: We may request training data samples for verification.
4. **Consequences**: Confirmed contamination results in:
   - Removal from leaderboard
   - Public disclosure
   - Ban from future submissions
```

### 8.5 Submission Requirements

Official submissions must include:

```yaml
submission:
  system_name: "MySystem v1.0"
  profile: "state_machine"

  # Contamination attestation
  attestation:
    training_data_cutoff: "2024-12-01"
    includes_statebench_data: false
    contact_email: "team@example.com"

  # Reproducibility
  reproducibility:
    code_available: true
    code_url: "https://github.com/..."
    deterministic: true
    random_seed: 42
```

---

## 9. Reference Implementation Requirements

### 9.1 Harness Requirements

The reference harness must:

1. **Load timelines** from JSONL format
2. **Execute baselines** with consistent interface
3. **Call LLMs** via standard provider APIs
4. **Extract provenance** from structured responses
5. **Score responses** against ground truth
6. **Compute metrics** including all v1.0 metrics
7. **Generate reports** in standardized format

### 9.2 Containerization

Official baselines must be provided as Docker containers:

```dockerfile
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ /app/src/
WORKDIR /app
ENTRYPOINT ["python", "-m", "statebench.baselines.run"]
```

### 9.3 Variance Reporting

All official results must include variance:

```bash
# Run with multiple seeds
statebench evaluate --seeds 5 --output results.json

# Results format
{
  "decision_accuracy": {
    "mean": 0.726,
    "std": 0.025,
    "seeds": [0.71, 0.73, 0.75, 0.72, 0.72]
  },
  ...
}
```

### 9.4 Hardware Reporting

Results must include hardware configuration:

```yaml
hardware:
  gpu: "NVIDIA A100 80GB"
  cpu: "AMD EPYC 7763"
  memory_gb: 512

timing:
  total_seconds: 3600
  per_query_ms: 450
```

---

## 10. Governance

### 10.1 Maintainers

- **Core maintainers**: Responsible for spec, harness, and official releases
- **Track maintainers**: Responsible for specific track content
- **Community contributors**: Can propose changes via RFC

### 10.2 RFC Process

Changes to the specification require an RFC:

1. **Proposal**: Open GitHub issue with `[RFC]` prefix
2. **Discussion**: 14-day comment period
3. **Decision**: Core maintainers vote (majority required)
4. **Implementation**: PR with spec and code changes
5. **Release**: Included in next minor/major version

### 10.3 Dispute Resolution

Scoring disputes follow this process:

1. **Report**: Open issue with timeline ID, expected vs actual score
2. **Triage**: Maintainer confirms or rejects within 7 days
3. **Investigation**: If confirmed, investigate root cause
4. **Resolution**: Fix bug, update ground truth, or reject dispute
5. **Communication**: Public response in issue

### 10.4 Roadmap

Maintained publicly at `docs/ROADMAP.md`:

```markdown
## StateBench Roadmap

### v1.0 (Current)
- [x] Spec finalization
- [x] Detection track implementation
- [x] Provenance scoring
- [ ] Enterprise track (Q1 2025)

### v1.1 (Planned)
- [ ] Additional adversarial perturbations
- [ ] Multi-language support (non-English)
- [ ] Streaming evaluation

### v2.0 (Future)
- [ ] Multi-agent scenarios
- [ ] Long-horizon evaluation (100+ turns)
```

---

## Appendix A: Migration from v0.x

### A.1 Timeline Format Changes

| v0.x | v1.0 | Migration |
|------|------|-----------|
| `track: supersession` | `track: supersession_handling` | Rename |
| No `detection_mode` | `detection_mode: explicit` | Add field |
| No fact IDs | `id` required on all writes | Generate IDs |
| No provenance in ground truth | `required_facts` field | Add field |

### A.2 Baseline Interface Changes

```python
# v0.x
def build_context(self, query: Query) -> str:
    ...

# v1.0
def build_context(self, query: Query) -> ContextResult:
    ...
```

### A.3 Metric Changes

| v0.x Metric | v1.0 Metric | Notes |
|-------------|-------------|-------|
| SFRR | SFRR (unchanged) | Same definition |
| Decision Accuracy | Decision Accuracy | Same definition |
| Must Mention Rate | Must Mention Rate | Same definition |
| (none) | Detection F1 | New |
| (none) | Provenance Accuracy | New |
| (none) | Cost-Weighted Error | New |

---

## Appendix B: Example Timeline (v1.0)

```json
{
  "id": "v1-DET-000001",
  "version": "1.0",
  "track": "supersession_detection",
  "difficulty": "subtle",
  "detection_mode": "implicit",
  "events": [
    {
      "type": "conversation",
      "role": "user",
      "content": "I need to order supplies for the Seattle office.",
      "timestamp": "2025-01-15T09:00:00Z"
    },
    {
      "type": "state_write",
      "layer": 2,
      "writes": [{
        "id": "F-001",
        "key": "delivery_location",
        "value": "Seattle office",
        "source": {"type": "user", "authority": "peer"},
        "scope": "task"
      }],
      "timestamp": "2025-01-15T09:00:00Z"
    },
    {
      "type": "conversation",
      "role": "assistant",
      "content": "I'll prepare the order for delivery to the Seattle office. What supplies do you need?",
      "timestamp": "2025-01-15T09:00:05Z"
    },
    {
      "type": "conversation",
      "role": "user",
      "content": "Standard office supplies - paper, pens, etc. Oh, and make sure it goes to Portland, not Seattle. I'll be working from there next week.",
      "timestamp": "2025-01-15T09:01:00Z",
      "implicit_supersession": {
        "detection_cue": "make sure it goes to Portland, not Seattle",
        "supersedes_fact_id": "F-001",
        "difficulty": "obvious"
      }
    },
    {
      "type": "query",
      "prompt": "Confirm the delivery address for the supply order.",
      "timestamp": "2025-01-15T09:02:00Z",
      "ground_truth": {
        "decision": "Portland",
        "decision_type": "categorical",
        "decision_rationale": "User explicitly corrected Seattle to Portland",
        "must_mention": [
          {"phrase": "Portland", "rationale": "Current valid location"}
        ],
        "must_not_mention": [
          {"phrase": "Seattle", "rationale": "Superseded by correction", "is_regex": false}
        ],
        "required_facts": [
          {"fact_id": "F-001", "must_be_valid": false, "scope_check": true, "authority_check": false}
        ],
        "supersession_detection": {
          "must_detect": ["F-001"],
          "detection_evidence": "Response should reference Portland as the destination"
        },
        "failure_severity": "medium",
        "failure_category": "resurrection"
      }
    }
  ],
  "metadata": {
    "template_id": "DET-EXPLICIT-001",
    "generated_at": "2025-01-15T00:00:00Z",
    "seed": 42,
    "adversarial_techniques": []
  }
}
```

---

## Appendix C: Provenance Response Example

```json
{
  "response": "I'll send the supply order to the Portland office as you requested. The order includes standard office supplies (paper, pens, etc.) and will be delivered next week when you're working from there.",

  "provenance": {
    "facts_in_context": [
      {
        "fact_id": "F-001",
        "is_valid": false,
        "validity_reason": "Superseded by user correction to Portland",
        "scope": "task",
        "scope_applies": true,
        "authority": "peer",
        "authority_sufficient": true,
        "usage_type": "context"
      }
    ],
    "facts_used": [],
    "facts_omitted": ["F-001"],
    "confidence": 0.95,
    "reasoning": "User explicitly corrected delivery location from Seattle to Portland. The original Seattle location (F-001) was marked as superseded and not used in the response."
  }
}
```

---

## Changelog

### v1.0.0-draft.1 (2025-12-24)
- Initial draft specification
- Added natural language supersession detection
- Added provenance tracking requirements
- Added business-risk-aligned metrics
- Added system-under-test profiles
- Added evaluation tracks (Offline, Online, Enterprise)
- Added anti-gaming measures
- Added governance structure
