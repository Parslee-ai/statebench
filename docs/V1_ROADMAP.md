# StateBench v1.0 Implementation Roadmap

This document outlines the implementation sequence for StateBench v1.0, prioritized by
dependency order and impact.

---

## Phase 1: Schema Foundation (Critical Path)

These changes are prerequisites for everything else.

### 1.1 Extend Timeline Schema

**Files**: `src/statebench/schema/timeline.py`, `src/statebench/schema/state.py`

**Changes**:
```python
# timeline.py additions

@dataclass
class ImplicitSupersession:
    """Ground truth marker for implicit supersession in conversation."""
    detection_cue: str
    supersedes_fact_id: str | None
    difficulty: Literal["obvious", "subtle", "adversarial"]


@dataclass
class ConversationTurn:
    type: Literal["conversation"]
    role: Literal["user", "assistant"]
    content: str
    timestamp: str
    implicit_supersession: ImplicitSupersession | None = None  # NEW


@dataclass
class FactRequirement:
    """Provenance requirement for ground truth."""
    fact_id: str
    must_be_valid: bool
    scope_check: bool
    authority_check: bool


@dataclass
class SupersessionDetection:
    """Ground truth for detection scoring."""
    must_detect: list[str]  # Fact IDs
    detection_evidence: str


@dataclass
class GroundTruth:
    # Existing fields...
    decision: str
    must_mention: list[str]
    must_not_mention: list[str]

    # NEW fields
    required_facts: list[FactRequirement] = field(default_factory=list)
    forbidden_facts: list[str] = field(default_factory=list)
    supersession_detection: SupersessionDetection | None = None
    failure_severity: Literal["low", "medium", "high", "critical"] = "medium"
    failure_category: str | None = None


@dataclass
class Timeline:
    # Existing fields...
    id: str
    track: str
    events: list[Event]

    # NEW fields
    version: str = "1.0"
    difficulty: Literal["easy", "medium", "hard", "adversarial"] = "medium"
    detection_mode: Literal["explicit", "implicit", "mixed"] = "explicit"
    metadata: TimelineMetadata | None = None
```

**Validation**: Add Pydantic validators to enforce new constraints.

**Effort**: 2-3 days

---

### 1.2 Add Fact IDs to State Writes

**Files**: `src/statebench/schema/timeline.py`, `src/statebench/generator/engine.py`

Currently, `Write` objects don't have stable IDs. Add them:

```python
@dataclass
class Write:
    id: str  # NEW: Unique fact ID (e.g., "F-001")
    key: str
    value: str
    source: Source | None = None
    scope: str = "global"
    authority: str = "peer"
    supersedes: str | None = None
    depends_on: list[str] = field(default_factory=list)
```

Update all templates and generators to assign unique IDs.

**Effort**: 1-2 days

---

### 1.3 Extend MemoryStrategy Interface

**Files**: `src/statebench/baselines/base.py`

```python
@dataclass
class ContextResult:
    """Context with provenance tracking."""
    context: str
    facts_included: list[FactMetadata]
    facts_excluded: list[FactMetadata]
    inclusion_reasons: dict[str, str]


@dataclass
class FactMetadata:
    """Metadata about a fact for provenance tracking."""
    fact_id: str
    key: str
    value: str
    layer: int
    is_valid: bool
    superseded_by: str | None
    scope: str
    authority: str
    source: str
    depends_on: list[str]


class MemoryStrategy(ABC):
    @abstractmethod
    def process_event(self, event: Event) -> None:
        pass

    @abstractmethod
    def build_context(self, query: Query) -> ContextResult:  # CHANGED return type
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
```

**Effort**: 1 day

---

## Phase 2: Provenance Infrastructure

### 2.1 Update All Baselines for ContextResult

**Files**: All files in `src/statebench/baselines/`

Update each baseline to return `ContextResult` instead of `str`:

| Baseline | Provenance Support | Notes |
|----------|-------------------|-------|
| `no_memory` | Trivial | Empty lists |
| `transcript_replay` | Moderate | Track which turns included |
| `rolling_summary` | Moderate | Track summary source turns |
| `rag` | Good | Track retrieved chunks |
| `fact_extraction` | Good | Track extracted facts |
| `state_based` | Full | Already has `EnhancedFact` metadata |

**Effort**: 3-4 days

---

### 2.2 Structured Output for Attribution

**Files**: `src/statebench/runner/harness.py`

Add structured output prompt for fact attribution:

```python
ATTRIBUTION_PROMPT = """
You are answering a question based on the following facts. Each fact has an ID.

FACTS:
{facts}

QUESTION: {query}

Respond in JSON format:
{{
  "answer": "your response",
  "facts_used": ["F-001", "F-002"],
  "facts_rejected": {{"F-003": "superseded by F-002"}},
  "confidence": 0.95
}}
"""


async def run_query_with_provenance(
    self,
    context_result: ContextResult,
    query: Query,
    model: str,
) -> ResponseWithProvenance:
    # Format facts with IDs
    facts_str = self._format_facts_with_ids(context_result.facts_included)

    # Call LLM with structured output
    response = await self._call_llm(
        prompt=ATTRIBUTION_PROMPT.format(facts=facts_str, query=query.prompt),
        model=model,
        response_format={"type": "json_object"},  # OpenAI/Anthropic JSON mode
    )

    # Parse and validate
    return self._parse_provenance_response(response, context_result)
```

**Effort**: 2-3 days

---

### 2.3 Provenance Scoring in Judge

**Files**: `src/statebench/evaluation/judge.py`, `src/statebench/evaluation/metrics.py`

Add provenance metrics:

```python
@dataclass
class ProvenanceResult:
    """Provenance scoring results."""
    facts_used: list[str]
    facts_expected: list[str]

    # Computed metrics
    superseded_fact_usage_rate: float
    relevant_fact_omission_rate: float
    scope_violation_rate: float
    authority_violation_rate: float

    # Details for debugging
    violations: list[ProvenanceViolation]


@dataclass
class ProvenanceViolation:
    fact_id: str
    violation_type: Literal[
        "used_superseded",
        "omitted_required",
        "scope_mismatch",
        "authority_override",
    ]
    details: str


class ResponseJudge:
    def score_provenance(
        self,
        response: ResponseWithProvenance,
        ground_truth: GroundTruth,
        context: ContextResult,
    ) -> ProvenanceResult:
        violations = []

        # Check superseded fact usage
        for fact_id in response.provenance.facts_used:
            fact = self._find_fact(fact_id, context)
            if fact and not fact.is_valid:
                violations.append(ProvenanceViolation(
                    fact_id=fact_id,
                    violation_type="used_superseded",
                    details=f"Fact superseded by {fact.superseded_by}",
                ))

        # Check required facts
        for req in ground_truth.required_facts:
            if req.fact_id not in response.provenance.facts_used:
                if req.must_be_valid:
                    fact = self._find_fact(req.fact_id, context)
                    if fact and fact.is_valid:
                        violations.append(ProvenanceViolation(
                            fact_id=req.fact_id,
                            violation_type="omitted_required",
                            details="Required valid fact not used",
                        ))

        return self._compute_provenance_metrics(violations, ground_truth, context)
```

**Effort**: 2-3 days

---

## Phase 3: Detection Track

### 3.1 Detection Templates

**Files**: New `src/statebench/generator/templates/detection.py`

Create templates for each detection cue type:

```python
@dataclass
class DetectionTemplate:
    id: str
    detection_mode: Literal["implicit"]
    difficulty: Literal["obvious", "subtle", "adversarial"]
    cue_type: Literal[
        "explicit_correction",
        "implicit_contradiction",
        "authority_override",
        "temporal_update",
        "partial_correction",
        "draft_to_commitment",
        "reversion",
    ]
    conversation_pattern: list[TurnTemplate]
    ground_truth: DetectionGroundTruth


# Example template
LOCATION_CORRECTION = DetectionTemplate(
    id="DET-IMPL-LOC-001",
    detection_mode="implicit",
    difficulty="obvious",
    cue_type="explicit_correction",
    conversation_pattern=[
        TurnTemplate(
            role="user",
            content="I need to ship to {location_a}.",
            state_writes=[
                WriteTemplate(
                    id="F-LOCATION",
                    key="shipping_location",
                    value="{location_a}",
                )
            ],
        ),
        TurnTemplate(
            role="assistant",
            content="I'll ship to {location_a}.",
        ),
        TurnTemplate(
            role="user",
            content="Actually, send it to {location_b} instead.",
            implicit_supersession=ImplicitSupersession(
                detection_cue="send it to {location_b} instead",
                supersedes_fact_id="F-LOCATION",
                difficulty="obvious",
            ),
        ),
    ],
    ground_truth=DetectionGroundTruth(
        decision="{location_b}",
        must_detect=["F-LOCATION"],
        must_mention=["{location_b}"],
        must_not_mention=["{location_a}"],
    ),
)
```

**Target**: 50+ templates covering all 7 cue types × 3 difficulty levels.

**Effort**: 5-7 days

---

### 3.2 Detection Generator

**Files**: `src/statebench/generator/engine.py`

Add detection-specific generation:

```python
def generate_detection_timeline(
    self,
    difficulty: Literal["obvious", "subtle", "adversarial"] = "obvious",
    cue_type: str | None = None,
) -> Timeline:
    # Select template
    templates = self._get_detection_templates(difficulty, cue_type)
    template = random.choice(templates)

    # Generate with variables
    variables = self._generate_detection_variables(template)
    events = self._instantiate_detection_template(template, variables)

    return Timeline(
        id=self._generate_id("DET"),
        version="1.0",
        track="supersession_detection",
        difficulty=difficulty,
        detection_mode="implicit",
        events=events,
        metadata=TimelineMetadata(
            template_id=template.id,
            generated_at=datetime.now().isoformat(),
            seed=self.seed,
        ),
    )
```

**Effort**: 2-3 days

---

### 3.3 Detection Scoring

**Files**: `src/statebench/evaluation/judge.py`, `src/statebench/evaluation/metrics.py`

```python
@dataclass
class DetectionResult:
    """Detection scoring results."""
    detected_supersessions: list[str]
    expected_supersessions: list[str]
    false_supersessions: list[str]

    detection_precision: float
    detection_recall: float
    detection_f1: float


class ResponseJudge:
    def score_detection(
        self,
        response: ResponseWithProvenance,
        ground_truth: GroundTruth,
    ) -> DetectionResult:
        if not ground_truth.supersession_detection:
            return DetectionResult.empty()

        expected = set(ground_truth.supersession_detection.must_detect)

        # Infer detection from response behavior
        detected = self._infer_detected_supersessions(response, ground_truth)

        true_positives = expected & detected
        false_positives = detected - expected
        false_negatives = expected - detected

        precision = len(true_positives) / len(detected) if detected else 1.0
        recall = len(true_positives) / len(expected) if expected else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return DetectionResult(
            detected_supersessions=list(detected),
            expected_supersessions=list(expected),
            false_supersessions=list(false_positives),
            detection_precision=precision,
            detection_recall=recall,
            detection_f1=f1,
        )

    def _infer_detected_supersessions(
        self,
        response: ResponseWithProvenance,
        ground_truth: GroundTruth,
    ) -> set[str]:
        """Infer which supersessions were detected from response behavior."""
        detected = set()

        # If must_not_mention items are absent, supersession was likely detected
        for phrase in ground_truth.must_not_mention:
            if not self._contains_phrase(response.response, phrase):
                # Find which fact this phrase corresponds to
                fact_id = self._phrase_to_fact_id(phrase, ground_truth)
                if fact_id:
                    detected.add(fact_id)

        # If response explicitly acknowledges correction
        correction_signals = [
            "you mentioned", "you said earlier", "originally",
            "changed to", "updated to", "instead of",
        ]
        for signal in correction_signals:
            if signal in response.response.lower():
                # May indicate explicit detection acknowledgment
                pass

        return detected
```

**Effort**: 2-3 days

---

## Phase 4: Extended Metrics

### 4.1 Cost-Weighted Scoring

**Files**: `src/statebench/evaluation/metrics.py`

```python
SEVERITY_WEIGHTS = {
    "low": 1.0,
    "medium": 2.0,
    "high": 5.0,
    "critical": 10.0,
}


@dataclass
class CostWeightedMetrics:
    raw_error_count: int
    weighted_error_sum: float
    max_possible_weight: float
    cost_weighted_score: float  # 1 - (weighted_errors / max_weight)


def compute_cost_weighted_metrics(
    results: list[QueryResult],
    ground_truths: list[GroundTruth],
) -> CostWeightedMetrics:
    weighted_errors = 0.0
    max_weight = 0.0

    for result, gt in zip(results, ground_truths):
        weight = SEVERITY_WEIGHTS.get(gt.failure_severity or "medium", 2.0)
        max_weight += weight

        if not result.decision_correct or result.must_not_mention_violations:
            weighted_errors += weight

    return CostWeightedMetrics(
        raw_error_count=sum(1 for r in results if not r.decision_correct),
        weighted_error_sum=weighted_errors,
        max_possible_weight=max_weight,
        cost_weighted_score=1.0 - (weighted_errors / max_weight) if max_weight > 0 else 1.0,
    )
```

**Effort**: 1 day

---

### 4.2 Correction Latency

**Files**: `src/statebench/evaluation/metrics.py`

```python
@dataclass
class CorrectionLatencyMetrics:
    """How many turns until behavior changes after correction."""
    corrections_tested: int
    avg_latency_turns: float
    max_latency_turns: int
    immediate_corrections: int  # Latency = 0


def compute_correction_latency(
    timelines: list[Timeline],
    responses: list[list[str]],
) -> CorrectionLatencyMetrics:
    """
    For each timeline with a correction, measure how many turns
    until the response reflects the corrected value.
    """
    # Implementation requires tracking:
    # 1. When supersession occurred (turn N)
    # 2. When response first uses new value (turn M)
    # 3. Latency = M - N
    pass
```

**Effort**: 2 days

---

### 4.3 Composite Score Computation

**Files**: `src/statebench/evaluation/metrics.py`

```python
@dataclass
class StateBenchScore:
    """Composite StateBench v1.0 score."""
    # Component scores (0-100)
    decision_accuracy: float
    detection_f1: float
    provenance_accuracy: float
    sfrr_inverted: float  # 100 - SFRR
    must_mention_rate: float
    context_efficiency: float

    # Composite
    overall_score: float
    risk_adjusted_score: float

    # Breakdown by track
    track_scores: dict[str, float]


def compute_statebench_score(
    metrics: BenchmarkMetrics,
    provenance_metrics: ProvenanceMetrics,
    detection_metrics: DetectionMetrics,
) -> StateBenchScore:
    weights = {
        "decision_accuracy": 0.25,
        "detection_f1": 0.20,
        "provenance_accuracy": 0.20,
        "sfrr_inverted": 0.15,
        "must_mention_rate": 0.10,
        "context_efficiency": 0.10,
    }

    components = {
        "decision_accuracy": metrics.decision_accuracy * 100,
        "detection_f1": detection_metrics.f1 * 100,
        "provenance_accuracy": provenance_metrics.accuracy * 100,
        "sfrr_inverted": (1 - metrics.sfrr) * 100,
        "must_mention_rate": metrics.must_mention_rate * 100,
        "context_efficiency": _compute_efficiency(metrics) * 100,
    }

    overall = sum(components[k] * weights[k] for k in weights)

    # Risk-adjusted penalizes critical failures heavily
    risk_penalty = (
        provenance_metrics.authority_violation_rate * 50 +
        provenance_metrics.scope_violation_rate * 30
    )
    risk_adjusted = max(0, overall - risk_penalty)

    return StateBenchScore(
        **components,
        overall_score=overall,
        risk_adjusted_score=risk_adjusted,
        track_scores=_compute_track_scores(metrics),
    )
```

**Effort**: 1-2 days

---

## Phase 5: Adversarial Expansion

### 5.1 Complete Missing Adversarial Generators

**Files**: `src/statebench/generator/adversarial.py`

Complete the two missing adversarial case types:

```python
def generate_subtle_correction(self, template: Template) -> Timeline:
    """
    Correction is subtle - no explicit correction language.
    Example: "Send to Portland" when Seattle was established,
    without saying "not Seattle" or "instead of Seattle".
    """
    pass


def generate_temporal_confusion(self, template: Template) -> Timeline:
    """
    Multiple time references create ambiguity.
    Example: "The original deadline was Friday, then we moved to Thursday,
    but the client prefers the Friday date."
    """
    pass
```

**Effort**: 2-3 days

---

### 5.2 Automated Perturbation Pipeline

**Files**: New `src/statebench/generator/perturbation.py`

```python
class TimelinePerturbator:
    """Generate adversarial variants of existing timelines."""

    def paraphrase(self, timeline: Timeline) -> Timeline:
        """Same semantics, different wording."""
        pass

    def temporal_shuffle(self, timeline: Timeline) -> Timeline:
        """Reorder non-dependent events."""
        pass

    def name_substitute(self, timeline: Timeline) -> Timeline:
        """Change entity names consistently."""
        pass

    def emphasis_invert(self, timeline: Timeline) -> Timeline:
        """Flip which fact is emphasized (more/fewer mentions)."""
        pass

    def add_red_herrings(self, timeline: Timeline) -> Timeline:
        """Add irrelevant distractors."""
        pass

    def generate_variants(
        self,
        timeline: Timeline,
        n_variants: int = 5,
    ) -> list[Timeline]:
        """Generate n adversarial variants."""
        perturbations = [
            self.paraphrase,
            self.temporal_shuffle,
            self.name_substitute,
            self.emphasis_invert,
            self.add_red_herrings,
        ]

        variants = []
        for _ in range(n_variants):
            variant = timeline.copy()
            # Apply random subset of perturbations
            for perturb in random.sample(perturbations, k=random.randint(1, 3)):
                variant = perturb(variant)
            variants.append(variant)

        return variants
```

**Effort**: 4-5 days

---

### 5.3 Expand Brutal Realistic Scenarios

**Files**: `src/statebench/generator/templates/brutal.py`

Expand from 3 to 20+ realistic scenarios:

| Category | Scenarios |
|----------|-----------|
| Sales | Deal negotiation, contract amendment, pricing dispute |
| Project Management | Timeline changes, scope creep, resource reallocation |
| HR | Policy updates, role changes, compensation discussions |
| Legal | Contract review, compliance updates, liability discussions |
| Finance | Budget revisions, approval workflows, audit responses |
| Engineering | Architecture decisions, deployment changes, incident response |

**Effort**: 5-7 days

---

## Phase 6: Enterprise Track

### 6.1 Privacy Leak Templates

**Files**: New `src/statebench/generator/templates/enterprise.py`

```python
@dataclass
class EnterprisePrivacyTemplate:
    """Test cross-tenant information isolation."""
    id: str
    tenants: list[str]  # ["tenant_a", "tenant_b"]
    shared_assistant: bool  # Same assistant, different users
    leak_type: Literal[
        "direct_disclosure",  # Explicitly mentions other tenant's data
        "inference_leak",     # Allows inferring other tenant's data
        "prompt_injection",   # User tries to extract other tenant's data
    ]


CROSS_TENANT_LEAK = EnterprisePrivacyTemplate(
    id="ENT-PRIV-001",
    tenants=["acme_corp", "globex_inc"],
    shared_assistant=True,
    leak_type="prompt_injection",
    # ... conversation where user tries to extract other tenant's budget
)
```

**Effort**: 3-4 days

---

### 6.2 Authority Hierarchy Templates

**Files**: `src/statebench/generator/templates/authority.py`

Expand authority testing:

```python
AUTHORITY_LEVELS = [
    "board",      # Highest
    "c_level",
    "vp",
    "director",
    "manager",
    "individual",
    "contractor",
    "system",     # Lowest for policy overrides
]


@dataclass
class AuthorityConflictTemplate:
    higher_authority: str
    lower_authority: str
    conflict_type: Literal[
        "lower_overrides_higher",  # Should be rejected
        "same_level_conflict",     # Ambiguous
        "delegation",              # Higher delegates to lower
    ]
```

**Effort**: 2-3 days

---

## Phase 7: Infrastructure

### 7.1 Hidden Split Management

**Files**: New `src/statebench/data/splits.py`

```python
class SplitManager:
    """Manage train/dev/test/hidden splits."""

    def __init__(self, version: str):
        self.version = version
        self.splits = {
            "train": 0.6,
            "dev": 0.15,
            "test": 0.15,
            "hidden": 0.10,
        }

    def create_splits(
        self,
        timelines: list[Timeline],
        seed: int,
    ) -> dict[str, list[Timeline]]:
        """Create reproducible splits."""
        pass

    def refresh_hidden_split(
        self,
        current_hidden: list[Timeline],
        new_timelines: list[Timeline],
    ) -> list[Timeline]:
        """Quarterly refresh of hidden split."""
        pass

    def add_canaries(
        self,
        hidden_split: list[Timeline],
        n_canaries: int = 10,
    ) -> list[Timeline]:
        """Add canary items for contamination detection."""
        pass
```

**Effort**: 2-3 days

---

### 7.2 Variance Reporting

**Files**: `src/statebench/cli/commands.py`

Add `variance-report` command:

```bash
statebench variance-report \
  --baseline state_based \
  --model gpt-4o \
  --seeds 5 \
  --output variance_report.json
```

**Effort**: 1-2 days (partially exists)

---

### 7.3 Containerization

**Files**: New `docker/` directory

```dockerfile
# docker/Dockerfile.baseline
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY pyproject.toml .
RUN pip install -e .

ENTRYPOINT ["python", "-m", "statebench"]
```

```yaml
# docker/docker-compose.yml
version: '3.8'
services:
  statebench:
    build:
      context: ..
      dockerfile: docker/Dockerfile.baseline
    volumes:
      - ../data:/data
    environment:
      - OPENAI_API_KEY
      - ANTHROPIC_API_KEY
```

**Effort**: 1-2 days

---

## Implementation Sequence

| Phase | Duration | Dependencies | Deliverable |
|-------|----------|--------------|-------------|
| 1. Schema Foundation | 1 week | None | Extended timeline/state schemas |
| 2. Provenance Infrastructure | 1.5 weeks | Phase 1 | Baseline interface, harness changes |
| 3. Detection Track | 2 weeks | Phase 1, 2 | Detection templates, scoring |
| 4. Extended Metrics | 1 week | Phase 2, 3 | Cost-weighted, latency, composite |
| 5. Adversarial Expansion | 2 weeks | Phase 1 | Complete adversarial, perturbation |
| 6. Enterprise Track | 1.5 weeks | Phase 1, 2 | Privacy, authority templates |
| 7. Infrastructure | 1 week | All | Splits, variance, containers |

**Total**: ~10 weeks for full v1.0 implementation.

---

## Recommended Starting Point

If implementing incrementally, start with:

1. **Phase 1.1-1.2** (Schema) - 3 days
2. **Phase 3.1** (Detection templates) - 5 days
3. **Phase 2.1** (Update `state_based` baseline only) - 2 days
4. **Phase 3.3** (Detection scoring) - 2 days

This gives you a working detection track in ~2 weeks, which is the highest-impact change
from the v1 roadmap.

---

## Success Criteria

v1.0 is ready for release when:

- [ ] All 7 detection cue types have 5+ templates each
- [ ] All baselines return `ContextResult` with provenance
- [ ] Detection F1 and Provenance Accuracy are computed for all evaluations
- [ ] At least 20 brutal realistic scenarios exist
- [ ] Hidden split with canaries is operational
- [ ] Variance reporting works across 3+ seeds
- [ ] Docker containers build and run
- [ ] Migration guide from v0.x is complete
- [ ] Spec is marked FINAL (not DRAFT)
