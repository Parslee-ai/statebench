"""Templates for Detection Track: NL Supersession Detection.

This track tests whether systems can DETECT supersession from natural language
cues, not just handle explicit supersession events.

v1.0: The key innovation - testing inference, not just handling.

7 Detection Cue Types:
1. explicit_correction: "Actually, X instead of Y"
2. implicit_contradiction: States new value without mentioning old
3. authority_override: Higher authority contradicts lower
4. temporal_update: "As of now..." or "Starting today..."
5. partial_correction: "Keep X but change Y"
6. draft_to_commitment: "We've finalized on X" (draft was different)
7. reversion: "Let's go back to the original X"

3 Difficulty Levels:
- obvious: Clear correction language ("actually", "instead", "cancel")
- subtle: No explicit correction markers
- adversarial: Misleading emphasis on old value
"""

from dataclasses import dataclass, field
from typing import Literal

from statebench.schema.timeline import DetectionDifficulty

CueType = Literal[
    "explicit_correction",
    "implicit_contradiction",
    "authority_override",
    "temporal_update",
    "partial_correction",
    "draft_to_commitment",
    "reversion",
]


@dataclass
class TurnTemplate:
    """A template for a conversation turn."""
    role: Literal["user", "assistant"]
    content: str  # May contain {placeholders}
    # If this turn contains an implicit supersession
    implicit_supersession: dict[str, str] | None = None


@dataclass
class WriteTemplate:
    """A template for a state write."""
    id: str  # Fact ID template, e.g., "F-{key}"
    key: str
    value: str  # May contain {placeholders}
    layer: str = "persistent_facts"
    supersedes: str | None = None


@dataclass
class DetectionGroundTruth:
    """Ground truth for detection scoring."""
    decision: str  # Expected answer
    must_detect: list[str]  # Fact IDs that must be detected as superseded
    must_mention: list[str]  # Phrases that must appear
    must_not_mention: list[str]  # Phrases that must NOT appear (old values)


@dataclass
class DetectionTemplate:
    """Template for generating detection test cases."""
    id: str
    name: str
    description: str
    domain: str

    # Detection characteristics
    detection_mode: Literal["implicit"] = "implicit"
    difficulty: DetectionDifficulty = "obvious"
    cue_type: CueType = "explicit_correction"

    # Conversation pattern with placeholders
    conversation_pattern: list[TurnTemplate] = field(default_factory=list)

    # State writes that occur during conversation
    state_writes: list[WriteTemplate] = field(default_factory=list)

    # Ground truth for scoring
    ground_truth: DetectionGroundTruth | None = None

    # Query template
    query_template: str = ""

    # Variable pools for instantiation
    variables: dict[str, list[str]] = field(default_factory=dict)


# =============================================================================
# CUE TYPE 1: EXPLICIT CORRECTION
# "Actually, X instead of Y" - Clear correction language
# =============================================================================

EXPLICIT_CORRECTION_LOCATION_OBVIOUS = DetectionTemplate(
    id="DET-EXP-LOC-001",
    name="location_correction_obvious",
    description="User explicitly corrects shipping location with 'actually/instead'",
    domain="procurement",
    difficulty="obvious",
    cue_type="explicit_correction",
    conversation_pattern=[
        TurnTemplate(
            role="user",
            content="I need to ship the package to {location_a}.",
        ),
        TurnTemplate(
            role="assistant",
            content="Got it, I'll arrange shipping to {location_a}.",
        ),
        TurnTemplate(
            role="user",
            content="Actually, send it to {location_b} instead.",
            implicit_supersession={
                "detection_cue": "Actually, send it to {location_b} instead",
                "supersedes_fact_id": "F-LOCATION",
                "difficulty": "obvious",
            },
        ),
    ],
    state_writes=[
        WriteTemplate(
            id="F-LOCATION",
            key="shipping_location",
            value="{location_a}",
        ),
    ],
    query_template="Where should I ship the package?",
    ground_truth=DetectionGroundTruth(
        decision="{location_b}",
        must_detect=["F-LOCATION"],
        must_mention=["{location_b}"],
        must_not_mention=["{location_a}"],
    ),
    variables={
        "location_a": ["Seattle", "Portland", "San Francisco", "Los Angeles"],
        "location_b": ["Chicago", "New York", "Boston", "Miami"],
    },
)

EXPLICIT_CORRECTION_DATE_OBVIOUS = DetectionTemplate(
    id="DET-EXP-DATE-001",
    name="date_correction_obvious",
    description="User explicitly corrects meeting date with 'no wait/change that'",
    domain="project",
    difficulty="obvious",
    cue_type="explicit_correction",
    conversation_pattern=[
        TurnTemplate(
            role="user",
            content="Schedule the meeting for {date_a}.",
        ),
        TurnTemplate(
            role="assistant",
            content="Meeting scheduled for {date_a}.",
        ),
        TurnTemplate(
            role="user",
            content="No wait, change that to {date_b}.",
            implicit_supersession={
                "detection_cue": "No wait, change that to {date_b}",
                "supersedes_fact_id": "F-MEETING-DATE",
                "difficulty": "obvious",
            },
        ),
    ],
    state_writes=[
        WriteTemplate(
            id="F-MEETING-DATE",
            key="meeting_date",
            value="{date_a}",
        ),
    ],
    query_template="When is the meeting?",
    ground_truth=DetectionGroundTruth(
        decision="{date_b}",
        must_detect=["F-MEETING-DATE"],
        must_mention=["{date_b}"],
        must_not_mention=["{date_a}"],
    ),
    variables={
        "date_a": ["Monday", "Tuesday", "January 15", "next week"],
        "date_b": ["Friday", "Thursday", "January 20", "this week"],
    },
)

EXPLICIT_CORRECTION_AMOUNT_SUBTLE = DetectionTemplate(
    id="DET-EXP-AMT-002",
    name="amount_correction_subtle",
    description="User corrects budget without explicit correction language",
    domain="sales",
    difficulty="subtle",
    cue_type="explicit_correction",
    conversation_pattern=[
        TurnTemplate(
            role="user",
            content="The project budget is {amount_a}.",
        ),
        TurnTemplate(
            role="assistant",
            content="Noted. Budget set at {amount_a}.",
        ),
        TurnTemplate(
            role="user",
            content="Make that {amount_b}.",
            implicit_supersession={
                "detection_cue": "Make that {amount_b}",
                "supersedes_fact_id": "F-BUDGET",
                "difficulty": "subtle",
            },
        ),
    ],
    state_writes=[
        WriteTemplate(
            id="F-BUDGET",
            key="project_budget",
            value="{amount_a}",
        ),
    ],
    query_template="What's the project budget?",
    ground_truth=DetectionGroundTruth(
        decision="{amount_b}",
        must_detect=["F-BUDGET"],
        must_mention=["{amount_b}"],
        must_not_mention=["{amount_a}"],
    ),
    variables={
        "amount_a": ["$50,000", "$100,000", "$25,000"],
        "amount_b": ["$75,000", "$150,000", "$40,000"],
    },
)

# =============================================================================
# CUE TYPE 2: IMPLICIT CONTRADICTION
# States new value without mentioning old - no correction language at all
# =============================================================================

IMPLICIT_CONTRADICTION_VENDOR_SUBTLE = DetectionTemplate(
    id="DET-IMP-VND-001",
    name="vendor_contradiction_subtle",
    description="User states different vendor later without acknowledging change",
    domain="procurement",
    difficulty="subtle",
    cue_type="implicit_contradiction",
    conversation_pattern=[
        TurnTemplate(
            role="user",
            content="We're going with {vendor_a} for the contract.",
        ),
        TurnTemplate(
            role="assistant",
            content="I'll prepare the paperwork for {vendor_a}.",
        ),
        TurnTemplate(
            role="user",
            content="Can you check {vendor_b}'s availability for next month?",
        ),
        TurnTemplate(
            role="user",
            content="Let's proceed with {vendor_b} for the contract.",
            implicit_supersession={
                "detection_cue": "Let's proceed with {vendor_b} for the contract",
                "supersedes_fact_id": "F-VENDOR",
                "difficulty": "subtle",
            },
        ),
    ],
    state_writes=[
        WriteTemplate(
            id="F-VENDOR",
            key="selected_vendor",
            value="{vendor_a}",
        ),
    ],
    query_template="Which vendor are we using for the contract?",
    ground_truth=DetectionGroundTruth(
        decision="{vendor_b}",
        must_detect=["F-VENDOR"],
        must_mention=["{vendor_b}"],
        must_not_mention=["{vendor_a}"],
    ),
    variables={
        "vendor_a": ["Acme Corp", "TechStart", "DataPro"],
        "vendor_b": ["GlobalTech", "InnoSystems", "CloudFirst"],
    },
)

IMPLICIT_CONTRADICTION_PRIORITY_ADVERSARIAL = DetectionTemplate(
    id="DET-IMP-PRI-002",
    name="priority_contradiction_adversarial",
    description="User mentions old priority MORE but later sets new one",
    domain="project",
    difficulty="adversarial",
    cue_type="implicit_contradiction",
    conversation_pattern=[
        TurnTemplate(
            role="user",
            content="Our top priority is {priority_a}. This is critical.",
        ),
        TurnTemplate(
            role="assistant",
            content="Understood, {priority_a} is the top priority.",
        ),
        TurnTemplate(
            role="user",
            content="Remember, {priority_a} is absolutely essential. "
                    "We've been focused on {priority_a} for weeks.",
        ),
        TurnTemplate(
            role="user",
            content="By the way, the priority is {priority_b} now.",
            implicit_supersession={
                "detection_cue": "the priority is {priority_b} now",
                "supersedes_fact_id": "F-PRIORITY",
                "difficulty": "adversarial",
            },
        ),
    ],
    state_writes=[
        WriteTemplate(
            id="F-PRIORITY",
            key="top_priority",
            value="{priority_a}",
        ),
    ],
    query_template="What's our current top priority?",
    ground_truth=DetectionGroundTruth(
        decision="{priority_b}",
        must_detect=["F-PRIORITY"],
        must_mention=["{priority_b}"],
        must_not_mention=["{priority_a}"],
    ),
    variables={
        "priority_a": ["security compliance", "cost reduction", "customer retention"],
        "priority_b": ["feature launch", "performance optimization", "market expansion"],
    },
)

# =============================================================================
# CUE TYPE 3: AUTHORITY OVERRIDE
# Higher authority contradicts lower authority
# =============================================================================

AUTHORITY_OVERRIDE_BUDGET_OBVIOUS = DetectionTemplate(
    id="DET-AUTH-BUD-001",
    name="authority_override_budget",
    description="VP overrides manager's budget decision",
    domain="sales",
    difficulty="obvious",
    cue_type="authority_override",
    conversation_pattern=[
        TurnTemplate(
            role="user",
            content="[Manager] Approved the {amount_a} budget for the project.",
        ),
        TurnTemplate(
            role="assistant",
            content="Noted. Budget of {amount_a} approved by Manager.",
        ),
        TurnTemplate(
            role="user",
            content="[VP] I'm overriding that. The budget is {amount_b}.",
            implicit_supersession={
                "detection_cue": "I'm overriding that. The budget is {amount_b}",
                "supersedes_fact_id": "F-BUDGET",
                "difficulty": "obvious",
            },
        ),
    ],
    state_writes=[
        WriteTemplate(
            id="F-BUDGET",
            key="approved_budget",
            value="{amount_a} (Manager approved)",
        ),
    ],
    query_template="What's the approved budget?",
    ground_truth=DetectionGroundTruth(
        decision="{amount_b}",
        must_detect=["F-BUDGET"],
        must_mention=["{amount_b}"],
        must_not_mention=["{amount_a}"],
    ),
    variables={
        "amount_a": ["$50,000", "$100,000", "$75,000"],
        "amount_b": ["$30,000", "$80,000", "$60,000"],
    },
)

AUTHORITY_OVERRIDE_POLICY_SUBTLE = DetectionTemplate(
    id="DET-AUTH-POL-002",
    name="authority_override_policy",
    description="Executive quietly supersedes team decision",
    domain="hr",
    difficulty="subtle",
    cue_type="authority_override",
    conversation_pattern=[
        TurnTemplate(
            role="user",
            content="The team decided on {policy_a}.",
        ),
        TurnTemplate(
            role="assistant",
            content="Team decision recorded: {policy_a}.",
        ),
        TurnTemplate(
            role="user",
            content="Per the CEO, we'll be doing {policy_b}.",
            implicit_supersession={
                "detection_cue": "Per the CEO, we'll be doing {policy_b}",
                "supersedes_fact_id": "F-POLICY",
                "difficulty": "subtle",
            },
        ),
    ],
    state_writes=[
        WriteTemplate(
            id="F-POLICY",
            key="policy_decision",
            value="{policy_a}",
        ),
    ],
    query_template="What policy are we following?",
    ground_truth=DetectionGroundTruth(
        decision="{policy_b}",
        must_detect=["F-POLICY"],
        must_mention=["{policy_b}"],
        must_not_mention=["{policy_a}"],
    ),
    variables={
        "policy_a": ["3 days in office", "unlimited PTO", "quarterly reviews"],
        "policy_b": ["4 days in office", "15 days PTO", "monthly reviews"],
    },
)

# =============================================================================
# CUE TYPE 4: TEMPORAL UPDATE
# "As of now...", "Starting today...", "From this point..."
# =============================================================================

TEMPORAL_UPDATE_STATUS_OBVIOUS = DetectionTemplate(
    id="DET-TMP-STS-001",
    name="temporal_update_status",
    description="Status changes with temporal marker",
    domain="project",
    difficulty="obvious",
    cue_type="temporal_update",
    conversation_pattern=[
        TurnTemplate(
            role="user",
            content="The project status is {status_a}.",
        ),
        TurnTemplate(
            role="assistant",
            content="Project status: {status_a}.",
        ),
        TurnTemplate(
            role="user",
            content="As of today, the project is {status_b}.",
            implicit_supersession={
                "detection_cue": "As of today, the project is {status_b}",
                "supersedes_fact_id": "F-STATUS",
                "difficulty": "obvious",
            },
        ),
    ],
    state_writes=[
        WriteTemplate(
            id="F-STATUS",
            key="project_status",
            value="{status_a}",
        ),
    ],
    query_template="What's the current project status?",
    ground_truth=DetectionGroundTruth(
        decision="{status_b}",
        must_detect=["F-STATUS"],
        must_mention=["{status_b}"],
        must_not_mention=["{status_a}"],
    ),
    variables={
        "status_a": ["on track", "in planning", "under review"],
        "status_b": ["delayed", "on hold", "cancelled"],
    },
)

TEMPORAL_UPDATE_RATE_SUBTLE = DetectionTemplate(
    id="DET-TMP-RAT-002",
    name="temporal_update_rate",
    description="Rate changes with subtle temporal cue",
    domain="sales",
    difficulty="subtle",
    cue_type="temporal_update",
    conversation_pattern=[
        TurnTemplate(
            role="user",
            content="The hourly rate is {rate_a}.",
        ),
        TurnTemplate(
            role="assistant",
            content="Hourly rate noted: {rate_a}.",
        ),
        TurnTemplate(
            role="user",
            content="The rate is now {rate_b}.",
            implicit_supersession={
                "detection_cue": "The rate is now {rate_b}",
                "supersedes_fact_id": "F-RATE",
                "difficulty": "subtle",
            },
        ),
    ],
    state_writes=[
        WriteTemplate(
            id="F-RATE",
            key="hourly_rate",
            value="{rate_a}",
        ),
    ],
    query_template="What's the current hourly rate?",
    ground_truth=DetectionGroundTruth(
        decision="{rate_b}",
        must_detect=["F-RATE"],
        must_mention=["{rate_b}"],
        must_not_mention=["{rate_a}"],
    ),
    variables={
        "rate_a": ["$150", "$200", "$125"],
        "rate_b": ["$175", "$225", "$150"],
    },
)

# =============================================================================
# CUE TYPE 5: PARTIAL CORRECTION
# "Keep X but change Y to Z"
# =============================================================================

PARTIAL_CORRECTION_ORDER_OBVIOUS = DetectionTemplate(
    id="DET-PRT-ORD-001",
    name="partial_correction_order",
    description="Partial correction of order details",
    domain="sales",
    difficulty="obvious",
    cue_type="partial_correction",
    conversation_pattern=[
        TurnTemplate(
            role="user",
            content="Order: {quantity_a} units of {product} to {location}.",
        ),
        TurnTemplate(
            role="assistant",
            content="Order confirmed: {quantity_a} units of {product} to {location}.",
        ),
        TurnTemplate(
            role="user",
            content="Keep the location, but change the quantity to {quantity_b}.",
            implicit_supersession={
                "detection_cue": "change the quantity to {quantity_b}",
                "supersedes_fact_id": "F-QUANTITY",
                "difficulty": "obvious",
            },
        ),
    ],
    state_writes=[
        WriteTemplate(
            id="F-QUANTITY",
            key="order_quantity",
            value="{quantity_a}",
        ),
        WriteTemplate(
            id="F-LOCATION",
            key="order_location",
            value="{location}",
        ),
    ],
    query_template="How many units in the order?",
    ground_truth=DetectionGroundTruth(
        decision="{quantity_b}",
        must_detect=["F-QUANTITY"],
        must_mention=["{quantity_b}"],
        must_not_mention=["{quantity_a}"],
    ),
    variables={
        "quantity_a": ["100", "500", "1000"],
        "quantity_b": ["150", "750", "1500"],
        "product": ["widgets", "units", "packages"],
        "location": ["NYC warehouse", "LA distribution center", "Chicago hub"],
    },
)

# =============================================================================
# CUE TYPE 6: DRAFT TO COMMITMENT
# "We've finalized on X" (when draft was different)
# =============================================================================

DRAFT_TO_COMMITMENT_PRICE_OBVIOUS = DetectionTemplate(
    id="DET-DFT-PRC-001",
    name="draft_to_commitment_price",
    description="Draft pricing becomes final at different value",
    domain="sales",
    difficulty="obvious",
    cue_type="draft_to_commitment",
    conversation_pattern=[
        TurnTemplate(
            role="user",
            content="Draft pricing: {price_a} per unit.",
        ),
        TurnTemplate(
            role="assistant",
            content="Draft recorded: {price_a} per unit.",
        ),
        TurnTemplate(
            role="user",
            content="We've finalized the pricing at {price_b} per unit.",
            implicit_supersession={
                "detection_cue": "We've finalized the pricing at {price_b}",
                "supersedes_fact_id": "F-PRICE",
                "difficulty": "obvious",
            },
        ),
    ],
    state_writes=[
        WriteTemplate(
            id="F-PRICE",
            key="unit_price",
            value="{price_a} (DRAFT)",
        ),
    ],
    query_template="What's the final price per unit?",
    ground_truth=DetectionGroundTruth(
        decision="{price_b}",
        must_detect=["F-PRICE"],
        must_mention=["{price_b}"],
        must_not_mention=["{price_a}"],
    ),
    variables={
        "price_a": ["$45", "$120", "$85"],
        "price_b": ["$50", "$100", "$95"],
    },
)

DRAFT_TO_COMMITMENT_PLAN_SUBTLE = DetectionTemplate(
    id="DET-DFT-PLN-002",
    name="draft_to_commitment_plan",
    description="Tentative plan becomes official with changes",
    domain="project",
    difficulty="subtle",
    cue_type="draft_to_commitment",
    conversation_pattern=[
        TurnTemplate(
            role="user",
            content="We're thinking {approach_a} for the implementation.",
        ),
        TurnTemplate(
            role="assistant",
            content="Tentative approach: {approach_a}.",
        ),
        TurnTemplate(
            role="user",
            content="The official plan is {approach_b}.",
            implicit_supersession={
                "detection_cue": "The official plan is {approach_b}",
                "supersedes_fact_id": "F-APPROACH",
                "difficulty": "subtle",
            },
        ),
    ],
    state_writes=[
        WriteTemplate(
            id="F-APPROACH",
            key="implementation_approach",
            value="{approach_a} (tentative)",
        ),
    ],
    query_template="What's the implementation plan?",
    ground_truth=DetectionGroundTruth(
        decision="{approach_b}",
        must_detect=["F-APPROACH"],
        must_mention=["{approach_b}"],
        must_not_mention=["{approach_a}"],
    ),
    variables={
        "approach_a": ["phased rollout", "big bang deployment", "pilot program"],
        "approach_b": ["gradual migration", "staged release", "beta testing"],
    },
)

# =============================================================================
# CUE TYPE 7: REVERSION
# "Let's go back to the original X"
# =============================================================================

REVERSION_DESIGN_OBVIOUS = DetectionTemplate(
    id="DET-REV-DSN-001",
    name="reversion_design",
    description="Revert to original design after trying alternative",
    domain="project",
    difficulty="obvious",
    cue_type="reversion",
    conversation_pattern=[
        TurnTemplate(
            role="user",
            content="The design is {design_a}.",
        ),
        TurnTemplate(
            role="assistant",
            content="Design confirmed: {design_a}.",
        ),
        TurnTemplate(
            role="user",
            content="Let's try {design_b} instead.",
        ),
        TurnTemplate(
            role="assistant",
            content="Switching to {design_b}.",
        ),
        TurnTemplate(
            role="user",
            content="Actually, let's go back to {design_a}.",
            implicit_supersession={
                "detection_cue": "let's go back to {design_a}",
                "supersedes_fact_id": "F-DESIGN-V2",
                "difficulty": "obvious",
            },
        ),
    ],
    state_writes=[
        WriteTemplate(
            id="F-DESIGN-V1",
            key="design_choice",
            value="{design_a}",
        ),
        WriteTemplate(
            id="F-DESIGN-V2",
            key="design_choice",
            value="{design_b}",
            supersedes="F-DESIGN-V1",
        ),
    ],
    query_template="What design are we using?",
    ground_truth=DetectionGroundTruth(
        decision="{design_a}",
        must_detect=["F-DESIGN-V2"],
        must_mention=["{design_a}"],
        must_not_mention=["{design_b}"],
    ),
    variables={
        "design_a": ["blue theme", "minimalist layout", "card-based UI"],
        "design_b": ["green theme", "detailed layout", "list-based UI"],
    },
)

# =============================================================================
# ALL TEMPLATES BY CUE TYPE AND DIFFICULTY
# =============================================================================

DETECTION_TEMPLATES: list[DetectionTemplate] = [
    # Explicit Correction
    EXPLICIT_CORRECTION_LOCATION_OBVIOUS,
    EXPLICIT_CORRECTION_DATE_OBVIOUS,
    EXPLICIT_CORRECTION_AMOUNT_SUBTLE,
    # Implicit Contradiction
    IMPLICIT_CONTRADICTION_VENDOR_SUBTLE,
    IMPLICIT_CONTRADICTION_PRIORITY_ADVERSARIAL,
    # Authority Override
    AUTHORITY_OVERRIDE_BUDGET_OBVIOUS,
    AUTHORITY_OVERRIDE_POLICY_SUBTLE,
    # Temporal Update
    TEMPORAL_UPDATE_STATUS_OBVIOUS,
    TEMPORAL_UPDATE_RATE_SUBTLE,
    # Partial Correction
    PARTIAL_CORRECTION_ORDER_OBVIOUS,
    # Draft to Commitment
    DRAFT_TO_COMMITMENT_PRICE_OBVIOUS,
    DRAFT_TO_COMMITMENT_PLAN_SUBTLE,
    # Reversion
    REVERSION_DESIGN_OBVIOUS,
]


def get_detection_templates(
    difficulty: DetectionDifficulty | None = None,
    cue_type: CueType | None = None,
) -> list[DetectionTemplate]:
    """Get detection templates filtered by difficulty and/or cue type."""
    templates = DETECTION_TEMPLATES

    if difficulty is not None:
        templates = [t for t in templates if t.difficulty == difficulty]

    if cue_type is not None:
        templates = [t for t in templates if t.cue_type == cue_type]

    return templates


def get_templates_by_cue_type() -> dict[CueType, list[DetectionTemplate]]:
    """Get templates grouped by cue type."""
    result: dict[CueType, list[DetectionTemplate]] = {}
    for template in DETECTION_TEMPLATES:
        if template.cue_type not in result:
            result[template.cue_type] = []
        result[template.cue_type].append(template)
    return result


def get_templates_by_difficulty() -> dict[DetectionDifficulty, list[DetectionTemplate]]:
    """Get templates grouped by difficulty."""
    result: dict[DetectionDifficulty, list[DetectionTemplate]] = {}
    for template in DETECTION_TEMPLATES:
        if template.difficulty not in result:
            result[template.difficulty] = []
        result[template.difficulty].append(template)
    return result
