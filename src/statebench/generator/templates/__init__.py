"""Template definitions for each benchmark track."""

from statebench.generator.templates.authority import (
    AUTHORITY_CONFLICT_TEMPLATES,
    AUTHORITY_LEVELS,
    AUTHORITY_RANK,
    AuthorityConflictTemplate,
    AuthorityInstruction,
    authority_outranks,
    get_authority_templates_by_conflict_type,
    get_authority_templates_by_domain,
)
from statebench.generator.templates.brutal import (
    BRUTAL_SCENARIOS,
    BrutalEvent,
    BrutalScenario,
    get_brutal_scenarios,
)
from statebench.generator.templates.causality import (
    AGGREGATION_TEMPLATES,
    CAUSALITY_TEMPLATES,
    CHAIN_DEPENDENCY_TEMPLATES,
    CONFLICTING_TEMPLATES,
    EDGE_CASE_TEMPLATES,
    HARD_CAUSALITY_TEMPLATES,
    MULTI_CONSTRAINT_TEMPLATES,
    AggregationTemplate,
    CausalityTemplate,
    ChainDependencyTemplate,
    ConflictingConstraintTemplate,
    EdgeCaseTemplate,
    MultiConstraintTemplate,
    get_causality_templates,
    get_hard_causality_templates,
    get_paired_test,
)
from statebench.generator.templates.commitment import (
    COMMITMENT_TEMPLATES,
    CommitmentTemplate,
    get_commitment_templates_by_domain,
)

# v1.0: Detection Track
from statebench.generator.templates.detection import (
    DETECTION_TEMPLATES,
    CueType,
    DetectionGroundTruth,
    DetectionTemplate,
    TurnTemplate,
    WriteTemplate,
    get_detection_templates,
    get_templates_by_cue_type,
    get_templates_by_difficulty,
)

# v1.0: Enterprise Track
from statebench.generator.templates.enterprise import (
    ENTERPRISE_PRIVACY_TEMPLATES,
    EnterprisePrivacyTemplate,
    TenantContext,
    get_enterprise_templates_by_domain,
    get_enterprise_templates_by_leak_type,
)
from statebench.generator.templates.environmental import (
    ENVIRONMENTAL_TEMPLATES,
    EnvironmentalTemplate,
    get_environmental_templates_by_domain,
)

# v0.2: Advanced failure mode templates
from statebench.generator.templates.hallucination import (
    HALLUCINATION_TEMPLATES,
    HallucinationTemplate,
    get_hallucination_templates,
)
from statebench.generator.templates.interruption import (
    INTERRUPTION_TEMPLATES,
    InterruptionTemplate,
    get_interruption_templates_by_domain,
)
from statebench.generator.templates.permission import (
    PERMISSION_TEMPLATES,
    PermissionTemplate,
    get_permission_templates_by_domain,
)
from statebench.generator.templates.repair import (
    REPAIR_CHAIN_TEMPLATES,
    RepairChain,
    get_repair_templates,
)
from statebench.generator.templates.scope_leak import (
    SCOPE_LEAK_TEMPLATES,
    ScopeLeakTemplate,
    get_scope_leak_templates,
    get_scope_leak_templates_by_type,
)
from statebench.generator.templates.supersession import (
    SUPERSESSION_TEMPLATES,
    SupersessionTemplate,
)
from statebench.generator.templates.supersession import (
    get_templates_by_domain as get_supersession_templates_by_domain,
)

__all__ = [
    # Track 1: Supersession
    "SupersessionTemplate",
    "SUPERSESSION_TEMPLATES",
    "get_supersession_templates_by_domain",
    # Track 2: Commitment
    "CommitmentTemplate",
    "COMMITMENT_TEMPLATES",
    "get_commitment_templates_by_domain",
    # Track 3: Interruption
    "InterruptionTemplate",
    "INTERRUPTION_TEMPLATES",
    "get_interruption_templates_by_domain",
    # Track 4: Permission
    "PermissionTemplate",
    "PERMISSION_TEMPLATES",
    "get_permission_templates_by_domain",
    # Track 5: Environmental
    "EnvironmentalTemplate",
    "ENVIRONMENTAL_TEMPLATES",
    "get_environmental_templates_by_domain",
    # Track 6: Hallucination Resistance
    "HallucinationTemplate",
    "HALLUCINATION_TEMPLATES",
    "get_hallucination_templates",
    # Track 7: Scope Leak
    "ScopeLeakTemplate",
    "SCOPE_LEAK_TEMPLATES",
    "get_scope_leak_templates",
    "get_scope_leak_templates_by_type",
    # Track 8: Causality
    "CausalityTemplate",
    "MultiConstraintTemplate",
    "EdgeCaseTemplate",
    "ConflictingConstraintTemplate",
    "ChainDependencyTemplate",
    "AggregationTemplate",
    "CAUSALITY_TEMPLATES",
    "HARD_CAUSALITY_TEMPLATES",
    "MULTI_CONSTRAINT_TEMPLATES",
    "EDGE_CASE_TEMPLATES",
    "CONFLICTING_TEMPLATES",
    "CHAIN_DEPENDENCY_TEMPLATES",
    "AGGREGATION_TEMPLATES",
    "get_causality_templates",
    "get_hard_causality_templates",
    "get_paired_test",
    # Track 9: Repair Propagation
    "RepairChain",
    "REPAIR_CHAIN_TEMPLATES",
    "get_repair_templates",
    # Track 10: Brutal Scenarios
    "BrutalScenario",
    "BrutalEvent",
    "BRUTAL_SCENARIOS",
    "get_brutal_scenarios",
    # v1.0: Detection Track
    "CueType",
    "DetectionGroundTruth",
    "DetectionTemplate",
    "DETECTION_TEMPLATES",
    "TurnTemplate",
    "WriteTemplate",
    "get_detection_templates",
    "get_templates_by_cue_type",
    "get_templates_by_difficulty",
    # v1.0: Enterprise Track
    "EnterprisePrivacyTemplate",
    "TenantContext",
    "ENTERPRISE_PRIVACY_TEMPLATES",
    "get_enterprise_templates_by_leak_type",
    "get_enterprise_templates_by_domain",
    "AUTHORITY_LEVELS",
    "AUTHORITY_RANK",
    "AuthorityConflictTemplate",
    "AuthorityInstruction",
    "AUTHORITY_CONFLICT_TEMPLATES",
    "authority_outranks",
    "get_authority_templates_by_conflict_type",
    "get_authority_templates_by_domain",
]
