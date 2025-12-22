"""Template definitions for each benchmark track."""

from statebench.generator.templates.supersession import (
    SupersessionTemplate,
    SUPERSESSION_TEMPLATES,
    get_templates_by_domain as get_supersession_templates_by_domain,
)
from statebench.generator.templates.commitment import (
    CommitmentTemplate,
    COMMITMENT_TEMPLATES,
    get_commitment_templates_by_domain,
)
from statebench.generator.templates.interruption import (
    InterruptionTemplate,
    INTERRUPTION_TEMPLATES,
    get_interruption_templates_by_domain,
)
from statebench.generator.templates.permission import (
    PermissionTemplate,
    PERMISSION_TEMPLATES,
    get_permission_templates_by_domain,
)
from statebench.generator.templates.environmental import (
    EnvironmentalTemplate,
    ENVIRONMENTAL_TEMPLATES,
    get_environmental_templates_by_domain,
)
# v0.2: Advanced failure mode templates
from statebench.generator.templates.hallucination import (
    HallucinationTemplate,
    HALLUCINATION_TEMPLATES,
    get_hallucination_templates,
)
from statebench.generator.templates.scope_leak import (
    ScopeLeakTemplate,
    SCOPE_LEAK_TEMPLATES,
    get_scope_leak_templates,
    get_scope_leak_templates_by_type,
)
from statebench.generator.templates.causality import (
    CausalityTemplate,
    MultiConstraintTemplate,
    EdgeCaseTemplate,
    ConflictingConstraintTemplate,
    ChainDependencyTemplate,
    AggregationTemplate,
    CAUSALITY_TEMPLATES,
    HARD_CAUSALITY_TEMPLATES,
    MULTI_CONSTRAINT_TEMPLATES,
    EDGE_CASE_TEMPLATES,
    CONFLICTING_TEMPLATES,
    CHAIN_DEPENDENCY_TEMPLATES,
    AGGREGATION_TEMPLATES,
    get_causality_templates,
    get_hard_causality_templates,
    get_paired_test,
)
from statebench.generator.templates.repair import (
    RepairChain,
    REPAIR_CHAIN_TEMPLATES,
    get_repair_templates,
)
from statebench.generator.templates.brutal import (
    BrutalScenario,
    BrutalEvent,
    BRUTAL_SCENARIOS,
    get_brutal_scenarios,
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
]
