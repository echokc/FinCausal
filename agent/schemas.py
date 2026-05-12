from __future__ import annotations

from enum import StrEnum
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Severity = Literal["low", "medium", "high", "critical"]


class ObligationCategory(StrEnum):
    """Taxonomy of causal-information obligations."""

    TEMPORAL_CAUSALITY = "temporal_causality"
    INFORMATION_BOUNDARY = "information_boundary"
    ESTIMATION_SEPARATION = "estimation_separation"
    STABLE_DECISION_MAPPING = "stable_decision_mapping"
    TARGET_DERIVED_FEATURES = "target_derived_features"
    RUNTIME_ISOLATION = "runtime_isolation"
    OUTPUT_ALIGNMENT = "output_alignment"
    RISK_OPTIMIZATION_INTEGRITY = "risk_optimization_integrity"


class InvariantCategory(StrEnum):
    """Taxonomy of behavioral invariants."""

    OUTPUT_ALIGNMENT = "output_alignment"
    CAUSAL_INVARIANCE = "causal_invariance"
    DETERMINISTIC_REPLAY = "deterministic_replay"
    INDEX_MONOTONICITY = "index_monotonicity"
    TRAIN_TEST_ISOLATION = "train_test_isolation"
    DECISION_LOCALITY = "decision_locality"
    CROSS_SECTIONAL_ISOLATION = "cross_sectional_isolation"


@dataclass(frozen=True)
class ToolError:
    type: str
    message: str
    recoverable: bool = True


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    result: Any = None
    error: ToolError | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(
        cls,
        result: Any,
        *,
        warnings: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            ok=True,
            result=result,
            warnings=warnings or [],
            metadata=metadata or {},
        )

    @classmethod
    def failure(
        cls,
        error_type: str,
        message: str,
        *,
        recoverable: bool = True,
        warnings: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            ok=False,
            error=ToolError(type=error_type, message=message, recoverable=recoverable),
            warnings=warnings or [],
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class Intent:
    task_type: str
    task_description: str
    domain: str
    input_data_description: str
    requested_output: str
    explicit_user_constraints: list[str] = field(default_factory=list)
    assumed_runtime_context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CausalObligation:
    id: str
    description: str


@dataclass(frozen=True)
class RequiredInvariant:
    id: str
    description: str
    check_type: str = "deterministic"


@dataclass(frozen=True)
class ObligationDef:
    id: str
    category: ObligationCategory
    description: str
    tags: list[str] = field(default_factory=list)
    severity: Severity = "high"
    applicable_domains: list[str] = field(default_factory=lambda: ["*"])


@dataclass(frozen=True)
class InvariantDef:
    id: str
    category: InvariantCategory
    description: str
    tags: list[str] = field(default_factory=list)
    check_type: str = "deterministic"


@dataclass(frozen=True)
class DecisionContext:
    decision_unit: str = "row"
    time_column: str | None = None
    decision_time: str | None = None
    output_used_at: str | None = None


@dataclass(frozen=True)
class EstimationScope:
    fit_scope: str = "task_dependent"
    forbid_full_sample_fit: bool = False


@dataclass(frozen=True)
class AgentOutputContract:
    variable_name: str
    kind: str
    accepted_names: list[str] = field(default_factory=list)
    alignment: str | None = None
    semantic: str | None = None


@dataclass(frozen=True)
class CausalContract:
    contract_id: str
    schema_version: str
    intent: Intent
    decision_context: DecisionContext
    allowed_information: list[CausalObligation]
    forbidden_information: list[CausalObligation]
    estimation_scope: EstimationScope
    required_invariants: list[RequiredInvariant]
    known_hazards: list[str]
    output_contract: AgentOutputContract
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GuardViolation:
    rule_id: str
    severity: Severity
    message: str
    line: int | None = None
    evidence: str | None = None


@dataclass(frozen=True)
class GuardResults:
    passed: bool
    violations: list[GuardViolation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rule_versions: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class InvariantCheckResult:
    invariant_id: str
    passed: bool
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InvariantResults:
    passed: bool
    checks: list[InvariantCheckResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CriticConcern:
    severity: Severity
    obligation_id: str
    claim: str
    code_evidence: str
    knowledge_evidence: str
    recommended_fix: str
    requires_repair: bool = False


@dataclass(frozen=True)
class CriticResults:
    passed: bool
    concerns: list[CriticConcern] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class AgentTrace:
    intent: dict[str, Any]
    causal_contract: dict[str, Any]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    generated_code_versions: list[dict[str, Any]] = field(default_factory=list)
    guard_results: list[dict[str, Any]] = field(default_factory=list)
    sandbox_results: list[dict[str, Any]] = field(default_factory=list)
    invariant_results: list[dict[str, Any]] = field(default_factory=list)
    critic_concerns: list[dict[str, Any]] = field(default_factory=list)
    repair_history: list[dict[str, Any]] = field(default_factory=list)
    layer1_eval: dict[str, Any] = field(default_factory=lambda: {"status": "skipped"})
    final_status: str = "unknown"


def to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    return value
