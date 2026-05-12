from __future__ import annotations

from agent.schemas import (
    CausalObligation,
    InvariantDef,
    InvariantCategory,
    ObligationCategory,
    ObligationDef,
    RequiredInvariant,
)

_OBLIGATIONS: dict[str, ObligationDef] = {
    # ── Temporal Causality ────────────────────────────────────────────────────
    "past_and_current_observations": ObligationDef(
        id="past_and_current_observations",
        category=ObligationCategory.TEMPORAL_CAUSALITY,
        description="For each decision, use only information available at or before the decision time.",
        tags=["point-in-time", "no-lookahead"],
        severity="critical",
        applicable_domains=["systematic_trading", "finance", "time_series"],
    ),
    "future_rows": ObligationDef(
        id="future_rows",
        category=ObligationCategory.TEMPORAL_CAUSALITY,
        description="Rows after a decision timestamp must not affect that decision or earlier outputs.",
        tags=["no-lookahead", "prefix-invariance"],
        severity="critical",
        applicable_domains=["systematic_trading", "finance", "time_series"],
    ),
    "forward_looking_bias": ObligationDef(
        id="forward_looking_bias",
        category=ObligationCategory.TEMPORAL_CAUSALITY,
        description="Do not use information from future time periods to influence current decisions.",
        tags=["no-lookahead", "point-in-time"],
        severity="critical",
        applicable_domains=["systematic_trading", "finance", "time_series"],
    ),
    "future_peer_information": ObligationDef(
        id="future_peer_information",
        category=ObligationCategory.TEMPORAL_CAUSALITY,
        description="Peer entity information unavailable at the current decision time must not influence current outputs.",
        tags=["cross-sectional", "no-lookahead"],
        severity="critical",
        applicable_domains=["systematic_trading", "finance"],
    ),
    "future_target_information": ObligationDef(
        id="future_target_information",
        category=ObligationCategory.TEMPORAL_CAUSALITY,
        description="Future target or label values must not influence feature construction or decisions.",
        tags=["no-target-leakage", "feature-engineering"],
        severity="critical",
        applicable_domains=["systematic_trading", "finance", "time_series"],
    ),
    "non_interference_between_decisions": ObligationDef(
        id="non_interference_between_decisions",
        category=ObligationCategory.TEMPORAL_CAUSALITY,
        description="Future or unrelated decisions must not retroactively alter earlier outputs.",
        tags=["no-lookahead", "prefix-invariance"],
        severity="critical",
        applicable_domains=["systematic_trading", "finance", "time_series"],
    ),
    # ── Information Boundary ──────────────────────────────────────────────────
    "task_available_information": ObligationDef(
        id="task_available_information",
        category=ObligationCategory.INFORMATION_BOUNDARY,
        description="Use only information available in the provided inputs and explicit user constraints.",
        tags=["input-boundary", "data-isolation"],
        severity="high",
        applicable_domains=["*"],
    ),
    "unstated_external_information": ObligationDef(
        id="unstated_external_information",
        category=ObligationCategory.INFORMATION_BOUNDARY,
        description="Do not use external data, hidden files, or assumptions not present in the user request.",
        tags=["input-boundary", "no-external-data"],
        severity="high",
        applicable_domains=["*"],
    ),
    "data_lineage_tracking": ObligationDef(
        id="data_lineage_tracking",
        category=ObligationCategory.INFORMATION_BOUNDARY,
        description="All intermediate data transformations must be traceable to declared inputs.",
        tags=["provenance", "reproducibility"],
        severity="medium",
        applicable_domains=["systematic_trading", "finance"],
    ),
    # ── Estimation Separation ─────────────────────────────────────────────────
    "full_sample_online_statistics": ObligationDef(
        id="full_sample_online_statistics",
        category=ObligationCategory.ESTIMATION_SEPARATION,
        description="Online row-level decisions must not depend on statistics estimated from the full dataframe.",
        tags=["online", "no-global-fit", "rolling"],
        severity="critical",
        applicable_domains=["systematic_trading", "finance", "time_series"],
    ),
    "estimation_separation": ObligationDef(
        id="estimation_separation",
        category=ObligationCategory.ESTIMATION_SEPARATION,
        description="Parameter estimation must use only the training prefix or rolling window, not the full sample or future data.",
        tags=["train-test-separation", "rolling-window", "expanding-window"],
        severity="critical",
        applicable_domains=["systematic_trading", "finance", "time_series"],
    ),
    "rolling_or_prefix_estimation": ObligationDef(
        id="rolling_or_prefix_estimation",
        category=ObligationCategory.ESTIMATION_SEPARATION,
        description="Temporal estimation procedures must use rolling, expanding, or prefix-aligned fitting scopes.",
        tags=["rolling-window", "expanding-window", "online"],
        severity="critical",
        applicable_domains=["systematic_trading", "finance", "time_series"],
    ),
    "training_evaluation_separation": ObligationDef(
        id="training_evaluation_separation",
        category=ObligationCategory.ESTIMATION_SEPARATION,
        description="Evaluation observations must not influence training-time parameter estimation or model selection.",
        tags=["train-test-separation", "no-data-leakage"],
        severity="critical",
        applicable_domains=["systematic_trading", "finance", "time_series"],
    ),
    "evaluation_after_decision": ObligationDef(
        id="evaluation_after_decision",
        category=ObligationCategory.ESTIMATION_SEPARATION,
        description="Evaluation outcomes must occur strictly after the associated decision point.",
        tags=["point-in-time", "no-lookahead"],
        severity="critical",
        applicable_domains=["systematic_trading", "finance", "time_series"],
    ),
    "risk_measure_alignment": ObligationDef(
        id="risk_measure_alignment",
        category=ObligationCategory.RISK_OPTIMIZATION_INTEGRITY,
        description="Risk measures must be computed using information available at the associated evaluation time.",
        tags=["point-in-time", "risk"],
        severity="critical",
        applicable_domains=["systematic_trading", "finance"],
    ),
    "post_optimization_evaluation_isolation": ObligationDef(
        id="post_optimization_evaluation_isolation",
        category=ObligationCategory.RISK_OPTIMIZATION_INTEGRITY,
        description="Post-optimization evaluation metrics must not influence earlier optimization stages.",
        tags=["optimization", "no-leakage"],
        severity="critical",
        applicable_domains=["systematic_trading", "finance"],
    ),
    # ── Stable Decision Mapping ───────────────────────────────────────────────
    "stable_decision_mapping": ObligationDef(
        id="stable_decision_mapping",
        category=ObligationCategory.STABLE_DECISION_MAPPING,
        description="The same input at the same decision time must produce the same output, regardless of surrounding context.",
        tags=["determinism", "repeatability"],
        severity="high",
        applicable_domains=["*"],
    ),
    # ── Output Alignment ─────────────────────────────────────────────────────────
    "row_output_alignment": ObligationDef(
        id="row_output_alignment",
        category=ObligationCategory.OUTPUT_ALIGNMENT,
        description="Row-level outputs must align with the corresponding input decision rows.",
        tags=["shape", "row-count"],
        severity="high",
        applicable_domains=["*"],
    ),
    "index_preservation": ObligationDef(
        id="index_preservation",
        category=ObligationCategory.OUTPUT_ALIGNMENT,
        description="Output indices and ordering must preserve the intended decision alignment.",
        tags=["index", "alignment"],
        severity="high",
        applicable_domains=["*"],
    ),
    "shape_consistency": ObligationDef(
        id="shape_consistency",
        category=ObligationCategory.OUTPUT_ALIGNMENT,
        description="Output shapes and dimensions must remain consistent with the declared output contract.",
        tags=["shape", "dimensionality"],
        severity="high",
        applicable_domains=["*"],
    ),
    # ── Target-Derived Features ───────────────────────────────────────────────
    "target_derived_features": ObligationDef(
        id="target_derived_features",
        category=ObligationCategory.TARGET_DERIVED_FEATURES,
        description="Do not construct features using future or target variable values that would not be available at decision time.",
        tags=["no-target-leakage", "feature-engineering"],
        severity="critical",
        applicable_domains=["systematic_trading", "finance"],
    ),
    "label_generation_alignment": ObligationDef(
        id="label_generation_alignment",
        category=ObligationCategory.TARGET_DERIVED_FEATURES,
        description="Generated labels must align with the intended prediction horizon and decision timing.",
        tags=["label", "horizon", "alignment"],
        severity="high",
        applicable_domains=["systematic_trading", "finance", "time_series"],
    ),
    # ── Runtime Isolation ─────────────────────────────────────────────────────
    "undeclared_runtime_dependencies": ObligationDef(
        id="undeclared_runtime_dependencies",
        category=ObligationCategory.RUNTIME_ISOLATION,
        description="All runtime dependencies (files, packages, environment variables) must be explicitly declared.",
        tags=["sandbox", "reproducibility"],
        severity="high",
        applicable_domains=["*"],
    ),
    "environment_independent_execution": ObligationDef(
        id="environment_independent_execution",
        category=ObligationCategory.RUNTIME_ISOLATION,
        description="Code execution should not depend on hidden environment state or machine-specific configuration.",
        tags=["reproducibility", "sandbox"],
        severity="medium",
        applicable_domains=["*"],
    ),
}

_INVARIANTS: dict[str, InvariantDef] = {
    # ── Output Alignment ──────────────────────────────────────────────────────
    "output_alignment": InvariantDef(
        id="output_alignment",
        category=InvariantCategory.OUTPUT_ALIGNMENT,
        description="Outputs must align with the intended decision rows, indices, and prediction horizon.",
        tags=["shape", "alignment"],
        check_type="deterministic",
    ),
    "row_alignment": InvariantDef(
        id="row_alignment",
        category=InvariantCategory.OUTPUT_ALIGNMENT,
        description="The output must have the same number of rows as the input for row-level decisions.",
        tags=["shape", "row-count"],
        check_type="deterministic",
    ),
    "index_alignment": InvariantDef(
        id="index_alignment",
        category=InvariantCategory.OUTPUT_ALIGNMENT,
        description="The output index must align with the input index, preserving temporal ordering.",
        tags=["index", "temporal-alignment"],
        check_type="deterministic",
    ),
    "shape_alignment": InvariantDef(
        id="shape_alignment",
        category=InvariantCategory.OUTPUT_ALIGNMENT,
        description="The output must satisfy the requested shape, dimensionality, and column contract.",
        tags=["shape", "columns", "dimensionality"],
        check_type="deterministic",
    ),
    "shape_consistency": InvariantDef(
        id="shape_consistency",
        category=InvariantCategory.OUTPUT_ALIGNMENT,
        description="Output shapes and dimensions must remain consistent across valid executions.",
        tags=["shape", "stability"],
        check_type="deterministic",
    ),
    # ── Index Monotonicity ────────────────────────────────────────────────────
    "index_monotonicity": InvariantDef(
        id="index_monotonicity",
        category=InvariantCategory.INDEX_MONOTONICITY,
        description="Temporal ordering and index monotonicity must be preserved throughout the decision pipeline.",
        tags=["temporal", "index", "ordering"],
        check_type="deterministic",
    ),
    # ── Causal Invariance ─────────────────────────────────────────────────────
    "prefix_invariance": InvariantDef(
        id="prefix_invariance",
        category=InvariantCategory.CAUSAL_INVARIANCE,
        description="Changing future rows must not change outputs for earlier decision times.",
        tags=["causal", "no-lookahead", "prefix"],
        check_type="light_synthetic",
    ),
    "future_perturbation_non_interference": InvariantDef(
        id="future_perturbation_non_interference",
        category=InvariantCategory.CAUSAL_INVARIANCE,
        description="Perturbing future observations should not alter outputs for earlier decisions.",
        tags=["causal", "robustness", "perturbation"],
        check_type="synthetic_intervention",
    ),
    "intervention_stability": InvariantDef(
        id="intervention_stability",
        category=InvariantCategory.CAUSAL_INVARIANCE,
        description="Small plausible perturbations to irrelevant future inputs should not produce unstable earlier decisions.",
        tags=["causal", "stability", "intervention"],
        check_type="light_intervention",
    ),
    # ── Train / Test Isolation ────────────────────────────────────────────────
    "train_test_isolation": InvariantDef(
        id="train_test_isolation",
        category=InvariantCategory.TRAIN_TEST_ISOLATION,
        description="Changes to evaluation-period observations should not alter training-period fitted parameters.",
        tags=["train-test", "isolation", "no-leakage"],
        check_type="synthetic_partition",
    ),
    # ── Decision Locality ─────────────────────────────────────────────────────
    "decision_locality": InvariantDef(
        id="decision_locality",
        category=InvariantCategory.DECISION_LOCALITY,
        description="A decision should depend only on causally reachable observations within the allowed information scope.",
        tags=["causal", "reachability", "information-boundary"],
        check_type="causal_dependency_probe",
    ),
    # ── Cross-Sectional Isolation ─────────────────────────────────────────────
    "cross_sectional_isolation": InvariantDef(
        id="cross_sectional_isolation",
        category=InvariantCategory.CROSS_SECTIONAL_ISOLATION,
        description="Entity-specific outputs should not depend on unavailable future information from peer entities.",
        tags=["cross-sectional", "isolation", "no-leakage"],
        check_type="synthetic_cross_section",
    ),
    # ── Deterministic Replay ──────────────────────────────────────────────────
    "deterministic_replay": InvariantDef(
        id="deterministic_replay",
        category=InvariantCategory.DETERMINISTIC_REPLAY,
        description="Repeated execution under identical inputs should produce identical outputs.",
        tags=["determinism", "reproducibility", "replay"],
        check_type="execution_consistency",
    ),
}


def get_obligation(item_id: str) -> CausalObligation:
    """Look up an obligation definition by ID. Raises KeyError if unknown."""
    if item_id not in _OBLIGATIONS:
        raise KeyError(f"Unknown obligation id: {item_id!r}")
    return CausalObligation(
        id=_OBLIGATIONS[item_id].id,
        description=_OBLIGATIONS[item_id].description,
    )


def get_invariant(item_id: str) -> RequiredInvariant:
    """Look up an invariant definition by ID. Raises KeyError if unknown."""
    if item_id not in _INVARIANTS:
        raise KeyError(f"Unknown invariant id: {item_id!r}")
    return RequiredInvariant(
        id=_INVARIANTS[item_id].id,
        description=_INVARIANTS[item_id].description,
        check_type=_INVARIANTS[item_id].check_type,
    )


def all_obligation_ids() -> set[str]:
    return set(_OBLIGATIONS)


def all_invariant_ids() -> set[str]:
    return set(_INVARIANTS)
