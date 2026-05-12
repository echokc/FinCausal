from __future__ import annotations

import uuid
from typing import Any

from agent.contract.catalog import get_invariant, get_obligation
from agent.contract.facts import infer_intent_facts
from agent.contract.kb import retrieve_contract_knowledge
from agent.contract.policy import ContractPolicy, build_contract_policy
from agent.schemas import (
    AgentOutputContract,
    CausalContract,
    DecisionContext,
    EstimationScope,
    Intent,
    ToolResult,
    to_dict,
)

CONTRACT_SCHEMA_VERSION = "0.1"


# ── Production entry point ──────────────────────────────────────────────────


def build_contract_from_intent(
    intent: Intent,
    *,
    use_local_knowledge: bool = True,
) -> CausalContract:
    """Build a CausalContract from a user Intent.

    Orchestrates four stages:
      1. infer_intent_facts   — extract structured facts from the Intent
      2. retrieve_contract_knowledge — look up relevant hazards from KB
      3. build_contract_policy — convert facts + knowledge into policy decisions
      4. assemble_contract     — pure data mapping from policy to CausalContract
    """
    facts = infer_intent_facts(intent)
    knowledge = retrieve_contract_knowledge(intent, facts, use_local_knowledge=use_local_knowledge)
    policy = build_contract_policy(facts, knowledge)
    return assemble_contract(intent, policy)


# ── Assembly (pure, dumb) ───────────────────────────────────────────────────


def assemble_contract(intent: Intent, policy: ContractPolicy) -> CausalContract:
    """Map a ContractPolicy onto a CausalContract with no if/else logic.

    Only does:
    - ID → definition lookups via catalog.py
    - Field mapping from policy fields to CausalContract constructor args
    - UUID generation for contract_id
    """
    allowed = [get_obligation(id) for id in policy.allowed_information_ids]
    forbidden = [get_obligation(id) for id in policy.forbidden_information_ids]
    invariants = [get_invariant(id) for id in policy.invariant_ids]

    return CausalContract(
        contract_id=f"contract_{uuid.uuid4().hex[:12]}",
        schema_version=CONTRACT_SCHEMA_VERSION,
        intent=intent,
        decision_context=DecisionContext(
            decision_unit=policy.decision_unit,
            time_column=policy.time_column,
            decision_time=policy.decision_time,
            output_used_at=policy.output_used_at,
        ),
        allowed_information=allowed,
        forbidden_information=forbidden,
        estimation_scope=EstimationScope(
            fit_scope=policy.fit_scope,
            forbid_full_sample_fit=policy.forbid_full_sample_fit,
        ),
        required_invariants=invariants,
        known_hazards=sorted(policy.known_hazard_ids),
        output_contract=AgentOutputContract(
            variable_name=policy.output_variable_name,
            kind=policy.output_kind,
            accepted_names=policy.accepted_names,
            alignment=policy.alignment,
            semantic=policy.semantic,
        ),
        metadata={"source": "prod_assembler"},
    )


# ── Validation ──────────────────────────────────────────────────────────────


def validate_contract_completeness(contract: CausalContract) -> ToolResult:
    errors: list[str] = []
    warnings: list[str] = []

    if not contract.contract_id:
        errors.append("contract_id is required.")
    if contract.schema_version != CONTRACT_SCHEMA_VERSION:
        warnings.append(
            f"Contract schema version {contract.schema_version!r} differs from supported {CONTRACT_SCHEMA_VERSION!r}."
        )
    if not contract.intent.task_type:
        errors.append("intent.task_type is required.")
    if not contract.allowed_information:
        errors.append("At least one allowed_information obligation is required.")
    if not contract.forbidden_information:
        errors.append("At least one forbidden_information obligation is required.")
    if not contract.required_invariants:
        errors.append("At least one required invariant is required.")
    if not contract.output_contract.variable_name:
        errors.append("output_contract.variable_name is required.")
    if not contract.output_contract.kind:
        errors.append("output_contract.kind is required.")

    ids = [item.id for item in contract.allowed_information + contract.forbidden_information]
    if len(ids) != len(set(ids)):
        errors.append("Obligation ids must be unique.")

    invariant_ids = [item.id for item in contract.required_invariants]
    if len(invariant_ids) != len(set(invariant_ids)):
        errors.append("Invariant ids must be unique.")

    if (
        contract.estimation_scope.forbid_full_sample_fit
        and "full_sample_online_statistics" not in ids
    ):
        warnings.append("Contract forbids full-sample fit but does not include full_sample_online_statistics.")

    if errors:
        return ToolResult.failure(
            "contract_validation_failed",
            " ".join(errors),
            recoverable=True,
            warnings=warnings,
            metadata={"tool_name": "validate_contract_completeness"},
        )

    return ToolResult.success(
        {"contract": to_dict(contract)},
        warnings=warnings,
        metadata={"tool_name": "validate_contract_completeness"},
    )


# ── Development / recipe helpers ────────────────────────────────────────────


def build_dev_intent_from_recipe(recipe: Any) -> Intent:
    """Development helper that derives Intent from a Layer 1 recipe fixture."""
    schema = recipe.schema_variants[recipe.default_schema_variant]
    output = recipe.output
    return Intent(
        task_type="generate_code",
        domain="systematic_trading",
        input_data_description=f"{recipe.behavior_key} using schema {recipe.default_schema_variant}: {schema}",
        requested_output=f"{output.kind} output `{output.variable_name}` for {output.semantic}",
        explicit_user_constraints=[
            "Use only the provided data bindings.",
            "Preserve causal information timing.",
            "Use pandas, numpy, os, math, statistics, and the Python standard library only.",
        ],
        assumed_runtime_context={"language": "python", "environment": "local_sandbox"},
        metadata={
            "behavior_key": recipe.behavior_key,
            "pillar": recipe.pillar,
            "difficulty": recipe.difficulty,
            "mechanism_variant": recipe.mechanism_variant,
        },
    )


def build_dev_contract_from_recipe(recipe: Any) -> CausalContract:
    """Development helper that derives a baseline CausalContract from a recipe.

    This is not the final Causal Contract Builder. It is deterministic
    scaffolding for testing downstream nodes: extraction, guardrails, sandbox
    execution, and trace recording.
    """
    intent = build_dev_intent_from_recipe(recipe)
    schema = recipe.schema_variants[recipe.default_schema_variant]
    time_column = _find_time_column(schema)
    output = recipe.output
    accepted_names = list(getattr(output, "accepted_names", []) or [])
    if output.variable_name not in accepted_names:
        accepted_names.insert(0, output.variable_name)

    hazards = _hazards_for_recipe(recipe)
    return CausalContract(
        contract_id=f"contract_{uuid.uuid4().hex[:12]}",
        schema_version=CONTRACT_SCHEMA_VERSION,
        intent=intent,
        decision_context=DecisionContext(
            decision_unit="row",
            time_column=time_column,
            decision_time=f"{time_column}[t]" if time_column else None,
            output_used_at=f"{time_column}[t]" if time_column else None,
        ),
        allowed_information=[
            _make_obligation(
                "past_and_current_observations",
                "For each decision, use only information available at or before the decision time.",
            )
        ],
        forbidden_information=[
            _make_obligation(
                "future_rows",
                "Rows after a decision timestamp must not affect that decision or earlier outputs.",
            ),
            _make_obligation(
                "full_sample_online_statistics",
                "Online row-level decisions must not depend on statistics estimated from the full dataframe.",
            ),
        ],
        estimation_scope=EstimationScope(
            fit_scope="rolling_or_prefix_only" if time_column else "task_dependent",
            forbid_full_sample_fit=bool(time_column),
        ),
        required_invariants=[
            _make_invariant(
                "row_alignment",
                "The output must have the same number of rows as the input for row-level decisions.",
                "deterministic",
            ),
            _make_invariant(
                "index_alignment",
                "The output index must align with the input index, preserving temporal ordering.",
                "deterministic",
            ),
            _make_invariant(
                "prefix_invariance",
                "Changing future rows should not change outputs for earlier decision times.",
                "light_synthetic",
            ),
        ],
        known_hazards=hazards,
        output_contract=AgentOutputContract(
            variable_name=output.variable_name,
            kind=output.kind,
            accepted_names=accepted_names,
            alignment="same_rows_as_input" if output.kind in {"series", "dataframe"} else None,
            semantic=output.semantic,
        ),
        metadata={
            "source": "development_recipe_helper",
            "behavior_key": recipe.behavior_key,
            "pillar": recipe.pillar,
            "schema_variant": recipe.default_schema_variant,
        },
    )


# Backward-compatible aliases.
build_intent_from_recipe = build_dev_intent_from_recipe
build_contract_from_recipe = build_dev_contract_from_recipe


# ── Recipe helpers (not used by prod path) ──────────────────────────────────


def _find_time_column(schema: dict[str, str]) -> str | None:
    for key in ("time_col", "timestamp_col", "date_col"):
        if key in schema:
            return schema[key]
    for key, value in schema.items():
        lowered = f"{key} {value}".lower()
        if "time" in lowered or "date" in lowered:
            return value
    return None


def _hazards_for_recipe(recipe: Any) -> list[str]:
    hazards = {
        "hardcoded_absolute_path",
        "forbidden_optional_import",
    }
    pillar = recipe.pillar.lower()
    traps = " ".join(str(trap) for trap in getattr(recipe, "known_traps", []))
    text = f"{recipe.behavior_key} {recipe.mechanism_variant} {pillar} {traps}".lower()

    if "temporal" in text or "leakage" in text or _find_time_column(recipe.schema_variants[recipe.default_schema_variant]):
        hazards.update(
            {
                "negative_shift",
                "backward_fill",
                "future_index_access",
                "global_quantile_for_online_signal",
                "global_fit_for_online_signal",
            }
        )
    if "regime" in text:
        hazards.add("future_regime_label")
    if "risk" in text or "tail" in text:
        hazards.add("risk_after_optimization")

    return sorted(hazards)


def _make_obligation(id: str, description: str) -> CausalObligation:
    from agent.schemas import CausalObligation

    return CausalObligation(id=id, description=description)


def _make_invariant(id: str, description: str, check_type: str) -> RequiredInvariant:
    from agent.schemas import RequiredInvariant

    return RequiredInvariant(id=id, description=description, check_type=check_type)
