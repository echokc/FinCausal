from __future__ import annotations

from dataclasses import dataclass, field

from agent.contract.facts import IntentFacts
from agent.contract.kb import ContractKnowledge


@dataclass(frozen=True)
class ContractPolicy:
    """Policy decisions expressed as IDs — no content, no CausalObligation objects.

    The assembler (contracts.py) maps these IDs to definitions via catalog.py.
    """

    decision_unit: str = "task"
    time_column: str | None = None
    decision_time: str | None = None
    output_used_at: str | None = None
    allowed_information_ids: list[str] = field(default_factory=list)
    forbidden_information_ids: list[str] = field(default_factory=list)
    invariant_ids: list[str] = field(default_factory=list)
    fit_scope: str = "task_dependent"
    forbid_full_sample_fit: bool = False
    known_hazard_ids: list[str] = field(default_factory=list)
    output_variable_name: str = "result"
    output_kind: str = "scalar"
    accepted_names: list[str] = field(default_factory=list)
    alignment: str | None = None
    semantic: str = ""


def build_contract_policy(facts: IntentFacts, knowledge: ContractKnowledge) -> ContractPolicy:
    """Convert facts and knowledge into policy decisions (IDs only).

    Every `if` in this function is a policy choice. The assembler downstream
    does not make any decisions — it only maps IDs to objects.
    """

    # ── Decision context (derived from temporal facts) ──────────────────────
    time_col = facts.time_column
    decision_unit = "row" if time_col else "task"
    decision_time = f"{time_col}[t]" if time_col else None
    output_used_at = f"{time_col}[t]" if time_col else None

    # ── Information obligations ─────────────────────────────────────────────
    allowed_ids: list[str] = ["task_available_information"]

    forbidden_ids: list[str] = ["unstated_external_information"]
    if time_col:
        forbidden_ids.extend(["future_rows", "full_sample_online_statistics"])

    # ── Invariants ──────────────────────────────────────────────────────────
    invariant_ids: list[str] = []
    if facts.output_kind in ("series", "dataframe"):
        invariant_ids.append("row_alignment")
        invariant_ids.append("index_alignment")
    else:
        invariant_ids.append("shape_alignment")
    if time_col:
        invariant_ids.append("prefix_invariance")

    # ── Estimation scope ────────────────────────────────────────────────────
    fit_scope = "rolling_or_prefix_only" if time_col else "task_dependent"
    forbid_full_sample = bool(time_col)

    # ── Known hazards ───────────────────────────────────────────────────────
    hazard_ids = set(knowledge.hazard_ids)
    if facts.is_temporal or time_col:
        hazard_ids.update(
            {
                "negative_shift",
                "backward_fill",
                "future_index_access",
                "global_quantile_for_online_signal",
                "global_fit_for_online_signal",
            }
        )
    if "regime" in facts.domain_hints:
        hazard_ids.add("future_regime_label")
    if "risk" in facts.domain_hints:
        hazard_ids.add("risk_after_optimization")

    # ── Output contract ────────────────────────────────────────────────────
    accepted_names: list[str] = []
    if facts.output_variable_name not in accepted_names:
        accepted_names.insert(0, facts.output_variable_name)

    alignment = None
    if facts.output_kind in ("series", "dataframe"):
        alignment = "same_rows_as_input"

    return ContractPolicy(
        decision_unit=decision_unit,
        time_column=time_col,
        decision_time=decision_time,
        output_used_at=output_used_at,
        allowed_information_ids=allowed_ids,
        forbidden_information_ids=forbidden_ids,
        invariant_ids=invariant_ids,
        fit_scope=fit_scope,
        forbid_full_sample_fit=forbid_full_sample,
        known_hazard_ids=sorted(hazard_ids),
        output_variable_name=facts.output_variable_name,
        output_kind=facts.output_kind,
        accepted_names=accepted_names,
        alignment=alignment,
        semantic=facts.semantic,
    )
