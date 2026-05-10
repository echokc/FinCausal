from typing import Any, Dict, Iterable, List


def causal_contract(
    *,
    allowed: List[str],
    forbidden: List[str],
    invariance: List[Dict[str, Any]],
    sensitivity: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "allowed_information": allowed,
        "forbidden_information": forbidden,
        "invariance_requirements": invariance,
        "sensitivity_requirements": sensitivity,
    }


def witness_map(
    *,
    interventions: List[Dict[str, Any]],
    must_match: List[Dict[str, Any]],
    inspect_windows: List[Dict[str, Any]],
    known_traps: List[Dict[str, Any]],
    **extra: Any,
) -> Dict[str, Any]:
    payload = {
        "interventions": interventions,
        "must_match": must_match,
        "inspect_windows": inspect_windows,
        "known_traps": known_traps,
    }
    payload.update(extra)
    return payload


def judge_config(
    *,
    required_output_semantics: Dict[str, Dict[str, Any]],
    probes: List[Dict[str, Any]],
    llm_judge_enabled: bool = True,
) -> Dict[str, Any]:
    return {
        "required_output_semantics": required_output_semantics,
        "probes": probes,
        "llm_judge": standard_llm_judge_config(enabled=llm_judge_enabled),
    }


def reference_behavior(
    *,
    causal_solution_strategy: str,
    expected_failure_strategy: str,
    positive_control_id: str,
    negative_control_id: str,
) -> Dict[str, Any]:
    return {
        "causal_solution_strategy": causal_solution_strategy,
        "expected_failure_strategy": expected_failure_strategy,
        "positive_control_id": positive_control_id,
        "negative_control_id": negative_control_id,
    }


def output_semantic(accepted_columns: Iterable[str], required: bool = True) -> Dict[str, Any]:
    return {
        "accepted_columns": list(dict.fromkeys(accepted_columns)),
        "required": required,
    }


def standard_llm_judge_config(enabled: bool = True) -> Dict[str, Any]:
    return {
        "enabled": enabled,
        "decision_labels": ["pass", "fail", "quarantine"],
        "must_audit_code": True,
        "must_check_probe_limitations": True,
    }
