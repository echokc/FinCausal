from typing import Any, Dict, Literal


def monotonic_response_probe(
    *,
    name: str,
    baseline: str,
    treatment: str,
    output_semantic: str,
    direction: Literal["increase", "decrease"],
    min_delta: float | None = None,
    ratio_range: tuple[float | None, float | None] | None = None,
    hard_fail: bool = True,
) -> Dict[str, Any]:
    return {
        "type": "monotonic_response",
        "name": name,
        "baseline_universe": baseline,
        "treatment_universe": treatment,
        "output_semantic": output_semantic,
        "direction": direction,
        "min_delta": min_delta,
        "ratio_range": ratio_range,
        "hard_fail": hard_fail,
    }


def leakage_sentinel_probe(
    *,
    name: str,
    clean_reference: str,
    leakage_universe: str,
    output_semantic: str,
    max_abs_delta: float,
    hard_fail: bool = True,
) -> Dict[str, Any]:
    return {
        "type": "leakage_sentinel",
        "name": name,
        "clean_reference_universe": clean_reference,
        "leakage_universe": leakage_universe,
        "output_semantic": output_semantic,
        "max_abs_delta": max_abs_delta,
        "hard_fail": hard_fail,
    }


def output_stability_probe(
    *,
    name: str,
    baseline: str,
    perturbed: str,
    output_semantic: str,
    max_mean_abs_delta: float,
    hard_fail: bool = True,
) -> Dict[str, Any]:
    return {
        "type": "output_stability",
        "name": name,
        "baseline_universe": baseline,
        "perturbed_universe": perturbed,
        "output_semantic": output_semantic,
        "max_mean_abs_delta": max_mean_abs_delta,
        "hard_fail": hard_fail,
    }


def leverage_bounds_probe(
    *,
    name: str,
    universe_name: str,
    output_semantic: str,
    max_leverage: float,
    fully_invested_tolerance: float,
    hard_fail: bool = True,
) -> Dict[str, Any]:
    return {
        "type": "leverage_bounds",
        "name": name,
        "universe": universe_name,
        "output_semantic": output_semantic,
        "max_leverage": max_leverage,
        "fully_invested_tolerance": fully_invested_tolerance,
        "hard_fail": hard_fail,
    }


def config_counterfactual_probe(
    *,
    name: str,
    baseline_config: str,
    counterfactual_config: str,
    output_semantic: str,
    direction: Literal["increase", "decrease", "change"],
    min_delta: float | None = None,
    hard_fail: bool = True,
) -> Dict[str, Any]:
    return {
        "type": "config_counterfactual",
        "name": name,
        "baseline_config": baseline_config,
        "counterfactual_config": counterfactual_config,
        "output_semantic": output_semantic,
        "direction": direction,
        "min_delta": min_delta,
        "hard_fail": hard_fail,
    }


def output_bounds_probe(
    *,
    name: str,
    universe_name: str,
    output_semantic: str,
    min_value: float | None = None,
    max_value: float | None = None,
    min_value_from_metadata: str | None = None,
    max_value_from_metadata: str | None = None,
    hard_fail: bool = True,
) -> Dict[str, Any]:
    return {
        "type": "output_bounds",
        "name": name,
        "universe": universe_name,
        "output_semantic": output_semantic,
        "min_value": min_value,
        "max_value": max_value,
        "min_value_from_metadata": min_value_from_metadata,
        "max_value_from_metadata": max_value_from_metadata,
        "hard_fail": hard_fail,
    }


def time_scaling_probe(
    *,
    name: str,
    universe_name: str,
    output_semantic: str,
    source_frequency: str,
    target_horizon: str,
    expected_scale: str,
    max_relative_error: float,
    hard_fail: bool = True,
) -> Dict[str, Any]:
    return {
        "type": "time_scaling",
        "name": name,
        "universe": universe_name,
        "output_semantic": output_semantic,
        "source_frequency": source_frequency,
        "target_horizon": target_horizon,
        "expected_scale": expected_scale,
        "max_relative_error": max_relative_error,
        "hard_fail": hard_fail,
    }


def derived_metric_monotonic_probe(
    *,
    name: str,
    baseline: str,
    treatment: str,
    metric: str,
    output_semantic: str,
    direction: Literal["increase", "decrease"],
    min_delta: float | None = None,
    ratio_range: tuple[float | None, float | None] | None = None,
    hard_fail: bool = True,
) -> Dict[str, Any]:
    return {
        "type": "derived_metric_monotonic",
        "name": name,
        "baseline_universe": baseline,
        "treatment_universe": treatment,
        "metric": metric,
        "output_semantic": output_semantic,
        "direction": direction,
        "min_delta": min_delta,
        "ratio_range": ratio_range,
        "hard_fail": hard_fail,
    }


def field_bounds_probe(
    *,
    name: str,
    universe_name: str,
    output_semantic: str,
    field: str,
    min_value: float | None = None,
    max_value: float | None = None,
    hard_fail: bool = True,
) -> Dict[str, Any]:
    return {
        "type": "field_bounds",
        "name": name,
        "universe": universe_name,
        "output_semantic": output_semantic,
        "field": field,
        "min_value": min_value,
        "max_value": max_value,
        "hard_fail": hard_fail,
    }


def field_monotonic_probe(
    *,
    name: str,
    baseline: str,
    treatment: str,
    output_semantic: str,
    field: str,
    direction: Literal["increase", "decrease"],
    min_delta: float | None = None,
    ratio_range: tuple[float | None, float | None] | None = None,
    hard_fail: bool = True,
) -> Dict[str, Any]:
    return {
        "type": "field_monotonic",
        "name": name,
        "baseline_universe": baseline,
        "treatment_universe": treatment,
        "output_semantic": output_semantic,
        "field": field,
        "direction": direction,
        "min_delta": min_delta,
        "ratio_range": ratio_range,
        "hard_fail": hard_fail,
    }


def predicate_matrix_probe(
    *,
    name: str,
    output_semantic: str,
    expectations: Dict[str, Dict[str, Any]],
    hard_fail: bool = True,
) -> Dict[str, Any]:
    return {
        "type": "predicate_matrix",
        "name": name,
        "output_semantic": output_semantic,
        "expectations": expectations,
        "hard_fail": hard_fail,
    }


def dataframe_window_threshold_probe(
    *,
    name: str,
    universe_name: str,
    output_semantic: str,
    field: str,
    start_idx: int | None = None,
    end_idx: int | None = None,
    start_idx_from_metadata: str | None = None,
    end_idx_from_metadata: str | None = None,
    min_abs_max: float | None = None,
    max_abs_max: float | None = None,
    hard_fail: bool = True,
) -> Dict[str, Any]:
    return {
        "type": "dataframe_window_threshold",
        "name": name,
        "universe": universe_name,
        "output_semantic": output_semantic,
        "field": field,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "start_idx_from_metadata": start_idx_from_metadata,
        "end_idx_from_metadata": end_idx_from_metadata,
        "min_abs_max": min_abs_max,
        "max_abs_max": max_abs_max,
        "hard_fail": hard_fail,
    }
