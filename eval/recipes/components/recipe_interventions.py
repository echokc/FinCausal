from typing import Any, Dict


def future_price_shock(
    *,
    price_col: str = "close",
    start_idx: int | None = None,
    start_fraction: float | None = None,
    multiplier: float | None = None,
    additive_noise_sigma: float | None = None,
) -> Dict[str, Any]:
    return {
        "type": "future_price_shock",
        "price_col": price_col,
        "start_idx": start_idx,
        "start_fraction": start_fraction,
        "multiplier": multiplier,
        "additive_noise_sigma": additive_noise_sigma,
    }


def leakage_last_point_sentinel(
    *,
    column: str = "close",
    value_strategy: str = "pre_mean",
) -> Dict[str, Any]:
    return {
        "type": "leakage_last_point_sentinel",
        "column": column,
        "index": -1,
        "value_strategy": value_strategy,
    }
