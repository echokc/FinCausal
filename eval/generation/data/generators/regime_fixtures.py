import os
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from eval.generation.data.fixture_builders import _fixture
from eval.generation.data.fixture_models import UniverseFixture


GeneratorFn = Callable[[int], List[UniverseFixture]]


def _s007(seed: int) -> List[UniverseFixture]:
    rng = np.random.default_rng(seed)
    n = 220
    ts = pd.date_range("2026-02-01", periods=n, freq="min")
    close = 100 + rng.normal(0, 0.05, n).cumsum()
    close[100] = 97.0
    close[101:110] = np.linspace(98.0, 105.0, 9)
    normal_volume = float(rng.uniform(60.0, 110.0))
    thin_volume = float(rng.uniform(0.8, 2.0))
    volume_normal = np.full(n, normal_volume)
    volume_thin = np.full(n, thin_volume)
    return [
        _fixture(
            "normal_liquidity",
            pd.DataFrame({"timestamp": ts, "close": close, "volume": volume_normal}),
            variant="realizable_liquidity",
            expected={"max_fill": min(10.0, normal_volume * 0.12)},
            volume=normal_volume,
        ),
        _fixture(
            "thin_liquidity",
            pd.DataFrame({"timestamp": ts, "close": close, "volume": volume_thin}),
            variant="thin_liquidity_illusion",
            expected={"max_fill": min(10.0, thin_volume * 0.12)},
            volume=thin_volume,
        ),
    ]


def _s008(seed: int) -> List[UniverseFixture]:
    rng = np.random.default_rng(seed)
    n = 260
    ts = pd.date_range("2026-03-01", periods=n, freq="min")
    trend_drop = float(rng.uniform(12.0, 20.0))
    shock_drop = float(rng.uniform(7.0, 12.0))
    trend = 100 - np.linspace(0, trend_drop, n) + rng.normal(0, 0.15, n)
    shock = 100 + rng.normal(0, 0.15, n)
    shock_start = int(rng.integers(220, 236))
    shock[shock_start:] -= shock_drop
    leakage = shock.copy()
    leakage[-1] = float(np.mean(leakage[:220]))
    return [
        _fixture("persistent_trend", pd.DataFrame({"timestamp": ts, "close": trend, "volume": 100.0}), variant="persistent_trend", expected={"position": "low"}, trend_drop=trend_drop),
        _fixture("transient_shock", pd.DataFrame({"timestamp": ts, "close": shock, "volume": 100.0}), variant="transient_vol_shock", expected={"position": "high"}, shock_start=shock_start, shock_drop=shock_drop),
        _fixture("leakage_sentinel", pd.DataFrame({"timestamp": ts, "close": leakage, "volume": 100.0}), variant="last_point_reversal_sentinel", expected={"prefix_invariant_to": "transient_shock"}, shock_start=shock_start, shock_drop=shock_drop),
    ]



def _correlated_ar_returns(
    rng: np.random.Generator,
    n: int,
    *,
    mean_a: float,
    mean_b: float,
    sigma_a: float,
    sigma_b: float,
    rho: float,
    phi_a: float,
    phi_b: float,
) -> tuple[np.ndarray, np.ndarray]:
    cov = [[sigma_a ** 2, rho * sigma_a * sigma_b], [rho * sigma_a * sigma_b, sigma_b ** 2]]
    innovations = rng.multivariate_normal([0.0, 0.0], cov, n)
    a = np.zeros(n)
    b = np.zeros(n)
    for idx in range(1, n):
        a[idx] = mean_a + phi_a * (a[idx - 1] - mean_a) + innovations[idx, 0]
        b[idx] = mean_b + phi_b * (b[idx - 1] - mean_b) + innovations[idx, 1]
    return a, b



def _regime_break_panel(seed: int, *, variant: str) -> List[UniverseFixture]:
    rng = np.random.default_rng(seed)
    n = 480
    ts = pd.date_range("2025-07-01", periods=n, freq="D")
    break_1 = 135
    drift_start = 235
    drift_end = 295
    break_3 = 365

    clean_a, clean_b = _correlated_ar_returns(
        rng,
        n,
        mean_a=0.0002,
        mean_b=0.0001,
        sigma_a=0.006,
        sigma_b=0.0055,
        rho=0.25,
        phi_a=0.10,
        phi_b=0.08,
    )
    shock_a = clean_a.copy()
    shock_b = clean_b.copy()

    # Regime 1: abrupt shift to high-volatility negative trend.
    seg_len = drift_start - break_1
    a1, b1 = _correlated_ar_returns(
        rng,
        seg_len,
        mean_a=-0.0024,
        mean_b=-0.0018,
        sigma_a=0.020,
        sigma_b=0.018,
        rho=0.15,
        phi_a=0.20,
        phi_b=0.16,
    )
    shock_a[break_1:drift_start] = a1
    shock_b[break_1:drift_start] = b1

    if variant == "vol_cluster":
        cluster_start = break_1 + 20
        cluster_end = cluster_start + 18
        shock_a[cluster_start:cluster_end] += rng.normal(0.0, 0.025, cluster_end - cluster_start)
        shock_b[cluster_start:cluster_end] += rng.normal(0.0, 0.023, cluster_end - cluster_start)

    # Regime 2: gradual drift in mean and volatility.
    drift_len = drift_end - drift_start
    drift_weight = np.linspace(0.0, 1.0, drift_len)
    drift_sigma_a = 0.020 - 0.011 * drift_weight
    drift_sigma_b = 0.018 - 0.010 * drift_weight
    drift_mean_a = -0.0024 + 0.0034 * drift_weight
    drift_mean_b = -0.0018 + 0.0027 * drift_weight
    if variant == "slow_drift":
        drift_sigma_a = 0.023 - 0.014 * drift_weight
        drift_sigma_b = 0.021 - 0.013 * drift_weight
        drift_mean_a = -0.0030 + 0.0048 * drift_weight
        drift_mean_b = -0.0025 + 0.0040 * drift_weight
    shock_a[drift_start:drift_end] = drift_mean_a + rng.normal(0.0, drift_sigma_a)
    shock_b[drift_start:drift_end] = drift_mean_b + rng.normal(0.0, drift_sigma_b)

    # Regime 3: durable recovery before the final persistence/correlation break.
    mid_len = break_3 - drift_end
    a2, b2 = _correlated_ar_returns(
        rng,
        mid_len,
        mean_a=0.0011,
        mean_b=0.0009,
        sigma_a=0.0085,
        sigma_b=0.0080,
        rho=0.30,
        phi_a=0.05,
        phi_b=0.05,
    )
    shock_a[drift_end:break_3] = a2
    shock_b[drift_end:break_3] = b2

    # Regime 4: persistence and correlation structure shift.
    final_len = n - break_3
    final_rho = 0.88 if variant == "correlation" else 0.70
    final_phi = 0.58 if variant in {"duration", "correlation"} else 0.45
    a3, b3 = _correlated_ar_returns(
        rng,
        final_len,
        mean_a=0.0004,
        mean_b=0.0003,
        sigma_a=0.012,
        sigma_b=0.011,
        rho=final_rho,
        phi_a=final_phi,
        phi_b=final_phi,
    )
    shock_a[break_3:] = a3
    shock_b[break_3:] = b3

    expected = {
        "break_1_idx": break_1,
        "break_1_window_start": break_1 - 18,
        "break_1_window_end": break_1 + 22,
        "break_2_idx": drift_start,
        "break_2_window_start": drift_start - 12,
        "break_2_window_end": drift_end + 14,
        "break_3_idx": break_3,
        "break_3_window_start": break_3 - 22,
        "break_3_window_end": break_3 + 24,
        "major_break_count": 3,
        "min_regime_count": 3,
    }

    def frame(ret_a, ret_b):
        price_a = 100.0 * np.exp(np.cumsum(ret_a))
        price_b = 90.0 * np.exp(np.cumsum(ret_b))
        return pd.DataFrame(
            {
                "timestamp": ts,
                "return_a": ret_a,
                "return_b": ret_b,
                "price_a": price_a,
                "price_b": price_b,
            }
        )

    return [
        _fixture(
            "clean",
            frame(clean_a, clean_b),
            variant="stable_single_regime",
            expected={"major_break_count": 0, "min_regime_count": 1},
        ),
        _fixture(
            "shock",
            frame(shock_a, shock_b),
            variant="multi_structural_break_sequence",
            expected=expected,
            regime_variant=variant,
        ),
    ]

def _s023(seed: int) -> List[UniverseFixture]:
    return _regime_break_panel(seed, variant="multi")


def _s024(seed: int) -> List[UniverseFixture]:
    return _regime_break_panel(seed, variant="slow_drift")


def _s025(seed: int) -> List[UniverseFixture]:
    return _regime_break_panel(seed, variant="correlation")


def _s026(seed: int) -> List[UniverseFixture]:
    return _regime_break_panel(seed, variant="vol_cluster")


def _s027(seed: int) -> List[UniverseFixture]:
    return _regime_break_panel(seed, variant="duration")


REGIME_GENERATORS: Dict[str, GeneratorFn] = {
    "s007_liquidity_illusion_realizable_pnl": _s007,
    "s008_vol_signature_misclassification": _s008,
    "s023_regime_multiple_structural_breaks_misclassification": _s023,
    "s024_regime_slow_drift_misclassification": _s024,
    "s025_regime_correlation_break_misclassification": _s025,
    "s026_regime_volatility_clustering_vs_regime_shift": _s026,
    "s027_regime_persistence_duration_misestimation": _s027,
}
