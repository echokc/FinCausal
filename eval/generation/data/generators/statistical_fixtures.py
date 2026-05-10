import argparse
import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from eval.generation.data.fixture_builders import _fixture
from eval.generation.data.fixture_models import UniverseFixture


GeneratorFn = Callable[[int], List[UniverseFixture]]


def _s003(seed: int) -> List[UniverseFixture]:
    rng = np.random.default_rng(seed)
    n_obs = int(rng.integers(28, 42))
    n_assets = 50
    returns = rng.normal(0.001, 0.02, (n_obs, n_assets))
    cols = [f"STK_{idx}" for idx in range(50)]
    cols = [f"STK_{idx}" for idx in range(n_assets)]
    base = pd.DataFrame(returns, columns=cols)
    noise_scale = float(rng.uniform(5e-7, 2e-6))
    micro_noise = base + rng.normal(0, noise_scale, base.shape)
    condition_number = float(np.linalg.cond(base.cov().to_numpy()))
    return [
        _fixture(
            "base_returns",
            base,
            variant="ill_conditioned_return_panel",
            expected={"max_leverage": 1.15},
            n_obs=n_obs,
            n_assets=n_assets,
            condition_number=condition_number,
        ),
        _fixture(
            "micro_noise",
            micro_noise,
            variant="micro_perturbed_return_panel",
            expected={"max_mean_abs_weight_delta": 0.05},
            noise_scale=noise_scale,
            n_obs=n_obs,
            n_assets=n_assets,
        ),
    ]






def _s009(seed: int) -> List[UniverseFixture]:
    rng = np.random.default_rng(seed)
    n = 300
    ts = pd.date_range("2026-04-01", periods=n, freq="min")
    trap_returns = rng.normal(0, 0.001, n)
    contemporaneous_strength = float(rng.uniform(180000, 320000))
    trap_volume = 100 + np.abs(trap_returns) * contemporaneous_strength
    leading_volume = 100 + rng.normal(0, 5, n)
    leading_returns = rng.normal(0, 0.001, n)
    spacing = int(rng.choice([18, 20, 22]))
    spike_idx = np.arange(40, 260, spacing)
    volume_spike = float(rng.uniform(420, 650))
    lead_return_bump = float(rng.uniform(0.008, 0.013))
    leading_volume[spike_idx] += volume_spike
    for idx in spike_idx:
        if idx + 1 < n:
            leading_returns[idx + 1] += lead_return_bump
    trap_close = 100 * np.exp(np.cumsum(trap_returns))
    leading_close = 100 * np.exp(np.cumsum(leading_returns))
    return [
        _fixture("contemporaneous_trap", pd.DataFrame({"timestamp": ts, "close": trap_close, "volume": trap_volume}), variant="contemporaneous_volume_trap", expected={"position": "low"}, contemporaneous_strength=contemporaneous_strength),
        _fixture("leading_signal", pd.DataFrame({"timestamp": ts, "close": leading_close, "volume": leading_volume}), variant="volume_leads_next_return", expected={"position": "high"}, spike_indices=spike_idx.tolist(), spacing=spacing, volume_spike=volume_spike, lead_return_bump=lead_return_bump),
    ]



def _s011(seed: int) -> List[UniverseFixture]:
    rng = np.random.default_rng(seed)
    n = 300
    ts = pd.date_range("2026-01-01", periods=n, freq="min")
    base = 50000 + rng.normal(0, 12, n).cumsum()
    poison_idx = int(rng.integers(60, 95))
    breakout_idx = int(rng.integers(135, 155))
    poison_price = float(rng.uniform(39200, 40500))
    strong_poison_price = float(rng.uniform(37000, 38400))
    breakout_price = float(rng.uniform(50600, 50900))
    poisoned = base.copy()
    poisoned[poison_idx] = poison_price
    poisoned[breakout_idx] = breakout_price
    clean = base.copy()
    clean[breakout_idx] = breakout_price
    strong = base.copy()
    strong[poison_idx] = strong_poison_price
    strong[breakout_idx] = breakout_price + 70
    no_signal = base.copy()
    no_signal[poison_idx] = poison_price
    multi = base.copy()
    multi_indices = [max(10, poison_idx - 20), poison_idx, min(n - 1, poison_idx + 20)]
    multi[multi_indices] = [poison_price - 300, poison_price + 1200, strong_poison_price]
    multi[breakout_idx] = breakout_price

    def frame(values):
        return pd.DataFrame({"timestamp": ts, "close": values})

    return [
        _fixture("poisoned_breakout", frame(poisoned), variant="poisoned_breakout", expected={"flag_at": {breakout_idx: True, poison_idx: False}}, poison_idx=poison_idx, breakout_idx=breakout_idx),
        _fixture("clean_breakout", frame(clean), variant="clean_breakout", expected={"flag_at": {breakout_idx: True}}, breakout_idx=breakout_idx),
        _fixture("strong_poison_breakout", frame(strong), variant="strong_poison_breakout", expected={"flag_at": {breakout_idx: True, poison_idx: False}}, poison_idx=poison_idx, breakout_idx=breakout_idx),
        _fixture("poison_without_signal", frame(no_signal), variant="poison_without_signal", expected={"total_flags": 0}, poison_idx=poison_idx),
        _fixture("multiple_outliers_breakout", frame(multi), variant="multiple_outliers_breakout", expected={"flag_at": {breakout_idx: True}}, poison_indices=multi_indices, breakout_idx=breakout_idx),
    ]






def _s018(seed: int) -> List[UniverseFixture]:
    return _nonstationary_panel(seed, variant="granger")


def _s019(seed: int) -> List[UniverseFixture]:
    return _nonstationary_panel(seed, variant="pairs")


def _s020(seed: int) -> List[UniverseFixture]:
    return _nonstationary_panel(seed, variant="factor")


def _s021(seed: int) -> List[UniverseFixture]:
    return _nonstationary_panel(seed, variant="residual")


def _s022(seed: int) -> List[UniverseFixture]:
    return _nonstationary_panel(seed, variant="lead_lag")



def _stationary_ar1(rng: np.random.Generator, n: int, *, phi: float, sigma: float) -> np.ndarray:
    values = np.zeros(n)
    innovations = rng.normal(0.0, sigma, n)
    for idx in range(1, n):
        values[idx] = phi * values[idx - 1] + innovations[idx]
    return values


def _nonstationary_panel(seed: int, *, variant: str) -> List[UniverseFixture]:
    rng = np.random.default_rng(seed)
    n = 420
    ts = pd.date_range("2026-06-01", periods=n, freq="D")
    shock_idx = int(rng.integers(210, 280))
    sigma = float(rng.uniform(0.7, 1.2))

    base_x = 100.0 + _stationary_ar1(rng, n, phi=0.55, sigma=sigma)
    base_z = 50.0 + _stationary_ar1(rng, n, phi=0.40, sigma=sigma * 0.8)
    base_y = 75.0 + 0.25 * (base_x - 100.0) + _stationary_ar1(rng, n, phi=0.45, sigma=sigma)
    base_a = 90.0 + _stationary_ar1(rng, n, phi=0.50, sigma=sigma)
    base_b = 110.0 + 0.35 * (base_a - 90.0) + _stationary_ar1(rng, n, phi=0.35, sigma=sigma)

    shock_x = base_x.copy()
    shock_z = base_z.copy()
    shock_y = base_y.copy()
    shock_a = base_a.copy()
    shock_b = base_b.copy()
    random_walk = np.cumsum(rng.normal(0.18, sigma * 0.55, n - shock_idx))
    secondary_walk = np.cumsum(rng.normal(0.12, sigma * 0.50, n - shock_idx))

    if variant == "pairs":
        shock_a[shock_idx:] = base_a[shock_idx] + random_walk
        shock_b[shock_idx:] = base_b[shock_idx] + secondary_walk + 0.65 * random_walk
        shock_y[shock_idx:] = base_y[shock_idx] + random_walk
        shock_x[shock_idx:] = base_x[shock_idx] + secondary_walk
    elif variant == "residual":
        shock_x[shock_idx:] = base_x[shock_idx] + random_walk
        shock_y[shock_idx:] = base_y[shock_idx] + 0.8 * random_walk + secondary_walk
        shock_a[shock_idx:] = base_a[shock_idx] + random_walk
        shock_b[shock_idx:] = base_b[shock_idx] + secondary_walk
    elif variant == "lead_lag":
        shock_x[shock_idx:] = base_x[shock_idx] + random_walk
        lagged_walk = np.r_[0.0, random_walk[:-1]]
        shock_y[shock_idx:] = base_y[shock_idx] + 0.75 * lagged_walk + secondary_walk
        shock_a[shock_idx:] = base_a[shock_idx] + random_walk
        shock_b[shock_idx:] = base_b[shock_idx] + lagged_walk
    elif variant == "factor":
        shock_x[shock_idx:] = base_x[shock_idx] + random_walk
        shock_z[shock_idx:] = base_z[shock_idx] + secondary_walk
        shock_y[shock_idx:] = base_y[shock_idx] + 0.6 * random_walk + 0.5 * secondary_walk
        shock_a[shock_idx:] = base_a[shock_idx] + random_walk
        shock_b[shock_idx:] = base_b[shock_idx] + secondary_walk
    else:
        shock_x[shock_idx:] = base_x[shock_idx] + random_walk
        shock_y[shock_idx:] = base_y[shock_idx] + 0.7 * random_walk + secondary_walk
        shock_a[shock_idx:] = base_a[shock_idx] + random_walk
        shock_b[shock_idx:] = base_b[shock_idx] + secondary_walk

    expected = {
        "shock_idx": shock_idx,
        "prefix_invariant_until": shock_idx - 1,
        "requires_differencing": True,
    }

    def frame(y, x, z, asset_a, asset_b):
        return pd.DataFrame(
            {
                "timestamp": ts,
                "y": y,
                "x": x,
                "z": z,
                "asset_a": asset_a,
                "asset_b": asset_b,
            }
        )

    return [
        _fixture(
            "clean",
            frame(base_y, base_x, base_z, base_a, base_b),
            variant="stationary_i0_panel",
            expected={**expected, "stationary": True},
            shock_idx=shock_idx,
            sigma=sigma,
        ),
        _fixture(
            "shock",
            frame(shock_y, shock_x, shock_z, shock_a, shock_b),
            variant="post_t_unit_root_injection",
            expected={**expected, "stationary": False},
            shock_idx=shock_idx,
            sigma=sigma,
            nonstationary_variant=variant,
        ),
    ]


STATISTICAL_GENERATORS: Dict[str, GeneratorFn] = {
    "s003_covariance_inversion_stability": _s003,
    "s009_volume_lead_lag_causality": _s009,
    "s011_outlier_robust_breakout_detection": _s011,
    "s018_statistical_spurious_granger_causality": _s018,
    "s019_statistical_ignored_cointegration_pairs_trading": _s019,
    "s020_statistical_spurious_factor_significance": _s020,
    "s021_statistical_non_stationary_residuals": _s021,
    "s022_statistical_false_lead_lag_relationships": _s022,
}
