import os
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from eval.generation.data.fixture_builders import fixture
from eval.generation.data.fixture_models import UniverseFixture


GeneratorFn = Callable[[int], List[UniverseFixture]]

def _s010(seed: int) -> List[UniverseFixture]:
    rng = np.random.default_rng(seed)
    n = 1008
    base = rng.normal(0.002, 0.0002, n)
    fat = base.copy()
    fat_idx = int(rng.integers(850, 990))
    fat_loss = float(-rng.uniform(0.40, 0.60))
    fat[fat_idx] = fat_loss
    mild = base.copy()
    mild_idx = int(rng.integers(550, 780))
    mild_loss = float(-rng.uniform(0.22, 0.30))
    mild[mild_idx] = mild_loss
    sharpe_hack = base.copy()
    sharpe_idx = int(rng.integers(420, 620))
    sharpe_hack[sharpe_idx] = fat_loss
    for idx in range(1, n):
        sharpe_hack[idx] += 0.25 * sharpe_hack[idx - 1]
    vol_spike = base.copy()
    vol_start = int(rng.integers(250, 360))
    vol_width = int(rng.integers(14, 28))
    vol_multiplier = float(rng.uniform(5.0, 9.0))
    vol_spike[vol_start:vol_start + vol_width] *= vol_multiplier

    def frame(values):
        return pd.DataFrame({"day": range(1, n + 1), "daily_return": values})

    return [
        fixture("fat_tail", frame(fat), variant="catastrophic_left_tail", expected={"endorsement": 0, "tail_loss": fat_loss}, shock_index=fat_idx, shock_return=fat_loss),
        fixture("clean", frame(base), variant="clean_high_sharpe", expected={"endorsement": 1}),
        fixture("mild_tail", frame(mild), variant="fiduciary_unacceptable_mild_tail", expected={"endorsement": 0, "tail_loss": mild_loss}, shock_index=mild_idx, shock_return=mild_loss),
        fixture("sharpe_hack", frame(sharpe_hack), variant="autocorrelated_sharpe_hack", expected={"endorsement": 0, "tail_loss": fat_loss}, shock_index=sharpe_idx, shock_return=fat_loss),
        fixture("vol_spike", frame(vol_spike), variant="temporary_vol_spike_no_tail", expected={"endorsement": 1}, vol_start=vol_start, vol_width=vol_width, vol_multiplier=vol_multiplier),
    ]

def _s012(seed: int) -> List[UniverseFixture]:
    rng = np.random.default_rng(seed)
    n = 720
    ts = pd.date_range("2026-04-15", periods=n, freq="min")

    base_sigma = float(rng.uniform(0.0006, 0.0011))
    normal_returns = rng.normal(0.00002, base_sigma, n)
    fat_tail_returns = rng.normal(0.00002, base_sigma, n)
    shock_count = int(rng.integers(2, 5))
    shock_indices = sorted(int(idx) for idx in rng.choice(np.arange(180, n - 30), size=shock_count, replace=False))
    shock_returns = -rng.uniform(0.045, 0.095, shock_count)
    if shock_count > 2:
        shock_returns[1] = float(rng.uniform(0.030, 0.055))
    fat_tail_returns[shock_indices] = shock_returns
    max_left_tail = float(abs(min(shock_returns)))
    expected_min_hedge = float(min(0.9, max(0.50, 0.12 + 8.0 * (max_left_tail - 0.01))))

    leakage_returns = fat_tail_returns.copy()
    sentinel_return = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.12, 0.18))
    leakage_returns[-1] = sentinel_return

    def frame(returns):
        close = 65000.0 * np.exp(np.cumsum(returns))
        return pd.DataFrame({"timestamp": ts, "close": close})

    return [
        fixture(
            "normal",
            frame(normal_returns),
            variant="normal_low_kurtosis",
            expected={"hedge_notional": {"min": 0.0, "max": 0.35}},
            base_sigma=base_sigma,
        ),
        fixture(
            "fat_tail_shock",
            frame(fat_tail_returns),
            variant="dynamic_fat_tail_shock",
            expected={"hedge_notional": {"min": expected_min_hedge, "max": 1.0}},
            base_sigma=base_sigma,
            shock_indices=shock_indices,
            shock_returns=shock_returns.tolist(),
            max_left_tail=max_left_tail,
        ),
        fixture(
            "leakage_sentinel",
            frame(leakage_returns),
            variant="future_point_leakage_sentinel",
            expected={"hedge_notional": {"min": expected_min_hedge, "max": 1.0}, "max_abs_delta_from_fat_tail_shock": 0.08},
            base_sigma=base_sigma,
            shock_indices=shock_indices,
            shock_returns=shock_returns.tolist(),
            sentinel_return=sentinel_return,
            max_left_tail=max_left_tail,
        ),
    ]


def _s028(seed: int) -> List[UniverseFixture]:
    rng = np.random.default_rng(seed)
    n = 720
    ts = pd.date_range("2026-04-20", periods=n, freq="min")
    shock_idx = int(rng.integers(590, 635))
    cluster_end = n

    base_sigma = float(rng.uniform(0.00055, 0.00095))
    cluster_sigma = float(rng.uniform(0.020, 0.027))
    stable_returns = rng.normal(0.00001, base_sigma, n)
    shock_returns = stable_returns.copy()
    shock_returns[shock_idx] = -float(rng.uniform(0.045, 0.065))

    clustered = np.zeros(cluster_end - shock_idx - 1)
    innovation_scale = cluster_sigma * float(rng.uniform(0.45, 0.65))
    for offset in range(len(clustered)):
        prev_abs = abs(clustered[offset - 1]) if offset > 0 else abs(shock_returns[shock_idx])
        local_sigma = min(cluster_sigma * 1.35, base_sigma + 0.55 * prev_abs + innovation_scale)
        clustered[offset] = rng.normal(-base_sigma * 0.25, local_sigma)
    if len(clustered) >= 8:
        extreme_offsets = rng.choice(np.arange(4, len(clustered)), size=3, replace=False)
        clustered[extreme_offsets] -= rng.uniform(0.020, 0.038, size=3)
    shock_returns[shock_idx + 1 : cluster_end] = clustered

    shock_returns[:shock_idx] = stable_returns[:shock_idx]

    def frame(returns):
        close = 65000.0 * np.exp(np.cumsum(returns))
        return pd.DataFrame({"timestamp": ts, "close": close})

    expected = {
        "tail_risk_estimate": {"min": 0.035, "max": 1.0},
        "shock_idx": shock_idx,
        "cluster_end": cluster_end,
        "cluster_sigma": cluster_sigma,
        "base_sigma": base_sigma,
    }

    return [
        fixture(
            "stable_low_vol",
            frame(stable_returns),
            variant="stable_low_volatility",
            expected={"tail_risk_estimate": {"min": 0.0, "max": 0.025}},
            base_sigma=base_sigma,
        ),
        fixture(
            "vol_cluster_shock",
            frame(shock_returns),
            variant="post_t_volatility_cluster",
            expected=expected,
            base_sigma=base_sigma,
            shock_idx=shock_idx,
            cluster_end=cluster_end,
            cluster_sigma=cluster_sigma,
        ),
    ]


def _s029(seed: int) -> List[UniverseFixture]:
    rng = np.random.default_rng(seed)
    n = 720
    n_assets = 5
    dates = pd.date_range("2026-04-20", periods=n, freq="D")
    cols = [f"asset_{idx + 1}" for idx in range(n_assets)]
    base_sigma = float(rng.uniform(0.006, 0.009))

    independent = rng.normal(0.0004, base_sigma, (n, n_assets))
    crisis = independent.copy()
    shock_start = int(rng.integers(560, 610))
    shock_len = n - shock_start
    shared_tail = rng.standard_t(df=3, size=shock_len) * float(rng.uniform(0.014, 0.020)) - 0.002
    idio = rng.normal(0.0, base_sigma * 0.45, (shock_len, n_assets))
    crisis[shock_start:] = shared_tail[:, None] + idio
    crash_offsets = rng.choice(np.arange(5, shock_len), size=4, replace=False)
    crisis[shock_start + crash_offsets] -= rng.uniform(0.035, 0.060, size=(4, 1))

    def frame(values):
        df = pd.DataFrame(values, columns=cols)
        df.insert(0, "date", dates)
        return df

    return [
        fixture(
            "diversified_normal",
            frame(independent),
            variant="diversified_independent_tails",
            expected={"portfolio_tail_risk": {"min": 0.0, "max": 0.035}},
            base_sigma=base_sigma,
        ),
        fixture(
            "tail_dependent_crisis",
            frame(crisis),
            variant="post_t_tail_dependent_crisis",
            expected={"portfolio_tail_risk": {"min": 0.060, "max": 1.0}, "shock_start": shock_start},
            base_sigma=base_sigma,
            shock_start=shock_start,
            crash_offsets=crash_offsets.tolist(),
        ),
    ]


def _s030(seed: int) -> List[UniverseFixture]:
    rng = np.random.default_rng(seed)
    n = 520
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    stable = rng.normal(0.0012, 0.010, n)
    fat_tail = rng.normal(0.0012, 0.010, n)
    shock_indices = sorted(int(idx) for idx in rng.choice(np.arange(80, n), size=6, replace=False))
    fat_tail[shock_indices] = -rng.uniform(0.080, 0.150, len(shock_indices))
    rebound_indices = rng.choice(np.setdiff1d(np.arange(80, n), shock_indices), size=3, replace=False)
    fat_tail[rebound_indices] = rng.uniform(0.030, 0.055, len(rebound_indices))

    def frame(values):
        return pd.DataFrame({"day": dates, "daily_return": values})

    return [
        fixture(
            "stable_edge",
            frame(stable),
            variant="stable_edge_low_tail",
            expected={"leverage_multiplier": {"min": 0.5, "max": 4.0}},
        ),
        fixture(
            "fat_tail_uncertain",
            frame(fat_tail),
            variant="fat_tail_uncertain_edge",
            expected={"leverage_multiplier": {"min": 0.0, "max": 0.6}, "shock_indices": shock_indices},
            shock_indices=shock_indices,
        ),
    ]


def _s031(seed: int) -> List[UniverseFixture]:
    rng = np.random.default_rng(seed)
    n = 420
    dates = pd.date_range("2025-06-01", periods=n, freq="D")
    quick = rng.normal(0.0010, 0.0015, n)
    prolonged = rng.normal(0.0008, 0.004, n)

    drawdown_start = int(rng.integers(130, 170))
    quick[drawdown_start : drawdown_start + 10] = -0.0065
    quick[drawdown_start + 10 : drawdown_start + 28] = 0.0060
    quick[drawdown_start + 28 :] = rng.normal(0.0012, 0.0012, n - drawdown_start - 28)

    prolonged[drawdown_start : drawdown_start + 10] = -0.0065
    prolonged[drawdown_start + 10 :] = rng.normal(0.00005, 0.002, n - drawdown_start - 10)
    prolonged[drawdown_start + 90 :] -= 0.00015

    def frame(values):
        return pd.DataFrame({"day": dates, "daily_return": values})

    return [
        fixture(
            "quick_recovery",
            frame(quick),
            variant="short_recovered_drawdown",
            expected={"drawdown_risk_score": {"min": 0.0, "max": 0.35}, "drawdown_start": drawdown_start},
        ),
        fixture(
            "prolonged_drawdown",
            frame(prolonged),
            variant="prolonged_unrecovered_drawdown",
            expected={"drawdown_risk_score": {"min": 0.60, "max": 1.0}, "drawdown_start": drawdown_start},
        ),
    ]


RISK_GENERATORS: Dict[str, GeneratorFn] = {
    "s010_fat_tail_fiduciary_discrimination": _s010,
    "s012_fat_tail_hedge_notional": _s012,
    "s028_tail_risk_volatility_clustering": _s028,
    "s029_tail_dependence_multi_asset_tail_risk": _s029,
    "s030_kelly_fat_tail_parameter_uncertainty_ruin": _s030,
    "s031_drawdown_duration_recovery_underestimation": _s031,
}
