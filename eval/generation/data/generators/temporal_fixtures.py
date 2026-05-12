import os
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from eval.generation.data.fixture_builders import fixture
from eval.generation.data.fixture_models import UniverseFixture

GeneratorFn = Callable[[int], List[UniverseFixture]]

def _rolling_stat_base(seed: int, *, n: int = 360) -> tuple[np.random.Generator, pd.DatetimeIndex, int, int, float, float, Dict[str, int]]:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-05-01", periods=n, freq="min")
    shock_idx = int(rng.integers(180, 240))
    shock_width = int(rng.integers(20, 61))
    base_sigma = float(rng.uniform(0.0008, 0.0014))
    expected = {
        "shock_idx": shock_idx,
        "pre_shock_window_start": max(0, shock_idx - 30),
        "pre_shock_window_end": shock_idx - 1,
        "post_shock_window_start": shock_idx + 5,
        "post_shock_window_end": min(n - 1, shock_idx + shock_width),
        "window": 30,
        "min_periods": 20,
    }
    return rng, ts, shock_idx, shock_width, base_sigma, float(n), expected


def _s001(seed: int) -> List[UniverseFixture]:
    rng = np.random.default_rng(seed)
    n = 1000
    shock_start = int(rng.integers(350, 650))
    shock_multiplier = float(rng.uniform(1.5, 2.8))
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    volatility = rng.gamma(2.0, 0.01, n)
    base = pd.DataFrame(
        {
            "date": dates,
            "volatility": volatility,
            "momentum_signal": rng.choice([-1, 1], n),
            "fwd_return": rng.normal(0, 0.01, n),
        }
    )
    base["regime_feature"] = base["volatility"]
    future = base.copy()
    future.loc[shock_start:, "regime_feature"] += float(base["regime_feature"].std() * shock_multiplier)
    return [
        fixture(
            "base_market",
            base,
            variant="quiet_market",
            expected={"prefix_invariant_until": shock_start - 1},
            n=n,
        ),
        fixture(
            "future_shock",
            future,
            variant="future_volatility_shock",
            expected={"prefix_invariant_until": shock_start - 1},
            shock_start=shock_start,
            shock_multiplier=shock_multiplier,
        ),
    ]


def _s002(seed: int) -> List[UniverseFixture]:
    rng = np.random.default_rng(seed)
    late_minutes = int(rng.integers(6, 16))
    pre_minutes = int(rng.integers(7, 18))
    trades = pd.DataFrame(
        [
            {"trade_id": "T101", "timestamp": "2026-03-07 10:00:05", "symbol": "BTC", "price": 62000.0, "side": "BUY"},
            {"trade_id": "T102", "timestamp": "2026-03-07 14:00:30", "symbol": "BTC", "price": 62150.0, "side": "SELL"},
        ]
    )
    news = pd.DataFrame(
        [
            {"news_id": "N_SH_01", "publish_time": "2026-03-07 08:00:00", "content": "FED Decision at 14:00"},
            {"news_id": "N_LG_01", "publish_time": f"2026-03-07 10:{late_minutes:02d}:00", "content": "Flash Crash Ex-post at 10:00"},
            {"news_id": "N_NOISE", "publish_time": f"2026-03-07 13:{60 - pre_minutes:02d}:00", "content": "Cafeteria lunch menu updated."},
        ]
    )
    contaminated_news = news.copy()
    contaminated_news.loc[contaminated_news["news_id"].eq("N_LG_01"), "content"] = "POISON_DATA_SHOCK reported later"
    return [
        fixture(
            "clean_news",
            {"clean_0.csv": trades, "clean_1.csv": news},
            variant="prior_news_only",
            expected={"T101_news_id": "N_SH_01", "T102_news_id": "N_NOISE"},
            late_news_minutes_after_trade=late_minutes,
        ),
        fixture(
            "contaminated_news",
            {"clean_0.csv": trades, "clean_1.csv": contaminated_news},
            variant="expost_poison_news",
            expected={"poison_news_id": "N_LG_01", "poison_must_not_match_trade": "T101"},
            late_news_minutes_after_trade=late_minutes,
        ),
    ]



def _s005(seed: int) -> List[UniverseFixture]:
    rng = np.random.default_rng(seed)
    n = 30 * 24 * 60
    ts = pd.date_range("2026-01-01", periods=n, freq="min")
    base_sigma = float(rng.uniform(0.0008, 0.0016))
    vol_multiplier = 2.0
    returns = rng.normal(0.0, base_sigma, n)
    base_price = 50000 * np.exp(np.cumsum(returns))
    scaled_price = 50000 * np.exp(np.cumsum(returns * vol_multiplier))
    return [
        fixture(
            "base_volatility",
            pd.DataFrame({"timestamp": ts, "price": base_price}),
            variant="base_minute_volatility",
            expected={"source_frequency": "1min"},
            base_sigma=base_sigma,
        ),
        fixture(
            "volatility_scaled_2x",
            pd.DataFrame({"timestamp": ts, "price": scaled_price}),
            variant="scaled_minute_volatility",
            expected={"volatility_multiplier": vol_multiplier},
            base_sigma=base_sigma,
            volatility_multiplier=vol_multiplier,
        ),
    ]




def _s013(seed: int) -> List[UniverseFixture]:
    rng = np.random.default_rng(seed)
    n = 360
    ts = pd.date_range("2026-05-01", periods=n, freq="min")
    shock_idx = int(rng.integers(180, 240))
    shock_width = int(rng.integers(20, 61))
    vol_multiplier = float(rng.uniform(4.0, 8.0))
    jump_return = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.015, 0.04))
    base_sigma = float(rng.uniform(0.0008, 0.0014))

    returns = rng.normal(0.0, base_sigma, n)
    shock_returns = returns.copy()
    shock_returns[shock_idx] += jump_return
    shock_end = min(n, shock_idx + shock_width)
    shock_returns[shock_idx + 1 : shock_end] = rng.normal(
        0.0,
        base_sigma * vol_multiplier,
        max(0, shock_end - shock_idx - 1),
    )
    if shock_end < n:
        decay = np.linspace(vol_multiplier, 1.0, n - shock_end)
        shock_returns[shock_end:] = rng.normal(0.0, base_sigma * decay)

    clean_close = 50000.0 * np.exp(np.cumsum(returns))
    shock_close = 50000.0 * np.exp(np.cumsum(shock_returns))
    shock_close[:shock_idx] = clean_close[:shock_idx]

    expected = {
        "shock_idx": shock_idx,
        "pre_shock_window_start": max(0, shock_idx - 30),
        "pre_shock_window_end": shock_idx - 1,
        "post_shock_window_start": shock_idx,
        "post_shock_window_end": min(n - 1, shock_idx + 10),
        "window": 30,
        "min_periods": 20,
    }

    def frame(close):
        return pd.DataFrame({"timestamp": ts, "close": close})

    return [
        fixture(
            "clean",
            frame(clean_close),
            variant="clean_gaussian_path",
            expected=expected,
            base_sigma=base_sigma,
        ),
        fixture(
            "shock",
            frame(shock_close),
            variant="post_t_volatility_shock",
            expected=expected,
            base_sigma=base_sigma,
            shock_idx=shock_idx,
            shock_width=shock_width,
            vol_multiplier=vol_multiplier,
            jump_return=jump_return,
        ),
    ]


def _s014(seed: int) -> List[UniverseFixture]:
    rng, ts, shock_idx, shock_width, base_sigma, _, expected = _rolling_stat_base(seed)
    n = len(ts)
    market = rng.normal(0.0, base_sigma, n)
    beta_clean = 1.0
    beta_shock = float(rng.uniform(2.0, 3.0))
    noise = rng.normal(0.0, base_sigma * 0.35, n)
    asset_a = beta_clean * market + noise
    asset_b_clean = market + rng.normal(0.0, base_sigma * 0.35, n)
    asset_b_shock = asset_b_clean.copy()
    shock_end = min(n, shock_idx + shock_width)
    asset_b_shock[shock_idx:shock_end] = beta_shock * market[shock_idx:shock_end] + rng.normal(0.0, base_sigma * 0.25, shock_end - shock_idx)

    def frame(asset_b):
        return pd.DataFrame(
            {
                "timestamp": ts,
                "asset_a": 100.0 * np.exp(np.cumsum(asset_a)),
                "asset_b": 100.0 * np.exp(np.cumsum(asset_b)),
                "market": 100.0 * np.exp(np.cumsum(market)),
            }
        )

    metadata = {**expected, "post_shock_window_start": shock_idx + 20}
    return [
        fixture("clean", frame(asset_b_clean), variant="stable_correlation_beta", expected=metadata, base_sigma=base_sigma, beta_clean=beta_clean),
        fixture("shock", frame(asset_b_shock), variant="post_t_beta_break", expected=metadata, base_sigma=base_sigma, beta_clean=beta_clean, beta_shock=beta_shock, shock_idx=shock_idx, shock_width=shock_width),
    ]


def _s015(seed: int) -> List[UniverseFixture]:
    rng, ts, shock_idx, shock_width, base_sigma, _, expected = _rolling_stat_base(seed)
    n = len(ts)
    clean_returns = rng.normal(0.0, base_sigma, n)
    shock_returns = clean_returns.copy()
    shock_end = min(n, shock_idx + shock_width)
    shock_returns[shock_idx] -= float(rng.uniform(0.025, 0.055))
    shock_returns[shock_idx + 1:shock_end] = rng.normal(-base_sigma * 0.2, base_sigma * float(rng.uniform(4.0, 7.0)), max(0, shock_end - shock_idx - 1))
    metadata = {**expected, "post_shock_window_start": shock_idx + 15}

    def frame(returns):
        return pd.DataFrame({"timestamp": ts, "close": 100.0 * np.exp(np.cumsum(returns))})

    return [
        fixture("clean", frame(clean_returns), variant="stable_var_cvar", expected=metadata, base_sigma=base_sigma),
        fixture("shock", frame(shock_returns), variant="post_t_tail_risk_spike", expected=metadata, base_sigma=base_sigma, shock_idx=shock_idx, shock_width=shock_width),
    ]


def _s016(seed: int) -> List[UniverseFixture]:
    rng, ts, shock_idx, shock_width, base_sigma, _, expected = _rolling_stat_base(seed)
    n = len(ts)
    clean_returns = rng.normal(0.00005, base_sigma, n)
    shock_returns = clean_returns.copy()
    shock_end = min(n, shock_idx + shock_width)
    shock_returns[shock_idx:shock_end] += -abs(float(rng.uniform(0.0015, 0.0035)))
    shock_returns[shock_idx] -= float(rng.uniform(0.018, 0.035))
    metadata = {**expected, "post_shock_window_start": shock_idx + 10}

    def frame(returns):
        return pd.DataFrame({"timestamp": ts, "close": 100.0 * np.exp(np.cumsum(returns))})

    return [
        fixture("clean", frame(clean_returns), variant="stable_drawdown_path", expected=metadata, base_sigma=base_sigma),
        fixture("shock", frame(shock_returns), variant="post_t_drawdown_break", expected=metadata, base_sigma=base_sigma, shock_idx=shock_idx, shock_width=shock_width),
    ]


def _s017(seed: int) -> List[UniverseFixture]:
    rng, ts, shock_idx, shock_width, base_sigma, _, expected = _rolling_stat_base(seed)
    n = len(ts)
    clean_returns = rng.normal(0.0, base_sigma, n)
    shock_returns = clean_returns.copy()
    shock_end = min(n, shock_idx + shock_width)
    tail = rng.standard_t(df=3, size=max(0, shock_end - shock_idx)) * base_sigma * float(rng.uniform(3.0, 5.0))
    tail[0] -= float(rng.uniform(0.018, 0.035))
    shock_returns[shock_idx:shock_end] = tail
    metadata = {**expected, "post_shock_window_start": shock_idx + 20}

    def frame(returns):
        return pd.DataFrame({"timestamp": ts, "close": 100.0 * np.exp(np.cumsum(returns))})

    return [
        fixture("clean", frame(clean_returns), variant="near_gaussian_moments", expected=metadata, base_sigma=base_sigma),
        fixture("shock", frame(shock_returns), variant="post_t_skew_kurtosis_tail", expected=metadata, base_sigma=base_sigma, shock_idx=shock_idx, shock_width=shock_width),
    ]


TEMPORAL_GENERATORS: Dict[str, GeneratorFn] = {
    "s001_global_quantile_leakage": _s001,
    "s002_expost_news_contamination": _s002,
    "s005_microstructure_volatility_scaling": _s005,
    "s013_temporal_rolling_zscore_causal_leakage": _s013,
    "s014_temporal_rolling_corr_beta_causal_leakage": _s014,
    "s015_temporal_rolling_var_cvar_causal_leakage": _s015,
    "s016_temporal_rolling_drawdown_causal_leakage": _s016,
    "s017_temporal_rolling_skew_kurtosis_causal_leakage": _s017,
}
