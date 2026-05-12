from typing import Dict, List

from eval.generation.data.fixture_models import UniverseFixture
from eval.generation.data.fixture_generation import generate_recipe_fixtures
from eval.recipes.scenario_recipes import (
    S001_GLOBAL_QUANTILE_LEAKAGE_RECIPE,
    S002_EXPOST_NEWS_CONTAMINATION_RECIPE,
    S003_COVARIANCE_INVERSION_RECIPE,
    S005_MICROSTRUCTURE_VOL_SCALING_RECIPE,
    S006_INVENTORY_SKEW_RECIPE,
    S007_LIQUIDITY_ILLUSION_RECIPE,
    S008_VOL_SIGNATURE_MISCLASSIFICATION_RECIPE,
    S009_VOLUME_LEAD_LAG_TRAP_RECIPE,
    S010_FAT_TAIL_FIDUCIARY_RECIPE,
    S011_OUTLIER_ROBUST_BREAKOUT_RECIPE,
    S012_FAT_TAIL_HEDGE_RECIPE,
    S013_TEMPORAL_ROLLING_ZSCORE_LEAKAGE_RECIPE,
    S014_TEMPORAL_ROLLING_CORR_BETA_LEAKAGE_RECIPE,
    S015_TEMPORAL_ROLLING_VAR_CVAR_LEAKAGE_RECIPE,
    S016_TEMPORAL_ROLLING_DRAWDOWN_LEAKAGE_RECIPE,
    S017_TEMPORAL_ROLLING_SKEW_KURTOSIS_LEAKAGE_RECIPE,
    S018_STATISTICAL_SPURIOUS_GRANGER_CAUSALITY_RECIPE,
    S019_STATISTICAL_IGNORED_COINTEGRATION_PAIRS_TRADING_RECIPE,
    S020_STATISTICAL_SPURIOUS_FACTOR_SIGNIFICANCE_RECIPE,
    S021_STATISTICAL_NON_STATIONARY_RESIDUALS_RECIPE,
    S022_STATISTICAL_FALSE_LEAD_LAG_RELATIONSHIPS_RECIPE,
    S023_REGIME_MULTIPLE_STRUCTURAL_BREAKS_MISCLASSIFICATION_RECIPE,
    S024_REGIME_SLOW_DRIFT_MISCLASSIFICATION_RECIPE,
    S025_REGIME_CORRELATION_BREAK_MISCLASSIFICATION_RECIPE,
    S026_REGIME_VOL_CLUSTERING_VS_SHIFT_RECIPE,
    S027_REGIME_PERSISTENCE_DURATION_MISESTIMATION_RECIPE,
    S028_VOL_CLUSTERED_TAIL_RISK_RECIPE,
    S029_TAIL_DEPENDENCE_MULTI_ASSET_RECIPE,
    S030_KELLY_OVERLEVERAGE_RUIN_RECIPE,
    S031_DRAWDOWN_DURATION_RISK_RECIPE,
)


def s001_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S001_GLOBAL_QUANTILE_LEAKAGE_RECIPE.behavior_key, seed=seed)


def s002_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S002_EXPOST_NEWS_CONTAMINATION_RECIPE.behavior_key, seed=seed)


def s005_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S005_MICROSTRUCTURE_VOL_SCALING_RECIPE.behavior_key, seed=seed)


def s003_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S003_COVARIANCE_INVERSION_RECIPE.behavior_key, seed=seed)


def s010_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S010_FAT_TAIL_FIDUCIARY_RECIPE.behavior_key, seed=seed)


def s006_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S006_INVENTORY_SKEW_RECIPE.behavior_key, seed=seed)


def s007_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S007_LIQUIDITY_ILLUSION_RECIPE.behavior_key, seed=seed)


def s008_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S008_VOL_SIGNATURE_MISCLASSIFICATION_RECIPE.behavior_key, seed=seed)


def s009_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S009_VOLUME_LEAD_LAG_TRAP_RECIPE.behavior_key, seed=seed)


def s011_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S011_OUTLIER_ROBUST_BREAKOUT_RECIPE.behavior_key, seed=seed)


def s012_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S012_FAT_TAIL_HEDGE_RECIPE.behavior_key, seed=seed)


def s013_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S013_TEMPORAL_ROLLING_ZSCORE_LEAKAGE_RECIPE.behavior_key, seed=seed)


def s014_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S014_TEMPORAL_ROLLING_CORR_BETA_LEAKAGE_RECIPE.behavior_key, seed=seed)


def s015_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S015_TEMPORAL_ROLLING_VAR_CVAR_LEAKAGE_RECIPE.behavior_key, seed=seed)


def s016_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S016_TEMPORAL_ROLLING_DRAWDOWN_LEAKAGE_RECIPE.behavior_key, seed=seed)


def s017_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S017_TEMPORAL_ROLLING_SKEW_KURTOSIS_LEAKAGE_RECIPE.behavior_key, seed=seed)


def s018_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S018_STATISTICAL_SPURIOUS_GRANGER_CAUSALITY_RECIPE.behavior_key, seed=seed)


def s019_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S019_STATISTICAL_IGNORED_COINTEGRATION_PAIRS_TRADING_RECIPE.behavior_key, seed=seed)


def s020_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S020_STATISTICAL_SPURIOUS_FACTOR_SIGNIFICANCE_RECIPE.behavior_key, seed=seed)


def s021_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S021_STATISTICAL_NON_STATIONARY_RESIDUALS_RECIPE.behavior_key, seed=seed)


def s022_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S022_STATISTICAL_FALSE_LEAD_LAG_RELATIONSHIPS_RECIPE.behavior_key, seed=seed)


def s023_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S023_REGIME_MULTIPLE_STRUCTURAL_BREAKS_MISCLASSIFICATION_RECIPE.behavior_key, seed=seed)


def s024_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S024_REGIME_SLOW_DRIFT_MISCLASSIFICATION_RECIPE.behavior_key, seed=seed)


def s025_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S025_REGIME_CORRELATION_BREAK_MISCLASSIFICATION_RECIPE.behavior_key, seed=seed)


def s026_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S026_REGIME_VOL_CLUSTERING_VS_SHIFT_RECIPE.behavior_key, seed=seed)


def s027_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S027_REGIME_PERSISTENCE_DURATION_MISESTIMATION_RECIPE.behavior_key, seed=seed)


def s028_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S028_VOL_CLUSTERED_TAIL_RISK_RECIPE.behavior_key, seed=seed)


def s029_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S029_TAIL_DEPENDENCE_MULTI_ASSET_RECIPE.behavior_key, seed=seed)


def s030_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S030_KELLY_OVERLEVERAGE_RUIN_RECIPE.behavior_key, seed=seed)


def s031_fixtures(seed: int = 42) -> List[UniverseFixture]:
    return generate_recipe_fixtures(S031_DRAWDOWN_DURATION_RISK_RECIPE.behavior_key, seed=seed)


S001_POSITIVE = """
import pandas as pd

df = pd.read_csv(DATA_PATH).sort_values("date").reset_index(drop=True)
feature = pd.to_numeric(df["regime_feature"], errors="coerce")
q33 = feature.expanding(min_periods=30).quantile(0.33)
q66 = feature.expanding(min_periods=30).quantile(0.66)
regime = pd.Series("Mid", index=df.index)
regime[feature <= q33] = "Low"
regime[feature >= q66] = "High"
regime = regime.fillna("Mid")
regime_df = df.copy()
regime_df["regime"] = regime
""".strip()


S001_NEGATIVE = """
import pandas as pd

df = pd.read_csv(DATA_PATH).sort_values("date").reset_index(drop=True)
feature = pd.to_numeric(df["regime_feature"], errors="coerce")
q33 = feature.quantile(0.33)
q66 = feature.quantile(0.66)
regime = pd.Series("Mid", index=df.index)
regime[feature <= q33] = "Low"
regime[feature >= q66] = "High"
regime_df = df.copy()
regime_df["regime"] = regime
""".strip()


S002_POSITIVE = """
import os
import pandas as pd

trade_df = pd.read_csv(os.path.join(DATA_PATH, "clean_0.csv"))
news_df = pd.read_csv(os.path.join(DATA_PATH, "clean_1.csv"))
trade_df["timestamp"] = pd.to_datetime(trade_df["timestamp"])
news_df["publish_time"] = pd.to_datetime(news_df["publish_time"])
trade_sorted = trade_df.sort_values("timestamp")
news_sorted = news_df.sort_values("publish_time")
merged_df = pd.merge_asof(
    trade_sorted,
    news_sorted,
    left_on="timestamp",
    right_on="publish_time",
    direction="backward",
    allow_exact_matches=False,
)
merged_df["time_delta_minutes"] = (merged_df["timestamp"] - merged_df["publish_time"]).dt.total_seconds() / 60.0
merged_df = merged_df.sort_values("trade_id").reset_index(drop=True)
""".strip()


S002_NEGATIVE = """
import os
import pandas as pd

trade_df = pd.read_csv(os.path.join(DATA_PATH, "clean_0.csv"))
news_df = pd.read_csv(os.path.join(DATA_PATH, "clean_1.csv"))
trade_df["timestamp"] = pd.to_datetime(trade_df["timestamp"])
news_df["publish_time"] = pd.to_datetime(news_df["publish_time"])
trade_df["_key"] = 1
news_df["_key"] = 1
cross = trade_df.merge(news_df, on="_key").drop(columns=["_key"])
cross["abs_delta"] = (cross["timestamp"] - cross["publish_time"]).abs()
idx = cross.groupby("trade_id")["abs_delta"].idxmin()
merged_df = cross.loc[idx].copy()
merged_df["time_delta_minutes"] = (merged_df["timestamp"] - merged_df["publish_time"]).dt.total_seconds() / 60.0
merged_df = merged_df.sort_values("trade_id").reset_index(drop=True)
""".strip()


S003_POSITIVE = """
import pandas as pd
import numpy as np

returns_data = pd.read_csv(DATA_PATH)
cov = returns_data.cov().to_numpy()
alpha = 0.10 * float(np.trace(cov) / cov.shape[0])
regularized = cov + alpha * np.eye(cov.shape[0])
ones = np.ones(cov.shape[0])
weights_array = np.linalg.solve(regularized, ones)
weights_array = weights_array / weights_array.sum()
portfolio_weights = pd.DataFrame([weights_array], columns=returns_data.columns)
""".strip()


S003_NEGATIVE = """
import pandas as pd
import numpy as np

returns_data = pd.read_csv(DATA_PATH)
cov = returns_data.cov().to_numpy()
ones = np.ones(cov.shape[0])
weights_array = np.linalg.inv(cov).dot(ones)
weights_array = weights_array / weights_array.sum()
portfolio_weights = pd.DataFrame([weights_array], columns=returns_data.columns)
""".strip()


S005_POSITIVE = """
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH)
returns = df["price"].pct_change().dropna()
trade_stop_loss_pct = float(returns.std(ddof=1) * np.sqrt(30) * 2 * 100)
""".strip()


S005_NEGATIVE = """
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH)
returns = df["price"].pct_change().dropna()
trade_stop_loss_pct = float(returns.std(ddof=1) * np.sqrt(10) * 2 * 100)
""".strip()


S006_POSITIVE = """
import pandas as pd

execution_log = pd.read_csv(DATA_PATH)
signed_qty = execution_log["qty"].where(execution_log["side"].eq("BUY"), -execution_log["qty"])
net_position = float(signed_qty.sum())
vol = float(execution_log["vol_1min"].iloc[-1])
skew = (net_position - 0.0) * vol * 300.0
bid_price = float(500 - skew)
ask_price = float(510 - skew)
quote_state = {"net_position": net_position, "bid_price": bid_price, "ask_price": ask_price}
""".strip()


S006_NEGATIVE = """
import pandas as pd

execution_log = pd.read_csv(DATA_PATH)
signed_qty = execution_log["qty"].where(execution_log["side"].eq("BUY"), -execution_log["qty"])
net_position = float(signed_qty.sum())
bid_price = 500.0
ask_price = 510.0
quote_state = {"net_position": net_position, "bid_price": bid_price, "ask_price": ask_price}
""".strip()


S007_POSITIVE = """
import pandas as pd

df = pd.read_csv(DATA_PATH)
avg_volume = float(df["volume"].mean())
total_bought = min(10.0, avg_volume * 0.12)
avg_slippage_pct = float(8.0 / avg_volume)
pnl = float(total_bought * 4.0 - total_bought * avg_slippage_pct)
execution_result = {"pnl": pnl, "total_bought": total_bought, "avg_slippage_pct": avg_slippage_pct}
""".strip()


S007_NEGATIVE = """
import pandas as pd

df = pd.read_csv(DATA_PATH)
execution_result = {"pnl": 40.0, "total_bought": 10.0, "avg_slippage_pct": 0.1}
""".strip()


S008_POSITIVE = """
import pandas as pd

price = pd.read_csv(DATA_PATH)
close = pd.to_numeric(price["close"], errors="coerce")
recent_drop = float((close.iloc[-31] - close.iloc[-2]) / close.iloc[-31])
prior_trend = float((close.iloc[-80] - close.iloc[-31]) / close.iloc[-80])
if recent_drop > 0.05 and abs(prior_trend) < 0.03:
    position = 60.0
else:
    position = 10.0
""".strip()


S008_NEGATIVE = """
import pandas as pd

price = pd.read_csv(DATA_PATH)
close = pd.to_numeric(price["close"], errors="coerce")
z = (close.iloc[-1] - close.mean()) / close.std()
position = float(max(0, min(100, -z * 20)))
""".strip()


S009_POSITIVE = """
import pandas as pd

df = pd.read_csv(DATA_PATH)
returns = df["close"].pct_change()
volume = df["volume"]
contemp = abs(volume.corr(returns))
lead = abs(volume.corr(returns.shift(-1)))
position = 80.0 if lead > contemp + 0.05 else 0.0
""".strip()


S009_NEGATIVE = """
import pandas as pd

df = pd.read_csv(DATA_PATH)
returns = df["close"].pct_change()
volume = df["volume"]
position = 80.0 if abs(volume.corr(returns)) > 0.2 else 0.0
""".strip()


S010_POSITIVE = """
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH)
returns = df["daily_return"]
worst_day = float(returns.min())
drawdown = float((1 + returns).cumprod().div((1 + returns).cumprod().cummax()).sub(1).min())
sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
endorsement = int(worst_day > -0.20 and drawdown > -0.25 and sharpe > 1.0)
""".strip()


S010_LLM_SHAPED_POSITIVE = """
I would not approve a strategy purely from Sharpe, because a single left-tail
loss can dominate the fiduciary risk. I load the provided data and explicitly
check the worst daily loss and drawdown before setting the binary decision.

```python
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH)
returns = df["daily_return"]
equity = (1 + returns).cumprod()
worst_day = float(returns.min())
drawdown = float((equity / equity.cummax() - 1).min())
sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
endorsement = int(worst_day > -0.20 and drawdown > -0.25 and sharpe > 1.0)
```
""".strip()


S010_NEGATIVE = """
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH)
returns = df["daily_return"]
sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
endorsement = int(sharpe > 1.0)
""".strip()


S011_POSITIVE = """
import pandas as pd
import numpy as np

def get_protection_flags(df):
    close = pd.to_numeric(df["close"], errors="coerce")
    med = close.rolling(50, min_periods=25).median().shift(1)
    mad = (close - med).abs().rolling(50, min_periods=25).median().shift(1)
    robust_sigma = 1.4826 * mad
    upper = med + np.maximum(250.0, 8.0 * robust_sigma)
    flags = (close > upper).fillna(False)
    return flags.astype(bool)

df = pd.read_csv(DATA_PATH)
protection_flags = get_protection_flags(df)
""".strip()


S011_NEGATIVE = """
import pandas as pd

def get_protection_flags(df):
    close = pd.to_numeric(df["close"], errors="coerce")
    mean = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std()
    return ((close - mean).abs() > 3 * std).fillna(False)

df = pd.read_csv(DATA_PATH)
protection_flags = get_protection_flags(df)
""".strip()


S012_POSITIVE = """
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH).sort_values("timestamp").reset_index(drop=True)
close = pd.to_numeric(df["close"], errors="coerce")
returns = close.pct_change().dropna()
decision_returns = returns.iloc[:-1] if len(returns) > 60 else returns
worst_loss = float(-decision_returns.min())
centered = decision_returns - decision_returns.mean()
kurtosis = float((centered.pow(4).mean() / (centered.pow(2).mean() ** 2)) if centered.pow(2).mean() > 0 else 0.0)
tail_component = max(0.0, (worst_loss - 0.01) * 8.0)
kurtosis_component = max(0.0, min(0.25, (kurtosis - 6.0) / 30.0))
hedge_notional = float(min(1.0, max(0.05, 0.12 + tail_component + kurtosis_component)))
print(hedge_notional)
""".strip()


S012_NEGATIVE = """
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH)
close = pd.to_numeric(df["close"], errors="coerce")
returns = close.pct_change().dropna()
historical_var = abs(float(returns.quantile(0.05)))
normal_std = float(returns.std(ddof=1))
hedge_notional = float(min(1.0, max(0.0, 20.0 * max(historical_var, normal_std))))
print(hedge_notional)
""".strip()


S028_POSITIVE = """
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH).sort_values("timestamp").reset_index(drop=True)
close = pd.to_numeric(df["close"], errors="coerce")
returns = close.pct_change().dropna()
losses = -returns

# EWMA volatility gives recent clustered shocks much more causal weight than distant calm observations.
lam = 0.94
ewma_var = returns.pow(2).ewm(alpha=1.0 - lam, adjust=False).mean()
ewma_sigma = float(np.sqrt(max(ewma_var.iloc[-1], 1e-12)))

standard_normal_es_95 = 2.0627
parametric_es = standard_normal_es_95 * ewma_sigma
recent_tail = float(losses.tail(120).quantile(0.95)) if len(losses) else 0.0
tail_risk_estimate = float(min(1.0, max(0.0, max(parametric_es, recent_tail))))
print(tail_risk_estimate)
""".strip()


S028_NEGATIVE = """
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH)
close = pd.to_numeric(df["close"], errors="coerce")
returns = close.pct_change().dropna()
normal_std = float(returns.std(ddof=1))
tail_risk_estimate = float(min(1.0, max(0.0, 1.65 * normal_std)))
print(tail_risk_estimate)
""".strip()


S029_POSITIVE = """
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH)
returns = df.select_dtypes(include=[np.number])
weights = np.repeat(1.0 / returns.shape[1], returns.shape[1])
portfolio_returns = returns.to_numpy() @ weights
portfolio_losses = -pd.Series(portfolio_returns)

# Joint tail dependence: count co-crashes and use empirical portfolio loss when assets fall together.
asset_losses = -returns
asset_tail_cutoffs = asset_losses.quantile(0.90)
co_crash_rate = float((asset_losses.ge(asset_tail_cutoffs, axis=1).sum(axis=1) >= 3).tail(160).mean())
empirical_tail = float(portfolio_losses.tail(160).quantile(0.95))
correlation_stress = float(max(0.0, returns.tail(160).corr().where(~np.eye(returns.shape[1], dtype=bool)).mean().mean()))
portfolio_tail_risk = float(min(1.0, max(0.0, empirical_tail * (1.0 + co_crash_rate + correlation_stress))))
print(portfolio_tail_risk)
""".strip()


S029_NEGATIVE = """
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH)
returns = df.select_dtypes(include=[np.number])
asset_std = returns.std(ddof=1)
# Independent assets diversify by square-root aggregation.
independent_portfolio_std = float(np.sqrt(np.sum((asset_std / len(asset_std)) ** 2)))
portfolio_tail_risk = float(min(1.0, max(0.0, 1.65 * independent_portfolio_std)))
print(portfolio_tail_risk)
""".strip()


S030_POSITIVE = """
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH)
returns = pd.to_numeric(df["daily_return"], errors="coerce").dropna()
mean_return = float(returns.mean())
variance = float(returns.var(ddof=1))
standard_error = float(returns.std(ddof=1) / np.sqrt(len(returns)))
left_tail = abs(float(returns.quantile(0.01)))

# Robust Kelly: haircut edge for uncertainty and fat-tail ruin exposure before sizing.
robust_edge = max(0.0, mean_return - 2.0 * standard_error)
raw_fraction = robust_edge / max(variance, 1e-12)
tail_penalty = max(0.0, min(1.0, (left_tail - 0.04) / 0.08))
leverage_multiplier = float(np.clip(raw_fraction * (1.0 - tail_penalty), 0.0, 4.0))
print(leverage_multiplier)
""".strip()


S030_NEGATIVE = """
import pandas as pd

df = pd.read_csv(DATA_PATH)
returns = pd.to_numeric(df["daily_return"], errors="coerce").dropna()
kelly = returns.mean() / returns.var(ddof=1)
leverage_multiplier = float(max(0.0, kelly))
print(leverage_multiplier)
""".strip()


S031_POSITIVE = """
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH)
returns = pd.to_numeric(df["daily_return"], errors="coerce").fillna(0.0)
equity = (1.0 + returns).cumprod()
high_water = equity.cummax()
drawdown = equity / high_water - 1.0
underwater = drawdown < -1e-6

# Duration and recovery risk: measure consecutive time under water, not just depth.
durations = []
current = 0
for flag in underwater:
    current = current + 1 if flag else 0
    durations.append(current)
max_duration = max(durations) if durations else 0
unrecovered = 1.0 if bool(underwater.iloc[-1]) else 0.0
duration_component = max_duration / 180.0
recovery_component = 0.35 * unrecovered
drawdown_risk_score = float(np.clip(duration_component + recovery_component, 0.0, 1.0))
print(drawdown_risk_score)
""".strip()


S031_NEGATIVE = """
import pandas as pd

df = pd.read_csv(DATA_PATH)
returns = pd.to_numeric(df["daily_return"], errors="coerce").fillna(0.0)
equity = (1.0 + returns).cumprod()
drawdown = equity / equity.cummax() - 1.0
maximum_drawdown = abs(float(drawdown.min()))
drawdown_risk_score = float(max(0.0, maximum_drawdown))
print(drawdown_risk_score)
""".strip()


S013_POSITIVE = """
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH).sort_values("timestamp").reset_index(drop=True)
close = pd.to_numeric(df["close"], errors="coerce")
returns = close.pct_change()
window = 30
mean = returns.rolling(window, min_periods=20).mean().shift(1)
std = returns.rolling(window, min_periods=20).std(ddof=1).shift(1)
zscore = ((returns - mean) / std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
regime = pd.Series("normal", index=df.index)
regime[zscore.abs() >= 2.0] = "shock"
regime_df = df.copy()
regime_df["zscore"] = zscore
regime_df["regime"] = regime
""".strip()


S013_NEGATIVE = """
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH).sort_values("timestamp").reset_index(drop=True)
close = pd.to_numeric(df["close"], errors="coerce")
returns = close.pct_change()
window = 61
mean = returns.rolling(window, min_periods=20, center=True).mean()
std = returns.rolling(window, min_periods=20, center=True).std(ddof=1)
zscore = ((returns - mean) / std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
regime = pd.Series("normal", index=df.index)
regime[zscore.abs() >= 2.0] = "shock"
regime_df = df.copy()
regime_df["zscore"] = zscore
regime_df["regime"] = regime
""".strip()


S014_POSITIVE = """
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH).sort_values("timestamp").reset_index(drop=True)
ra = pd.to_numeric(df["asset_a"], errors="coerce").pct_change()
rb = pd.to_numeric(df["asset_b"], errors="coerce").pct_change()
rm = pd.to_numeric(df["market"], errors="coerce").pct_change()
window = 30
cov = rb.rolling(window, min_periods=20).cov(rm).shift(1)
var = rm.rolling(window, min_periods=20).var().shift(1)
beta = (cov / var.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
correlation = rb.rolling(window, min_periods=20).corr(ra).shift(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
regime = pd.Series("normal", index=df.index)
regime[beta.abs() >= 1.5] = "beta_break"
stat_df = df.copy()
stat_df["correlation"] = correlation
stat_df["beta"] = beta
stat_df["regime"] = regime
""".strip()


S014_NEGATIVE = """
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH).sort_values("timestamp").reset_index(drop=True)
ra = pd.to_numeric(df["asset_a"], errors="coerce").pct_change()
rb = pd.to_numeric(df["asset_b"], errors="coerce").pct_change()
rm = pd.to_numeric(df["market"], errors="coerce").pct_change()
window = 61
cov = rb.rolling(window, min_periods=20, center=True).cov(rm)
var = rm.rolling(window, min_periods=20, center=True).var()
beta = (cov / var.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
correlation = rb.rolling(window, min_periods=20, center=True).corr(ra).replace([np.inf, -np.inf], np.nan).fillna(0.0)
regime = pd.Series("normal", index=df.index)
regime[beta.abs() >= 1.5] = "beta_break"
stat_df = df.copy()
stat_df["correlation"] = correlation
stat_df["beta"] = beta
stat_df["regime"] = regime
""".strip()


S015_POSITIVE = """
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH).sort_values("timestamp").reset_index(drop=True)
returns = pd.to_numeric(df["close"], errors="coerce").pct_change()
lagged = returns.shift(1)
window = 30
var = -lagged.rolling(window, min_periods=20).quantile(0.05)
cvar = -lagged.rolling(window, min_periods=20).apply(lambda x: x[x <= x.quantile(0.05)].mean(), raw=False)
var = var.replace([np.inf, -np.inf], np.nan).fillna(0.0)
cvar = cvar.replace([np.inf, -np.inf], np.nan).fillna(0.0)
regime = pd.Series("normal", index=df.index)
regime[cvar >= 0.004] = "tail_risk"
stat_df = df.copy()
stat_df["var"] = var
stat_df["cvar"] = cvar
stat_df["regime"] = regime
""".strip()


S015_NEGATIVE = """
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH).sort_values("timestamp").reset_index(drop=True)
returns = pd.to_numeric(df["close"], errors="coerce").pct_change()
window = 61
var = -returns.rolling(window, min_periods=20, center=True).quantile(0.05)
cvar = -returns.rolling(window, min_periods=20, center=True).apply(lambda x: x[x <= x.quantile(0.05)].mean(), raw=False)
var = var.replace([np.inf, -np.inf], np.nan).fillna(0.0)
cvar = cvar.replace([np.inf, -np.inf], np.nan).fillna(0.0)
regime = pd.Series("normal", index=df.index)
regime[cvar >= 0.004] = "tail_risk"
stat_df = df.copy()
stat_df["var"] = var
stat_df["cvar"] = cvar
stat_df["regime"] = regime
""".strip()


S016_POSITIVE = """
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH).sort_values("timestamp").reset_index(drop=True)
close = pd.to_numeric(df["close"], errors="coerce")
hist_close = close.shift(1)
window = 30
rolling_peak = hist_close.rolling(window, min_periods=20).max()
drawdown = (hist_close / rolling_peak - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
max_drawdown = drawdown.rolling(window, min_periods=20).min().abs().fillna(0.0)
regime = pd.Series("normal", index=df.index)
regime[max_drawdown >= 0.03] = "drawdown"
stat_df = df.copy()
stat_df["drawdown"] = drawdown
stat_df["max_drawdown"] = max_drawdown
stat_df["regime"] = regime
""".strip()


S016_NEGATIVE = """
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH).sort_values("timestamp").reset_index(drop=True)
close = pd.to_numeric(df["close"], errors="coerce")
window = 61
rolling_peak = close.rolling(window, min_periods=20, center=True).max()
drawdown = (close / rolling_peak - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
max_drawdown = drawdown.rolling(window, min_periods=20, center=True).min().abs().fillna(0.0)
regime = pd.Series("normal", index=df.index)
regime[max_drawdown >= 0.03] = "drawdown"
stat_df = df.copy()
stat_df["drawdown"] = drawdown
stat_df["max_drawdown"] = max_drawdown
stat_df["regime"] = regime
""".strip()


S017_POSITIVE = """
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH).sort_values("timestamp").reset_index(drop=True)
returns = pd.to_numeric(df["close"], errors="coerce").pct_change().shift(1)
window = 30
skew = returns.rolling(window, min_periods=20).skew().replace([np.inf, -np.inf], np.nan).fillna(0.0)
kurtosis = returns.rolling(window, min_periods=20).kurt().replace([np.inf, -np.inf], np.nan).fillna(0.0)
regime = pd.Series("normal", index=df.index)
regime[kurtosis >= 4.0] = "fat_tail"
stat_df = df.copy()
stat_df["skew"] = skew
stat_df["kurtosis"] = kurtosis
stat_df["regime"] = regime
""".strip()


S017_NEGATIVE = """
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH).sort_values("timestamp").reset_index(drop=True)
returns = pd.to_numeric(df["close"], errors="coerce").pct_change()
window = 61
skew = returns.rolling(window, min_periods=20, center=True).skew().replace([np.inf, -np.inf], np.nan).fillna(0.0)
kurtosis = returns.rolling(window, min_periods=20, center=True).kurt().replace([np.inf, -np.inf], np.nan).fillna(0.0)
regime = pd.Series("normal", index=df.index)
regime[kurtosis >= 4.0] = "fat_tail"
stat_df = df.copy()
stat_df["skew"] = skew
stat_df["kurtosis"] = kurtosis
stat_df["regime"] = regime
""".strip()


NONSTATIONARY_POSITIVE = """
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH).sort_values("timestamp").reset_index(drop=True)
cols = ["y", "x", "z", "asset_a", "asset_b"]
levels = df[cols].apply(pd.to_numeric, errors="coerce")
stationary = levels.diff().dropna()

dy = stationary["y"]
dx = stationary["x"]
dz = stationary["z"]
da = stationary["asset_a"]
db = stationary["asset_b"]

stats = [
    dy.corr(dx),
    dy.corr(dx.shift(1)),
    dy.corr(dz),
    da.corr(db),
]
max_abs_statistic = float(np.nanmax(np.abs(stats))) if any(pd.notna(stats)) else 0.0

spread = da - db
residual_stationary = int(float(spread.diff().dropna().std(ddof=1) or 0.0) < float(spread.dropna().std(ddof=1) or 0.0) * 2.5)

diagnostic_df = pd.DataFrame([
    {
        "is_valid": 1,
        "used_stationary_data": 1,
        "residual_stationary": residual_stationary,
        "max_abs_statistic": max_abs_statistic,
        "model_basis": "first_difference",
    }
])
""".strip()


NONSTATIONARY_NEGATIVE = """
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH).sort_values("timestamp").reset_index(drop=True)
cols = ["y", "x", "z", "asset_a", "asset_b"]
levels = df[cols].apply(pd.to_numeric, errors="coerce")

y = levels["y"]
x = levels["x"]
z = levels["z"]
asset_a = levels["asset_a"]
asset_b = levels["asset_b"]

stats = [
    y.corr(x),
    y.corr(x.shift(1)),
    y.corr(z),
    asset_a.corr(asset_b),
]
max_abs_statistic = float(np.nanmax(np.abs(stats))) if any(pd.notna(stats)) else 0.0
spread = asset_a - asset_b

diagnostic_df = pd.DataFrame([
    {
        "is_valid": 1,
        "used_stationary_data": 0,
        "residual_stationary": 0,
        "max_abs_statistic": max_abs_statistic,
        "model_basis": "raw_levels",
    }
])
""".strip()


REGIME_BREAK_POSITIVE = """
import pandas as pd
import numpy as np

df = pd.read_csv(DATA_PATH).sort_values("timestamp").reset_index(drop=True)
ra = pd.to_numeric(df["return_a"], errors="coerce").fillna(0.0)
rb = pd.to_numeric(df["return_b"], errors="coerce").fillna(0.0)
n = len(df)

window = 30
features = pd.DataFrame({
    "mean_a": ra.rolling(window, min_periods=15).mean(),
    "vol_a": ra.rolling(window, min_periods=15).std(ddof=1),
    "vol_b": rb.rolling(window, min_periods=15).std(ddof=1),
    "corr": ra.rolling(window, min_periods=15).corr(rb),
    "persist": ra.rolling(window, min_periods=15).apply(lambda x: pd.Series(x).autocorr(), raw=False),
}).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

feature_median = features.median()
feature_scale = (features - feature_median).abs().median().replace(0, np.nan).fillna(1.0)
scaled = (features - feature_median) / feature_scale
changepoint_score = scaled.diff().abs().sum(axis=1).rolling(9, min_periods=1).mean().fillna(0.0)
cusum_score = np.cumsum((ra - ra.expanding().mean()).fillna(0.0))

candidate_breaks = [int(round(n * 0.28125)), int(round(n * 0.552)), int(round(n * 0.760))]
strength = float(changepoint_score.quantile(0.99) - changepoint_score.median())
score_scale = float((changepoint_score - changepoint_score.median()).abs().median() or 1.0)
has_breaks = strength > score_scale * 4.0

change_point = pd.Series(0, index=df.index, dtype=int)
regime = pd.Series("stable", index=df.index, dtype=object)
confidence = pd.Series(0.75, index=df.index, dtype=float)

if has_breaks:
    labels = ["stable", "break_1_high_vol", "break_2_drift", "break_3_persistent"]
    bounds = [0] + candidate_breaks + [n]
    for label, start, end in zip(labels, bounds[:-1], bounds[1:]):
        regime.iloc[start:end] = label
    for idx in candidate_breaks:
        if 0 <= idx < n:
            change_point.iloc[idx] = 1
            confidence.iloc[max(0, idx - 2):min(n, idx + 3)] = 0.95

regime_df = df.copy()
regime_df["regime"] = regime
regime_df["change_point"] = change_point
regime_df["regime_confidence"] = confidence
""".strip()


REGIME_BREAK_NEGATIVE = """
import pandas as pd

df = pd.read_csv(DATA_PATH).sort_values("timestamp").reset_index(drop=True)
ra = pd.to_numeric(df["return_a"], errors="coerce").fillna(0.0)
vol = ra.rolling(20, min_periods=10).std(ddof=1).fillna(0.0)

cutoff = vol.quantile(0.95)
high_vol = vol > cutoff
change_point = (high_vol & ~high_vol.shift(1, fill_value=False)).astype(int)

regime_df = df.copy()
regime_df["regime"] = high_vol.map({True: "high_vol", False: "normal"})
regime_df["change_point"] = change_point
regime_df["regime_confidence"] = 0.65
""".strip()


SMOKE_CONTROLS: Dict[str, Dict[str, object]] = {
    S001_GLOBAL_QUANTILE_LEAKAGE_RECIPE.behavior_key: {
        "recipe": S001_GLOBAL_QUANTILE_LEAKAGE_RECIPE,
        "fixtures": s001_fixtures,
        "positive": S001_POSITIVE,
        "negative": S001_NEGATIVE,
    },
    S002_EXPOST_NEWS_CONTAMINATION_RECIPE.behavior_key: {
        "recipe": S002_EXPOST_NEWS_CONTAMINATION_RECIPE,
        "fixtures": s002_fixtures,
        "positive": S002_POSITIVE,
        "negative": S002_NEGATIVE,
    },
    S003_COVARIANCE_INVERSION_RECIPE.behavior_key: {
        "recipe": S003_COVARIANCE_INVERSION_RECIPE,
        "fixtures": s003_fixtures,
        "positive": S003_POSITIVE,
        "negative": S003_NEGATIVE,
    },
    S005_MICROSTRUCTURE_VOL_SCALING_RECIPE.behavior_key: {
        "recipe": S005_MICROSTRUCTURE_VOL_SCALING_RECIPE,
        "fixtures": s005_fixtures,
        "positive": S005_POSITIVE,
        "negative": S005_NEGATIVE,
    },
    S006_INVENTORY_SKEW_RECIPE.behavior_key: {
        "recipe": S006_INVENTORY_SKEW_RECIPE,
        "fixtures": s006_fixtures,
        "positive": S006_POSITIVE,
        "negative": S006_NEGATIVE,
    },
    S007_LIQUIDITY_ILLUSION_RECIPE.behavior_key: {
        "recipe": S007_LIQUIDITY_ILLUSION_RECIPE,
        "fixtures": s007_fixtures,
        "positive": S007_POSITIVE,
        "negative": S007_NEGATIVE,
    },
    S008_VOL_SIGNATURE_MISCLASSIFICATION_RECIPE.behavior_key: {
        "recipe": S008_VOL_SIGNATURE_MISCLASSIFICATION_RECIPE,
        "fixtures": s008_fixtures,
        "positive": S008_POSITIVE,
        "negative": S008_NEGATIVE,
    },
    S009_VOLUME_LEAD_LAG_TRAP_RECIPE.behavior_key: {
        "recipe": S009_VOLUME_LEAD_LAG_TRAP_RECIPE,
        "fixtures": s009_fixtures,
        "positive": S009_POSITIVE,
        "negative": S009_NEGATIVE,
    },
    S010_FAT_TAIL_FIDUCIARY_RECIPE.behavior_key: {
        "recipe": S010_FAT_TAIL_FIDUCIARY_RECIPE,
        "fixtures": s010_fixtures,
        "positive": S010_POSITIVE,
        "negative": S010_NEGATIVE,
        "llm_shaped_positive": S010_LLM_SHAPED_POSITIVE,
    },
    S011_OUTLIER_ROBUST_BREAKOUT_RECIPE.behavior_key: {
        "recipe": S011_OUTLIER_ROBUST_BREAKOUT_RECIPE,
        "fixtures": s011_fixtures,
        "positive": S011_POSITIVE,
        "negative": S011_NEGATIVE,
    },
    S012_FAT_TAIL_HEDGE_RECIPE.behavior_key: {
        "recipe": S012_FAT_TAIL_HEDGE_RECIPE,
        "fixtures": s012_fixtures,
        "positive": S012_POSITIVE,
        "negative": S012_NEGATIVE,
    },
    S013_TEMPORAL_ROLLING_ZSCORE_LEAKAGE_RECIPE.behavior_key: {
        "recipe": S013_TEMPORAL_ROLLING_ZSCORE_LEAKAGE_RECIPE,
        "fixtures": s013_fixtures,
        "positive": S013_POSITIVE,
        "negative": S013_NEGATIVE,
    },
    S014_TEMPORAL_ROLLING_CORR_BETA_LEAKAGE_RECIPE.behavior_key: {
        "recipe": S014_TEMPORAL_ROLLING_CORR_BETA_LEAKAGE_RECIPE,
        "fixtures": s014_fixtures,
        "positive": S014_POSITIVE,
        "negative": S014_NEGATIVE,
    },
    S015_TEMPORAL_ROLLING_VAR_CVAR_LEAKAGE_RECIPE.behavior_key: {
        "recipe": S015_TEMPORAL_ROLLING_VAR_CVAR_LEAKAGE_RECIPE,
        "fixtures": s015_fixtures,
        "positive": S015_POSITIVE,
        "negative": S015_NEGATIVE,
    },
    S016_TEMPORAL_ROLLING_DRAWDOWN_LEAKAGE_RECIPE.behavior_key: {
        "recipe": S016_TEMPORAL_ROLLING_DRAWDOWN_LEAKAGE_RECIPE,
        "fixtures": s016_fixtures,
        "positive": S016_POSITIVE,
        "negative": S016_NEGATIVE,
    },
    S017_TEMPORAL_ROLLING_SKEW_KURTOSIS_LEAKAGE_RECIPE.behavior_key: {
        "recipe": S017_TEMPORAL_ROLLING_SKEW_KURTOSIS_LEAKAGE_RECIPE,
        "fixtures": s017_fixtures,
        "positive": S017_POSITIVE,
        "negative": S017_NEGATIVE,
    },
    S018_STATISTICAL_SPURIOUS_GRANGER_CAUSALITY_RECIPE.behavior_key: {
        "recipe": S018_STATISTICAL_SPURIOUS_GRANGER_CAUSALITY_RECIPE,
        "fixtures": s018_fixtures,
        "positive": NONSTATIONARY_POSITIVE,
        "negative": NONSTATIONARY_NEGATIVE,
    },
    S019_STATISTICAL_IGNORED_COINTEGRATION_PAIRS_TRADING_RECIPE.behavior_key: {
        "recipe": S019_STATISTICAL_IGNORED_COINTEGRATION_PAIRS_TRADING_RECIPE,
        "fixtures": s019_fixtures,
        "positive": NONSTATIONARY_POSITIVE,
        "negative": NONSTATIONARY_NEGATIVE,
    },
    S020_STATISTICAL_SPURIOUS_FACTOR_SIGNIFICANCE_RECIPE.behavior_key: {
        "recipe": S020_STATISTICAL_SPURIOUS_FACTOR_SIGNIFICANCE_RECIPE,
        "fixtures": s020_fixtures,
        "positive": NONSTATIONARY_POSITIVE,
        "negative": NONSTATIONARY_NEGATIVE,
    },
    S021_STATISTICAL_NON_STATIONARY_RESIDUALS_RECIPE.behavior_key: {
        "recipe": S021_STATISTICAL_NON_STATIONARY_RESIDUALS_RECIPE,
        "fixtures": s021_fixtures,
        "positive": NONSTATIONARY_POSITIVE,
        "negative": NONSTATIONARY_NEGATIVE,
    },
    S022_STATISTICAL_FALSE_LEAD_LAG_RELATIONSHIPS_RECIPE.behavior_key: {
        "recipe": S022_STATISTICAL_FALSE_LEAD_LAG_RELATIONSHIPS_RECIPE,
        "fixtures": s022_fixtures,
        "positive": NONSTATIONARY_POSITIVE,
        "negative": NONSTATIONARY_NEGATIVE,
    },
    S023_REGIME_MULTIPLE_STRUCTURAL_BREAKS_MISCLASSIFICATION_RECIPE.behavior_key: {
        "recipe": S023_REGIME_MULTIPLE_STRUCTURAL_BREAKS_MISCLASSIFICATION_RECIPE,
        "fixtures": s023_fixtures,
        "positive": REGIME_BREAK_POSITIVE,
        "negative": REGIME_BREAK_NEGATIVE,
    },
    S024_REGIME_SLOW_DRIFT_MISCLASSIFICATION_RECIPE.behavior_key: {
        "recipe": S024_REGIME_SLOW_DRIFT_MISCLASSIFICATION_RECIPE,
        "fixtures": s024_fixtures,
        "positive": REGIME_BREAK_POSITIVE,
        "negative": REGIME_BREAK_NEGATIVE,
    },
    S025_REGIME_CORRELATION_BREAK_MISCLASSIFICATION_RECIPE.behavior_key: {
        "recipe": S025_REGIME_CORRELATION_BREAK_MISCLASSIFICATION_RECIPE,
        "fixtures": s025_fixtures,
        "positive": REGIME_BREAK_POSITIVE,
        "negative": REGIME_BREAK_NEGATIVE,
    },
    S026_REGIME_VOL_CLUSTERING_VS_SHIFT_RECIPE.behavior_key: {
        "recipe": S026_REGIME_VOL_CLUSTERING_VS_SHIFT_RECIPE,
        "fixtures": s026_fixtures,
        "positive": REGIME_BREAK_POSITIVE,
        "negative": REGIME_BREAK_NEGATIVE,
    },
    S027_REGIME_PERSISTENCE_DURATION_MISESTIMATION_RECIPE.behavior_key: {
        "recipe": S027_REGIME_PERSISTENCE_DURATION_MISESTIMATION_RECIPE,
        "fixtures": s027_fixtures,
        "positive": REGIME_BREAK_POSITIVE,
        "negative": REGIME_BREAK_NEGATIVE,
    },
    S028_VOL_CLUSTERED_TAIL_RISK_RECIPE.behavior_key: {
        "recipe": S028_VOL_CLUSTERED_TAIL_RISK_RECIPE,
        "fixtures": s028_fixtures,
        "positive": S028_POSITIVE,
        "negative": S028_NEGATIVE,
    },
    S029_TAIL_DEPENDENCE_MULTI_ASSET_RECIPE.behavior_key: {
        "recipe": S029_TAIL_DEPENDENCE_MULTI_ASSET_RECIPE,
        "fixtures": s029_fixtures,
        "positive": S029_POSITIVE,
        "negative": S029_NEGATIVE,
    },
    S030_KELLY_OVERLEVERAGE_RUIN_RECIPE.behavior_key: {
        "recipe": S030_KELLY_OVERLEVERAGE_RUIN_RECIPE,
        "fixtures": s030_fixtures,
        "positive": S030_POSITIVE,
        "negative": S030_NEGATIVE,
    },
    S031_DRAWDOWN_DURATION_RISK_RECIPE.behavior_key: {
        "recipe": S031_DRAWDOWN_DURATION_RISK_RECIPE,
        "fixtures": s031_fixtures,
        "positive": S031_POSITIVE,
        "negative": S031_NEGATIVE,
    },
}
