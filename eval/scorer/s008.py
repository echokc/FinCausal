import pandas as pd
import numpy as np
import os
import re
import shutil
import traceback
from datetime import datetime, timedelta

from core.sandbox import run_in_docker_sandbox, format_result

def setup_s008_data():
    np.random.seed(42)
    n = 30 * 24 * 60
    ts = [datetime(2025, 4, 1) + timedelta(minutes=i) for i in range(n)]

    def ou(mean=65000, theta=0.01, sigma=280):
        x = np.zeros(n)
        x[0] = mean
        for t in range(1, n):
            x[t] = x[t-1] + theta * (mean - x[t-1]) + sigma * np.random.normal()
        return x

    trend = ou(theta=0.002, sigma=180) + np.linspace(0, 9200, n)
    df_trend = pd.DataFrame({'timestamp': ts, 'close': trend, 'volume': np.random.lognormal(10, 1.1, n).astype(int)})

    shock = ou(theta=0.015, sigma=250)
    shock[-1440:] *= 0.91
    df_shock = pd.DataFrame({'timestamp': ts, 'close': shock, 'volume': np.random.lognormal(10, 1.2, n).astype(int)})

    leak = ou(theta=0.014, sigma=260)
    leak[-1440:] *= 0.905
    leak[-1] = leak[:-1].mean()   # if leaked → z≈0 → position≈0
    df_leak = pd.DataFrame({'timestamp': ts, 'close': leak, 'volume': np.random.lognormal(10, 1.15, n).astype(int)})

    return df_trend, df_shock, df_leak
    
def score_s008(generated_code: str, reasoning: str) -> dict:
    df_trend, df_shock, df_leak = setup_s008_data()

    suffix = """
# === S008 Evaluation Snippet (Docker-safe) ===
import pandas as pd
eval_result = pd.DataFrame([{'position': float(position) if 'position' in locals() else None}])
"""

    full_code = generated_code + suffix

    try:
        res_trend = run_in_docker_sandbox(full_code, {'DATA_PATH': df_trend}, result_keys=['eval_result'])
        res_shock = run_in_docker_sandbox(full_code, {'DATA_PATH': df_shock}, result_keys=['eval_result'])
        res_leak  = run_in_docker_sandbox(full_code, {'DATA_PATH': df_leak},  result_keys=['eval_result'])

        if any(r is None for r in (res_trend, res_shock, res_leak)):
            return format_result(status="ERROR", reason="Docker sandbox failed")

        pos_trend = float(res_trend.iloc[0]['position'])
        pos_shock = float(res_shock.iloc[0]['position'])
        pos_leak  = float(res_leak.iloc[0]['position'])

        import numpy as np
        metrics = {"pos_trend": round(pos_trend,2), "pos_shock": round(pos_shock,2), "pos_leak": round(pos_leak,2), "delta": round(pos_shock-pos_trend,2), "cdi": 0.0}

        score_cdi = 0
        score_util = 0
        score_reason = 0
        details = {}

        # Regime CDI (original)
        if np.isclose(pos_trend, pos_shock, atol=20, rtol=0.25):
            score_cdi = 0
            details['regime_cdi'] = f"CRITICAL FAIL (0/50): regime-blind (delta={metrics['delta']})"
        elif abs(pos_trend) <= 22 and pos_shock >= 48:
            score_cdi = 50
            metrics['cdi'] = 1.0
            details['regime_cdi'] = f"PASS (+50): full regime isolation"
        else:
            score_cdi = 0
            details['regime_cdi'] = "FAIL (0/50): wrong regime response"

        # NEW: Temporal Causality / Leakage CDI (one-vote-fail)
        if pos_leak >= 45:   # correct: no leakage
            score_cdi += 10
            details['leakage_cdi'] = f"PASS (+10): no look-ahead (leak={pos_leak:.1f})"
        else:
            score_cdi = 0
            details['leakage_cdi'] = f"CRITICAL FAIL (0/100 total): look-ahead bias detected (leak={pos_leak:.1f})"
            return format_result(
            total=0,
            status="FAIL",
            pillars={"regime_cdi": score_cdi},
            metrics=metrics,
            details=details
        )

        # Utility
        if abs(pos_trend) <= 25: score_util += 15
        if pos_shock >= 42: score_util += 15
        details['utility'] = f"Utility +{score_util}/30"

        # Reasoning
        r = reasoning.lower()
        if any(w in r for w in ['regime','trend','persistent','momentum']) and any(w in r for w in ['shock','transient','spike','decay','revert']):
            score_reason = 10
            details['reasoning'] = "PASS (+10): explicit regime-driver analysis"
        else:
            details['reasoning'] = "FAIL (0/10): no causality insight"

        total = score_cdi + score_util + score_reason

        return format_result(
            total=total,
            status="PASS" if total >= 80 else "FAIL",
            pillars={"regime_cdi": score_cdi, "functional_utility": score_util, "reasoning_awareness": score_reason},
            metrics=metrics,
            details=details
        )

    except Exception as e:
        return format_result(status="ERROR", reason=str(e))