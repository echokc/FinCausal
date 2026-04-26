import pandas as pd
import numpy as np
import os
import re
import shutil
import traceback

from core.sandbox import run_in_docker_sandbox, format_result

def setup_s012_data():
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    np.random.seed(42)
    n = 5000
    ts = [datetime(2025, 4, 1) + timedelta(minutes=i) for i in range(n)]

    def ou(mean=65000, theta=0.01, sigma=280):
        x = np.zeros(n)
        x[0] = mean
        for t in range(1, n):
            x[t] = x[t-1] + theta * (mean - x[t-1]) + sigma * np.random.normal()
        return x

    # 1. Normal (low kurtosis)
    normal = ou(theta=0.008, sigma=220)
    df_normal = pd.DataFrame({'timestamp': ts, 'close': normal})

    # 2. Shock universe: fat-tail after T=2500 (12-sigma jump + vol*8)
    shock = ou(theta=0.008, sigma=220)
    shock[2500:] *= 1.15
    shock[3000] *= 1.28   # 12-sigma jump
    shock[2500:] += np.random.normal(0, 800, n-2500)
    df_shock = pd.DataFrame({'timestamp': ts, 'close': shock})

    # 3. Leakage test: extreme last point forces collapse if full-df leaked
    leak = ou(theta=0.008, sigma=220)
    leak[2500:] *= 1.15
    leak[-1] = leak[:-1].mean()   # leaked kurtosis → hedge collapses
    df_leak = pd.DataFrame({'timestamp': ts, 'close': leak})

    return df_normal, df_shock, df_leak


def score_s012(generated_code: str, reasoning: str) -> dict:
    df_normal, df_shock, df_leak = setup_s012_data()

    suffix = """
# === S012 Evaluation Snippet (Docker-safe) ===
import pandas as pd
eval_result = pd.DataFrame([{'hedge_notional': float(hedge_notional) if 'hedge_notional' in locals() else None}])
"""

    full_code = generated_code + suffix

    try:
        res_normal = run_in_docker_sandbox(full_code, {'DATA_PATH': df_normal}, result_keys=['eval_result'])
        res_shock  = run_in_docker_sandbox(full_code, {'DATA_PATH': df_shock},  result_keys=['eval_result'])
        res_leak   = run_in_docker_sandbox(full_code, {'DATA_PATH': df_leak},   result_keys=['eval_result'])

        if any(r is None for r in (res_normal, res_shock, res_leak)):
            return format_result(status="ERROR", reason="Docker sandbox failed")

        h_normal = float(res_normal.iloc[0]['hedge_notional'])
        h_shock  = float(res_shock.iloc[0]['hedge_notional'])
        h_leak   = float(res_leak.iloc[0]['hedge_notional'])

        metrics = {"h_normal": round(h_normal,3), "h_shock": round(h_shock,3), "h_leak": round(h_leak,3),
                   "delta": round(h_shock - h_normal,3), "cdi": 0.0}

        score_cdi = 0
        score_util = 0
        score_reason = 0
        details = {}

        # 1. Dynamic fat-tail response
        if h_normal <= 0.35 and h_shock >= 0.50:
            score_cdi += 50
            metrics['cdi'] = 1.0
            details['dynamic_cdi'] = f"PASS (+50): fat-tail detected & hedged (delta={metrics['delta']})"
        else:
            score_cdi = 0
            details['dynamic_cdi'] = f"CRITICAL FAIL (0/80): static VaR or no tail response"

        # 2. Temporal Causality / no leakage (one-vote-fail)
        if abs(h_leak - h_shock) <= 0.08:   # correct: no look-ahead
            score_cdi += 30
            details['leakage_cdi'] = f"PASS (+30): no look-ahead (leak delta={abs(h_leak-h_shock):.3f})"
        else:
            score_cdi = 0
            details['leakage_cdi'] = f"CRITICAL FAIL (0/80 total): look-ahead bias detected"
            return format_result(
                total=0,
                status="FAIL",
                pillars={"risk_causality": score_cdi},
                metrics=metrics,
                details=details
            )

        if 0.10 <= h_normal <= 0.40: score_util += 8
        if h_shock >= 0.50: score_util += 7
        details['utility'] = f"Utility +{score_util}/15"

        r = reasoning.lower()
        if any(w in r for w in ['kurtosis','evt','fat-tail','cornish','higher moment','convexity']):
            score_reason = 5
            details['reasoning'] = "PASS (+5): explicit tail-risk awareness"
        else:
            details['reasoning'] = "FAIL (0/5): no fat-tail insight"

        total = score_cdi + score_util + score_reason

        return format_result(
            total=total,
            status="PASS" if total >= 80 else "FAIL",
            pillars={"risk_causality": score_cdi, "functional_utility": score_util, "reasoning_awareness": score_reason},
            metrics=metrics,
            details=details
        )

    except Exception as e:
        return format_result(status="ERROR", reason=str(e))