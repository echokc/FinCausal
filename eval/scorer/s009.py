import pandas as pd
import numpy as np
import os
import re
import shutil
import traceback

from core.sandbox import run_in_docker_sandbox, format_result

def setup_s009_data():

    n = 2800
    rng = np.random.default_rng(42)

    # 1. Trap universe (volume lags + momentum continuation → predictive > contemp)
    returns_trap = rng.normal(0, 0.00055, n)
    volumes_trap = rng.uniform(1.8, 5.8, n)

    for i in range(220, n-3, 260):
        volumes_trap[i] = 70.0
        returns_trap[i] = 0.0075          # modest concurrent
        returns_trap[i+1] = 0.0125        # strong continuation

    # Add mild momentum to flip relative correlation
    for i in range(1, n):
        returns_trap[i] += 0.33 * returns_trap[i-1]

    df_trap = pd.DataFrame({
        'timestamp': pd.date_range('2026-01-01', periods=n, freq='1min'),
        'close': 50000 * np.exp(np.cumsum(returns_trap)),
        'volume': volumes_trap
    })

    # 2. Lead universe (clean leading signal)
    returns_lead = rng.normal(0, 0.00055, n)
    volumes_lead = rng.uniform(1.8, 5.8, n)

    for i in range(220, n-2, 260):
        volumes_lead[i] = 70.0
        returns_lead[i+1] = 0.014

    df_lead = pd.DataFrame({
        'timestamp': pd.date_range('2026-01-01', periods=n, freq='1min'),
        'close': 50000 * np.exp(np.cumsum(returns_lead)),
        'volume': volumes_lead
    })

    return df_trap, df_lead

def score_s009(generated_code: str, reasoning: str) -> dict:
    df_trap, df_lead = setup_s009_data()

    suffix = """
# === S009 Evaluation Snippet (Docker-safe) ===
import pandas as pd
eval_result = pd.DataFrame([{'position': float(position) if 'position' in locals() else None}])
"""

    full_code = generated_code + suffix

    try:
        res_trap = run_in_docker_sandbox(full_code, {'DATA_PATH': df_trap}, result_keys=['eval_result'])
        res_lead = run_in_docker_sandbox(full_code, {'DATA_PATH': df_lead}, result_keys=['eval_result'])

        if res_trap is None or res_lead is None:
            return format_result(status="ERROR", reason="Docker sandbox failed")

        pos_trap = float(res_trap.iloc[0]['position'])
        pos_lead = float(res_lead.iloc[0]['position'])

        import numpy as np
        metrics = {
            "pos_trap": round(pos_trap, 2),
            "pos_lead": round(pos_lead, 2),
            "delta": round(pos_lead - pos_trap, 2),
            "cdi": 0.0
        }

        score_cdi = 0
        score_util = 0
        details = {}

        # Pure A/B CDI (no code string match, only positions)
        if abs(pos_trap) <= 10 and pos_lead >= 50:
            score_cdi = 65
            metrics['cdi'] = 1.0
            details['causality_cdi'] = f"PASS (+65): correct inversion (trap={pos_trap:.1f} | lead={pos_lead:.1f})"
        elif abs(pos_trap) <= 20 and pos_lead >= 40:
            score_cdi = 45
            details['causality_cdi'] = "PARTIAL (+45): weak differentiation"
        elif np.isclose(pos_trap, pos_lead, atol=20):
            score_cdi = 0
            details['causality_cdi'] = f"CRITICAL FAIL (0/65): causality-blind (same behavior)"
        else:
            score_cdi = 0
            details['causality_cdi'] = "FAIL (0/65): wrong discrimination"

        # Utility (still pure behavior)
        if abs(pos_trap) <= 12: score_util += 15
        if pos_lead >= 45: score_util += 15
        details['utility'] = f"Utility +{score_util}/30"

        total = score_cdi + score_util

        return format_result(
            total=total,
            status="PASS" if total >= 80 else "FAIL",
            pillars={"causality_cdi": score_cdi, "functional_utility": score_util},
            metrics=metrics,
            details=details
        )

    except Exception as e:
        return format_result(status="ERROR", reason=str(e))