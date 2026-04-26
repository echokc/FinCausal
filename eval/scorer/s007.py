import pandas as pd
import numpy as np
import os
import re
import shutil
import traceback

from core.sandbox import run_in_docker_sandbox, format_result

def setup_s007_normal_liquidity() -> pd.DataFrame:
    """Control group: high liquidity"""
    dates = pd.date_range('2025-04-21 10:00', periods=300, freq='1min')
    df = pd.DataFrame({
        'timestamp': dates,
        'close': 50000 + np.sin(np.arange(300) / 15) * 1500,   # ← 振幅加大到 1500
        'volume': [80.0] * 300,
    })
    return df


def setup_s007_thin_liquidity() -> pd.DataFrame:
    """Treatment group: extremely low liquidity"""
    dates = pd.date_range('2025-04-21 10:00', periods=300, freq='1min')
    df = pd.DataFrame({
        'timestamp': dates,
        'close': 50000 + np.sin(np.arange(300) / 15) * 1500,   # ← 振幅加大到 1500
        'volume': [1.2] * 300,
    })
    return df


def score_s007(generated_code: str, reasoning: str) -> dict:
    df_normal = setup_s007_normal_liquidity()
    df_thin = setup_s007_thin_liquidity()

    suffix = """
# === Evaluation Snippet ===
import pandas as pd
df = pd.read_csv(DATA_PATH)
result = calculate_realizable_pnl(df)

# Handle multiple possible return formats
if isinstance(result, dict):
    pnl = float(result.get('pnl', 0.0))
    total_bought = float(result.get('total_bought', 0.0))
    avg_slippage_pct = float(result.get('avg_slippage_pct', 0.0))
else:
    # fallback if model only returns a number
    pnl = float(result)
    total_bought = 0.0
    avg_slippage_pct = 0.0

eval_result = pd.DataFrame([{
    'pnl': pnl,
    'total_bought': total_bought,
    'avg_slippage_pct': avg_slippage_pct
}])
"""

    full_code = generated_code + suffix

    try:
        res_normal = run_in_docker_sandbox(full_code, {'DATA_PATH': df_normal}, result_keys=['eval_result'])
        res_thin   = run_in_docker_sandbox(full_code, {'DATA_PATH': df_thin},   result_keys=['eval_result'])

        if res_normal is None or res_thin is None:
            return format_result(status="ERROR", reason="Docker sandbox execution failed")

        # Extract results
        pnl_normal = float(res_normal.iloc[0]['pnl'])
        bought_normal = float(res_normal.iloc[0]['total_bought'])
        slip_normal = float(res_normal.iloc[0]['avg_slippage_pct'])

        pnl_thin = float(res_thin.iloc[0]['pnl'])
        bought_thin = float(res_thin.iloc[0]['total_bought'])
        slip_thin = float(res_thin.iloc[0]['avg_slippage_pct'])

        metrics = {
            "pnl_normal": round(pnl_normal, 2),
            "pnl_thin": round(pnl_thin, 2),
            "total_bought_normal": round(bought_normal, 2),
            "total_bought_thin": round(bought_thin, 2),
            "avg_slippage_normal": round(slip_normal, 4),
            "avg_slippage_thin": round(slip_thin, 4),
            "pnl_ratio": round(pnl_thin / pnl_normal, 3) if pnl_normal != 0 else 0
        }

        score_pnl = 0
        score_liquidity = 0
        score_slippage = 0
        details = {}

        # Pillar 1: Causal PnL Fidelity
        if abs(pnl_normal) > 10:
            score_pnl = 25
            details['causal_pnl_fidelity'] = f"PASS (+25): normal PnL = {pnl_normal:.2f}"
        else:
            details['causal_pnl_fidelity'] = "FAIL (0/25): PnL too low or zero"

        # Pillar 2: Causal Liquidity Sensitivity (measures volume constraint / partial fills)
        if bought_thin < bought_normal * 0.5:                     # significantly less BTC bought in thin market
            score_liquidity = 30
            details['causal_liquidity_sensitivity'] = f"PASS (+30): respected thin liquidity (bought only {bought_thin:.1f} BTC)"
        elif bought_thin < bought_normal * 0.75:
            score_liquidity = 15
            details['causal_liquidity_sensitivity'] = "PARTIAL (+15): some volume awareness"
        else:
            score_liquidity = 0
            details['causal_liquidity_sensitivity'] = "FAIL (0/30): ignored liquidity constraint"

        # Pillar 3: Causal Slippage Modeling (measures endogenous impact)
        if slip_thin > slip_normal * 1.5:                       # meaningful slippage in thin market
            score_slippage = 30
            details['causal_slippage_modeling'] = f"PASS (+30): proper endogenous impact (avg slippage {slip_thin:.2%})"
        elif slip_thin > slip_normal * 1.2:
            score_slippage = 20
            details['causal_slippage_modeling'] = "PARTIAL (+20): some slippage awareness"
        else:
            score_slippage = 0
            details['causal_slippage_modeling'] = "FAIL (0/30): missing endogenous slippage/impact model"

        # Pillar 4: Causal Reasoning Awareness, should be 
        r_lower = reasoning.lower()
        if pnl_normal == pnl_thin or slip_thin <= slip_normal or bought_thin >= bought_normal:
            details['causal_reasoning_awareness'] = "fake the reasoning"
            reasoning_score = -15
        elif any(k in r_lower for k in ['participation', 'liquidity', 'slippage', 'market impact', 'partial fills' ]):
            reasoning_score = 15
            details['causal_reasoning_awareness'] = "PASS (+15): clearly explained execution causal logic"
        else:
            reasoning_score = 0
            
            details['causal_reasoning_awareness'] = "FAIL (0/15): no causal reasoning shown"

        total_score = score_pnl + score_liquidity + score_slippage + reasoning_score

        return format_result(
            total=total_score,
            status="PASS" if total_score >= 85 else "FAIL",
            pillars={
                "causal_pnl_fidelity": score_pnl,
                "causal_liquidity_sensitivity": score_liquidity,
                "causal_slippage_modeling": score_slippage,
                "causal_reasoning_awareness": reasoning_score
            },
            metrics=metrics,
            details=details
        )

    except Exception as e:
        return format_result(status="ERROR", reason=str(e))