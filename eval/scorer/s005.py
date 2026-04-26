import pandas as pd
import numpy as np
import os
import re
import shutil
import traceback

from core.sandbox import run_in_docker_sandbox, format_result

def setup_s005_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_mins = 30 * 24 * 60
    returns = rng.normal(0.00005, 0.0006, n_mins)
    df = pd.DataFrame({
        'timestamp': pd.date_range('2026-01-01', periods=n_mins, freq='1min'),
        'price': 50000 * np.exp(np.cumsum(returns))
    })
    return df
    
def score_s005(generated_code: str, reasoning: str) -> dict:

    df_a = setup_s005_data()
  
    df_b = df_a.copy()
    returns = df_b['price'].pct_change()
    df_b['price'] = df_b['price'].iloc[0] * (1 + returns * 2).cumprod()

    suffix = """
# Evaluation snippet
import pandas as pd
eval_result = pd.DataFrame([{'val': float(trade_stop_loss_pct)}])
"""
    full_code = generated_code + suffix

    try:
        res_a = run_in_docker_sandbox(full_code, {'DATA_PATH': df_a}, result_keys=['eval_result'])
        res_b = run_in_docker_sandbox(full_code, {'DATA_PATH': df_b}, result_keys=['eval_result'])

        val_a = float(res_a.iloc[0]['val'])
        val_b = float(res_b.iloc[0]['val'])
        
    
        ratio = val_b / val_a if val_a != 0 else 0
        
        metrics = {"sl_a": val_a, "sl_b": val_b, "ratio": ratio}
        details = {}
        score_logic = 0
        score_scaling = 0

        # causal 
        if abs(ratio - 2.0) < 0.01:
            score_logic = 50
            details['logic'] = "PASS (+50): Logic is data-driven. Correctly responded to volatility scaling."
        elif abs(ratio - 1.0) < 0.01:
            return format_result(status="FAIL", reason="Hardcoded output: Model ignored the data values.", metrics=metrics)
        else:
            details['logic'] = f"CAUTION: Unexpected scaling ratio {ratio:.2f}."

        # time scaling 
        σ_1m = df_a['price'].pct_change().std()
        theoretical_sl = σ_1m * np.sqrt(30) * 2 * 100 # 2 sigma, percentage
        
        error_pct = abs(val_a - theoretical_sl) / theoretical_sl
        
        if error_pct < 0.05:
            score_scaling = 50
            details['scaling'] = "PASS (+50): Accurate Square Root of Time scaling."
        elif abs(val_a - (σ_1m * 2 * 100)) < 0.05:
            details['scaling'] = "FAIL (0/50): No time scaling. Used 1-minute volatility."
        elif abs(val_a - (σ_1m * 30 * 2 * 100)) < 0.05:
            details['scaling'] = "FAIL (0/50): Linear scaling error (multiplied by 30 instead of sqrt(30))."
        else:
            details['scaling'] = f"FAIL (0/50): Significant math error. Deviation: {error_pct:.2%}"

        total_score = score_logic + score_scaling
        return format_result(
            total=total_score,
            status="PASS" if total_score >= 100 else "FAIL",
            pillars={"causal_logic": score_logic, "time_scaling": score_scaling},
            metrics=metrics,
            details=details
        )

    except Exception as e:
        return format_result(status="ERROR", reason=str(e))