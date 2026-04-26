import pandas as pd
import numpy as np
import os
import re
import shutil
import traceback

from core.sandbox import run_in_docker_sandbox, format_result

def setup_s006_neutral_data() -> pd.DataFrame:
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-04-21 10:00', periods=20, freq='min'),
        'side': ['BUY', 'SELL'] * 10,
        'qty': [0.5] * 20,
        'price': [505.0] * 20,
        'vol_1min': [0.0018] * 20,   
    })
    return df


def setup_s006_long_lowvol_data() -> pd.DataFrame:
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-04-21 10:00', periods=25, freq='min'),
        'side': ['BUY'] * 25,
        'qty': [0.5] * 25,
        'price': [505.0] * 25,
        'vol_1min': [0.0012] * 25,      
    })
    return df


def setup_s006_long_highvol_data() -> pd.DataFrame:
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-04-21 10:00', periods=25, freq='min'),
        'side': ['BUY'] * 25,
        'qty': [0.5] * 25,
        'price': [505.0] * 25,
        'vol_1min': [0.0036] * 25,      
    })
    return df


def score_s006(generated_code: str, reasoning: str) -> dict:
    df_neu = setup_s006_neutral_data()
    df_low = setup_s006_long_lowvol_data()
    df_high = setup_s006_long_highvol_data()

    suffix = """
# === Evaluation Snippet ===
import pandas as pd
eval_result = pd.DataFrame([{
    'net_position': float(net_position) if 'net_position' in locals() else None,
    'bid_price': float(bid_price) if 'bid_price' in locals() else None,
    'ask_price': float(ask_price) if 'ask_price' in locals() else None,
}])
"""

    full_code = generated_code + suffix

    try:
        res_neu = run_in_docker_sandbox(full_code, {'DATA_PATH': df_neu}, result_keys=['eval_result'])
        res_low = run_in_docker_sandbox(full_code, {'DATA_PATH': df_low}, result_keys=['eval_result'])
        res_high = run_in_docker_sandbox(full_code, {'DATA_PATH': df_high}, result_keys=['eval_result'])

        if res_neu is None or res_low is None or res_high is None:
            return format_result(status="ERROR", reason="Docker sandbox execution failed")

        # Extract results
        n_neu = float(res_neu.iloc[0]['net_position'])
        b_neu = float(res_neu.iloc[0]['bid_price'])
        a_neu = float(res_neu.iloc[0]['ask_price'])

        n_low = float(res_low.iloc[0]['net_position'])
        b_low = float(res_low.iloc[0]['bid_price'])
        a_low = float(res_low.iloc[0]['ask_price'])

        n_high = float(res_high.iloc[0]['net_position'])
        b_high = float(res_high.iloc[0]['bid_price'])
        a_high = float(res_high.iloc[0]['ask_price'])

        mid_neu = (b_neu + a_neu) / 2
        mid_low = (b_low + a_low) / 2
        mid_high = (b_high + a_high) / 2

        drop_low = mid_neu - mid_low
        drop_high = mid_neu - mid_high
        vol_scaling_ratio = drop_high / drop_low if drop_low != 0 else 0

        metrics = {
            "net_neutral": round(n_neu, 3),
            "bid_neutral": round(b_neu, 2),
            "ask_neutral": round(a_neu, 2),
            "net_long_low": round(n_low, 3),
            "bid_long_low": round(b_low, 2),
            "ask_long_low": round(a_low, 2),
            "net_long_high": round(n_high, 3),
            "bid_long_high": round(b_high, 2),
            "ask_long_high": round(a_high, 2),
            "mid_drop_lowvol": round(drop_low, 3),
            "mid_drop_highvol": round(drop_high, 3),
            "vol_scaling_ratio": round(vol_scaling_ratio, 3)
        }

        score_net = 0
        score_skew = 0
        score_vol = 0
        details = {}

        # Pillar 1: Net Position Accuracy 
        if abs(n_neu) <= 0.8:
            score_net += 10
            details['net_neutral'] = f"PASS (+10): neutral ≈ {n_neu:.2f}"
        if abs(n_low - 12.5) <= 0.8 and abs(n_high - 12.5) <= 0.8:
            score_net += 15
            details['net_long'] = f"PASS (+15): long ≈ +{n_low:.2f}"

        # Pillar 2: Downward Inventory Skew 
        if drop_low > 4.0 and b_low < b_neu - 3.0 and a_low < a_neu - 3.0:
            score_skew = 35
            details['skew'] = f"PASS (+35): correct downward skew (mid decrease {drop_low:.2f})"
        elif drop_low > 2.0:
            score_skew = 20
            details['skew'] = "PARTIAL (+20): partial skew"
        else:
            score_skew = 0
            details['skew'] = "FAIL (0/35): no inventory skew"

        # Pillar 3: Volatility Scaling
        if 2.6 < vol_scaling_ratio < 3.4:
            score_vol = 30
            details['vol_scaling'] = f"PASS (+30): volatility scaling (ratio={vol_scaling_ratio:.2f})"
        elif 1.5 < vol_scaling_ratio < 4.0:
            score_vol = 15
            details['vol_scaling'] = f"PARTIAL (+15): scaling but not accurate"
        else:
            score_vol = 0
            details['vol_scaling'] = "FAIL (0/30): lack volatility scaling"

        # Pillar 4: Reasoning Awareness
        r_lower = reasoning.lower()
        if any(k in r_lower for k in ['inventory', 'skew', 'volatility', 'deviation', 'systematic', 'risk']):
            reasoning_score = 10
            details['reasoning'] = "PASS (+10): recog systematic inventory skew "
        else:
            reasoning_score = 0
            details['reasoning'] = "FAIL (0/10)"

        total_score = score_net + score_skew + score_vol + reasoning_score

        return format_result(
            total=total_score,
            status="PASS" if total_score >= 85 else "FAIL",
            pillars={
                "net_position_accuracy": score_net,
                "inventory_skew": score_skew,
                "volatility_scaling": score_vol,
                "reasoning_awareness": reasoning_score
            },
            metrics=metrics,
            details=details
        )

    except Exception as e:
        return format_result(status="ERROR", reason=str(e))