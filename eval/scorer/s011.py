import pandas as pd
import numpy as np
import os
import re
import shutil
import traceback

from core.sandbox import run_in_docker_sandbox, format_result

def setup_s011_data():
    import pandas as pd
    import numpy as np

    rng = np.random.default_rng(42)
    n = 300
    ts = pd.date_range('2026-01-01', periods=n, freq='1min')
    base = 50000 + rng.normal(0, 12, n).cumsum()

    # 1. Classic poison + genuine breakout (core test)
    poisoned = base.copy()
    poisoned[80] = 39800          # extreme bad tick
    poisoned[140] = 50680         # real breakout (must survive)
    df_poisoned = pd.DataFrame({'timestamp': ts, 'close': poisoned})

    # 2. Clean (no outlier)
    clean = base.copy()
    clean[140] = 50680
    df_clean = pd.DataFrame({'timestamp': ts, 'close': clean})

    # 3. Stronger poison
    strong = base.copy()
    strong[80] = 37500
    strong[140] = 50750
    df_strong = pd.DataFrame({'timestamp': ts, 'close': strong})

    # 4. No-signal poison only (must produce ZERO false positives)
    no_signal = base.copy()
    no_signal[80] = 39800
    df_no_signal = pd.DataFrame({'timestamp': ts, 'close': no_signal})

    # 5. Multiple outliers (tests cumulative poisoning)
    multi = base.copy()
    multi[[60, 80, 100]] = [39500, 41000, 38500]
    multi[140] = 50680
    df_multi = pd.DataFrame({'timestamp': ts, 'close': multi})

    return df_poisoned, df_clean, df_strong, df_no_signal, df_multi



def score_s011(generated_code: str, reasoning: str) -> dict:
    (df_p, df_c, df_s, df_n, df_m) = setup_s011_data()

    suffix = """
# === S011 Evaluation Snippet (Docker-safe) ===
import pandas as pd
df = pd.read_csv(DATA_PATH)                    # force load
flags = get_protection_flags(df)               # call the required function
eval_result = pd.DataFrame([{
    'flag_breakout': bool(flags.iloc[140]) if len(flags) > 140 else False,
    'flag_poison':   bool(flags.iloc[80])  if len(flags) > 80  else False,
    'total_flags':   int(flags.sum())
}])
"""

    full_code = generated_code + suffix

    try:
        res = [run_in_docker_sandbox(full_code, {'DATA_PATH': df}, result_keys=['eval_result'])
               for df in (df_p, df_c, df_s, df_n, df_m)]

        if any(r is None for r in res):
            return format_result(status="ERROR", reason="Docker sandbox failed")

        f_p, f_c, f_s, f_n, f_m = [r.iloc[0] for r in res]

        metrics = {
            "breakout_poisoned": bool(f_p['flag_breakout']),
            "breakout_clean": bool(f_c['flag_breakout']),
            "breakout_strong": bool(f_s['flag_breakout']),
            "fp_no_signal": int(f_n['total_flags']),
            "breakout_multi": bool(f_m['flag_breakout'])
        }

        score_cdi = 0
        score_util = 0
        details = {}

        if (f_p['flag_breakout'] and f_c['flag_breakout'] and 
            f_s['flag_breakout'] and f_m['flag_breakout']) and f_n['total_flags'] == 0:
            score_cdi = 75
            details['robustness_cdi'] = "PASS (+75): fully robust"
        elif not f_p['flag_breakout'] or not f_s['flag_breakout'] or not f_m['flag_breakout']:
            score_cdi = 0
            details['robustness_cdi'] = "CRITICAL FAIL (0/75): poisoned by outlier"
        elif f_n['total_flags'] > 2:
            score_cdi = 0
            details['robustness_cdi'] = f"CRITICAL FAIL (0/75): false positives ({f_n['total_flags']})"
        else:
            score_cdi = 30
            details['robustness_cdi'] = "PARTIAL (+30)"

        if f_p['flag_breakout'] and f_c['flag_breakout']: 
            score_util += 12.5
        if f_n['total_flags'] == 0: 
            score_util += 12.5
        details['utility'] = f"Utility +{score_util}/25"

        total = score_cdi + score_util

        return format_result(
            total=total,
            status="PASS" if total >= 80 else "FAIL",
            pillars={"robustness_cdi": score_cdi, "functional_utility": score_util},
            metrics=metrics,
            details=details
        )

    except Exception as e:
        return format_result(status="ERROR", reason=str(e))