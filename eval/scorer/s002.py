import pandas as pd
import numpy as np
import os
import re
import shutil
import traceback

from core.sandbox import run_in_docker_sandbox, format_result

def score_s002(generated_code: str, reasoning: str) -> dict:
    try:
        trades_base = pd.DataFrame([
            {"trade_id": "T101", "timestamp": "2026-03-07 10:00:05", "symbol": "BTC"},
            {"trade_id": "T102", "timestamp": "2026-03-07 14:00:30", "symbol": "BTC"}
        ])
        news_base = pd.DataFrame([
            {"news_id": "N_SH_01", "publish_time": "2026-03-07 08:00:00", "content": "FED Decision at 14:00"},
            {"news_id": "N_LG_01", "publish_time": "2026-03-07 10:10:00", "content": "Flash Crash Ex-post at 10:00"},
            {"news_id": "N_NOISE", "publish_time": "2026-03-07 13:50:00", "content": "Cafeteria lunch menu updated."}
        ])

        news_b = news_base.copy()
        news_b.loc[news_b['news_id'] == 'N_SH_01', 'publish_time'] = "2026-03-07 10:05:00"
     
        news_c = news_base.copy()
        news_c.loc[news_c['news_id'] == 'N_LG_01', 'content'] = "POISON_DATA_SHOCK"

        res_a = run_in_docker_sandbox(generated_code, {'TRADE_PATH': trades_base, 'NEWS_PATH': news_base})
        res_b = run_in_docker_sandbox(generated_code, {'TRADE_PATH': trades_base, 'NEWS_PATH': news_b})
        res_c = run_in_docker_sandbox(generated_code, {'TRADE_PATH': trades_base, 'NEWS_PATH': news_c})

        if res_a is None or res_b is None or res_c is None:
            return format_result(status="FAIL", reason="Execution Error: No output DataFrame detected.")

        def get_content(df, tid):
            row = df[df['trade_id'] == tid]
            return str(row.to_dict(orient='records')).lower()

        score_physical = 0
        score_semantic = 0
        score_noise = 0
        penalty_eng = 0
        details = {}

        # Physical CDI
        t101_a = get_content(res_a, 'T101')
        t101_c = get_content(res_c, 'T101')
        if t101_a == t101_c and 'n_lg_01' not in t101_a:
            score_physical = 30
            details['physical_cdi'] = "PASS (+30): No look-ahead leakage detected."
        else:
            return format_result(
                status="CRITICAL FAIL", 
                reason="Physical CDI Violation: T101 contaminated by future news."
            )

        # Semantic CDI
        t102_a = get_content(res_a, 'T102')
        has_anchor_t102 = 'n_sh_01' in t102_a
        wrong_anchor_t101 = 'n_sh_01' in t101_a
        
        if has_anchor_t102 and not wrong_anchor_t101:
            score_semantic = 40
            details['semantic_cdi'] = "PASS (+40): Correctly identified 08:00 as anchor for T102 but not for T101."
        else:
            reason = []
            if not has_anchor_t102: reason.append("T102 missed 08:00 anchor")
            if wrong_anchor_t101: reason.append("T101 wrongly associated 08:00 pre-announcement")
            details['semantic_cdi'] = f"FAIL: {', '.join(reason)}"

        # Noise CDI
        if 'n_noise' not in t102_a:
            score_noise = 30
            details['noise_cdi'] = "PASS (+30): Filtered out irrelevant semantic noise."
        else:
            details['noise_cdi'] = "FAIL: Semantic pollution detected (Lunch Menu associated)."

        # Engineering check
        r = reasoning.lower()
        if ".assign(key=1)" in generated_code or "iterrows" in generated_code:
            penalty_eng = -20
            details['engineering'] = "WARNING (-20): Inefficient implementation (Cross-join/Iterrows)."

        total_score = max(0, score_physical + score_semantic + score_noise + penalty_eng)
        status = "PASS" if total_score >= 80 else "CAUTION" if total_score >= 60 else "FAIL"

        return format_result(
            total=total_score,
            status=status,
            reason="Evaluation Completed",
            pillars={
                "physical_cdi": score_physical,
                "semantic_cdi": score_semantic,
                "noise_cdi": score_noise,
                "engineering_penalty": penalty_eng
            },
            metrics={
                "physical_leakage": 'n_lg_01' in t101_a,
                "anchor_alignment": has_anchor_t102 and not wrong_anchor_t101,
                "noise_leakage": 'n_noise' in t102_a
            },
            details=details
        )

    except Exception as e:
        traceback.print_exc()
        return format_result(
            status="ERROR", 
            reason=str(e), 
            details={"traceback": traceback.format_exc()}
        )