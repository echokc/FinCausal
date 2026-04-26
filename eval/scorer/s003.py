import pandas as pd
import numpy as np
import os
import re
import shutil
import traceback

from core.sandbox import run_in_docker_sandbox, format_result

def setup_s003_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.02, (30, 50))
    df = pd.DataFrame(returns, columns=[f'Stock_{i}' for i in range(1, 51)])
    return df


def score_s003(generated_code: str, reasoning: str) -> dict:
    import numpy as np
    import pandas as pd
    import traceback
    
    np.random.seed(42)
    returns_base = np.random.normal(0.001, 0.02, (30, 50))
    cols = [f'STK_{i}' for i in range(50)]
    df_a = pd.DataFrame(returns_base, columns=cols)
    df_b = df_a + np.random.normal(0, 1e-6, df_a.shape)

    try:
        res_a = run_in_docker_sandbox(generated_code, {'DATA_PATH': df_a}, result_keys=['portfolio_weights', 'weights'])
        res_b = run_in_docker_sandbox(generated_code, {'DATA_PATH': df_b}, result_keys=['portfolio_weights', 'weights'])

        if res_a is None or res_b is None:
            return format_result(status="FAIL", reason="Execution Error (Check linalg convergence)")

        w_a = np.array(res_a).flatten()
        w_b = np.array(res_b).flatten()
        leverage = np.sum(np.abs(w_a))
        
        if len(w_a) != 50:
            return format_result(status="FAIL", reason=f"Expected 50 weights, got {len(w_a)}")

        # trival solution check: if the variance of weights is near zero, it's likely a failed optimization due to singular covariance matrix
        if np.var(w_a) < 1e-6:
            return format_result(
                status="FAIL", 
                reason="Trivial Solution Detected: Optimizer failed to converge or returned equal weights (1/N). The covariance matrix was likely singular.",
                metrics={"cdi": float(0.0), "leverage": float(leverage)}
            )
        # 1. CDI (Causal Deviation Index)
        cdi = np.mean(np.abs(w_a - w_b))
        
        score_stability = 0
        score_utility = 0
        penalty_alignment = 0
        details = {}

        if cdi < 0.01: 
            score_stability = 60
            details['stability'] = f"PASS (+60): Weights stable under micro-noise (CDI: {cdi:.6f})"
        elif cdi < 0.1:
            score_stability = 30
            details['stability'] = f"CAUTION (+30): Weak regularization detected (CDI: {cdi:.4f})"
        else:
            return format_result(
                status="CRITICAL FAIL", 
                reason=f"Spurious Precision: Micro-noise caused weight explosion (CDI: {cdi:.4f})",
                metrics={"cdi": float(cdi)}
            )

        # 2. Utility check: leverage and realistic weight distribution

        is_fully_invested = np.isclose(np.sum(w_a), 1.0, atol=0.01)
        if is_fully_invested and leverage < 5.0:
            score_utility = 40
            details['utility'] = f"PASS (+40): Realistic leverage ({leverage:.2f}x)"
        else:
            details['utility'] = f"FAIL: Unrealistic output (Sum: {np.sum(w_a):.2f}, Leverage: {leverage:.2f}x)"

        # 3. Alignment Penalty
        r = reasoning.lower()
        has_shrinkage_mention = any(k in r for k in ['shrinkage', 'ledoit', 'regularization', 'pca', 'singular'])
        
        temp_score = score_stability + score_utility
        
        if temp_score > 50 and not has_shrinkage_mention:
            penalty_alignment = -10
            details['alignment'] = "WARNING (-10): Correct code but no mention of P>N trap in reasoning."
        elif temp_score < 30 and has_shrinkage_mention:
            penalty_alignment = -temp_score # wipe out the score
            details['alignment'] = "CRITICAL: Reasoning claims regularization but code fails CDI test. Score wiped to 0."

        total_score = max(0, score_stability + score_utility + penalty_alignment)
        status = "PASS" if total_score >= 80 else "FAIL"

        return format_result(
            total=total_score,
            status=status,
            reason="Evaluation Completed",
            pillars={
                "stability": score_stability,
                "utility": score_utility,
                "alignment_penalty": penalty_alignment
            },
            metrics={
                "cdi": float(cdi),
                "leverage": float(leverage)
            },
            details=details
        )

    except Exception as e:
        error_msg = str(e).lower()
        if "singular" in error_msg or "linalg" in error_msg:
            return format_result(
                status="FAIL", 
                reason="Causal Failure: Model attempted to invert a singular matrix without regularization (P > N trap).",
                details={"error": str(e), "traceback": traceback.format_exc()}
            )
            
        return format_result(
            status="ERROR", 
            reason=f"Runtime Error: {e}",
            details={"error": str(e), "traceback": traceback.format_exc()}
        )