import argparse
import json
from typing import Any, Dict

from eval.scoring.generic_recipe_scorer import GenericRecipeScorer, RecipeScore
from smoketest.smoke_controls import SMOKE_CONTROLS


def _score_to_dict(score: RecipeScore) -> Dict[str, Any]:
    return {
        "total": score.total,
        "status": score.status,
        "decision": score.decision,
        "failure_type": score.failure_type,
        "vote_summary": score.vote_summary,
        "outputs": score.outputs,
        "errors": score.errors,
        "probe_results": score.probe_results,
        "judge_verdict": score.judge_verdict,
        "failure_origin": score.failure_origin,
        "diagnostics": score.diagnostics,
        "extracted_code": score.extracted_code,
    }


def run_recipe_smoke(behavior_key: str | None = None, timeout_seconds: int = 10) -> Dict[str, Any]:
    scorer = GenericRecipeScorer(timeout_seconds=timeout_seconds)
    keys = [behavior_key] if behavior_key else list(SMOKE_CONTROLS)
    results: Dict[str, Any] = {}

    for key in keys:
        control = SMOKE_CONTROLS[key]
        recipe = control["recipe"]
        fixtures = control["fixtures"]()
        positive = scorer.score(recipe, control["positive"], fixtures)
        negative = scorer.score(recipe, control["negative"], fixtures)
        llm_shaped = None
        if "llm_shaped_positive" in control:
            llm_shaped = scorer.score_raw_response(recipe, control["llm_shaped_positive"], fixtures)
        results[key] = {
            "positive": _score_to_dict(positive),
            "negative": _score_to_dict(negative),
            "ok": (
                positive.status == "PASS"
                and negative.status == "FAIL"
                and (llm_shaped is None or llm_shaped.status == "PASS")
            ),
        }
        if llm_shaped is not None:
            results[key]["llm_shaped_positive"] = _score_to_dict(llm_shaped)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run generic recipe smoke controls.")
    parser.add_argument("--behavior-key", choices=sorted(SMOKE_CONTROLS), default=None)
    parser.add_argument("--timeout-seconds", type=int, default=10)
    args = parser.parse_args()

    results = run_recipe_smoke(args.behavior_key, timeout_seconds=args.timeout_seconds)
    print(json.dumps(results, indent=2, default=str))
    return 0 if all(result["ok"] for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
