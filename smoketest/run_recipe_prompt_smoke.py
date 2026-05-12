import argparse
import json
from typing import Any, Dict

from eval.generation.prompts.prompt_builders import build_prompt_from_recipe, prompt_quality_checks
from smoketest.smoke_controls import SMOKE_CONTROLS


def run_recipe_prompt_smoke(behavior_key: str | None = None) -> Dict[str, Any]:
    keys = [behavior_key] if behavior_key else list(SMOKE_CONTROLS)
    results = {}
    for key in keys:
        recipe = SMOKE_CONTROLS[key]["recipe"]
        prompt = build_prompt_from_recipe(recipe)
        checks = prompt_quality_checks(recipe, prompt)
        results[key] = {
            "ok": all(checks.values()),
            "checks": checks,
            "prompt_preview": prompt[:1200],
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test generic recipe prompt generation.")
    parser.add_argument("--behavior-key", choices=sorted(SMOKE_CONTROLS), default=None)
    args = parser.parse_args()

    results = run_recipe_prompt_smoke(args.behavior_key)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0 if all(result["ok"] for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
