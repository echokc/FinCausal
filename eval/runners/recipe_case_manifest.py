import argparse
import json

from eval.evaluation.case_manifest import build_and_write_recipe_case
from eval.smoketest.smoke_controls import SMOKE_CONTROLS


def main() -> int:
    parser = argparse.ArgumentParser(description="Build recipe case manifests.")
    parser.add_argument("--behavior-key", choices=sorted(SMOKE_CONTROLS), default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="/tmp/fincausal_recipe_cases")
    args = parser.parse_args()

    keys = [args.behavior_key] if args.behavior_key else list(SMOKE_CONTROLS)
    results = {}
    for key in keys:
        recipe = SMOKE_CONTROLS[key]["recipe"]
        build = build_and_write_recipe_case(recipe, output_root=args.output_root, seed=args.seed)
        results[key] = {
            "manifest_path": build.manifest_path,
            "case_id": build.manifest["case_id"],
            "status": build.manifest["quality_report"]["status"],
            "validation_errors": build.manifest["quality_report"].get("validation_errors", []),
            "data_paths": build.data_paths,
        }

    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    return 0 if all(not result["validation_errors"] for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
