import argparse
import json
import os

from eval.generation.data.fixture_generation import generate_recipe_fixtures
from eval.generation.data.fixture_io import write_recipe_fixtures
from eval.generation.data.fixture_quality import data_quality_report
from eval.generation.data.fixture_registry import RECIPE_DATA_GENERATORS


def main() -> int:
    from smoketest.smoke_controls import SMOKE_CONTROLS

    parser = argparse.ArgumentParser(description="Generate recipe universe data.")
    parser.add_argument("--behavior-key", choices=sorted(RECIPE_DATA_GENERATORS), default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    keys = [args.behavior_key] if args.behavior_key else list(RECIPE_DATA_GENERATORS)
    reports = {}
    for key in keys:
        recipe = SMOKE_CONTROLS[key]["recipe"]
        fixtures = generate_recipe_fixtures(key, seed=args.seed)
        quality = data_quality_report(recipe, fixtures)
        payload = {"quality": quality.__dict__}
        if args.output_root:
            payload["paths"] = write_recipe_fixtures(fixtures, os.path.join(args.output_root, key))
        reports[key] = payload

    print(json.dumps(reports, indent=2, default=str))
    return 0 if all(report["quality"]["missing_universes"] == [] and report["quality"]["extra_universes"] == [] for report in reports.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
