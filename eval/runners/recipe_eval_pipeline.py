import argparse
import json

from eval.evaluation.pipeline import run_recipe_eval_pipeline
from eval.smoketest.smoke_controls import SMOKE_CONTROLS


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the recipe eval orchestrator.")
    parser.add_argument("--behavior-key", choices=sorted(SMOKE_CONTROLS), default=None)
    parser.add_argument(
        "--candidate-source",
        choices=["controls", "llm_shaped_controls", "llm"],
        default="controls",
    )
    parser.add_argument("--llm-samples", type=int, default=1)
    parser.add_argument("--config-path", default="config.yaml")
    parser.add_argument("--timeout-seconds", type=int, default=10)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--case-manifest-root", default=None)
    parser.add_argument("--repair-attempts", type=int, default=1)
    args = parser.parse_args()

    records = run_recipe_eval_pipeline(
        behavior_key=args.behavior_key,
        candidate_source=args.candidate_source,
        llm_samples=args.llm_samples,
        config_path=args.config_path,
        timeout_seconds=args.timeout_seconds,
        output_path=args.output_path,
        case_manifest_root=args.case_manifest_root,
        repair_attempts=args.repair_attempts,
    )
    print(json.dumps(records, indent=2, ensure_ascii=False, default=str))
    failed = [record for record in records if record["decision"] != "pass"]
    return 1 if args.candidate_source == "llm" and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
