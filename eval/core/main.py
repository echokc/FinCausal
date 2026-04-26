import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from dotenv import load_dotenv
from scenarios.registry import SCENARIOS
from llm_factory import build_llm, load_config
from runner import EvalRunner

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(description="FinCausal-Eval")
    parser.add_argument("--scenarios", nargs="*", help="Scenario IDs to run (default: all)")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--force", action="store_true", help="Bypass cache")
    args = parser.parse_args()

    config = load_config("config.yaml")
    llm = build_llm(config)

    runner = EvalRunner(
        llm=llm,
        cache_dir=config["eval"]["cache_dir"],
        results_dir=config["eval"]["results_dir"],
    )
    runner.run_eval(
        scenarios=SCENARIOS,
        scenario_ids=args.scenarios,
        use_cache=not args.force,
        save_results=not args.no_save,
    )


if __name__ == "__main__":
    main()