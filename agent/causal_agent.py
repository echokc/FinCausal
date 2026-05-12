import argparse
import json

from agent.graph import run_minimal_agent_flow
from agent.schemas import Intent
from eval.generation.data.fixture_generation import generate_recipe_fixtures
from smoketest.smoke_controls import SMOKE_CONTROLS


def _mock_llm(code: str):
    def invoke(prompt: str) -> str:
        return f"```python\n{code}\n```"

    return invoke


def _mock_critic_no_concerns(prompt: str) -> str:
    return '{"concerns": []}'


def _mock_critic_actionable_once():
    calls = {"count": 0}

    def invoke(prompt: str) -> str:
        calls["count"] += 1
        if calls["count"] > 1:
            return '{"concerns": []}'
        return json.dumps(
            {
                "concerns": [
                    {
                        "severity": "high",
                        "obligation_id": "full_sample_online_statistics",
                        "claim": "The critic requires a repair for demonstration.",
                        "code_evidence": "mock evidence",
                        "knowledge_evidence": "Online decisions must be point-in-time.",
                        "recommended_fix": "Use rolling or expanding estimates.",
                        "requires_repair": True,
                    }
                ]
            }
        )

    return invoke


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the minimal Layer 2 causal agent spine.")
    parser.add_argument("--behavior-key", choices=sorted(SMOKE_CONTROLS), default="s001_global_quantile_leakage")
    parser.add_argument("--candidate", choices=["positive", "negative"], default="positive")
    parser.add_argument("--contract-source", choices=["dev_recipe", "intent"], default="dev_recipe")
    parser.add_argument("--generate", action="store_true", help="Generate code from the contract instead of using a control.")
    parser.add_argument("--mock-generator", action="store_true", help="Use the positive control as a deterministic mock LLM.")
    parser.add_argument("--mock-repair", action="store_true", help="Use the positive control as a deterministic mock repair LLM.")
    parser.add_argument("--run-critic", action="store_true")
    parser.add_argument("--mock-critic", choices=["none", "actionable"], default=None)
    parser.add_argument("--max-repair-attempts", type=int, default=0)
    parser.add_argument("--config-path", default="config.yaml")
    parser.add_argument("--timeout-seconds", type=int, default=10)
    args = parser.parse_args()

    control = SMOKE_CONTROLS[args.behavior_key]
    recipe = control["recipe"]
    code = control[args.candidate]
    try:
        fixtures = generate_recipe_fixtures(recipe.behavior_key)
    except KeyError:
        fixtures = control["fixtures"]()

    intent = None
    if args.contract_source == "intent":
        intent = Intent(
            task_type="generate_code",
            domain="systematic_trading",
            input_data_description=(
                "Daily time-indexed market feature data with columns date, regime_feature, and fwd_return."
            ),
            requested_output="A dataframe named regime_df aligned to input rows with a regime label column.",
            explicit_user_constraints=[
                "Use point-in-time information only.",
                "Avoid full-sample statistics for online row decisions.",
            ],
            assumed_runtime_context={"language": "python", "environment": "local_sandbox"},
            metadata={
                "time_column": "date",
                "output_variable_name": "regime_df",
                "output_kind": "dataframe",
            },
        )

    result = run_minimal_agent_flow(
        recipe=recipe if args.contract_source == "dev_recipe" else None,
        intent=intent,
        code="" if args.generate else code,
        user_task=(
            "Generate point-in-time regime labels from date-sorted regime_feature data and assign regime_df."
        ),
        generate_code=args.generate,
        llm_invoke=_mock_llm(code) if args.generate and args.mock_generator else None,
        repair_llm_invoke=_mock_llm(control["positive"]) if args.mock_repair else None,
        critic_llm_invoke=(
            _mock_critic_no_concerns
            if args.mock_critic == "none"
            else _mock_critic_actionable_once()
            if args.mock_critic == "actionable"
            else None
        ),
        run_critic=args.run_critic,
        max_repair_attempts=args.max_repair_attempts,
        config_path=args.config_path,
        fixtures=fixtures,
        timeout_seconds=args.timeout_seconds,
        contract_source=args.contract_source,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0 if result["final_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
