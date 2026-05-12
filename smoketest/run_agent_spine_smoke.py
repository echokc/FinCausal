import json

from agent.contract.contracts import build_contract_from_intent, validate_contract_completeness
from agent.graph import run_minimal_agent_flow
from agent.guardrails import run_hard_guardrails
from agent.invariants import run_invariant_checks
from agent.retrieval import (
    retrieve_causal_principles,
    retrieve_hazard_definitions,
    retrieve_hazard_ids_for_intent,
    retrieve_similar_contracts,
)
from agent.schemas import Intent
from eval.generation.data.fixture_generation import generate_recipe_fixtures
from smoketest.smoke_controls import SMOKE_CONTROLS


def _actionable_critic_once():
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
                        "claim": "Mock actionable concern.",
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
    intent = Intent(
        task_type="generate_code",
        domain="systematic_trading",
        input_data_description="Daily time-indexed feature table with date, regime_feature, and fwd_return columns.",
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

    contract = build_contract_from_intent(intent, use_local_knowledge=True)
    validation = validate_contract_completeness(contract)
    hazard_ids = retrieve_hazard_ids_for_intent(intent, severity=["high", "critical"])
    principles = retrieve_causal_principles("point in time online quantile threshold", domain="systematic_trading")
    hazards = retrieve_hazard_definitions("full sample quantile online regime", severity=["high", "critical"])
    similar = retrieve_similar_contracts(intent, top_k=1)

    bad_code = """
import pandas as pd

df = pd.read_csv(DATA_PATH)
q = df["regime_feature"].quantile(0.95)
regime_df = df.copy()
""".strip()
    good_code = """
import pandas as pd

df = pd.read_csv(DATA_PATH)
q = df["regime_feature"].expanding(min_periods=20).quantile(0.95)
regime_df = df.copy()
""".strip()

    bad_guard = run_hard_guardrails(bad_code, contract)
    good_guard = run_hard_guardrails(good_code, contract)
    fixtures = generate_recipe_fixtures("s001_global_quantile_leakage")
    invariant_result = run_invariant_checks(good_code, contract, fixtures)
    control = SMOKE_CONTROLS["s001_global_quantile_leakage"]
    generated_flow = run_minimal_agent_flow(
        intent=intent,
        user_task="Generate point-in-time regime labels and assign regime_df.",
        generate_code=True,
        llm_invoke=lambda prompt: f"```python\n{control['positive']}\n```",
        fixtures=fixtures,
        contract_source="intent",
    )
    repaired_flow = run_minimal_agent_flow(
        intent=intent,
        code=control["negative"],
        repair_llm_invoke=lambda prompt: f"```python\n{control['positive']}\n```",
        max_repair_attempts=1,
        fixtures=fixtures,
        contract_source="intent",
    )
    critic_clean_flow = run_minimal_agent_flow(
        intent=intent,
        code=control["positive"],
        critic_llm_invoke=lambda prompt: '{"concerns": []}',
        run_critic=True,
        fixtures=fixtures,
        contract_source="intent",
    )
    critic_repaired_flow = run_minimal_agent_flow(
        intent=intent,
        code=control["positive"],
        critic_llm_invoke=_actionable_critic_once(),
        repair_llm_invoke=lambda prompt: f"```python\n{control['positive']}\n```",
        run_critic=True,
        max_repair_attempts=1,
        fixtures=fixtures,
        contract_source="intent",
    )

    checks = {
        "contract_valid": validation.ok,
        "retrieved_temporal_hazard": "global_quantile_for_online_signal" in hazard_ids,
        "principles_nonempty": principles.ok and bool(principles.result["items"]),
        "hazards_nonempty": hazards.ok and bool(hazards.result["items"]),
        "similar_contracts_nonempty": similar.ok and bool(similar.result["items"]),
        "bad_code_blocked": bad_guard.ok and not bad_guard.result["passed"],
        "good_code_passes": good_guard.ok and good_guard.result["passed"],
        "invariants_pass": invariant_result.ok and invariant_result.result["passed"],
        "output_contract_checked": invariant_result.ok
        and any(item["invariant_id"] == "output_contract" for item in invariant_result.result["checks"]),
        "prefix_invariance_checked": invariant_result.ok
        and any(item["invariant_id"] == "prefix_invariance" for item in invariant_result.result["checks"]),
        "generated_flow_passes": generated_flow["final_status"] == "pass",
        "generated_flow_used_generator": generated_flow["generation_result"].get("ok") is True,
        "repaired_flow_passes": repaired_flow["final_status"] == "pass",
        "repaired_flow_used_repair": repaired_flow["repair_result"].get("ok") is True,
        "critic_clean_flow_passes": critic_clean_flow["final_status"] == "pass",
        "critic_clean_flow_used_critic": critic_clean_flow["critic_result"].get("ok") is True,
        "critic_repaired_flow_passes": critic_repaired_flow["final_status"] == "pass",
        "critic_repaired_flow_used_repair": critic_repaired_flow["repair_result"].get("ok") is True,
    }
    output = {
        "ok": all(checks.values()),
        "checks": checks,
        "contract_source": contract.metadata.get("source"),
        "known_hazards": contract.known_hazards,
        "bad_guard_violations": bad_guard.result["violations"] if bad_guard.ok else [],
        "invariant_result": invariant_result.result if invariant_result.ok else None,
        "generated_flow_status": generated_flow["final_status"],
        "repaired_flow_status": repaired_flow["final_status"],
        "critic_clean_flow_status": critic_clean_flow["final_status"],
        "critic_repaired_flow_status": critic_repaired_flow["final_status"],
    }
    print(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    return 0 if output["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
