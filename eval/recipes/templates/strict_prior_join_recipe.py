import os
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from eval.case_schema.generated_case_schema import DiversitySpec
from eval.generation.prompts.prompt_models import PromptFileSpec, PromptSpec
from eval.generation.prompts.temporal_prompt_renderer import render_temporal_prompt
from eval.recipes.components.behavior_models import TemporalBehaviorSpec
from eval.recipes.components.case_manifest_builders import all_files_exist, column_metadata, schema_file, schema_manifest
from eval.recipes.components.judge_contract_builders import (
    causal_contract,
    judge_config,
    output_semantic,
    reference_behavior,
    witness_map,
)
from eval.recipes.components.prompt_models import PromptVariant


@dataclass(frozen=True)
class StrictPriorJoinRecipe:
    behavior_spec: TemporalBehaviorSpec
    schema_variants: Dict[str, Dict[str, Any]]
    prompt_variants: Dict[str, PromptVariant]
    distribution_variants: List[str]
    mechanism_variant: str
    intervention_name: str
    intervention_type: str
    output_dataframe_name: str
    required_output_columns: List[str]
    allowed_information: List[str]
    forbidden_information: List[str]
    invariance_requirement: Dict[str, Any]
    sensitivity_requirement: Dict[str, Any]
    known_traps: List[Dict[str, str]]
    causal_solution_strategy: str
    expected_failure_strategy: str
    positive_control_id: str
    negative_control_id: str

    @property
    def pillar(self) -> str:
        return self.behavior_spec.pillar

    @property
    def default_prompt_variant(self) -> str:
        return self.behavior_spec.default_prompt_variant

    @property
    def default_schema_variant(self) -> str:
        return self.behavior_spec.default_schema_variant

    def sample_diversity_spec(self, random_module) -> DiversitySpec:
        prompt_variant = random_module.choice(list(self.prompt_variants))
        prompt = self.prompt_variants[prompt_variant]
        return DiversitySpec(
            schema_variant=random_module.choice(list(self.schema_variants)),
            prompt_variant=prompt_variant,
            distribution_variant=random_module.choice(self.distribution_variants),
            mechanism_variant=self.mechanism_variant,
            difficulty_level=prompt.difficulty_level,
            surface_domain=random_module.choice(prompt.surface_domains),
        )

    def schema_variant(self, params: Dict[str, Any]) -> Dict[str, Any]:
        diversity = params.get("diversity", {})
        return self.schema_variants.get(
            diversity.get("schema_variant", self.default_schema_variant),
            self.schema_variants[self.default_schema_variant],
        )

    def build_prompt(self, params: Any, data_path: str) -> str:
        params_dict = params.__dict__ if hasattr(params, "__dict__") else params
        diversity = params_dict.get("diversity", {})
        schema_variant = self.schema_variant(params_dict)
        prompt = self.prompt_variants.get(
            diversity.get("prompt_variant", self.default_prompt_variant),
            self.prompt_variants[self.default_prompt_variant],
        )
        trade_id_col = schema_variant["trade_id_col"]
        trade_time_col = schema_variant["trade_time_col"]
        news_id_col = schema_variant["news_id_col"]
        news_time_col = schema_variant["news_time_col"]
        content_col = schema_variant["content_col"]
        return render_temporal_prompt(
            PromptSpec(
                framing=prompt.framing,
                data_path=data_path,
                files=[
                    PromptFileSpec(
                        variable="trade_df",
                        filename="clean_0.csv",
                        columns=[trade_id_col, trade_time_col, "symbol", "price", "side"],
                    ),
                    PromptFileSpec(
                        variable="news_df",
                        filename="clean_1.csv",
                        columns=[news_id_col, news_time_col, content_col],
                    ),
                ],
                load_snippet=(
                    "trade_df = pd.read_csv(os.path.join(data_path, 'clean_0.csv'))\n"
                    "news_df  = pd.read_csv(os.path.join(data_path, 'clean_1.csv'))"
                ),
                task_steps=[
                    (
                        f"Produce a merged DataFrame (`{self.output_dataframe_name}`) where each trade row "
                        "is joined to news published strictly before that trade's timestamp."
                    )
                ],
                critical_requirements=[
                    prompt.causal_requirement,
                    f"Use `{news_time_col} < {trade_time_col}` with no exceptions.",
                    "No ex-post news: news content must not describe events that occurred before the trade but were reported after.",
                    "Handle trades with no prior news by filling news columns with NaN.",
                ],
                required_output_columns=self.required_output_columns,
                output_dataframe_name=self.output_dataframe_name,
            )
        )

    def schema_manifest(self, variant_a: Dict[str, Any], variant_b: Dict[str, Any]) -> Dict[str, Any]:
        schema_variant = self.schema_variant(variant_a["params"])
        return schema_manifest(
            [
                schema_file(
                    filename="clean_0.csv",
                    role=schema_variant["trade_file_role"],
                    primary_time_column=schema_variant["trade_time_col"],
                    entity_column=schema_variant["trade_id_col"],
                    columns={
                        "trade_id": column_metadata("entity_id", "current"),
                        "timestamp": column_metadata("decision_timestamp", "current"),
                        schema_variant["trade_id_col"]: column_metadata("entity_id", "current"),
                        schema_variant["trade_time_col"]: column_metadata("decision_timestamp", "current"),
                        "symbol": column_metadata("asset", "current"),
                        "price": column_metadata("market_value", "current"),
                        "side": column_metadata("trade_side", "current"),
                    },
                ),
                schema_file(
                    filename="clean_1.csv",
                    role=schema_variant["news_file_role"],
                    primary_time_column=schema_variant["news_time_col"],
                    entity_column=schema_variant["news_id_col"],
                    columns={
                        "news_id": column_metadata("entity_id", "current"),
                        "publish_time": column_metadata("information_available_time", "current"),
                        "content": column_metadata("text_signal", "available_at_publish_time"),
                        schema_variant["news_id_col"]: column_metadata("entity_id", "current"),
                        schema_variant["news_time_col"]: column_metadata("information_available_time", "current"),
                        schema_variant["content_col"]: column_metadata("text_signal", "available_at_publish_time"),
                    },
                ),
            ]
        )

    def _contamination_context(self, variant_a: Dict[str, Any], variant_b: Dict[str, Any]) -> Dict[str, Any]:
        params = variant_a["params"]
        news_b = pd.read_csv(os.path.join(variant_b["data_path"], "clean_1.csv"))
        contamination_records = params.get("contamination_records", [])
        contaminated_news_ids = [record["news_id"] for record in contamination_records]
        affected_trade_ids = [record["target_trade_id"] for record in contamination_records]
        suspicious_rows = news_b[news_b["news_id"].isin(contaminated_news_ids)].index.tolist()
        first_suspicious = int(min(suspicious_rows)) if suspicious_rows else 0
        last_suspicious = int(max(suspicious_rows)) if suspicious_rows else min(len(news_b) - 1, 10)
        return {
            "params": params,
            "contamination_records": contamination_records,
            "contaminated_news_ids": contaminated_news_ids,
            "affected_trade_ids": affected_trade_ids,
            "suspicious_rows": suspicious_rows,
            "first_suspicious": first_suspicious,
            "last_suspicious": last_suspicious,
        }

    def witness_map(self, variant_a: Dict[str, Any], variant_b: Dict[str, Any]) -> Dict[str, Any]:
        ctx = self._contamination_context(variant_a, variant_b)
        return witness_map(
            interventions=[
                {
                    "name": self.intervention_name,
                    "universe": "B",
                    "type": self.intervention_type,
                    "contaminated_news_ids": ctx["contaminated_news_ids"],
                    "affected_trade_ids": ctx["affected_trade_ids"],
                    "contamination_count": len(ctx["contamination_records"]),
                    "contamination_ratio": ctx["params"].get("contamination_ratio"),
                }
            ],
            must_match=[
                {
                    "name": "strict_prior_news_join",
                    "rule": "For every non-null joined news row, publish_time must be strictly before timestamp.",
                    "output_semantics": ["news_join", "time_delta_minutes"],
                    "max_violation_count": 0,
                }
            ],
            inspect_windows=[
                {
                    "name": "polluted_news_rows",
                    "rows": {"start_idx": ctx["first_suspicious"], "end_idx": ctx["last_suspicious"]},
                    "file": "clean_1.csv",
                    "universe": "B",
                    "reason": "Injected future-published news rows are the highest-risk evidence region.",
                }
            ],
            known_traps=self.known_traps,
            ground_truth_pollution=ctx["contamination_records"][:50],
        )

    def judge_config(self, variant_a: Dict[str, Any], variant_b: Dict[str, Any]) -> Dict[str, Any]:
        schema_variant = self.schema_variant(variant_a["params"])
        return judge_config(
            required_output_semantics={
                "trade_timestamp": output_semantic(["timestamp", "trade_timestamp", schema_variant["trade_time_col"]]),
                "news_publish_time": output_semantic(
                    ["publish_time", "news_publish_time", schema_variant["news_time_col"]],
                    required=False,
                ),
                "join_time_delta": output_semantic(
                    ["time_delta_minutes", "lag_minutes", "news_lag_minutes"],
                    required=False,
                ),
                "news_id": output_semantic(["news_id", "matched_news_id", schema_variant["news_id_col"]], required=False),
                "trade_id": output_semantic(["trade_id", "matched_trade_id", schema_variant["trade_id_col"]], required=False),
            },
            probes=[
                {
                    "type": "timestamp_order",
                    "left_semantic": "news_publish_time",
                    "right_semantic": "trade_timestamp",
                    "operator": "<",
                    "allow_null_left": True,
                    "hard_fail": True,
                },
                {"type": "nonnegative_delta", "target_semantic": "join_time_delta", "hard_fail": True},
                {
                    "type": "contaminated_entity_exclusion",
                    "entity_semantic": "news_id",
                    "intervention_ref": self.intervention_name,
                    "hard_fail": True,
                },
            ],
        )

    def reference_behavior(self, variant_a: Dict[str, Any], variant_b: Dict[str, Any]) -> Dict[str, Any]:
        return reference_behavior(
            causal_solution_strategy=self.causal_solution_strategy,
            expected_failure_strategy=self.expected_failure_strategy,
            positive_control_id=self.positive_control_id,
            negative_control_id=self.negative_control_id,
        )

    def causal_contract(self, variant_a: Dict[str, Any], variant_b: Dict[str, Any]) -> Dict[str, Any]:
        return causal_contract(
            allowed=self.allowed_information,
            forbidden=self.forbidden_information,
            invariance=[self.invariance_requirement],
            sensitivity=[self.sensitivity_requirement],
        )

    def quality_checks(self, variant_a: Dict[str, Any], variant_b: Dict[str, Any]) -> Dict[str, Any]:
        ctx = self._contamination_context(variant_a, variant_b)
        trades_a = pd.read_csv(os.path.join(variant_a["data_path"], "clean_0.csv"))
        trades_b = pd.read_csv(os.path.join(variant_b["data_path"], "clean_0.csv"))
        news_a = pd.read_csv(os.path.join(variant_a["data_path"], "clean_1.csv"))
        news_b = pd.read_csv(os.path.join(variant_b["data_path"], "clean_1.csv"))
        return {
            "data_written": all_files_exist(variant_a["data_path"], ["clean_0.csv", "clean_1.csv"])
            and all_files_exist(variant_b["data_path"], ["clean_0.csv", "clean_1.csv"]),
            "a_b_shape_compatible": len(trades_a) == len(trades_b) and len(news_a) == len(news_b),
            "intervention_applied": len(ctx["contamination_records"]) > 0,
            "witness_rows_exist": bool(ctx["suspicious_rows"]),
        }

    def task_columns(self, variant_a: Dict[str, Any], variant_b: Dict[str, Any]) -> Dict[str, Any]:
        schema_variant = self.schema_variant(variant_a["params"])
        return {
            "trade_id": schema_variant["trade_id_col"],
            "trade_time": schema_variant["trade_time_col"],
            "news_id": schema_variant["news_id_col"],
            "news_time": schema_variant["news_time_col"],
            "content": schema_variant["content_col"],
        }

    def prompt_leakage_values(self, variant_a: Dict[str, Any], variant_b: Dict[str, Any]) -> List[Any]:
        ctx = self._contamination_context(variant_a, variant_b)
        return ctx["contaminated_news_ids"][:5] + ctx["affected_trade_ids"][:5]
