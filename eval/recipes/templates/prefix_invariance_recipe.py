import os
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from eval.case_schema.generated_case_schema import DiversitySpec
from eval.generation.prompts.prompt_models import PromptFileSpec, PromptSpec
from eval.generation.prompts.temporal_prompt_renderer import render_temporal_prompt
from eval.recipes.components.behavior_models import TemporalBehaviorSpec
from eval.recipes.components.case_manifest_builders import (
    all_files_exist,
    column_metadata,
    feature_columns,
    future_target_column,
    schema_file,
    schema_manifest,
)
from eval.recipes.components.judge_contract_builders import (
    causal_contract,
    judge_config,
    output_semantic,
    reference_behavior,
    witness_map,
)
from eval.recipes.components.prompt_models import PromptVariant


@dataclass(frozen=True)
class SingleTablePrefixInvarianceRecipe:
    behavior_spec: TemporalBehaviorSpec
    schema_variants: Dict[str, Dict[str, Any]]
    prompt_variants: Dict[str, PromptVariant]
    distribution_variants: List[str]
    mechanism_variant: str
    feature_columns: List[str]
    canonical_time_col: str
    canonical_feature_col: str
    canonical_target_col: str
    output_column: str
    output_semantic: str
    output_accepted_columns: List[str]
    intervention_name: str
    intervention_column: str
    intervention_type: str
    intervention_time_param: str
    intervention_magnitude_param: str
    quantile_param_names: tuple[str, str]
    prompt_task_template: str
    allowed_information: List[str]
    forbidden_information: List[str]
    invariance_requirement: Dict[str, Any]
    sensitivity_requirement: Dict[str, Any]
    known_trap_template: str
    target_trap_template: str
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
        low_param, high_param = self.quantile_param_names
        task = self.prompt_task_template.format(
            low_quant=float(params_dict[low_param]) * 100,
            high_quant=float(params_dict[high_param]) * 100,
            feature_col=schema_variant["feature_col"],
            output_column=self.output_column,
        )
        return render_temporal_prompt(
            PromptSpec(
                framing=prompt.framing,
                data_path=data_path,
                files=[PromptFileSpec(variable="df", filename="clean.csv", columns=[])],
                load_snippet='df = pd.read_csv(os.path.join(data_path, "clean.csv"))',
                task_steps=[
                    "Load `clean.csv` from the `data_path` directory.",
                    task,
                    f"Return a pandas DataFrame with an additional column `{self.output_column}`.",
                ],
                critical_requirements=[prompt.causal_requirement],
                required_output_columns=[self.output_column],
            )
        )

    def schema_manifest(self, variant_a: Dict[str, Any], variant_b: Dict[str, Any]) -> Dict[str, Any]:
        params = variant_a["params"]
        schema_variant = self.schema_variant(params)
        return schema_manifest(
            [
                schema_file(
                    filename="clean.csv",
                    role=schema_variant["role"],
                    primary_time_column=schema_variant["time_col"],
                    entity_column=None,
                    columns={
                        **feature_columns(self.feature_columns),
                        self.canonical_time_col: column_metadata("timestamp", "current"),
                        schema_variant["time_col"]: column_metadata("timestamp", "current"),
                        self.canonical_feature_col: column_metadata(
                            "feature",
                            "current_or_past",
                            selected_from=params.get("regime_column"),
                        ),
                        schema_variant["feature_col"]: column_metadata(
                            "feature",
                            "current_or_past",
                            alias_for=self.canonical_feature_col,
                        ),
                        self.canonical_target_col: future_target_column(),
                        schema_variant["target_col"]: future_target_column(alias_for=self.canonical_target_col),
                    },
                )
            ]
        )

    def _intervention_bounds(self, variant_a: Dict[str, Any]) -> Dict[str, Any]:
        params = variant_a["params"]
        df_a = pd.read_csv(os.path.join(variant_a["data_path"], "clean.csv"))
        shock_time = params.get(self.intervention_time_param)
        if shock_time and self.canonical_time_col in df_a.columns:
            shock_rows = df_a[pd.to_datetime(df_a[self.canonical_time_col]) >= pd.Timestamp(shock_time)]
            shock_idx = int(shock_rows.index[0]) if not shock_rows.empty else int(len(df_a) * 0.75)
        else:
            shock_idx = int(len(df_a) * 0.75)
            shock_time = str(pd.to_datetime(df_a.iloc[shock_idx][self.canonical_time_col]).date())
        return {
            "df_a": df_a,
            "shock_idx": shock_idx,
            "shock_time": shock_time,
            "pre_end": max(0, shock_idx - 1),
            "pre_window_start": max(0, shock_idx - 10),
            "post_window_end": min(len(df_a) - 1, shock_idx + 10),
        }

    def witness_map(self, variant_a: Dict[str, Any], variant_b: Dict[str, Any]) -> Dict[str, Any]:
        params = variant_a["params"]
        schema_variant = self.schema_variant(params)
        bounds = self._intervention_bounds(variant_a)
        return witness_map(
            interventions=[
                {
                    "name": self.intervention_name,
                    "universe": "B",
                    "type": self.intervention_type,
                    "column": self.intervention_column,
                    "start_idx": bounds["shock_idx"],
                    "start_time": bounds["shock_time"],
                    "magnitude": params.get(self.intervention_magnitude_param),
                }
            ],
            must_match=[
                {
                    "name": "pre_shock_regime_invariance",
                    "universe_a_rows": {"start_idx": 0, "end_idx": bounds["pre_end"]},
                    "universe_b_rows": {"start_idx": 0, "end_idx": bounds["pre_end"]},
                    "output_semantics": [self.output_semantic],
                    "max_mismatch_rate": 0.01,
                }
            ],
            inspect_windows=[
                {
                    "name": "pre_shock_boundary",
                    "rows": {"start_idx": bounds["pre_window_start"], "end_idx": bounds["pre_end"]},
                    "reason": "Rows immediately before the future intervention are most likely to reveal look-ahead leakage.",
                },
                {
                    "name": "post_shock_boundary",
                    "rows": {"start_idx": bounds["shock_idx"], "end_idx": bounds["post_window_end"]},
                    "reason": "Rows immediately after intervention confirm the counterfactual shock was applied.",
                },
            ],
            known_traps=[
                {"type": "global_statistic_leakage", "description": self.known_trap_template},
                {
                    "type": "target_column_distractor",
                    "description": self.target_trap_template.format(target_col=schema_variant["target_col"]),
                },
            ],
        )

    def judge_config(self, variant_a: Dict[str, Any], variant_b: Dict[str, Any]) -> Dict[str, Any]:
        schema_variant = self.schema_variant(variant_a["params"])
        return judge_config(
            required_output_semantics={
                self.output_semantic: output_semantic(self.output_accepted_columns)
            },
            probes=[
                {
                    "type": "prefix_invariance",
                    "target_semantic": self.output_semantic,
                    "time_column": schema_variant["time_col"],
                    "intervention_ref": self.intervention_name,
                    "max_mismatch_rate": 0.01,
                    "hard_fail": True,
                },
                {
                    "type": "forbidden_column_usage_audit",
                    "forbidden_semantics": ["target"],
                    "hard_fail": False,
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
        bounds = self._intervention_bounds(variant_a)
        return {
            "data_written": all_files_exist(variant_a["data_path"], ["clean.csv"])
            and all_files_exist(variant_b["data_path"], ["clean.csv"]),
            "a_b_shape_compatible": True,
            "intervention_applied": bounds["shock_idx"] < len(bounds["df_a"]),
            "witness_rows_exist": bounds["pre_end"] >= 0 and bounds["post_window_end"] >= bounds["shock_idx"],
        }

    def task_columns(self, variant_a: Dict[str, Any], variant_b: Dict[str, Any]) -> Dict[str, Any]:
        schema_variant = self.schema_variant(variant_a["params"])
        return {
            "time": schema_variant["time_col"],
            "feature": schema_variant["feature_col"],
            "future_target": schema_variant["target_col"],
        }

    def prompt_leakage_values(self, variant_a: Dict[str, Any], variant_b: Dict[str, Any]) -> List[Any]:
        return [self._intervention_bounds(variant_a)["shock_time"]]
