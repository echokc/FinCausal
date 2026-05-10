from typing import Any, Dict, List, Literal

from eval.recipes.components.behavior_models import TemporalBehaviorSpec
from eval.recipes.components.prompt_models import PromptVariant
from eval.recipes.recipe_models import (
    MultiUniverseOutputRecipe,
    MultiUniverseScalarRecipe,
    OutputRecipe,
    ScalarOutputRecipe,
    UniverseRecipe,
)
from eval.recipes.templates.prefix_invariance_recipe import SingleTablePrefixInvarianceRecipe
from eval.recipes.templates.strict_prior_join_recipe import StrictPriorJoinRecipe


def universe(
    name: str,
    *,
    role: Literal["baseline", "treatment", "leakage", "stress", "positive", "negative", "trap"],
    description: str,
    interventions: List[Dict[str, Any]] | None = None,
    expected: Dict[str, Any] | None = None,
) -> UniverseRecipe:
    return UniverseRecipe(
        name=name,
        role=role,
        description=description,
        interventions=interventions or [],
        expected=expected or {},
    )


def scalar_output(
    variable_name: str,
    *,
    semantic: str,
    valid_range: tuple[float, float] | None = None,
    accepted_names: List[str] | None = None,
) -> ScalarOutputRecipe:
    return ScalarOutputRecipe(
        variable_name=variable_name,
        semantic=semantic,
        valid_range=valid_range,
        accepted_names=accepted_names or [variable_name],
    )


def output(
    variable_name: str,
    *,
    semantic: str,
    kind: Literal["scalar", "vector", "dataframe", "series", "dict"],
    valid_range: tuple[float, float] | None = None,
    shape: tuple[int, ...] | None = None,
    accepted_names: List[str] | None = None,
) -> OutputRecipe:
    return OutputRecipe(
        variable_name=variable_name,
        semantic=semantic,
        kind=kind,
        valid_range=valid_range,
        shape=shape,
        accepted_names=accepted_names or [variable_name],
    )


def multi_universe_scalar_recipe(
    *,
    behavior_key: str,
    pillar: str,
    difficulty: str,
    default_schema_variant: str,
    default_prompt_variant: str,
    schema_variants: Dict[str, Dict[str, str]],
    prompt_variants: Dict[str, PromptVariant],
    distribution_variants: List[str],
    mechanism_variant: str,
    output: ScalarOutputRecipe,
    universes: List[UniverseRecipe],
    probes: List[Dict[str, Any]],
    task_steps: List[str],
    load_snippet: str = "price = pd.read_csv(DATA_PATH)",
    known_traps: List[Dict[str, str]] | None = None,
) -> MultiUniverseScalarRecipe:
    return MultiUniverseScalarRecipe(
        behavior_key=behavior_key,
        pillar=pillar,
        difficulty=difficulty,
        schema_variants=schema_variants,
        default_schema_variant=default_schema_variant,
        prompt_variants=prompt_variants,
        default_prompt_variant=default_prompt_variant,
        distribution_variants=distribution_variants,
        mechanism_variant=mechanism_variant,
        output=output,
        universes=universes,
        probes=probes,
        task_steps=task_steps,
        load_snippet=load_snippet,
        known_traps=known_traps or [],
    )


def multi_universe_output_recipe(
    *,
    behavior_key: str,
    pillar: str,
    difficulty: str,
    default_schema_variant: str,
    default_prompt_variant: str,
    schema_variants: Dict[str, Dict[str, str]],
    prompt_variants: Dict[str, PromptVariant],
    distribution_variants: List[str],
    mechanism_variant: str,
    output: OutputRecipe,
    universes: List[UniverseRecipe],
    probes: List[Dict[str, Any]],
    task_steps: List[str],
    load_snippet: str,
    known_traps: List[Dict[str, str]] | None = None,
) -> MultiUniverseOutputRecipe:
    return MultiUniverseOutputRecipe(
        behavior_key=behavior_key,
        pillar=pillar,
        difficulty=difficulty,
        schema_variants=schema_variants,
        default_schema_variant=default_schema_variant,
        prompt_variants=prompt_variants,
        default_prompt_variant=default_prompt_variant,
        distribution_variants=distribution_variants,
        mechanism_variant=mechanism_variant,
        output=output,
        universes=universes,
        probes=probes,
        task_steps=task_steps,
        load_snippet=load_snippet,
        known_traps=known_traps or [],
    )


def single_table_prefix_invariance_recipe(
    *,
    behavior_key: str,
    pillar: str,
    difficulty: str,
    default_schema_variant: str,
    default_prompt_variant: str,
    schema_variants: Dict[str, Dict[str, Any]],
    prompt_variants: Dict[str, PromptVariant],
    distribution_variants: List[str],
    mechanism_variant: str,
    feature_columns: List[str],
    output_column: str,
    output_semantic: str,
    output_accepted_columns: List[str],
    intervention_name: str,
    intervention_column: str,
    intervention_type: str,
    intervention_time_param: str,
    intervention_magnitude_param: str,
    quantile_param_names: tuple[str, str],
    prompt_task_template: str,
    canonical_time_col: str = "date",
    canonical_feature_col: str = "regime_feature",
    canonical_target_col: str = "fwd_return",
    allowed_information: List[str] | None = None,
    forbidden_information: List[str] | None = None,
    invariance_requirement: Dict[str, Any] | None = None,
    sensitivity_requirement: Dict[str, Any] | None = None,
    known_trap_template: str | None = None,
    target_trap_template: str | None = None,
    causal_solution_strategy: str | None = None,
    expected_failure_strategy: str | None = None,
    positive_control_id: str | None = None,
    negative_control_id: str | None = None,
) -> SingleTablePrefixInvarianceRecipe:
    return SingleTablePrefixInvarianceRecipe(
        behavior_spec=TemporalBehaviorSpec(
            behavior_key=behavior_key,
            pillar=pillar,
            difficulty=difficulty,
            default_schema_variant=default_schema_variant,
            default_prompt_variant=default_prompt_variant,
        ),
        schema_variants=schema_variants,
        prompt_variants=prompt_variants,
        distribution_variants=distribution_variants,
        mechanism_variant=mechanism_variant,
        feature_columns=feature_columns,
        canonical_time_col=canonical_time_col,
        canonical_feature_col=canonical_feature_col,
        canonical_target_col=canonical_target_col,
        output_column=output_column,
        output_semantic=output_semantic,
        output_accepted_columns=output_accepted_columns,
        intervention_name=intervention_name,
        intervention_column=intervention_column,
        intervention_type=intervention_type,
        intervention_time_param=intervention_time_param,
        intervention_magnitude_param=intervention_magnitude_param,
        quantile_param_names=quantile_param_names,
        prompt_task_template=prompt_task_template,
        allowed_information=allowed_information
        or [
            "For each row, use only rows available at or before that row's timestamp.",
            f"Current and historical values of {canonical_feature_col} may be used.",
        ],
        forbidden_information=forbidden_information
        or [
            "Future rows when assigning a current or past output.",
            f"Forward-looking target columns such as {canonical_target_col}.",
            "Universe B intervention metadata, shock time, or future shock magnitude.",
        ],
        invariance_requirement=invariance_requirement
        or {
            "name": "prefix_stability",
            "description": "Universe B future intervention must not change pre-intervention outputs.",
            "scope": "rows_before_intervention",
            "target_outputs": [output_semantic],
        },
        sensitivity_requirement=sensitivity_requirement
        or {
            "name": "post_intervention_response_allowed",
            "description": "Outputs at or after the intervention may differ between universes.",
            "scope": "rows_at_or_after_intervention",
        },
        known_trap_template=known_trap_template
        or "Full-column statistics can let future shocked rows alter pre-intervention outputs.",
        target_trap_template=target_trap_template
        or "{target_col} is present but is future-only and must not be used for this decision.",
        causal_solution_strategy=causal_solution_strategy
        or "Compute decision inputs using only history available up to each row.",
        expected_failure_strategy=expected_failure_strategy
        or "Compute statistics using the full dataset before assigning outputs to all rows.",
        positive_control_id=positive_control_id or "prefix_stable_reference",
        negative_control_id=negative_control_id or "global_statistic_reference",
    )


def strict_prior_join_recipe(
    *,
    behavior_key: str,
    pillar: str,
    difficulty: str,
    default_schema_variant: str,
    default_prompt_variant: str,
    schema_variants: Dict[str, Dict[str, Any]],
    prompt_variants: Dict[str, PromptVariant],
    distribution_variants: List[str],
    mechanism_variant: str,
    intervention_name: str,
    output_dataframe_name: str,
    required_output_columns: List[str],
    intervention_type: str = "timestamp_and_semantic_pollution",
    allowed_information: List[str] | None = None,
    forbidden_information: List[str] | None = None,
    invariance_requirement: Dict[str, Any] | None = None,
    sensitivity_requirement: Dict[str, Any] | None = None,
    known_traps: List[Dict[str, str]] | None = None,
    causal_solution_strategy: str | None = None,
    expected_failure_strategy: str | None = None,
    positive_control_id: str | None = None,
    negative_control_id: str | None = None,
) -> StrictPriorJoinRecipe:
    return StrictPriorJoinRecipe(
        behavior_spec=TemporalBehaviorSpec(
            behavior_key=behavior_key,
            pillar=pillar,
            difficulty=difficulty,
            default_schema_variant=default_schema_variant,
            default_prompt_variant=default_prompt_variant,
        ),
        schema_variants=schema_variants,
        prompt_variants=prompt_variants,
        distribution_variants=distribution_variants,
        mechanism_variant=mechanism_variant,
        intervention_name=intervention_name,
        intervention_type=intervention_type,
        output_dataframe_name=output_dataframe_name,
        required_output_columns=required_output_columns,
        allowed_information=allowed_information
        or [
            "For each decision row, only information records available strictly before the decision timestamp may be joined.",
            "Decision rows with no prior information may retain null joined fields.",
        ],
        forbidden_information=forbidden_information
        or [
            "Information records published at or after the decision timestamp.",
            "Ex-post content that was not available at the decision timestamp.",
            "Nearest-neighbor joins that can select future information.",
        ],
        invariance_requirement=invariance_requirement
        or {
            "name": "strict_information_availability",
            "description": "Every joined non-null information row must be strictly before the decision timestamp.",
            "scope": "all_output_rows",
            "target_outputs": ["information_join", "time_delta_minutes"],
        },
        sensitivity_requirement=sensitivity_requirement
        or {
            "name": "future_information_rejection",
            "description": "Injected future-published information must not be joined to past decisions.",
            "scope": "contaminated_records",
        },
        known_traps=known_traps
        or [
            {
                "type": "nearest_timestamp_join",
                "description": "Nearest or absolute-distance joins may select future-published records.",
            },
            {
                "type": "inclusive_or_reversed_time_filter",
                "description": "Using information_time >= decision_time or allowing exact/future matches violates availability.",
            },
        ],
        causal_solution_strategy=causal_solution_strategy
        or "Use a strict prior as-of join with backward direction and no exact or future matches.",
        expected_failure_strategy=expected_failure_strategy
        or "Use nearest joins, cross joins with absolute time distance, or any join that can select future records.",
        positive_control_id=positive_control_id or "strict_backward_asof_reference",
        negative_control_id=negative_control_id or "nearest_timestamp_reference",
    )
