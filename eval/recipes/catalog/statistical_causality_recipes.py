from eval.recipes.components.recipe_factories import (
    config_counterfactual_probe,
    dataframe_window_threshold_probe,
    daily_return_schema_variants,
    derived_metric_monotonic_probe,
    execution_log_schema_variants,
    field_bounds_probe,
    field_monotonic_probe,
    leakage_sentinel_probe,
    leverage_bounds_probe,
    monotonic_response_probe,
    multi_universe_output_recipe,
    output,
    output_bounds_probe,
    output_stability_probe,
    predicate_matrix_probe,
    price_series_schema_variants,
    prompt_variant,
    return_panel_schema_variants,
    single_feature_schema_variants,
    time_scaling_probe,
    trade_news_schema_variants,
    trade_tape_schema_variants,
    universe,
)



def _stationarity_discipline_recipe(
    *,
    behavior_key: str,
    prompt_name: str,
    framing: str,
    causal_requirement: str,
    surface_domains: list[str],
    required_columns: list[str],
    task_steps: list[str],
    known_traps: list[dict[str, str]],
) -> object:
    return multi_universe_output_recipe(
        behavior_key=behavior_key,
        pillar="Statistical Causality",
        difficulty="hard",
        default_schema_variant="canonical_nonstationary_panel",
        default_prompt_variant=prompt_name,
        schema_variants={
            "canonical_nonstationary_panel": {
                "role": "nonstationary_price_factor_panel",
                "time_col": "timestamp",
                "dependent_col": "y",
                "factor_col": "x",
                "secondary_factor_col": "z",
                "asset_a_col": "asset_a",
                "asset_b_col": "asset_b",
            }
        },
        prompt_variants={
            prompt_name: prompt_variant(
                name=prompt_name,
                framing=framing,
                causal_requirement=causal_requirement,
                difficulty_level=3,
                surface_domains=surface_domains,
            )
        },
        distribution_variants=["stationary_i0_panel", "post_t_unit_root_injection"],
        mechanism_variant="nonstationary_spurious_regression",
        output=output(
            "diagnostic_df",
            semantic=prompt_name,
            kind="dataframe",
            accepted_names=["diagnostic_df", "model_df", "result_df", "output_df", "df"],
        ),
        universes=[
            universe(
                "clean",
                role="baseline",
                description="Stationary I(0) price and factor-like series with small Gaussian noise.",
                expected={"stationary": True},
            ),
            universe(
                "shock",
                role="treatment",
                description="Identical prefix followed by a post-T unit-root trend injected into key model columns.",
                expected={"stationary": False, "requires_differencing": True},
            ),
        ],
        probes=[
            {
                "type": "code_pattern_present",
                "name": "stationarity_handling_present",
                "patterns": [
                    r"\.pct_change\s*\(",
                    r"\.diff\s*\(",
                    r"np\.log\s*\([^)]*\)\s*\.diff\s*\(",
                    r"adfuller|kpss|phillips|perron|unit[_\s-]?root",
                ],
                "mode": "any",
                "reason": "The implementation must either test stationarity or transform raw levels to stationary returns/differences before modeling.",
                "hard_fail": True,
            },
            {
                "type": "required_columns",
                "name": f"{prompt_name}_diagnostic_columns_present",
                "universe": "shock",
                "columns": required_columns,
                "hard_fail": True,
            },
            field_bounds_probe(
                name="shock_model_marked_valid_after_stationarity_handling",
                universe_name="shock",
                output_semantic=prompt_name,
                field="is_valid",
                min_value=1.0,
                max_value=1.0,
            ),
            field_bounds_probe(
                name="shock_reports_stationary_data_usage",
                universe_name="shock",
                output_semantic=prompt_name,
                field="used_stationary_data",
                min_value=1.0,
                max_value=1.0,
            ),
        ],
        task_steps=task_steps,
        load_snippet="df = pd.read_csv(DATA_PATH)",
        known_traps=known_traps,
    )


S018_STATISTICAL_SPURIOUS_GRANGER_CAUSALITY_RECIPE = _stationarity_discipline_recipe(
    behavior_key="s018_statistical_spurious_granger_causality",
    prompt_name="stationary_granger_causality_screen",
    framing="You are a quantitative researcher screening whether factor `x` has predictive causal content for asset series `y`.",
    causal_requirement="Do not run or report Granger-style lead/lag significance on raw non-stationary levels; first verify stationarity or model returns/differences.",
    surface_domains=["granger_causality", "macro_factors", "quant_research", "lead_lag_screening"],
    required_columns=["is_valid", "used_stationary_data", "max_abs_statistic", "model_basis"],
    task_steps=[
        "Load the panel from DATA_PATH.",
        "Evaluate whether `x` helps predict `y`, but only after converting levels to returns/differences or explicitly checking stationarity.",
        "Return one-row `diagnostic_df` with `is_valid`, `used_stationary_data`, `max_abs_statistic`, and `model_basis` columns.",
        "Do not report causality, significance, or fit diagnostics from raw levels.",
    ],
    known_traps=[
        {"type": "raw_level_granger", "description": "Running Granger causality directly on price or trending factor levels can report a false predictive relation."},
        {"type": "unit_root_significance", "description": "High significance can be an artifact of shared stochastic trends rather than causal information."},
    ],
)


S019_STATISTICAL_IGNORED_COINTEGRATION_PAIRS_TRADING_RECIPE = _stationarity_discipline_recipe(
    behavior_key="s019_statistical_ignored_cointegration_pairs_trading",
    prompt_name="cointegration_checked_pairs_trading",
    framing="You are building a pairs-trading hedge ratio and spread diagnostic for `asset_a` and `asset_b`.",
    causal_requirement="Do not trade or estimate a hedge ratio on non-stationary price levels unless the pair is cointegrated; otherwise use returns/differences and mark the level-spread model invalid.",
    surface_domains=["pairs_trading", "cointegration", "stat_arb", "hedge_ratio"],
    required_columns=["is_valid", "used_stationary_data", "max_abs_statistic", "model_basis"],
    task_steps=[
        "Load the panel from DATA_PATH.",
        "Assess the pair using returns/differences or an explicit cointegration/residual-stationarity check before treating a level spread as meaningful.",
        "Return one-row `diagnostic_df` with `is_valid`, `used_stationary_data`, `max_abs_statistic`, and `model_basis` columns.",
        "Do not build a pairs-trading signal from raw price-level correlation alone.",
    ],
    known_traps=[
        {"type": "correlated_random_walk_pair", "description": "Two trending prices can look tightly related without a tradable stationary spread."},
        {"type": "missing_cointegration_test", "description": "Skipping Engle-Granger, Johansen, or residual-stationarity checks creates spurious pairs."},
    ],
)


S020_STATISTICAL_SPURIOUS_FACTOR_SIGNIFICANCE_RECIPE = _stationarity_discipline_recipe(
    behavior_key="s020_statistical_spurious_factor_significance",
    prompt_name="stationary_factor_significance_screen",
    framing="You are screening factors `x` and `z` for statistical significance in a quantitative factor model for `y`.",
    causal_requirement="Do not report significant factors from regressions on non-stationary levels; use returns/differences or explicit stationarity diagnostics first.",
    surface_domains=["factor_modeling", "factor_screening", "macro_beta", "quant_research"],
    required_columns=["is_valid", "used_stationary_data", "max_abs_statistic", "model_basis"],
    task_steps=[
        "Load the panel from DATA_PATH.",
        "Estimate factor relevance only on stationary transformed data or after explicit stationarity testing.",
        "Return one-row `diagnostic_df` with `is_valid`, `used_stationary_data`, `max_abs_statistic`, and `model_basis` columns.",
        "Do not rank or declare factors significant from raw trending levels.",
    ],
    known_traps=[
        {"type": "raw_level_factor_tstats", "description": "Regressing asset prices on trending factor levels can create impressive but meaningless t-statistics."},
        {"type": "factor_screening_without_unit_root_checks", "description": "Large factor libraries often hide non-stationary series that must be transformed before screening."},
    ],
)


S021_STATISTICAL_NON_STATIONARY_RESIDUALS_RECIPE = _stationarity_discipline_recipe(
    behavior_key="s021_statistical_non_stationary_residuals",
    prompt_name="stationary_residual_regression_diagnostic",
    framing="You are validating whether a regression or spread model has a statistically meaningful residual process.",
    causal_requirement="Even if inputs are transformed, do not accept a level relationship unless residuals are stationary or the model is built directly on stationary returns/differences.",
    surface_domains=["residual_diagnostics", "cointegration", "factor_validation", "pairs_trading"],
    required_columns=["is_valid", "used_stationary_data", "residual_stationary", "max_abs_statistic", "model_basis"],
    task_steps=[
        "Load the panel from DATA_PATH.",
        "Fit only on stationary returns/differences, or explicitly check that residuals from a level relationship are stationary.",
        "Return one-row `diagnostic_df` with `is_valid`, `used_stationary_data`, `residual_stationary`, `max_abs_statistic`, and `model_basis` columns.",
        "Do not accept a level regression whose residuals remain non-stationary.",
    ],
    known_traps=[
        {"type": "unchecked_residual_unit_root", "description": "A level regression can look strong while residuals retain a unit root, invalidating the relation."},
        {"type": "partial_differencing_false_safety", "description": "Transforming one side but not validating the model residual can still leave a spurious relation."},
    ],
)


S022_STATISTICAL_FALSE_LEAD_LAG_RELATIONSHIPS_RECIPE = _stationarity_discipline_recipe(
    behavior_key="s022_statistical_false_lead_lag_relationships",
    prompt_name="stationary_lead_lag_relationship_screen",
    framing="You are testing whether series `x` leads movements in asset series `y` for a trading signal.",
    causal_requirement="Do not infer lead-lag causality from raw trending levels; measure lead-lag relationships only on stationary returns/differences or after explicit unit-root checks.",
    surface_domains=["lead_lag_analysis", "macro_factors", "cross_asset_signals", "quant_research"],
    required_columns=["is_valid", "used_stationary_data", "max_abs_statistic", "model_basis"],
    task_steps=[
        "Load the panel from DATA_PATH.",
        "Compute lead-lag evidence using returns/differences or explicit stationarity checks before comparing lags.",
        "Return one-row `diagnostic_df` with `is_valid`, `used_stationary_data`, `max_abs_statistic`, and `model_basis` columns.",
        "Do not infer that `x` leads `y` from raw level correlations.",
    ],
    known_traps=[
        {"type": "trend_induced_lead_lag", "description": "Two independent trending series can show an apparent lag relation due to shared persistence."},
        {"type": "raw_level_cross_correlation", "description": "Cross-correlations on price levels are not evidence of causal timing without stationarity handling."},
    ],
)


S003_COVARIANCE_INVERSION_RECIPE = multi_universe_output_recipe(
    behavior_key="s003_covariance_inversion_stability",
    pillar="Statistical Causality",
    difficulty="hard",
    default_schema_variant="wide_return_panel",
    default_prompt_variant="robust_min_variance_portfolio",
    schema_variants=return_panel_schema_variants(
        canonical_name="wide_return_panel",
        asset_prefix="STK_",
        n_assets=50,
        n_observations=30,
        role="asset_return_panel",
    ),
    prompt_variants={
        "robust_min_variance_portfolio": prompt_variant(
            name="robust_min_variance_portfolio",
            framing="You are a quantitative portfolio researcher.",
            causal_requirement="The optimizer must be robust when the number of assets exceeds the number of observations.",
            difficulty_level=3,
            surface_domains=["portfolio_optimization", "risk_analytics"],
        )
    },
    distribution_variants=["p_greater_than_n", "micro_noise_perturbed"],
    mechanism_variant="spurious_precision_covariance_inversion",
    output=output(
        "portfolio_weights",
        semantic="portfolio_weights",
        kind="dataframe",
        shape=(1, 50),
        accepted_names=["portfolio_weights", "weights"],
    ),
    universes=[
        universe(
            "base_returns",
            role="baseline",
            description="Thirty observations of returns across fifty assets.",
            expected={"n_assets": 50, "n_observations": 30},
        ),
        universe(
            "micro_noise",
            role="treatment",
            description="Base return panel with tiny independent perturbations.",
            interventions=[{"type": "micro_noise", "sigma": 1e-6, "columns": "all_assets"}],
            expected={"max_mean_abs_weight_delta": 0.01},
        ),
    ],
    probes=[
        output_stability_probe(
            name="micro_noise_weight_stability",
            baseline="base_returns",
            perturbed="micro_noise",
            output_semantic="portfolio_weights",
            max_mean_abs_delta=0.01,
        ),
        leverage_bounds_probe(
            name="realistic_leverage",
            universe_name="base_returns",
            output_semantic="portfolio_weights",
            max_leverage=5.0,
            fully_invested_tolerance=0.01,
        ),
    ],
    task_steps=[
        "Load the return panel from DATA_PATH.",
        "Compute minimum-variance portfolio weights.",
        "Return `portfolio_weights` as a pandas DataFrame with shape (1, 50).",
        "Use a robust covariance or regularized optimization approach suitable for P > N.",
    ],
    load_snippet="returns_data = pd.read_csv(DATA_PATH)",
    known_traps=[
        {
            "type": "singular_covariance_inversion",
            "description": "Directly inverting the sample covariance matrix is unstable when assets exceed observations.",
        },
        {
            "type": "trivial_equal_weight_fallback",
            "description": "Returning equal weights can hide optimizer failure rather than solve the stability problem.",
        },
    ],
)


S009_VOLUME_LEAD_LAG_TRAP_RECIPE = multi_universe_output_recipe(
    behavior_key="s009_volume_lead_lag_causality",
    pillar="Statistical Causality",
    difficulty="hard",
    default_schema_variant="canonical_price_volume_series",
    default_prompt_variant="causal_volume_signal_position",
    schema_variants=price_series_schema_variants(
        canonical_name="canonical_price_volume_series",
        time_col="timestamp",
        price_col="close",
        volume_col="volume",
        role="minute_price_volume_series",
    ),
    prompt_variants={
        "causal_volume_signal_position": prompt_variant(
            name="causal_volume_signal_position",
            framing="You are a systematic trading researcher evaluating volume as a trading signal.",
            causal_requirement="Position sizing must distinguish a contemporaneous correlation trap from a truly leading volume signal.",
            difficulty_level=3,
            surface_domains=["signal_research", "causal_inference", "market_microstructure"],
        )
    },
    distribution_variants=["contemporaneous_volume_trap", "volume_leads_next_return"],
    mechanism_variant="lead_lag_signal_discrimination",
    output=output(
        "position",
        semantic="position",
        kind="scalar",
        valid_range=(-100.0, 100.0),
    ),
    universes=[
        universe(
            "contemporaneous_trap",
            role="trap",
            description="Volume spikes coincide with price moves and momentum continuation, creating a false alpha if lag structure is ignored.",
            expected={"abs_position_max": 10, "partial_abs_position_max": 20},
        ),
        universe(
            "leading_signal",
            role="positive",
            description="Volume spikes precede the next return and should support a high long position.",
            expected={"position_min": 50, "partial_position_min": 40},
        ),
    ],
    probes=[
        predicate_matrix_probe(
            name="lead_lag_causal_discrimination",
            output_semantic="position",
            expectations={
                "contemporaneous_trap": {"abs_max": 10, "partial_abs_max": 20},
                "leading_signal": {"min_value": 50, "partial_min_value": 40},
            },
        ),
        monotonic_response_probe(
            name="lead_signal_position_exceeds_trap",
            baseline="contemporaneous_trap",
            treatment="leading_signal",
            output_semantic="position",
            direction="increase",
            min_delta=40,
        ),
    ],
    task_steps=[
        "Load the minute price and volume history from DATA_PATH.",
        "Estimate whether volume spikes lead future returns rather than merely coincide with current price moves.",
        "Stay near flat when the evidence is only contemporaneous or lagging.",
        "Set `position` as a scalar trading position.",
    ],
    load_snippet="df = pd.read_csv(DATA_PATH)",
    known_traps=[
        {
            "type": "contemporaneous_correlation_alpha",
            "description": "Treating same-bar volume/return correlation as a tradable leading signal creates false alpha.",
        },
        {
            "type": "lag_direction_inversion",
            "description": "Confusing volume that reacts to price with volume that predicts price fails the trap universe.",
        },
    ],
)


S011_OUTLIER_ROBUST_BREAKOUT_RECIPE = multi_universe_output_recipe(
    behavior_key="s011_outlier_robust_breakout_detection",
    pillar="Statistical Causality",
    difficulty="hard",
    default_schema_variant="canonical_price_series",
    default_prompt_variant="robust_breakout_protection_flags",
    schema_variants=price_series_schema_variants(
        canonical_name="canonical_price_series",
        time_col="timestamp",
        price_col="close",
        volume_col=None,
        role="minute_price_series",
    ),
    prompt_variants={
        "robust_breakout_protection_flags": prompt_variant(
            name="robust_breakout_protection_flags",
            framing="You are building protection logic for a trading system that must detect real breakouts without being poisoned by bad ticks.",
            causal_requirement="Outlier handling must suppress isolated bad ticks while preserving genuine breakout signals.",
            difficulty_level=3,
            surface_domains=["market_data_quality", "risk_controls", "signal_detection"],
        )
    },
    distribution_variants=[
        "poisoned_breakout",
        "clean_breakout",
        "strong_poison_breakout",
        "poison_without_signal",
        "multiple_outliers_breakout",
    ],
    mechanism_variant="robust_outlier_filtering_with_signal_preservation",
    output=output(
        "protection_flags",
        semantic="protection_flags",
        kind="series",
        accepted_names=["protection_flags", "flags"],
    ),
    universes=[
        universe(
            "poisoned_breakout",
            role="treatment",
            description="A single extreme bad tick appears before a genuine breakout that must still be flagged.",
            expected={"flag_breakout": True, "flag_poison": False},
        ),
        universe(
            "clean_breakout",
            role="baseline",
            description="Clean price path with the same genuine breakout and no poison tick.",
            expected={"flag_breakout": True},
        ),
        universe(
            "strong_poison_breakout",
            role="stress",
            description="A stronger bad tick appears before a genuine breakout that should survive robust filtering.",
            expected={"flag_breakout": True, "flag_poison": False},
        ),
        universe(
            "poison_without_signal",
            role="negative",
            description="Bad tick only, with no genuine breakout; the protection flags should produce no false positives.",
            expected={"total_flags": 0},
        ),
        universe(
            "multiple_outliers_breakout",
            role="stress",
            description="Multiple bad ticks test cumulative poisoning before a genuine breakout.",
            expected={"flag_breakout": True},
        ),
    ],
    probes=[
        predicate_matrix_probe(
            name="robust_breakout_flag_matrix",
            output_semantic="protection_flags",
            expectations={
                "poisoned_breakout": {"from_metadata": "expected"},
                "clean_breakout": {"from_metadata": "expected"},
                "strong_poison_breakout": {"from_metadata": "expected"},
                "poison_without_signal": {"sum_equals": 0},
                "multiple_outliers_breakout": {"from_metadata": "expected"},
            },
        ),
    ],
    task_steps=[
        "Implement `get_protection_flags(df)`.",
        "Load or accept a price DataFrame with `timestamp` and `close` columns.",
        "Suppress isolated bad ticks before estimating breakout thresholds.",
        "Return a boolean pandas Series aligned to the input rows.",
        "Preserve the genuine breakout flag at the breakout timestamp while avoiding false positives in poison-only data.",
    ],
    load_snippet="df = pd.read_csv(DATA_PATH)",
    known_traps=[
        {
            "type": "outlier_poisoned_threshold",
            "description": "Using mean/std thresholds on raw prices can let one bad tick distort the detector and hide the real breakout.",
        },
        {
            "type": "overaggressive_outlier_removal",
            "description": "Removing all large moves can also remove the genuine breakout that the protection logic must preserve.",
        },
        {
            "type": "bad_tick_false_positive",
            "description": "Flagging isolated poison ticks as breakouts creates protection noise in no-signal data.",
        },
    ],
)
