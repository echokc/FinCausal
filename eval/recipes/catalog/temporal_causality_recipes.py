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



S013_TEMPORAL_ROLLING_ZSCORE_LEAKAGE_RECIPE = multi_universe_output_recipe(
    behavior_key="s013_temporal_rolling_zscore_causal_leakage",
    pillar="Temporal Causality",
    difficulty="hard",
    default_schema_variant="canonical_price_series",
    default_prompt_variant="causal_rolling_zscore_regime_detection",
    schema_variants=price_series_schema_variants(
        canonical_name="canonical_price_series",
        time_col="timestamp",
        price_col="close",
        volume_col=None,
        role="minute_price_series",
    ),
    prompt_variants={
        "causal_rolling_zscore_regime_detection": prompt_variant(
            name="causal_rolling_zscore_regime_detection",
            framing="You are a quantitative researcher building rolling z-score regime features for a trading model.",
            causal_requirement="The z-score and regime label at time t must use only window statistics available through time t-1.",
            difficulty_level=3,
            surface_domains=["quant_research", "regime_detection", "high_frequency_factors"],
        )
    },
    distribution_variants=["clean_gaussian_path", "post_t_volatility_shock"],
    mechanism_variant="rolling_zscore_lookahead_leakage",
    output=output(
        "regime_df",
        semantic="rolling_zscore_regime",
        kind="dataframe",
        accepted_names=["regime_df", "result_df", "output_df", "df"],
    ),
    universes=[
        universe(
            "clean",
            role="baseline",
            description="Gaussian minute price path without structural breaks.",
            expected={"pre_shock_regime_stable": True},
        ),
        universe(
            "shock",
            role="treatment",
            description="Identical prefix, followed by a volatility and price-jump shock that must only affect post-shock labels.",
            expected={"pre_shock_regime_stable": True, "post_shock_detection": True},
        ),
    ],
    probes=[
        {
            "type": "code_pattern_absent",
            "name": "no_centered_rolling_window",
            "patterns": [r"center\s*=\s*True"],
            "reason": "Centered rolling windows use future observations for pre-shock z-scores.",
            "hard_fail": True,
        },
        {
            "type": "required_columns",
            "name": "rolling_zscore_columns_present",
            "universe": "shock",
            "columns": ["zscore", "regime"],
            "hard_fail": True,
        },
        {
            "type": "prefix_invariance",
            "name": "pre_shock_regime_invariance",
            "baseline_universe": "clean",
            "treatment_universe": "shock",
            "field": "regime",
            "start_idx_from_metadata": "expected.pre_shock_window_start",
            "end_idx_from_metadata": "expected.pre_shock_window_end",
            "max_mismatch_rate": 0.0,
            "hard_fail": True,
        },
        dataframe_window_threshold_probe(
            name="post_shock_zscore_detects_regime_change",
            universe_name="shock",
            output_semantic="rolling_zscore_regime",
            field="zscore",
            start_idx_from_metadata="expected.post_shock_window_start",
            end_idx_from_metadata="expected.post_shock_window_end",
            min_abs_max=2.0,
        ),
    ],
    task_steps=[
        "Load the price history from DATA_PATH.",
        "Calculate returns from the `close` column.",
        "Compute rolling z-scores using lagged rolling mean and standard deviation; statistics for time t must be shifted so they use data through t-1 only.",
        "Do not use centered rolling windows, unshifted rolling statistics, global thresholds, or future rows.",
        "Return `regime_df` as a DataFrame aligned to the input rows with at least `zscore` and `regime` columns.",
    ],
    load_snippet="df = pd.read_csv(DATA_PATH)",
    known_traps=[
        {
            "type": "unshifted_rolling_statistics",
            "description": "Using rolling mean/std at time t without `.shift(1)` includes the current shock observation in the denominator and contaminates the feature.",
        },
        {
            "type": "centered_rolling_window",
            "description": "`rolling(..., center=True)` leaks future observations into pre-shock z-scores and regime labels.",
        },
        {
            "type": "future_threshold_calibration",
            "description": "Global or expanding thresholds can let post-shock volatility alter labels before the shock is observable.",
        },
    ],
)


def _rolling_stat_leakage_recipe(
    *,
    behavior_key: str,
    prompt_name: str,
    stat_label: str,
    causal_requirement: str,
    surface_domains: list[str],
    columns: list[str],
    invariant_field: str,
    post_field: str,
    post_min_abs: float,
    task_steps: list[str],
    known_traps: list[dict[str, str]],
) -> object:
    return multi_universe_output_recipe(
        behavior_key=behavior_key,
        pillar="Temporal Causality",
        difficulty="hard",
        default_schema_variant="canonical_price_series",
        default_prompt_variant=prompt_name,
        schema_variants=price_series_schema_variants(
            canonical_name="canonical_price_series",
            time_col="timestamp",
            price_col="close",
            volume_col=None,
            role="rolling_stat_panel",
        ),
        prompt_variants={
            prompt_name: prompt_variant(
                name=prompt_name,
                framing=f"You are a quantitative researcher building {stat_label} features for live trading and risk workflows.",
                causal_requirement=causal_requirement,
                difficulty_level=3,
                surface_domains=surface_domains,
            )
        },
        distribution_variants=["clean_prefix", "post_t_statistical_break"],
        mechanism_variant="rolling_stat_lookahead_leakage",
        output=output(
            "stat_df",
            semantic=prompt_name,
            kind="dataframe",
            accepted_names=["stat_df", "regime_df", "result_df", "output_df", "df"],
        ),
        universes=[
            universe(
                "clean",
                role="baseline",
                description="Clean path without a structural break in the target rolling statistic.",
                expected={"pre_shock_stat_stable": True},
            ),
            universe(
                "shock",
                role="treatment",
                description="Identical prefix, followed by a post-T statistical break that should only affect post-shock rows.",
                expected={"pre_shock_stat_stable": True, "post_shock_detection": True},
            ),
        ],
        probes=[
            {
                "type": "code_pattern_absent",
                "name": "no_centered_rolling_window",
                "patterns": [r"center\s*=\s*True"],
                "reason": "Centered rolling windows leak future observations into pre-shock rolling statistics.",
                "hard_fail": True,
            },
            {
                "type": "required_columns",
                "name": f"{prompt_name}_columns_present",
                "universe": "shock",
                "columns": columns,
                "hard_fail": True,
            },
            {
                "type": "prefix_invariance",
                "name": "pre_shock_stat_invariance",
                "baseline_universe": "clean",
                "treatment_universe": "shock",
                "field": invariant_field,
                "start_idx_from_metadata": "expected.pre_shock_window_start",
                "end_idx_from_metadata": "expected.pre_shock_window_end",
                "max_mismatch_rate": 0.0,
                "hard_fail": True,
            },
            dataframe_window_threshold_probe(
                name="post_shock_stat_detects_break",
                universe_name="shock",
                output_semantic=prompt_name,
                field=post_field,
                start_idx_from_metadata="expected.post_shock_window_start",
                end_idx_from_metadata="expected.post_shock_window_end",
                min_abs_max=post_min_abs,
            ),
        ],
        task_steps=task_steps,
        load_snippet="df = pd.read_csv(DATA_PATH)",
        known_traps=known_traps,
    )


S014_TEMPORAL_ROLLING_CORR_BETA_LEAKAGE_RECIPE = _rolling_stat_leakage_recipe(
    behavior_key="s014_temporal_rolling_corr_beta_causal_leakage",
    prompt_name="causal_rolling_correlation_beta",
    stat_label="rolling correlation and beta",
    causal_requirement="Rolling correlation, beta, and hedge-ratio estimates at time t must use only observations available through time t-1.",
    surface_domains=["pairs_trading", "stat_arb", "factor_exposure", "dynamic_hedging"],
    columns=["correlation", "beta", "regime"],
    invariant_field="beta",
    post_field="beta",
    post_min_abs=1.5,
    task_steps=[
        "Load the panel from DATA_PATH with `timestamp`, `asset_a`, `asset_b`, and `market` columns.",
        "Compute returns for the assets and market.",
        "Estimate rolling correlation and beta using lagged rolling covariance and variance; estimates for time t must use data through t-1 only.",
        "Do not use centered rolling windows, unshifted rolling statistics, or full-sample correlation/beta.",
        "Return `stat_df` aligned to input rows with at least `correlation`, `beta`, and `regime` columns.",
    ],
    known_traps=[
        {"type": "centered_rolling_beta", "description": "Centered rolling windows leak post-break returns into pre-break beta estimates."},
        {"type": "full_sample_hedge_ratio", "description": "A full-sample hedge ratio uses future exposure changes and inflates stat-arb stability."},
    ],
)


S015_TEMPORAL_ROLLING_VAR_CVAR_LEAKAGE_RECIPE = _rolling_stat_leakage_recipe(
    behavior_key="s015_temporal_rolling_var_cvar_causal_leakage",
    prompt_name="causal_rolling_var_cvar",
    stat_label="rolling historical VaR and CVaR",
    causal_requirement="Rolling VaR/CVaR at time t must be estimated only from returns available through time t-1.",
    surface_domains=["risk_management", "position_sizing", "regulatory_reporting"],
    columns=["var", "cvar", "regime"],
    invariant_field="cvar",
    post_field="cvar",
    post_min_abs=0.004,
    task_steps=[
        "Load the price history from DATA_PATH.",
        "Calculate returns from `close`.",
        "Estimate rolling historical VaR and CVaR with lagged windows; risk values for time t must use data through t-1 only.",
        "Do not use centered windows, unshifted rolling quantiles, or full-sample quantile thresholds.",
        "Return `stat_df` aligned to input rows with at least `var`, `cvar`, and `regime` columns.",
    ],
    known_traps=[
        {"type": "unshifted_rolling_quantile", "description": "Unshifted rolling quantiles include the current loss in the risk estimate for that same row."},
        {"type": "full_sample_var", "description": "Full-sample VaR/CVaR uses future tail events before they occur."},
    ],
)


S016_TEMPORAL_ROLLING_DRAWDOWN_LEAKAGE_RECIPE = _rolling_stat_leakage_recipe(
    behavior_key="s016_temporal_rolling_drawdown_causal_leakage",
    prompt_name="causal_rolling_max_drawdown",
    stat_label="rolling maximum drawdown",
    causal_requirement="Rolling drawdown and max-drawdown estimates at time t must use only price history available through time t-1.",
    surface_domains=["strategy_evaluation", "risk_control", "live_monitoring"],
    columns=["drawdown", "max_drawdown", "regime"],
    invariant_field="max_drawdown",
    post_field="max_drawdown",
    post_min_abs=0.03,
    task_steps=[
        "Load the price history from DATA_PATH.",
        "Compute rolling drawdown and maximum drawdown from the `close` column.",
        "Lag rolling peak and drawdown statistics so the decision for time t uses data through t-1 only.",
        "Do not use centered windows or full-series future peaks/troughs.",
        "Return `stat_df` aligned to input rows with at least `drawdown`, `max_drawdown`, and `regime` columns.",
    ],
    known_traps=[
        {"type": "future_peak_trough_drawdown", "description": "Using future peaks or troughs can flag drawdown regimes before the drawdown is observable."},
        {"type": "centered_drawdown_window", "description": "Centered windows leak future losses into pre-break drawdown estimates."},
    ],
)


S017_TEMPORAL_ROLLING_SKEW_KURTOSIS_LEAKAGE_RECIPE = _rolling_stat_leakage_recipe(
    behavior_key="s017_temporal_rolling_skew_kurtosis_causal_leakage",
    prompt_name="causal_rolling_skew_kurtosis",
    stat_label="rolling skewness and kurtosis",
    causal_requirement="Rolling skewness and kurtosis at time t must use only returns available through time t-1.",
    surface_domains=["regime_detection", "fat_tail_modeling", "option_pricing"],
    columns=["skew", "kurtosis", "regime"],
    invariant_field="kurtosis",
    post_field="kurtosis",
    post_min_abs=4.0,
    task_steps=[
        "Load the price history from DATA_PATH.",
        "Calculate returns from `close`.",
        "Estimate rolling skewness and kurtosis with lagged windows; moment estimates for time t must use data through t-1 only.",
        "Do not use centered windows, unshifted rolling moments, or full-sample moment thresholds.",
        "Return `stat_df` aligned to input rows with at least `skew`, `kurtosis`, and `regime` columns.",
    ],
    known_traps=[
        {"type": "future_tail_moment_leakage", "description": "Future tail observations can inflate pre-shock skewness or kurtosis if windows are centered or unlagged."},
        {"type": "global_moment_threshold", "description": "Full-sample moment thresholds leak future fat-tail regimes into earlier labels."},
    ],
)


S001_GLOBAL_QUANTILE_LEAKAGE_RECIPE = multi_universe_output_recipe(
    behavior_key="s001_global_quantile_leakage",
    pillar="Temporal Causality",
    difficulty="medium",
    default_schema_variant="canonical_single_file",
    default_prompt_variant="point_in_time_regime_quantiles",
    schema_variants=single_feature_schema_variants(
        canonical_name="canonical_single_file",
        canonical_time_col="date",
        canonical_feature_col="regime_feature",
        canonical_target_col="fwd_return",
        canonical_role="market_features",
        aliases={},
    ),
    prompt_variants={
        "point_in_time_regime_quantiles": prompt_variant(
            name="point_in_time_regime_quantiles",
            framing="You are a rigorous quantitative researcher.",
            causal_requirement="For each row, regime labels must use only information available at or before that row.",
            difficulty_level=2,
            surface_domains=["quant_research", "risk_analytics"],
        )
    },
    distribution_variants=["quiet_market", "future_volatility_shock"],
    mechanism_variant="global_statistic_prefix_leakage",
    output=output(
        "regime_df",
        semantic="regime_label",
        kind="dataframe",
        accepted_names=["regime_df", "result_df", "output_df", "df"],
    ),
    universes=[
        universe(
            "base_market",
            role="baseline",
            description="Market feature panel before counterfactual intervention.",
        ),
        universe(
            "future_shock",
            role="leakage",
            description="Same prefix as baseline, with a large future volatility shock after the cutoff.",
            interventions=[{"type": "future_regime_feature_shock", "start_idx": 400}],
        ),
    ],
    probes=[
        {
            "type": "required_columns",
            "name": "regime_column_present",
            "universe": "base_market",
            "columns": ["regime"],
            "hard_fail": True,
        },
        {
            "type": "prefix_invariance",
            "name": "pre_shock_regime_invariance",
            "baseline_universe": "base_market",
            "treatment_universe": "future_shock",
            "field": "regime",
            "start_idx": 0,
            "end_idx": 399,
            "end_idx_from_metadata": "expected.prefix_invariant_until",
            "max_mismatch_rate": 0.01,
            "hard_fail": True,
        },
        {
            "type": "field_unique_count",
            "name": "nontrivial_regime_labels",
            "universe": "base_market",
            "field": "regime",
            "min_unique": 2,
            "hard_fail": False,
        },
    ],
    task_steps=[
        "Load the market feature table from DATA_PATH.",
        "Use the `regime_feature` column to assign each row to Low / Mid / High regimes using point-in-time quantiles.",
        "Return `regime_df` as a pandas DataFrame aligned to input rows and containing a `regime` column.",
        "Do not use full-column quantiles, future rows, or `fwd_return` when labeling a row.",
    ],
    load_snippet="df = pd.read_csv(DATA_PATH)",
    known_traps=[
        {
            "type": "global_statistic_leakage",
            "description": "Full-column quantiles let future shocked rows alter pre-shock regime labels.",
        },
        {
            "type": "target_column_distractor",
            "description": "`fwd_return` is future-only and must not be used for regime assignment.",
        },
    ],
)


S002_EXPOST_NEWS_CONTAMINATION_RECIPE = multi_universe_output_recipe(
    behavior_key="s002_expost_news_contamination",
    pillar="Temporal Causality",
    difficulty="hard",
    default_schema_variant="canonical_trade_news",
    default_prompt_variant="strict_prior_news_join",
    schema_variants=trade_news_schema_variants(
        canonical_name="canonical_trade_news",
        trade_file_role="trades",
        news_file_role="news",
        trade_time_col="timestamp",
        news_time_col="publish_time",
        trade_id_col="trade_id",
        news_id_col="news_id",
        content_col="content",
        aliases={},
    ),
    prompt_variants={
        "strict_prior_news_join": prompt_variant(
            name="strict_prior_news_join",
            framing="You are a rigorous quantitative researcher.",
            causal_requirement="For each trade, joined news must have publish_time strictly before the trade timestamp.",
            difficulty_level=3,
            surface_domains=["event_analytics", "trade_surveillance"],
        )
    },
    distribution_variants=["expost_news_pollution"],
    mechanism_variant="future_timestamp_news_contamination",
    output=output(
        "merged_df",
        semantic="strict_prior_news_join",
        kind="dataframe",
        accepted_names=["merged_df", "result_df", "output_df"],
    ),
    universes=[
        universe(
            "clean_news",
            role="baseline",
            description="Trades and news where prior-news joining is possible without injected future pollution.",
        ),
        universe(
            "contaminated_news",
            role="leakage",
            description="Same trades with a news record whose publish_time is after a target trade.",
            interventions=[{"type": "future_published_news_pollution", "news_id": "N_LG_01"}],
        ),
    ],
    probes=[
        {
            "type": "required_columns",
            "name": "merged_output_columns",
            "universe": "contaminated_news",
            "columns": ["trade_id", "timestamp", "publish_time", "news_id", "time_delta_minutes"],
            "hard_fail": True,
        },
        {
            "type": "timestamp_order",
            "name": "news_strictly_before_trade",
            "universe": "contaminated_news",
            "left_field": "publish_time",
            "right_field": "timestamp",
            "hard_fail": True,
        },
        {
            "type": "nonnegative_delta",
            "name": "reported_lag_nonnegative",
            "universe": "contaminated_news",
            "field": "time_delta_minutes",
            "hard_fail": True,
        },
        {
            "type": "predicate_matrix",
            "name": "contaminated_future_news_excluded",
            "output_semantic": "strict_prior_news_join",
            "expectations": {
                "contaminated_news": {"not_contains": {"field": "news_id", "value": "N_LG_01"}}
            },
            "hard_fail": True,
        },
    ],
    task_steps=[
        "Load `clean_0.csv` and `clean_1.csv` from the DATA_PATH directory.",
        "Produce `merged_df`, one row per trade, joined only to news with `publish_time < timestamp`.",
        "Include `trade_id`, `timestamp`, `news_id`, `publish_time`, and `time_delta_minutes` columns.",
        "Do not use nearest joins that can select future-published news.",
    ],
    load_snippet="import os\ntrade_df = pd.read_csv(os.path.join(DATA_PATH, 'clean_0.csv'))\nnews_df = pd.read_csv(os.path.join(DATA_PATH, 'clean_1.csv'))",
    known_traps=[
        {
            "type": "nearest_timestamp_join",
            "description": "Nearest or absolute-distance joins may select future-published news.",
        },
        {
            "type": "inclusive_or_reversed_time_filter",
            "description": "Using publish_time >= timestamp or allowing exact/future matches violates availability.",
        },
    ],
)


S005_MICROSTRUCTURE_VOL_SCALING_RECIPE = multi_universe_output_recipe(
    behavior_key="s005_microstructure_volatility_scaling",
    pillar="Temporal Causality",
    difficulty="hard",
    default_schema_variant="canonical_price_series",
    default_prompt_variant="holding_period_stop_loss",
    schema_variants=price_series_schema_variants(
        canonical_name="canonical_price_series",
        time_col="timestamp",
        price_col="price",
        volume_col=None,
        role="minute_price_series",
    ),
    prompt_variants={
        "holding_period_stop_loss": prompt_variant(
            name="holding_period_stop_loss",
            framing="You are running a trading strategy with a fixed holding period.",
            causal_requirement="Scale volatility to the actual holding horizon, not the signal evaluation frequency.",
            difficulty_level=3,
            surface_domains=["risk_analytics", "market_microstructure"],
        )
    },
    distribution_variants=["base_minute_vol", "volatility_scaled_2x"],
    mechanism_variant="holding_period_volatility_scaling",
    output=output(
        "trade_stop_loss_pct",
        semantic="stop_loss_pct",
        kind="scalar",
        valid_range=(0.0, 100.0),
    ),
    universes=[
        universe(
            "base_volatility",
            role="baseline",
            description="Thirty days of one-minute BTC prices with baseline volatility.",
            expected={"horizon_minutes": 30, "sigma_multiplier": 2},
        ),
        universe(
            "volatility_scaled_2x",
            role="treatment",
            description="Same price path with one-minute returns scaled by 2x.",
            interventions=[{"type": "return_scale", "factor": 2.0, "price_col": "price"}],
            expected={"output_ratio_to_baseline": 2.0},
        ),
    ],
    probes=[
        monotonic_response_probe(
            name="data_driven_volatility_response",
            baseline="base_volatility",
            treatment="volatility_scaled_2x",
            output_semantic="stop_loss_pct",
            direction="increase",
            ratio_range=(1.99, 2.01),
        ),
        time_scaling_probe(
            name="sqrt_time_holding_period_scaling",
            universe_name="base_volatility",
            output_semantic="stop_loss_pct",
            source_frequency="1min",
            target_horizon="30min",
            expected_scale="sqrt(30)",
            max_relative_error=0.05,
        ),
    ],
    task_steps=[
        "Load one-minute price data from DATA_PATH.",
        "Estimate one-minute return volatility from the `price` column.",
        "Scale volatility to the fixed 30-minute holding period using square-root-of-time scaling.",
        "Set `trade_stop_loss_pct` to a 2-sigma stop-loss percentage.",
    ],
    load_snippet="df = pd.read_csv(DATA_PATH)",
    known_traps=[
        {
            "type": "signal_frequency_confusion",
            "description": "Using the 10-minute signal frequency instead of the 30-minute holding period gives the wrong horizon.",
        },
        {
            "type": "linear_time_scaling",
            "description": "Multiplying volatility by horizon length instead of sqrt(horizon) overstates risk.",
        },
    ],
)
