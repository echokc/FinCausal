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



def _regime_changepoint_recipe(
    *,
    behavior_key: str,
    prompt_name: str,
    framing: str,
    causal_requirement: str,
    surface_domains: list[str],
    known_traps: list[dict[str, str]],
) -> object:
    return multi_universe_output_recipe(
        behavior_key=behavior_key,
        pillar="Regime Causality",
        difficulty="hard",
        default_schema_variant="canonical_regime_panel",
        default_prompt_variant=prompt_name,
        schema_variants={
            "canonical_regime_panel": {
                "role": "multi_asset_return_regime_panel",
                "time_col": "timestamp",
                "return_a_col": "return_a",
                "return_b_col": "return_b",
                "price_a_col": "price_a",
                "price_b_col": "price_b",
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
        distribution_variants=["stable_single_regime", "multi_structural_break_sequence"],
        mechanism_variant="multi_break_regime_misclassification",
        output=output(
            "regime_df",
            semantic=prompt_name,
            kind="dataframe",
            accepted_names=["regime_df", "diagnostic_df", "result_df", "output_df", "df"],
        ),
        universes=[
            universe(
                "clean",
                role="baseline",
                description="Stable return panel with constant mean, volatility, persistence, and cross-asset correlation.",
                expected={"major_break_count": 0},
            ),
            universe(
                "shock",
                role="treatment",
                description="Identical prefix followed by multiple abrupt and gradual structural breaks across mean, volatility, persistence, and correlation.",
                expected={"major_break_count": 3, "min_regime_count": 3},
            ),
        ],
        probes=[
            {
                "type": "code_pattern_present",
                "name": "changepoint_method_present",
                "patterns": [
                    r"changepoint|change[_-]?point[_-]?(?:detection|method|algorithm|score)",
                    r"\bpelt\b|binary[_\s-]?segmentation|ruptures|hmm|bai|perron|bayesian",
                    r"\bcusum\b|cumulative\s+sum|np\.cumsum",
                ],
                "mode": "any",
                "reason": "The implementation should use a change-point style method rather than a single fixed threshold rule.",
                "hard_fail": True,
            },
            {
                "type": "code_pattern_absent",
                "name": "no_single_fixed_threshold_rule",
                "patterns": [
                    r"vol(?:atility)?\s*>\s*[0-9.]+",
                    r"std\s*\([^)]*\)\s*>\s*[0-9.]+",
                    r"if\s+[^:\n]*(?:vol|std|sigma)[^:\n]*>\s*[0-9.]+",
                ],
                "reason": "A fixed volatility threshold is not enough to detect multiple abrupt, gradual, correlation, and persistence breaks.",
                "hard_fail": False,
            },
            {
                "type": "required_columns",
                "name": f"{prompt_name}_columns_present",
                "universe": "shock",
                "columns": ["regime", "change_point", "regime_confidence"],
                "hard_fail": True,
            },
            {
                "type": "field_unique_count",
                "name": "clean_has_minimal_false_regimes",
                "universe": "clean",
                "field": "regime",
                "max_unique": 2,
                "hard_fail": True,
            },
            {
                "type": "field_unique_count",
                "name": "shock_has_multiple_regimes",
                "universe": "shock",
                "field": "regime",
                "min_unique": 3,
                "hard_fail": True,
            },
            dataframe_window_threshold_probe(
                name="detects_first_structural_break",
                universe_name="shock",
                output_semantic=prompt_name,
                field="change_point",
                start_idx_from_metadata="expected.break_1_window_start",
                end_idx_from_metadata="expected.break_1_window_end",
                min_abs_max=1.0,
            ),
            dataframe_window_threshold_probe(
                name="detects_second_structural_break_or_drift",
                universe_name="shock",
                output_semantic=prompt_name,
                field="change_point",
                start_idx_from_metadata="expected.break_2_window_start",
                end_idx_from_metadata="expected.break_2_window_end",
                min_abs_max=1.0,
            ),
            dataframe_window_threshold_probe(
                name="detects_third_persistence_or_correlation_break",
                universe_name="shock",
                output_semantic=prompt_name,
                field="change_point",
                start_idx_from_metadata="expected.break_3_window_start",
                end_idx_from_metadata="expected.break_3_window_end",
                min_abs_max=1.0,
            ),
            dataframe_window_threshold_probe(
                name="clean_avoids_major_false_change_points",
                universe_name="clean",
                output_semantic=prompt_name,
                field="change_point",
                start_idx=0,
                end_idx=479,
                max_abs_max=0.0,
            ),
        ],
        task_steps=[
            "Load the return panel from DATA_PATH.",
            "Detect and label regimes using a change-point method that can capture multiple abrupt breaks, gradual drifts, correlation shifts, and persistence changes.",
            "Return `regime_df` aligned to the input rows with at least `regime`, `change_point`, and `regime_confidence` columns.",
            "Avoid single fixed-threshold volatility rules and validate that regimes persist across segments rather than flipping on isolated noisy observations.",
        ],
        load_snippet="df = pd.read_csv(DATA_PATH)",
        known_traps=known_traps,
    )


S023_REGIME_MULTIPLE_STRUCTURAL_BREAKS_MISCLASSIFICATION_RECIPE = _regime_changepoint_recipe(
    behavior_key="s023_regime_multiple_structural_breaks_misclassification",
    prompt_name="multi_structural_break_regime_detection",
    framing="You are a quantitative researcher detecting market regimes across a multi-asset return panel.",
    causal_requirement="The regime detector must identify multiple structural breaks, including abrupt breaks, gradual drifts, and persistent changes in volatility, trend, and correlation.",
    surface_domains=["regime_detection", "change_point_detection", "multi_asset_research", "risk_models"],
    known_traps=[
        {"type": "single_break_detector", "description": "Detecting only the largest abrupt break misses earlier or later structural changes."},
        {"type": "fixed_volatility_threshold", "description": "A fixed volatility cutoff cannot distinguish gradual drift, correlation breaks, and persistence changes."},
        {"type": "regime_chatter", "description": "Labels that flip on isolated noisy rows do not demonstrate persistent regimes."},
    ],
)


S024_REGIME_SLOW_DRIFT_MISCLASSIFICATION_RECIPE = _regime_changepoint_recipe(
    behavior_key="s024_regime_slow_drift_misclassification",
    prompt_name="slow_drift_regime_detection",
    framing="You are detecting regimes where volatility and trend can drift gradually over several weeks before stabilizing.",
    causal_requirement="The implementation must detect gradual regime drift as a structural change, not only abrupt jumps.",
    surface_domains=["regime_detection", "slow_drift", "volatility_models", "trend_following"],
    known_traps=[
        {"type": "abrupt_only_detector", "description": "Rules tuned to flash-crash jumps often miss slow volatility or trend transitions."},
        {"type": "drift_as_noise", "description": "Smoothing away persistent drift leaves the strategy in the wrong regime for too long."},
    ],
)


S025_REGIME_CORRELATION_BREAK_MISCLASSIFICATION_RECIPE = _regime_changepoint_recipe(
    behavior_key="s025_regime_correlation_break_misclassification",
    prompt_name="correlation_break_regime_detection",
    framing="You are monitoring cross-asset regimes where crisis periods can sharply change correlation structure.",
    causal_requirement="The detector must capture breaks in cross-asset correlation, not only univariate volatility or mean shifts.",
    surface_domains=["correlation_breaks", "risk_parity", "cross_asset_allocation", "cta"],
    known_traps=[
        {"type": "univariate_only_regime_detection", "description": "Looking only at one asset's volatility can miss a correlation structure break."},
        {"type": "crisis_correlation_blindness", "description": "Risk parity and allocation models fail if they ignore correlation spikes or collapses."},
    ],
)


S026_REGIME_VOL_CLUSTERING_VS_SHIFT_RECIPE = _regime_changepoint_recipe(
    behavior_key="s026_regime_volatility_clustering_vs_regime_shift",
    prompt_name="vol_clustering_vs_regime_shift_detection",
    framing="You are distinguishing transient volatility clustering from durable structural regime shifts.",
    causal_requirement="The implementation must avoid treating short GARCH-like volatility clusters as persistent structural breaks, while still detecting true durable shifts.",
    surface_domains=["volatility_clustering", "risk_model_switching", "regime_detection", "garch"],
    known_traps=[
        {"type": "cluster_as_regime_shift", "description": "Temporary volatility clusters can be mistaken for durable structural regimes."},
        {"type": "true_shift_as_cluster", "description": "A persistent post-break volatility level should not be dismissed as transient clustering."},
    ],
)


S027_REGIME_PERSISTENCE_DURATION_MISESTIMATION_RECIPE = _regime_changepoint_recipe(
    behavior_key="s027_regime_persistence_duration_misestimation",
    prompt_name="persistent_duration_regime_detection",
    framing="You are estimating regime labels and durations for live strategy switching.",
    causal_requirement="The detector must estimate persistent regime durations without switching too early, too late, or repeatedly within a true segment.",
    surface_domains=["regime_duration", "strategy_switching", "risk_control", "change_point_detection"],
    known_traps=[
        {"type": "duration_underestimation", "description": "Detecting a break but ending the new regime too early causes premature strategy switching."},
        {"type": "duration_overestimation", "description": "Keeping a stale regime label after the next break leaves the strategy using outdated assumptions."},
    ],
)


S007_LIQUIDITY_ILLUSION_RECIPE = multi_universe_output_recipe(
    behavior_key="s007_liquidity_illusion_realizable_pnl",
    pillar="Regime Causality",
    difficulty="hard",
    default_schema_variant="canonical_price_series",
    default_prompt_variant="realizable_execution_pnl",
    schema_variants=price_series_schema_variants(
        canonical_name="canonical_price_series",
        time_col="timestamp",
        price_col="close",
        volume_col="volume",
        role="liquidity_aware_price_series",
    ),
    prompt_variants={
        "realizable_execution_pnl": prompt_variant(
            name="realizable_execution_pnl",
            framing="You are a professional trading agent evaluating whether a backtest is physically realizable.",
            causal_requirement="PnL must account for liquidity, partial fills, and endogenous slippage rather than assuming infinite execution.",
            difficulty_level=3,
            surface_domains=["backtesting", "execution", "market_microstructure"],
        )
    },
    distribution_variants=["normal_liquidity", "thin_liquidity"],
    mechanism_variant="liquidity_constraint_and_slippage",
    output=output(
        "execution_result",
        semantic="execution_result",
        kind="dict",
        accepted_names=["execution_result", "realizable_pnl", "result"],
    ),
    universes=[
        universe(
            "normal_liquidity",
            role="baseline",
            description="Flash-dip strategy with enough volume to execute meaningful size.",
            expected={"volume_level": 80.0, "pnl_abs_min": 10.0},
        ),
        universe(
            "thin_liquidity",
            role="treatment",
            description="Same price path under extremely thin volume where fills should shrink and slippage should rise.",
            interventions=[{"type": "liquidity_scale", "volume_col": "volume", "level": 1.2}],
            expected={"total_bought_ratio_max": 0.5, "slippage_ratio_min": 1.5},
        ),
    ],
    probes=[
        field_bounds_probe(
            name="normal_pnl_nontrivial",
            universe_name="normal_liquidity",
            output_semantic="execution_result",
            field="pnl",
            min_value=10.0,
        ),
        field_monotonic_probe(
            name="thin_liquidity_reduces_fills",
            baseline="normal_liquidity",
            treatment="thin_liquidity",
            output_semantic="execution_result",
            field="total_bought",
            direction="decrease",
            ratio_range=(0.0, 0.5),
        ),
        field_monotonic_probe(
            name="thin_liquidity_increases_slippage",
            baseline="normal_liquidity",
            treatment="thin_liquidity",
            output_semantic="execution_result",
            field="avg_slippage_pct",
            direction="increase",
            ratio_range=(1.5, None),
        ),
    ],
    task_steps=[
        "Implement `calculate_realizable_pnl(df)` for the dip-buying strategy.",
        "Respect available volume and partial fills when buying.",
        "Model slippage or impact as liquidity becomes thin.",
        "Return a dict with `pnl`, `total_bought`, and `avg_slippage_pct`.",
    ],
    load_snippet="df = pd.read_csv(DATA_PATH)",
    known_traps=[
        {
            "type": "infinite_liquidity_assumption",
            "description": "Computing PnL from price differences alone ignores whether the strategy could actually buy size.",
        },
        {
            "type": "exogenous_fill_price",
            "description": "Using close prices as guaranteed execution prices ignores endogenous slippage and impact.",
        },
    ],
)


S008_VOL_SIGNATURE_MISCLASSIFICATION_RECIPE = multi_universe_output_recipe(
    behavior_key="s008_vol_signature_misclassification",
    pillar="Regime Causality",
    difficulty="hard",
    default_schema_variant="canonical_price_series",
    default_prompt_variant="regime_aware_position_sizing",
    schema_variants=price_series_schema_variants(
        canonical_name="canonical_price_series",
        time_col="timestamp",
        price_col="close",
        volume_col="volume",
        role="regime_price_series",
    ),
    prompt_variants={
        "regime_aware_position_sizing": prompt_variant(
            name="regime_aware_position_sizing",
            framing="You are a professional systematic trading agent running a validated mean-reversion strategy.",
            causal_requirement="Position sizing must distinguish persistent trend regimes from transient shock regimes without using future leakage.",
            difficulty_level=3,
            surface_domains=["regime_detection", "systematic_trading"],
        )
    },
    distribution_variants=["persistent_trend", "transient_shock", "leakage_sentinel"],
    mechanism_variant="regime_discrimination_with_leakage_sentinel",
    output=output(
        "position",
        semantic="position",
        kind="scalar",
        valid_range=(0.0, 100.0),
    ),
    universes=[
        universe(
            "persistent_trend",
            role="baseline",
            description="Trending market where mean-reversion size should remain modest.",
            expected={"position_abs_max": 22},
        ),
        universe(
            "transient_shock",
            role="treatment",
            description="Transient shock regime where mean-reversion position should increase.",
            interventions=[{"type": "transient_price_shock", "window": "last_day", "multiplier": 0.91}],
            expected={"position_min": 48},
        ),
        universe(
            "leakage_sentinel",
            role="leakage",
            description="Sentinel path where final future point collapses full-series z-score leakage.",
            interventions=[{"type": "last_point_mean_sentinel", "price_col": "close"}],
            expected={"position_min": 45},
        ),
    ],
    probes=[
        output_bounds_probe(
            name="trend_position_modest",
            universe_name="persistent_trend",
            output_semantic="position",
            min_value=-22,
            max_value=22,
        ),
        monotonic_response_probe(
            name="shock_regime_position_increases",
            baseline="persistent_trend",
            treatment="transient_shock",
            output_semantic="position",
            direction="increase",
            min_delta=26,
        ),
        leakage_sentinel_probe(
            name="no_full_series_zscore_leakage",
            clean_reference="transient_shock",
            leakage_universe="leakage_sentinel",
            output_semantic="position",
            max_abs_delta=20,
        ),
    ],
    task_steps=[
        "Load the price history from DATA_PATH.",
        "Implement `get_position_size(hist_df, max_position=100)`.",
        "Distinguish persistent trend from transient shock before sizing a mean-reversion trade.",
        "Store the final position in variable `position`.",
    ],
    load_snippet="price = pd.read_csv(DATA_PATH)",
    known_traps=[
        {
            "type": "regime_blind_mean_reversion",
            "description": "Applying symmetric mean reversion to both trend and shock regimes ignores causal regime differences.",
        },
        {
            "type": "full_series_zscore_leakage",
            "description": "Using full-history statistics including sentinel future points can collapse the signal.",
        },
    ],
)
