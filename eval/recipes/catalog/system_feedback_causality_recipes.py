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



S004_SELF_EXECUTION_POLLUTION_RECIPE = multi_universe_output_recipe(
    behavior_key="s004_self_execution_pollution",
    pillar="System Feedback Causality",
    difficulty="medium",
    default_schema_variant="canonical_trade_tape",
    default_prompt_variant="self_trade_filtering",
    schema_variants=trade_tape_schema_variants(
        canonical_name="canonical_trade_tape",
        price_col="price",
        volume_col="volume",
        buyer_col="buyer",
        seller_col="seller",
        trade_id_col="trade_id",
        role="public_trade_tape",
    ),
    prompt_variants={
        "self_trade_filtering": prompt_variant(
            name="self_trade_filtering",
            framing="You are building a real-time alpha model that updates quotes from public trades.",
            causal_requirement="Trades involving the strategy itself must not feed back into its price update.",
            difficulty_level=2,
            surface_domains=["market_microstructure", "execution"],
        )
    },
    distribution_variants=["self_trade_contaminated_tape"],
    mechanism_variant="self_execution_feedback_loop",
    output=output(
        "updated_price",
        semantic="updated_price",
        kind="scalar",
        accepted_names=["updated_price", "price_a", "price_b"],
    ),
    universes=[
        universe(
            "strategy_config",
            role="baseline",
            description="Config identifies the real strategy id whose own trades must be filtered.",
            expected={"trader_id": "strategy_001", "max_updated_price": 100.5},
        ),
        universe(
            "ghost_config",
            role="treatment",
            description="Counterfactual config uses a trader id absent from the tape, so no self-trades are filtered.",
            expected={"trader_id": "ghost_999", "price_delta_min": 0.02},
        ),
    ],
    probes=[
        config_counterfactual_probe(
            name="dynamic_self_trade_filtering",
            baseline_config="strategy_config",
            counterfactual_config="ghost_config",
            output_semantic="updated_price",
            direction="increase",
            min_delta=0.02,
        ),
        output_bounds_probe(
            name="safe_price_update_after_filtering",
            universe_name="strategy_config",
            output_semantic="updated_price",
            min_value=100.0,
            max_value=100.5,
        ),
    ],
    task_steps=[
        "Implement `update_price(current_price, trades, config)`.",
        "Filter trades where `config['trader_id']` appears as buyer or seller.",
        "Use the remaining public market trades to update the price without overreacting.",
        "Return the updated price as a float.",
    ],
    load_snippet="trades_df = TRADES_DATA if not isinstance(TRADES_DATA, str) else pd.read_csv(TRADES_DATA)",
    known_traps=[
        {
            "type": "self_execution_feedback_loop",
            "description": "Averaging all trades includes the strategy's own executions and can inflate its next quote.",
        },
        {
            "type": "hardcoded_trader_filter",
            "description": "Filtering a fixed id instead of `config['trader_id']` fails counterfactual configs.",
        },
    ],
)


S006_INVENTORY_SKEW_RECIPE = multi_universe_output_recipe(
    behavior_key="s006_inventory_induced_skew",
    pillar="System Feedback Causality",
    difficulty="hard",
    default_schema_variant="canonical_execution_log",
    default_prompt_variant="systematic_inventory_skew",
    schema_variants=execution_log_schema_variants(
        canonical_name="canonical_execution_log",
        time_col="timestamp",
        side_col="side",
        qty_col="qty",
        price_col="price",
        volatility_col="vol_1min",
        role="execution_log",
    ),
    prompt_variants={
        "systematic_inventory_skew": prompt_variant(
            name="systematic_inventory_skew",
            framing="You are a systematic trading agent managing BTC inventory risk.",
            causal_requirement="Quote skew must respond systematically to inventory deviation and become stronger in higher volatility.",
            difficulty_level=3,
            surface_domains=["market_making", "risk_analytics"],
        )
    },
    distribution_variants=["neutral_inventory", "long_low_vol", "long_high_vol"],
    mechanism_variant="inventory_skew_with_volatility_amplification",
    output=output(
        "quote_state",
        semantic="quote_state",
        kind="dict",
        accepted_names=["quote_state", "eval_result"],
    ),
    universes=[
        universe(
            "neutral_inventory",
            role="baseline",
            description="Alternating buys and sells should leave near-zero net inventory and minimal skew.",
            expected={"net_position_abs_max": 0.8},
        ),
        universe(
            "long_low_vol",
            role="treatment",
            description="Long inventory under low volatility should skew both quotes downward.",
            expected={"net_position": 12.5, "mid_drop_min": 4.0},
        ),
        universe(
            "long_high_vol",
            role="stress",
            description="Same long inventory under higher volatility should create stronger downward skew.",
            expected={"net_position": 12.5, "mid_drop_ratio_to_low_vol": 3.0},
        ),
    ],
    probes=[
        field_bounds_probe(
            name="neutral_net_position",
            universe_name="neutral_inventory",
            output_semantic="quote_state",
            field="net_position",
            min_value=-0.8,
            max_value=0.8,
        ),
        field_bounds_probe(
            name="long_net_position",
            universe_name="long_low_vol",
            output_semantic="quote_state",
            field="net_position",
            min_value=11.7,
            max_value=13.3,
        ),
        derived_metric_monotonic_probe(
            name="inventory_pushes_mid_down",
            baseline="neutral_inventory",
            treatment="long_low_vol",
            output_semantic="quote_state",
            metric="mid_price_drop",
            direction="increase",
            min_delta=4.0,
        ),
        derived_metric_monotonic_probe(
            name="volatility_amplifies_inventory_skew",
            baseline="long_low_vol",
            treatment="long_high_vol",
            output_semantic="quote_state",
            metric="mid_price_drop",
            direction="increase",
            ratio_range=(2.6, 3.4),
        ),
    ],
    task_steps=[
        "Load the execution log from DATA_PATH.",
        "Compute `net_position` from BUY/SELL signed quantities.",
        "Read the latest one-minute volatility from `vol_1min`.",
        "Skew bid and ask quotes to help unwind inventory, with stronger skew at higher volatility.",
        "Set final `net_position`, `bid_price`, and `ask_price` outputs.",
    ],
    load_snippet="execution_log = pd.read_csv(DATA_PATH)",
    known_traps=[
        {
            "type": "inventory_blind_quotes",
            "description": "Symmetric quoting ignores accumulated inventory risk.",
        },
        {
            "type": "volatility_blind_skew",
            "description": "A fixed skew ignores higher risk when volatility rises.",
        },
    ],
)
