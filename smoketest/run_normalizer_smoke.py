"""Integration tests for the Intent Normalizer node.

Tests the normalizer in isolation (heuristic fallback) and integrated
into the LangGraph pipeline. These tests do NOT require an LLM —
they exercise the keyword-based fallback path.
"""

from __future__ import annotations

from agent.normalizer import normalize_intent
from agent.graph import build_minimal_agent_graph
from agent.schemas import Intent, to_dict



SYNTHETIC_TASKS = [
    {
        "name": "rolling_zscore_regime",
        "text": (
            "Build a rolling z-score regime detection signal from minute-bar price data. "
            "The data has a timestamp column. Use a rolling window of 20 periods and "
            "shift the statistics by 1 period to avoid look ahead bias. "
            "Return a series called 'signal' with the z-score values."
        ),
        "expected_domain": "systematic_trading",
        "expected_time_column": "timestamp",
        "expected_output_kind": "series",
        "expected_output_var": "signal",
        "expected_temporal_constraint": True,
    },
    {
        "name": "risk_parity_portfolio",
        "text": (
            "Compute risk-parity portfolio weights for a universe of 10 equities "
            "using daily return data indexed by date. Use expanding window covariance "
            "estimation. Return a weight vector for each rebalance date."
        ),
        "expected_domain": "portfolio_construction",
        "expected_time_column": "date",
        "expected_output_kind": "vector",
        "expected_output_var": "weights",
        "expected_temporal_constraint": False,
    },
    {
        "name": "factor_alpha_signal",
        "text": (
            "Generate a long-short factor alpha signal from a panel of stock returns "
            "with sector labels. No time-based look-ahead is allowed — use only "
            "point-in-time information. Output a dataframe with columns 'stock_id', "
            "'date', and 'alpha'."
        ),
        "expected_domain": "factor_research",
        "expected_time_column": "date",
        "expected_output_kind": "dataframe",
        "expected_output_var": "result_df",
        "expected_temporal_constraint": True,
    },
    {
        "name": "var_calculation",
        "text": (
            "Calculate the 95% Value-at-Risk for a portfolio return series. "
            "The input is a single column of daily P&L data without timestamps."
        ),
        "expected_domain": "portfolio_construction",
        "expected_time_column": None,
        "expected_output_kind": "series",
        "expected_output_var": "signal",
        "expected_temporal_constraint": False,
    },
    {
        "name": "momentum_cross_over",
        "text": (
            "Implement a dual moving average crossover signal for intraday price data. "
            "The data has minute-level timestamps. Use a fast 5-period and slow 20-period "
            "SMA. Only use information available at the decision time — no future leakage. "
            "Return a series called 'signal'."
        ),
        "expected_domain": "systematic_trading",
        "expected_time_column": "timestamp",
        "expected_output_kind": "series",
        "expected_output_var": "signal",
        "expected_temporal_constraint": True,
    },
]


def test_normalize_heuristic_fallback():
    """The normalizer must always produce a valid Intent from the heuristic path."""
    for task in SYNTHETIC_TASKS:
        result = normalize_intent(task["text"], llm_invoke=None)
        assert result.ok, f"{task['name']}: heuristic should succeed, got {result.error}"
        intent: Intent = result.result["intent"]
        assert isinstance(intent, Intent), f"{task['name']}: expected Intent, got {type(intent)}"
        assert intent.task_type == "generate_code", f"{task['name']}: task_type mismatch"


def test_normalize_domain_detection():
    """Domain should be correctly detected from keywords."""
    cases = [
        ("rolling zscore for trading", "systematic_trading"),
        ("risk parity var calculation", "risk_management"),
        ("portfolio weights with allocation", "portfolio_construction"),
        ("factor model alpha signal", "factor_research"),
        ("generic financial data", "systematic_trading"),
    ]
    for text, expected_domain in cases:
        result = normalize_intent(text, llm_invoke=None)
        assert result.ok
        assert result.result["intent"].domain == expected_domain, (
            f"Expected domain={expected_domain!r} for {text!r}, got {result.result['intent'].domain!r}"
        )


def test_normalize_output_kind():
    """Output kind and variable name should be inferred correctly."""
    cases = [
        ("return a dataframe with columns", "dataframe", "result_df"),
        ("generate a signal series", "series", "signal"),
        ("compute a single scalar value", "scalar", "result"),
        ("output a weight vector", "vector", "weights"),
        ("calculate the zscore and regime", "series", "signal"),
    ]
    for text, expected_kind, expected_var in cases:
        result = normalize_intent(text, llm_invoke=None)
        assert result.ok
        meta = result.result["intent"].metadata
        assert meta.get("output_kind") == expected_kind, (
            f"Expected output_kind={expected_kind!r} for {text!r}, got {meta.get('output_kind')!r}"
        )
        assert meta.get("output_variable_name") == expected_var, (
            f"Expected output_variable={expected_var!r} for {text!r}, got {meta.get('output_variable_name')!r}"
        )


def test_normalize_temporal_constraint():
    """When lookahead/future/temporal keywords are present, the constraint should be added."""
    text = "point-in-time factor signals without future leakage"
    result = normalize_intent(text, llm_invoke=None)
    assert result.ok
    constraints = result.result["intent"].explicit_user_constraints
    temporal_found = any("timing" in c.lower() for c in constraints)
    assert temporal_found, f"Expected temporal constraint in {constraints}"
    assert result.result["intent"].metadata.get("pillar") == "temporal_causality"


def test_normalize_time_column_detection():
    """Time column should be inferred from metadata keywords."""
    cases = [
        ("data has a timestamp column", "timestamp"),
        ("daily data indexed by date", "date"),
        ("intraday data with time index", "time")
    ]
    for text, expected_col in cases:
        result = normalize_intent(text, llm_invoke=None)
        assert result.ok
        meta = result.result["intent"].metadata
        if expected_col is None:
            assert "time_column" not in meta or meta["time_column"] is None
        else:
            assert meta.get("time_column") == expected_col, (
                f"Expected time_column={expected_col!r} for {text!r}, got {meta.get('time_column')!r}"
            )


def test_normalize_empty_input():
    """Empty input should return a failure."""
    result = normalize_intent("", llm_invoke=None)
    assert not result.ok
    assert result.error is not None
    assert "empty" in result.error.message.lower()

    result = normalize_intent("   ", llm_invoke=None)
    assert not result.ok


def test_normalize_graph_entry_point():
    """The graph should route through normalizer and produce a valid contract."""
    app = build_minimal_agent_graph()
    state = app.invoke(
        {
            "user_text": "Build a momentum signal from daily price data with a timestamp column. "
                         "Use a 12-month rolling lookback. No future leakage allowed.",
            "contract_source": "intent",
            "generate_code": False,
            "repair_attempts": 0,
            "max_repair_attempts": 0,
            "fixtures": [],
            "timeout_seconds": 10,
        }
    )

    # Normalizer should have produced an Intent
    intent = state.get("intent")
    assert intent is not None, "Graph did not produce an intent"
    assert isinstance(intent, Intent), f"Expected Intent, got {type(intent)}"
    # 'momentum' doesn't trigger factor_detection, but 'signal' doesn't either
    # Default domain is systematic_trading — the text says "trading" via "momentum" context
    # Actually 'signal' matches factor_research, but 'momentum signal' has 'factor' indirectly
    # The heuristic catches 'factor' only when that exact word appears, so this defaults
    assert intent.domain == "systematic_trading", f"Unexpected domain: {intent.domain}"
    assert "momentum" in intent.input_data_description.lower() or "momentum" in intent.requested_output.lower()

    # Contract Builder should have consumed the intent
    contract = state.get("contract")
    assert contract is not None, "Graph did not produce a contract"
    assert contract.contract_id is not None

    # Known hazards should include temporal ones (timestamp + no future leakage)
    assert "negative_shift" in contract.known_hazards
    assert "backward_fill" in contract.known_hazards


def test_normalize_graph_empty_text():
    """Empty user_text should route to finalize with normalization_failed status."""
    app = build_minimal_agent_graph()
    state = app.invoke(
        {
            "user_text": "",
            "contract_source": "intent",
            "repair_attempts": 0,
            "max_repair_attempts": 0,
            "fixtures": [],
            "timeout_seconds": 10,
        }
    )
    assert state.get("final_status") == "normalization_failed", (
        f"Expected normalization_failed, got {state.get('final_status')}"
    )
    assert state.get("contract") is None
    assert state.get("intent") is None


def test_normalize_trace_recording():
    """The normalizer step should be recorded in AgentTrace tool_calls."""
    app = build_minimal_agent_graph()
    state = app.invoke(
        {
            "user_text": "Compute risk-parity weights for 10 equities with daily date index.",
            "contract_source": "intent",
            "generate_code": False,
            "repair_attempts": 0,
            "max_repair_attempts": 0,
            "fixtures": [],
            "timeout_seconds": 10,
        }
    )
    trace = state.get("trace")
    assert trace is not None
    assert len(trace.tool_calls) >= 1
    first_call = trace.tool_calls[0]
    assert first_call.get("tool_name") == "normalize_intent"
    assert first_call.get("ok") is True


def test_normalize_produces_valid_intent_for_dev_recipe():
    """The dev_recipe path (bypassing normalizer) should still work unchanged."""
    from eval.recipes.catalog.temporal_causality_recipes import (
        S013_TEMPORAL_ROLLING_ZSCORE_LEAKAGE_RECIPE as recipe,
    )

    app = build_minimal_agent_graph()
    state = app.invoke(
        {
            "recipe": recipe,
            "contract_source": "dev_recipe",
            "generate_code": False,
            "repair_attempts": 0,
            "max_repair_attempts": 0,
            "fixtures": [],
            "timeout_seconds": 10,
        }
    )
    assert state.get("contract") is not None
    assert "negative_shift" in state["contract"].known_hazards


def test_normalize_heuristic_all_synthetic():
    """Run all synthetic tasks through the heuristic normalizer and validate key fields."""
    for task in SYNTHETIC_TASKS:
        result = normalize_intent(task["text"], llm_invoke=None)
        assert result.ok, f"{task['name']}: heuristic failed"

        intent = result.result["intent"]
        meta = intent.metadata

        # Domain
        assert intent.domain == task["expected_domain"], (
            f"{task['name']}: domain mismatch: expected {task['expected_domain']}, got {intent.domain}"
        )

        # Time column
        if task["expected_time_column"] is not None:
            assert meta.get("time_column") == task["expected_time_column"], (
                f"{task['name']}: expected time_column {task['expected_time_column']}, got {meta.get('time_column')}"
            )

        # Output kind
        assert meta.get("output_kind") == task["expected_output_kind"], (
            f"{task['name']}: expected output_kind {task['expected_output_kind']}, got {meta.get('output_kind')}"
        )

        # Output variable
        assert meta.get("output_variable_name") == task["expected_output_var"], (
            f"{task['name']}: expected output_var {task['expected_output_var']}, got {meta.get('output_variable_name')}"
        )

        # Temporal constraint
        has_temporal = any("timing" in c.lower() for c in intent.explicit_user_constraints)
        assert has_temporal == task["expected_temporal_constraint"], (
            f"{task['name']}: temporal constraint mismatch: expected={task['expected_temporal_constraint']}, got={has_temporal}"
        )


if __name__ == "__main__":
    import sys

    tests = [
        ("test_normalize_heuristic_fallback", test_normalize_heuristic_fallback),
        ("test_normalize_domain_detection", test_normalize_domain_detection),
        ("test_normalize_output_kind", test_normalize_output_kind),
        ("test_normalize_temporal_constraint", test_normalize_temporal_constraint),
        ("test_normalize_time_column_detection", test_normalize_time_column_detection),
        ("test_normalize_empty_input", test_normalize_empty_input),
        ("test_normalize_graph_entry_point", test_normalize_graph_entry_point),
        ("test_normalize_graph_empty_text", test_normalize_graph_empty_text),
        ("test_normalize_trace_recording", test_normalize_trace_recording),
        ("test_normalize_produces_valid_intent_for_dev_recipe", test_normalize_produces_valid_intent_for_dev_recipe),
        ("test_normalize_heuristic_all_synthetic", test_normalize_heuristic_all_synthetic),
    ]

    failed = 0
    for name, func in tests:
        try:
            func()
            print(f"  PASS  {name}")
        except Exception as exc:
            print(f"  FAIL  {name}: {exc}")
            failed += 1

    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
