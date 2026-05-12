from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.testing import assert_frame_equal

from agent.schemas import (
    CausalContract,
    InvariantCheckResult,
    InvariantResults,
    ToolResult,
    to_dict,
)
from eval.execution.execution_models import InputBinding
from eval.scoring.generic_recipe_scorer import GenericRecipeScorer


def run_invariant_checks(
    code: str,
    contract: CausalContract,
    fixtures: list[Any],
    *,
    timeout_seconds: int = 10,
    mode: str = "basic",
    strict_mode: bool = False,
    checks: list[str] | None = None,
) -> ToolResult:
    selected = checks or [item.id for item in contract.required_invariants]
    results: list[InvariantCheckResult] = []
    warnings: list[str] = []

    alias_map = {"output_contract": {"row_alignment", "index_alignment", "shape_alignment"}}

    effective = set()
    for item in selected:
        if item in alias_map:
            effective.update(alias_map[item])
        else:
            effective.add(item)

    if effective & {"row_alignment", "index_alignment", "shape_alignment", "output_alignment"}:
        results.append(_check_output_contract(code, contract, fixtures, timeout_seconds))

    if "prefix_invariance" in effective:
        prefix_result = _check_prefix_invariance(code, contract, fixtures, timeout_seconds)
        if prefix_result is None:
            message = "Prefix invariance skipped because the fixture or contract is not time-indexed dataframe data."
            if strict_mode:
                results.append(
                    InvariantCheckResult(
                        invariant_id="prefix_invariance",
                        passed=False,
                        message=message,
                    )
                )
            else:
                warnings.append(message)
        else:
            results.append(prefix_result)

    aggregate = InvariantResults(
        passed=all(item.passed for item in results),
        checks=results,
        warnings=warnings,
    )
    return ToolResult.success(
        to_dict(aggregate),
        warnings=warnings,
        metadata={
            "tool_name": "run_invariant_checks",
            "mode": mode,
            "strict_mode": strict_mode,
            "checks": selected,
        },
    )


def _check_output_contract(
    code: str,
    contract: CausalContract,
    fixtures: list[Any],
    timeout_seconds: int,
) -> InvariantCheckResult:
    if not fixtures:
        return InvariantCheckResult(
            invariant_id="output_contract",
            passed=False,
            message="No fixture provided for output contract check.",
        )

    execution = _execute_for_fixture(code, contract, fixtures[0], timeout_seconds)
    if not execution["success"]:
        return InvariantCheckResult(
            invariant_id="output_contract",
            passed=False,
            message=execution["error"] or "Execution failed.",
            evidence={"execution": execution},
        )

    rows = execution["rows"]
    columns = execution["columns"]
    output = contract.output_contract
    passed = True
    messages: list[str] = []

    if output.kind in {"series", "dataframe"} and _fixture_row_count(fixtures[0]) is not None:
        expected_rows = _fixture_row_count(fixtures[0])
        if rows != expected_rows:
            passed = False
            messages.append(f"Expected {expected_rows} output rows, got {rows}.")

    accepted = set(output.accepted_names or [output.variable_name])
    if output.kind in {"series", "scalar", "dict"} and columns and not accepted.intersection(columns):
        passed = False
        messages.append(f"Output columns {columns} do not include accepted names {sorted(accepted)}.")

    return InvariantCheckResult(
        invariant_id="output_contract",
        passed=passed,
        message="Output contract satisfied." if passed else " ".join(messages),
        evidence={"rows": rows, "columns": columns},
    )


def _check_prefix_invariance(
    code: str,
    contract: CausalContract,
    fixtures: list[Any],
    timeout_seconds: int,
) -> InvariantCheckResult | None:
    if not fixtures:
        return None
    fixture = fixtures[0]
    if not isinstance(getattr(fixture, "data", None), pd.DataFrame):
        return None
    time_column = contract.decision_context.time_column
    if not time_column or time_column not in fixture.data.columns:
        return None
    if len(fixture.data) < 20:
        return None

    original = fixture.data.sort_values(time_column).reset_index(drop=True)
    split_idx = max(1, len(original) // 2)
    perturbed = original.copy(deep=True)
    future_slice = perturbed.index[split_idx:]
    _perturb_future_rows(perturbed, future_slice, exclude={time_column})

    original_result = _execute_for_dataframe(code, contract, original, timeout_seconds)
    perturbed_result = _execute_for_dataframe(code, contract, perturbed, timeout_seconds)
    if not original_result["success"] or not perturbed_result["success"]:
        return InvariantCheckResult(
            invariant_id="prefix_invariance",
            passed=False,
            message="Execution failed during prefix invariance check.",
            evidence={"original": original_result, "perturbed": perturbed_result},
        )

    original_df = original_result["output_df"]
    perturbed_df = perturbed_result["output_df"]
    if original_df is None or perturbed_df is None:
        return InvariantCheckResult(
            invariant_id="prefix_invariance",
            passed=False,
            message="Prefix invariance check did not produce comparable outputs.",
        )

    prefix_original = original_df.iloc[:split_idx].reset_index(drop=True)
    prefix_perturbed = perturbed_df.iloc[:split_idx].reset_index(drop=True)
    passed = _frames_equal_ignoring_dtype(prefix_original, prefix_perturbed)
    return InvariantCheckResult(
        invariant_id="prefix_invariance",
        passed=passed,
        message=(
            "Future perturbation did not alter prior outputs."
            if passed
            else "Changing future rows altered outputs before the perturbation boundary."
        ),
        evidence={
            "split_idx": split_idx,
            "prefix_rows_compared": len(prefix_original),
            "original_columns": list(prefix_original.columns),
            "perturbed_columns": list(prefix_perturbed.columns),
        },
    )


def _execute_for_fixture(
    code: str,
    contract: CausalContract,
    fixture: Any,
    timeout_seconds: int,
) -> dict[str, Any]:
    scorer = GenericRecipeScorer(timeout_seconds=timeout_seconds)
    binding = scorer._fixture_binding(fixture)
    return _execute_with_binding(code, contract, binding, timeout_seconds)


def _execute_for_dataframe(
    code: str,
    contract: CausalContract,
    df: pd.DataFrame,
    timeout_seconds: int,
) -> dict[str, Any]:
    binding = InputBinding(name="DATA_PATH", kind="file", data=df)
    return _execute_with_binding(code, contract, binding, timeout_seconds)


def _execute_with_binding(
    code: str,
    contract: CausalContract,
    binding: InputBinding,
    timeout_seconds: int,
) -> dict[str, Any]:
    scorer = GenericRecipeScorer(timeout_seconds=timeout_seconds)
    collector_code = scorer._with_output_collector(code, contract.output_contract.accepted_names)
    result = scorer.executor.run_with_bindings(collector_code, [binding])
    return {
        "success": result.success,
        "error": result.error,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "rows": None if result.output_df is None else len(result.output_df),
        "columns": [] if result.output_df is None else list(result.output_df.columns),
        "output_df": result.output_df,
    }


def _fixture_row_count(fixture: Any) -> int | None:
    data = getattr(fixture, "data", None)
    if isinstance(data, pd.DataFrame):
        return len(data)
    return None


def _perturb_future_rows(df: pd.DataFrame, rows: pd.Index, *, exclude: set[str]) -> None:
    for column in df.columns:
        if column in exclude:
            continue
        series = df[column]
        if pd.api.types.is_bool_dtype(series):
            df.loc[rows, column] = ~series.loc[rows]
        elif pd.api.types.is_numeric_dtype(series):
            df[column] = df[column].astype(float)
            series = df[column]
            scale = series.std()
            if pd.isna(scale) or scale == 0:
                scale = 1.0
            df.loc[rows, column] = series.loc[rows] + (100.0 * scale)
        else:
            df.loc[rows, column] = "__future_perturbed__"


def _frames_equal_ignoring_dtype(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    try:
        assert_frame_equal(left, right, check_dtype=False, check_like=False)
        return True
    except AssertionError:
        return False
