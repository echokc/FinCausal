from __future__ import annotations

from typing import Callable

from agent.schemas import CausalContract, ToolResult


def build_repair_prompt(
    *,
    contract: CausalContract,
    current_code: str,
    failure_summary: str,
    attempt: int,
) -> str:
    output = contract.output_contract
    accepted_names = ", ".join(f"`{name}`" for name in output.accepted_names)
    return f"""Your previous Python code failed a public Layer 2 validation gate.

Rewrite the code as a complete replacement. Fix only the concrete failures below.
Do not infer hidden evaluation details. Do not hardcode fixture-specific values.

## Attempt
{attempt}

## Failure Summary
{failure_summary}

## Causal Contract
- Decision unit: `{contract.decision_context.decision_unit}`
- Time column: `{contract.decision_context.time_column}`
- Decision time: `{contract.decision_context.decision_time}`
- Estimation scope: `{contract.estimation_scope.fit_scope}`
- Forbid full-sample fit: `{contract.estimation_scope.forbid_full_sample_fit}`

Forbidden information:
{_bullets(item.description for item in contract.forbidden_information)}

Required invariants:
{_bullets(f"{item.id}: {item.description}" for item in contract.required_invariants)}

Known hazards:
{_bullets(contract.known_hazards)}

## Output Contract
- Output kind: `{output.kind}`
- Primary output variable: `{output.variable_name}`
- Accepted output variable names: {accepted_names}
- Alignment: `{output.alignment}`

## Current Code
```python
{current_code.strip()}
```

## Response Format
Return exactly one fenced Python code block and no prose outside the block.
"""


def repair_code_from_failures(
    *,
    contract: CausalContract,
    current_code: str,
    failure_summary: str,
    llm_invoke: Callable[[str], str],
    attempt: int,
) -> ToolResult:
    prompt = build_repair_prompt(
        contract=contract,
        current_code=current_code,
        failure_summary=failure_summary,
        attempt=attempt,
    )
    try:
        raw_response = llm_invoke(prompt)
    except Exception as exc:
        return ToolResult.failure(
            type(exc).__name__,
            str(exc),
            recoverable=True,
            metadata={"tool_name": "repair_code_from_failures", "attempt": attempt},
        )

    return ToolResult.success(
        {
            "raw_response": raw_response,
            "prompt": prompt,
            "attempt": attempt,
            "failure_summary": failure_summary,
        },
        metadata={
            "tool_name": "repair_code_from_failures",
            "attempt": attempt,
            "prompt_chars": len(prompt),
            "response_chars": len(raw_response),
        },
    )


def _bullets(rows) -> str:
    materialized = list(rows)
    if not materialized:
        return "- None."
    return "\n".join(f"- {row}" for row in materialized)
