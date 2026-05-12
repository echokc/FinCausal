from __future__ import annotations

from typing import Callable

from agent.retrieval import retrieve_hazard_definitions
from agent.schemas import CausalContract, ToolResult, to_dict


def build_code_generation_prompt(contract: CausalContract, *, user_task: str = "") -> str:
    hazard_result = retrieve_hazard_definitions(
        " ".join(contract.known_hazards),
        severity=["high", "critical"],
        top_k=8,
    )
    hazards = hazard_result.result["items"] if hazard_result.ok else []
    hazard_rows = "\n".join(
        f"- `{item['id']}` ({item['severity']}): {item['description']}" for item in hazards
    )
    output = contract.output_contract
    accepted_names = ", ".join(f"`{name}`" for name in output.accepted_names)
    task_text = user_task.strip() or contract.intent.requested_output

    return f"""You are generating Python code for a causally constrained financial research task.

## User Task
{task_text}

## Causal Contract
- Decision unit: `{contract.decision_context.decision_unit}`
- Time column: `{contract.decision_context.time_column}`
- Decision time: `{contract.decision_context.decision_time}`
- Estimation scope: `{contract.estimation_scope.fit_scope}`
- Forbid full-sample fit: `{contract.estimation_scope.forbid_full_sample_fit}`

Allowed information:
{_bullets(item.description for item in contract.allowed_information)}

Forbidden information:
{_bullets(item.description for item in contract.forbidden_information)}

Required invariants:
{_bullets(f"{item.id}: {item.description}" for item in contract.required_invariants)}

Known hazards:
{hazard_rows or "- No hazard definitions retrieved."}

## Output Contract
- Output kind: `{output.kind}`
- Primary output variable: `{output.variable_name}`
- Accepted output variable names: {accepted_names}
- Output semantic: `{output.semantic}`
- Alignment: `{output.alignment}`

## Runtime Contract
- The input data is available as `DATA_PATH`.
- Use only pandas, numpy, os, math, statistics, and the Python standard library.
- Do not hardcode file paths.
- Do not overwrite `DATA_PATH`.
- Assign the final answer to one accepted output variable.
- Do not only define functions; compute the final output at top level.

## Response Format
Return exactly one fenced Python code block and no prose outside the block.
"""


def generate_code_from_contract(
    contract: CausalContract,
    *,
    llm_invoke: Callable[[str], str],
    user_task: str = "",
) -> ToolResult:
    prompt = build_code_generation_prompt(contract, user_task=user_task)
    try:
        raw_response = llm_invoke(prompt)
    except Exception as exc:
        return ToolResult.failure(
            type(exc).__name__,
            str(exc),
            recoverable=True,
            metadata={"tool_name": "generate_code_from_contract"},
        )

    return ToolResult.success(
        {
            "raw_response": raw_response,
            "prompt": prompt,
            "contract_id": contract.contract_id,
            "contract": to_dict(contract),
        },
        metadata={
            "tool_name": "generate_code_from_contract",
            "response_chars": len(raw_response),
            "prompt_chars": len(prompt),
        },
    )


def _bullets(rows) -> str:
    materialized = list(rows)
    if not materialized:
        return "- None."
    return "\n".join(f"- {row}" for row in materialized)
