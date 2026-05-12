from __future__ import annotations

from typing import Iterable

from agent.schemas import ToolResult
from eval.execution.code_extraction import extract_candidate_code
from eval.execution.execution_models import InputBinding, execution_to_dict
from eval.execution.local_subprocess_executor import LocalSubprocessSandboxExecutor


def extract_code_from_response(
    raw_response: str,
    *,
    expected_language: str = "python",
    require_single_block: bool = True,
) -> ToolResult:
    if expected_language.lower() != "python":
        return ToolResult.failure(
            "unsupported_language",
            f"Only Python extraction is supported; got {expected_language!r}.",
            recoverable=False,
            metadata={"tool_name": "extract_code_from_response"},
        )
    code = extract_candidate_code(raw_response)
    if not code:
        return ToolResult.failure(
            "code_extraction_failed",
            "No executable code was found in the response.",
            recoverable=True,
            metadata={"tool_name": "extract_code_from_response"},
        )
    if require_single_block and raw_response.count("```") > 2:
        return ToolResult.failure(
            "multiple_code_blocks",
            "Expected exactly one code block, but multiple fenced blocks were found.",
            recoverable=True,
            metadata={"tool_name": "extract_code_from_response"},
        )
    return ToolResult.success(
        {"code": code},
        metadata={"tool_name": "extract_code_from_response", "code_chars": len(code)},
    )


def execute_in_sandbox(
    code: str,
    bindings: Iterable[InputBinding],
    *,
    timeout_seconds: int = 10,
    purpose: str = "final_validation",
) -> ToolResult:
    try:
        executor = LocalSubprocessSandboxExecutor(timeout_seconds=timeout_seconds)
        result = executor.run_with_bindings(code, list(bindings))
    except Exception as exc:
        return ToolResult.failure(
            type(exc).__name__,
            str(exc),
            recoverable=True,
            metadata={"tool_name": "execute_in_sandbox", "purpose": purpose},
        )

    return ToolResult.success(
        execution_to_dict(result),
        metadata={
            "tool_name": "execute_in_sandbox",
            "purpose": purpose,
            "timeout_seconds": timeout_seconds,
            "success": result.success,
        },
    )

