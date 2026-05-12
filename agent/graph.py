from __future__ import annotations

from typing import Any, Callable, TypedDict

from langgraph.graph import END, StateGraph

from agent.codegen import generate_code_from_contract
from agent.contract.contracts import (
    build_contract_from_intent,
    build_dev_contract_from_recipe,
    validate_contract_completeness,
)
from agent.critic import actionable_critic_summary, critique_code
from agent.guardrails import run_hard_guardrails
from agent.invariants import run_invariant_checks
from agent.normalizer import normalize_intent
from agent.repair import repair_code_from_failures
from agent.schemas import AgentTrace, CausalContract, Intent, ToolResult, to_dict
from agent.tools import execute_in_sandbox, extract_code_from_response
from eval.scoring.generic_recipe_scorer import GenericRecipeScorer


class AgentState(TypedDict, total=False):
    recipe: Any
    intent: Intent
    raw_response: str
    code: str
    user_text: str
    user_task: str
    normalize_llm_invoke: Callable[[str], str]
    generate_code: bool
    llm_invoke: Callable[[str], str]
    repair_llm_invoke: Callable[[str], str]
    critic_llm_invoke: Callable[[str], str]
    config_path: str
    run_critic: bool
    repair_attempts: int
    max_repair_attempts: int
    failure_summary: str
    fixtures: list[Any]
    timeout_seconds: int
    contract_source: str
    contract: CausalContract
    extracted_code: str
    validation_result: dict[str, Any]
    generation_result: dict[str, Any]
    extraction_result: dict[str, Any]
    guard_result: dict[str, Any]
    sandbox_result: dict[str, Any]
    invariant_result: dict[str, Any]
    critic_result: dict[str, Any]
    repair_result: dict[str, Any]
    trace: AgentTrace
    final_status: str


def build_minimal_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("normalize_intent", _normalize_intent_node)
    graph.add_node("build_contract", _build_contract_node)
    graph.add_node("generate_code", _generate_code_node)
    graph.add_node("extract_code", _extract_code_node)
    graph.add_node("run_guardrails", _run_guardrails_node)
    graph.add_node("execute_sandbox", _execute_sandbox_node)
    graph.add_node("run_invariants", _run_invariants_node)
    graph.add_node("run_critic", _run_critic_node)
    graph.add_node("repair_code", _repair_code_node)
    graph.add_node("finalize", _finalize_node)

    graph.set_entry_point("normalize_intent")
    graph.add_conditional_edges(
        "normalize_intent",
        _after_normalize,
        {"build_contract": "build_contract", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "build_contract",
        _after_contract,
        {"generate": "generate_code", "extract": "extract_code"},
    )
    graph.add_edge("generate_code", "extract_code")
    graph.add_edge("extract_code", "run_guardrails")
    graph.add_conditional_edges(
        "run_guardrails",
        _after_guardrails,
        {"sandbox": "execute_sandbox", "repair": "repair_code", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "execute_sandbox",
        _after_sandbox,
        {"invariants": "run_invariants", "repair": "repair_code", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "run_invariants",
        _after_invariants,
        {"critic": "run_critic", "repair": "repair_code", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "run_critic",
        _after_critic,
        {"repair": "repair_code", "finalize": "finalize"},
    )
    graph.add_edge("repair_code", "extract_code")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_minimal_agent_flow(
    *,
    recipe: Any | None = None,
    intent: Intent | None = None,
    raw_response: str = "",
    code: str = "",
    user_text: str = "",
    user_task: str = "",
    generate_code: bool = False,
    llm_invoke: Callable[[str], str] | None = None,
    normalize_llm_invoke: Callable[[str], str] | None = None,
    repair_llm_invoke: Callable[[str], str] | None = None,
    critic_llm_invoke: Callable[[str], str] | None = None,
    run_critic: bool = False,
    max_repair_attempts: int = 0,
    config_path: str = "config.yaml",
    fixtures: list[Any] | None = None,
    timeout_seconds: int = 10,
    contract_source: str = "dev_recipe",
) -> dict[str, Any]:
    app = build_minimal_agent_graph()
    state = app.invoke(
        {
            "recipe": recipe,
            "intent": intent,
            "raw_response": raw_response,
            "code": code,
            "user_text": user_text,
            "user_task": user_task,
            "generate_code": generate_code,
            "llm_invoke": llm_invoke,
            "normalize_llm_invoke": normalize_llm_invoke,
            "repair_llm_invoke": repair_llm_invoke,
            "critic_llm_invoke": critic_llm_invoke,
            "run_critic": run_critic,
            "repair_attempts": 0,
            "max_repair_attempts": max_repair_attempts,
            "config_path": config_path,
            "fixtures": fixtures or [],
            "timeout_seconds": timeout_seconds,
            "contract_source": contract_source,
        }
    )
    return _state_to_public_result(state)


def _normalize_intent_node(state: AgentState) -> AgentState:
    source = state.get("contract_source", "dev_recipe")
    if source in ("intent"):
        llm_invoke = state.get("normalize_llm_invoke") or state.get("llm_invoke")
        user_text = state.get("user_text") or ""

        result = normalize_intent(user_text, llm_invoke=llm_invoke)

        trace = AgentTrace(
            intent=result.result.get("intent") if result.ok else {},
            causal_contract={},
            tool_calls=[_tool_call_record(result)],
        )

        intent = result.result["intent"] if result.ok else None
        return {
            **state,
            "intent": intent,
            "trace": trace,
        }
    else:
        # If not normalizing intent, skip directly to contract building
        trace = AgentTrace(
            intent=to_dict(state.get("intent")) if state.get("intent") else {},
            causal_contract={},
            tool_calls=[],
        )
        return {
            **state,
            "trace": trace,
        }


def _build_contract_node(state: AgentState) -> AgentState:
    source = state.get("contract_source", "dev_recipe")
    if source in ("intent"):
        if not state.get("intent"):
            raise ValueError("contract_source='intent' requires state['intent'].")
        contract = build_contract_from_intent(state["intent"])
    else:
        if not state.get("recipe"):
            raise ValueError("contract_source='dev_recipe' requires state['recipe'].")
        contract = build_dev_contract_from_recipe(state["recipe"])
    validation = validate_contract_completeness(contract)
    trace = state.get("trace")
    if trace is None:
        trace = AgentTrace(
            intent=to_dict(contract.intent),
            causal_contract=to_dict(contract),
            tool_calls=[_tool_call_record(validation)],
        )
    else:
        trace.intent = to_dict(contract.intent)
        trace.causal_contract = to_dict(contract)
        trace.tool_calls.append(_tool_call_record(validation))
    return {
        **state,
        "contract": contract,
        "validation_result": to_dict(validation),
        "trace": trace,
    }


def _generate_code_node(state: AgentState) -> AgentState:
    llm_invoke = state.get("llm_invoke")
    if llm_invoke is None:
        from eval.evaluation.candidate_sources import build_llm_invoker

        llm_invoke = build_llm_invoker(state.get("config_path", "config.yaml"))

    result = generate_code_from_contract(
        state["contract"],
        llm_invoke=llm_invoke,
        user_task=state.get("user_task", ""),
    )
    trace = state["trace"]
    trace.tool_calls.append(_tool_call_record(result))
    raw_response = result.result["raw_response"] if result.ok else ""
    trace.generated_code_versions.append(
        {
            "source": "code_generator",
            "raw_response": raw_response,
            "generation_ok": result.ok,
        }
    )
    return {
        **state,
        "raw_response": raw_response,
        "generation_result": to_dict(result),
        "trace": trace,
    }


def _run_critic_node(state: AgentState) -> AgentState:
    llm_invoke = state.get("critic_llm_invoke") or state.get("llm_invoke")
    if llm_invoke is None:
        from eval.evaluation.candidate_sources import build_llm_invoker

        llm_invoke = build_llm_invoker(state.get("config_path", "config.yaml"))

    result = critique_code(
        contract=state["contract"],
        code=state.get("extracted_code", ""),
        guard_result=state.get("guard_result", {}),
        invariant_result=state.get("invariant_result", {}),
        llm_invoke=llm_invoke,
    )
    trace = state["trace"]
    trace.tool_calls.append(_tool_call_record(result))
    if result.ok:
        critic_payload = result.result["critic"]
        trace.critic_concerns.extend(critic_payload.get("concerns", []))
    else:
        trace.critic_concerns.append({"critic_error": to_dict(result.error)})
    return {
        **state,
        "critic_result": to_dict(result),
        "trace": trace,
    }


def _repair_code_node(state: AgentState) -> AgentState:
    attempt = int(state.get("repair_attempts", 0)) + 1
    llm_invoke = state.get("repair_llm_invoke") or state.get("llm_invoke")
    if llm_invoke is None:
        from eval.evaluation.candidate_sources import build_llm_invoker

        llm_invoke = build_llm_invoker(state.get("config_path", "config.yaml"))

    failure_summary = state.get("failure_summary") or _failure_summary(state)
    result = repair_code_from_failures(
        contract=state["contract"],
        current_code=state.get("extracted_code", "") or state.get("code", ""),
        failure_summary=failure_summary,
        llm_invoke=llm_invoke,
        attempt=attempt,
    )
    trace = state["trace"]
    trace.tool_calls.append(_tool_call_record(result))
    raw_response = result.result["raw_response"] if result.ok else ""
    trace.repair_history.append(
        {
            "attempt": attempt,
            "ok": result.ok,
            "failure_summary": failure_summary,
            "raw_response": raw_response,
        }
    )
    if result.ok:
        trace.generated_code_versions.append(
            {
                "source": "repair_loop",
                "raw_response": raw_response,
                "generation_ok": True,
                "attempt": attempt,
            }
        )
    return {
        **state,
        "raw_response": raw_response,
        "code": "",
        "repair_attempts": attempt,
        "repair_result": to_dict(result),
        "failure_summary": "",
        "trace": trace,
    }


def _extract_code_node(state: AgentState) -> AgentState:
    raw = state.get("raw_response") or state.get("code") or ""
    result = extract_code_from_response(raw, require_single_block=False)
    trace = state["trace"]
    trace.tool_calls.append(_tool_call_record(result))
    code = result.result["code"] if result.ok else ""
    trace.generated_code_versions.append(
        {
            "source": "provided_candidate",
            "code": code,
            "extraction_ok": result.ok,
        }
    )
    return {
        **state,
        "extracted_code": code,
        "extraction_result": to_dict(result),
        "trace": trace,
    }


def _run_guardrails_node(state: AgentState) -> AgentState:
    if not state.get("extracted_code"):
        result = ToolResult.failure(
            "missing_code",
            "Guardrails cannot run because no code was extracted.",
            recoverable=True,
            metadata={"tool_name": "run_hard_guardrails"},
        )
    else:
        result = run_hard_guardrails(state["extracted_code"], state["contract"])

    trace = state["trace"]
    trace.tool_calls.append(_tool_call_record(result))
    trace.guard_results.append(to_dict(result.result) if result.ok else to_dict(result))
    return {
        **state,
        "guard_result": to_dict(result),
        "trace": trace,
    }


def _execute_sandbox_node(state: AgentState) -> AgentState:
    fixtures = state.get("fixtures") or []
    if not fixtures:
        result = ToolResult.failure(
            "missing_fixture",
            "No fixture was provided for sandbox execution.",
            recoverable=True,
            metadata={"tool_name": "execute_in_sandbox", "purpose": "final_validation"},
        )
    else:
        scorer = GenericRecipeScorer(timeout_seconds=state.get("timeout_seconds", 10))
        fixture = fixtures[0]
        binding = scorer._fixture_binding(fixture)
        accepted_names = state["contract"].output_contract.accepted_names
        collector_code = scorer._with_output_collector(state["extracted_code"], accepted_names)
        result = execute_in_sandbox(
            collector_code,
            [binding],
            timeout_seconds=state.get("timeout_seconds", 10),
            purpose="final_validation",
        )

    trace = state["trace"]
    trace.tool_calls.append(_tool_call_record(result))
    trace.sandbox_results.append(to_dict(result.result) if result.ok else to_dict(result))
    return {
        **state,
        "sandbox_result": to_dict(result),
        "trace": trace,
    }


def _run_invariants_node(state: AgentState) -> AgentState:
    result = run_invariant_checks(
        state["extracted_code"],
        state["contract"],
        state.get("fixtures") or [],
        timeout_seconds=state.get("timeout_seconds", 10),
        mode="basic",
        strict_mode=False,
    )
    trace = state["trace"]
    trace.tool_calls.append(_tool_call_record(result))
    trace.invariant_results.append(to_dict(result.result) if result.ok else to_dict(result))
    return {
        **state,
        "invariant_result": to_dict(result),
        "trace": trace,
    }


def _finalize_node(state: AgentState) -> AgentState:
    final_status = "pass"
    validation = state.get("validation_result", {})
    generation = state.get("generation_result", {})
    extraction = state.get("extraction_result", {})
    guard = state.get("guard_result", {})
    sandbox = state.get("sandbox_result", {})
    invariants = state.get("invariant_result", {})
    critic = state.get("critic_result", {})

    if state.get("intent") is None and state.get("contract") is None:
        trace = state.get("trace")
        if trace:
            final_status = "normalization_failed"
    elif not validation.get("ok", False):
        final_status = "contract_failed"
    elif generation and not generation.get("ok", False):
        final_status = "generation_failed"
    elif not extraction.get("ok", False):
        final_status = "code_extraction_failed"
    elif not guard.get("ok", False):
        final_status = "guardrail_failed"
    elif guard.get("result", {}).get("passed") is False:
        final_status = "guardrail_failed"
    elif sandbox and not sandbox.get("ok", False):
        final_status = "sandbox_failed"
    elif sandbox and not sandbox.get("result", {}).get("success", False):
        final_status = "sandbox_failed"
    elif invariants and not invariants.get("ok", False):
        final_status = "invariant_failed"
    elif invariants and invariants.get("result", {}).get("passed") is False:
        final_status = "invariant_failed"
    elif critic and critic.get("ok", False) and critic.get("result", {}).get("critic", {}).get("passed") is False:
        final_status = "critic_concern"

    trace = state["trace"]
    trace.final_status = final_status
    return {**state, "final_status": final_status, "trace": trace}


def _after_normalize(state: AgentState) -> str:
    if state.get("contract_source") == "intent" and state.get("intent") is None:
        return "finalize"
    return "build_contract"


def _after_guardrails(state: AgentState) -> str:
    guard = state.get("guard_result", {})
    if not guard.get("ok", False):
        return _repair_or_finalize(state, _failure_summary(state))
    if guard.get("result", {}).get("passed") is False:
        return _repair_or_finalize(state, _failure_summary(state))
    return "sandbox"


def _after_contract(state: AgentState) -> str:
    if state.get("generate_code"):
        return "generate"
    return "extract"


def _after_sandbox(state: AgentState) -> str:
    sandbox = state.get("sandbox_result", {})
    if not sandbox.get("ok", False):
        return _repair_or_finalize(state, _failure_summary(state))
    if not sandbox.get("result", {}).get("success", False):
        return _repair_or_finalize(state, _failure_summary(state))
    return "invariants"


def _after_invariants(state: AgentState) -> str:
    invariants = state.get("invariant_result", {})
    if not invariants.get("ok", False):
        return _repair_or_finalize(state, _failure_summary(state))
    if invariants.get("result", {}).get("passed") is False:
        return _repair_or_finalize(state, _failure_summary(state))
    if state.get("run_critic"):
        return "critic"
    return "finalize"


def _after_critic(state: AgentState) -> str:
    critic = state.get("critic_result", {})
    if not critic.get("ok", False):
        return "finalize"
    summary = actionable_critic_summary(critic)
    if summary:
        return _repair_or_finalize(state, summary)
    return "finalize"


def _repair_or_finalize(state: AgentState, failure_summary: str) -> str:
    if int(state.get("repair_attempts", 0)) >= int(state.get("max_repair_attempts", 0)):
        return "finalize"
    state["failure_summary"] = failure_summary
    return "repair"


def _failure_summary(state: AgentState) -> str:
    trace = state.get("trace")
    if trace and trace.tool_calls:
        first_call = trace.tool_calls[0]
        if first_call.get("tool_name") == "normalize_intent" and not first_call.get("ok"):
            return f"Intent normalization failed: {first_call.get('error', {}).get('message', 'unknown error')}"

    extraction = state.get("extraction_result", {})
    if extraction and not extraction.get("ok", False):
        return f"Code extraction failed: {extraction.get('error')}"

    guard = state.get("guard_result", {})
    if guard and (not guard.get("ok", False) or guard.get("result", {}).get("passed") is False):
        violations = guard.get("result", {}).get("violations", [])
        lines = [
            f"- {item.get('rule_id')}: {item.get('message')} Evidence: {item.get('evidence')}"
            for item in violations
        ]
        return "Guardrail validation failed:\n" + ("\n".join(lines) or str(guard.get("error")))

    sandbox = state.get("sandbox_result", {})
    if sandbox and (not sandbox.get("ok", False) or not sandbox.get("result", {}).get("success", False)):
        error = sandbox.get("error") or sandbox.get("result", {}).get("error")
        stderr = sandbox.get("result", {}).get("stderr")
        return f"Sandbox execution failed: {error or stderr or sandbox}"

    invariants = state.get("invariant_result", {})
    if invariants and (not invariants.get("ok", False) or invariants.get("result", {}).get("passed") is False):
        checks = invariants.get("result", {}).get("checks", [])
        failed = [item for item in checks if not item.get("passed")]
        lines = [f"- {item.get('invariant_id')}: {item.get('message')}" for item in failed]
        return "Invariant validation failed:\n" + ("\n".join(lines) or str(invariants.get("error")))

    critic = state.get("critic_result", {})
    if critic:
        summary = actionable_critic_summary(critic)
        if summary:
            return summary

    return "Validation failed without a more specific public error."


def _tool_call_record(result: ToolResult) -> dict[str, Any]:
    return {
        "tool_name": result.metadata.get("tool_name", "unknown"),
        "ok": result.ok,
        "error": to_dict(result.error) if result.error else None,
        "warnings": result.warnings,
        "metadata": result.metadata,
    }


def _state_to_public_result(state: AgentState) -> dict[str, Any]:
    return {
        "final_status": state.get("final_status", "unknown"),
        "intent": to_dict(state.get("intent")) if state.get("intent") else state.get("trace", {}).intent,
        "contract": to_dict(state.get("contract")) if state.get("contract") else {},
        "generation_result": state.get("generation_result", {}),
        "extracted_code": state.get("extracted_code", ""),
        "guard_result": state.get("guard_result", {}),
        "sandbox_result": state.get("sandbox_result", {}),
        "invariant_result": state.get("invariant_result", {}),
        "critic_result": state.get("critic_result", {}),
        "repair_result": state.get("repair_result", {}),
        "trace": to_dict(state.get("trace")),
    }
