from __future__ import annotations

import json
import re
from typing import Any, Callable

from agent.retrieval import retrieve_causal_principles, retrieve_hazard_definitions
from agent.schemas import CausalContract, CriticConcern, CriticResults, Severity, ToolResult, to_dict


ALLOWED_SEVERITIES: set[Severity] = {"low", "medium", "high", "critical"}


def build_critic_prompt(
    *,
    contract: CausalContract,
    code: str,
    guard_result: dict[str, Any],
    invariant_result: dict[str, Any],
) -> str:
    principle_result = retrieve_causal_principles(
        " ".join(contract.known_hazards + [contract.intent.requested_output]),
        domain=contract.intent.domain,
        top_k=4,
    )
    hazard_result = retrieve_hazard_definitions(
        " ".join(contract.known_hazards),
        severity=["high", "critical"],
        top_k=8,
    )
    principles = principle_result.result["items"] if principle_result.ok else []
    hazards = hazard_result.result["items"] if hazard_result.ok else []
    evidence = {
        "principles": principles,
        "hazards": hazards,
    }
    return f"""You are a bounded causal red-team critic for generated financial Python code.

You are not the final judge. Deterministic guardrails and invariant checks have already run.
Only report concrete, evidence-backed concerns. Do not speculate without code evidence.

## Causal Contract
{json.dumps(to_dict(contract), indent=2, ensure_ascii=False)}

## Code
```python
{code.strip()}
```

## Deterministic Results
Guardrails:
{json.dumps(guard_result, indent=2, ensure_ascii=False, default=str)}

Invariants:
{json.dumps(invariant_result, indent=2, ensure_ascii=False, default=str)}

## Knowledge Evidence
{json.dumps(evidence, indent=2, ensure_ascii=False)}

## Response Format
Return exactly one JSON object and no prose. Shape:
{{
  "concerns": [
    {{
      "severity": "low|medium|high|critical",
      "obligation_id": "string",
      "claim": "string",
      "code_evidence": "string",
      "knowledge_evidence": "string",
      "recommended_fix": "string",
      "requires_repair": true
    }}
  ]
}}

If there are no concrete concerns, return {{"concerns": []}}.
"""


def critique_code(
    *,
    contract: CausalContract,
    code: str,
    guard_result: dict[str, Any],
    invariant_result: dict[str, Any],
    llm_invoke: Callable[[str], str],
) -> ToolResult:
    prompt = build_critic_prompt(
        contract=contract,
        code=code,
        guard_result=guard_result,
        invariant_result=invariant_result,
    )
    try:
        raw_response = llm_invoke(prompt)
        payload = _parse_json_object(raw_response)
        concerns = [_parse_concern(item) for item in payload.get("concerns", [])]
    except Exception as exc:
        return ToolResult.failure(
            type(exc).__name__,
            str(exc),
            recoverable=True,
            metadata={"tool_name": "critique_code", "prompt_chars": len(prompt)},
        )

    results = CriticResults(
        passed=not any(item.requires_repair and item.severity in {"high", "critical"} for item in concerns),
        concerns=concerns,
    )
    return ToolResult.success(
        {
            "raw_response": raw_response,
            "prompt": prompt,
            "critic": to_dict(results),
        },
        metadata={
            "tool_name": "critique_code",
            "prompt_chars": len(prompt),
            "response_chars": len(raw_response),
        },
    )


def actionable_critic_summary(critic_result: dict[str, Any]) -> str:
    concerns = critic_result.get("result", {}).get("critic", {}).get("concerns", [])
    actionable = [
        item
        for item in concerns
        if item.get("requires_repair") and item.get("severity") in {"high", "critical"}
    ]
    if not actionable:
        return ""
    rows = []
    for item in actionable:
        rows.append(
            "- {severity} {obligation_id}: {claim} Evidence: {code_evidence} Fix: {recommended_fix}".format(
                severity=item.get("severity"),
                obligation_id=item.get("obligation_id"),
                claim=item.get("claim"),
                code_evidence=item.get("code_evidence"),
                recommended_fix=item.get("recommended_fix"),
            )
        )
    return "LLM causal critic found actionable concerns:\n" + "\n".join(rows)


def _parse_json_object(raw_response: str) -> dict[str, Any]:
    text = raw_response.strip()
    fenced = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    return json.loads(text)


def _parse_concern(item: dict[str, Any]) -> CriticConcern:
    severity = str(item.get("severity", "low")).lower()
    if severity not in ALLOWED_SEVERITIES:
        severity = "low"
    return CriticConcern(
        severity=severity,  # type: ignore[arg-type]
        obligation_id=str(item.get("obligation_id", "")),
        claim=str(item.get("claim", "")),
        code_evidence=str(item.get("code_evidence", "")),
        knowledge_evidence=str(item.get("knowledge_evidence", "")),
        recommended_fix=str(item.get("recommended_fix", "")),
        requires_repair=bool(item.get("requires_repair", False)),
    )
