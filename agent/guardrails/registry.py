from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from agent.schemas import CausalContract, GuardResults, GuardViolation, Severity, ToolResult, to_dict


@dataclass(frozen=True)
class GuardrailRule:
    id: str
    severity: Severity
    pillar: str
    description: str
    version: str
    check: Callable[[str, CausalContract], list[GuardViolation]]
    enabled_by_default: bool = True


def run_hard_guardrails(
    code: str,
    contract: CausalContract,
    *,
    enabled_rule_ids: list[str] | None = None,
    severity_threshold: Severity = "high",
) -> ToolResult:
    selected = _select_rules(contract, enabled_rule_ids, severity_threshold)
    violations: list[GuardViolation] = []
    warnings: list[str] = []
    for rule in selected:
        try:
            violations.extend(rule.check(code, contract))
        except Exception as exc:
            return ToolResult.failure(
                "guardrail_rule_failed",
                f"Rule {rule.id} failed: {type(exc).__name__}: {exc}",
                recoverable=False,
                metadata={"tool_name": "run_hard_guardrails", "rule_id": rule.id},
            )

    result = GuardResults(
        passed=not violations,
        violations=violations,
        warnings=warnings,
        rule_versions={rule.id: rule.version for rule in selected},
    )
    return ToolResult.success(
        to_dict(result),
        metadata={"tool_name": "run_hard_guardrails", "rules_run": [rule.id for rule in selected]},
    )


def _select_rules(
    contract: CausalContract,
    enabled_rule_ids: list[str] | None,
    severity_threshold: Severity,
) -> list[GuardrailRule]:
    rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    hazards = set(contract.known_hazards)
    selected = []
    for rule in RULES:
        if enabled_rule_ids is not None and rule.id not in enabled_rule_ids:
            continue
        if enabled_rule_ids is None and not rule.enabled_by_default:
            continue
        if enabled_rule_ids is None and rule.id not in hazards and rule.severity != "critical":
            continue
        if rank[rule.severity] < rank[severity_threshold]:
            continue
        selected.append(rule)
    return selected


def _regex_violations(
    code: str,
    *,
    rule_id: str,
    severity: Severity,
    pattern: str,
    message: str,
    flags: int = 0,
) -> list[GuardViolation]:
    violations: list[GuardViolation] = []
    for line_no, line in enumerate(code.splitlines(), start=1):
        if re.search(pattern, line, flags):
            violations.append(
                GuardViolation(
                    rule_id=rule_id,
                    severity=severity,
                    message=message,
                    line=line_no,
                    evidence=line.strip(),
                )
            )
    return violations


def _no_negative_shift(code: str, contract: CausalContract) -> list[GuardViolation]:
    return _regex_violations(
        code,
        rule_id="negative_shift",
        severity="critical",
        pattern=r"\.shift\s*\(\s*-\s*\d+",
        message="Negative shift can move future values into current decisions.",
    )


def _no_backward_fill(code: str, contract: CausalContract) -> list[GuardViolation]:
    violations = _regex_violations(
        code,
        rule_id="backward_fill",
        severity="high",
        pattern=r"\.(bfill|backfill)\s*\(",
        message="Backward fill can leak future observations into earlier rows.",
    )
    violations.extend(
        _regex_violations(
            code,
            rule_id="backward_fill",
            severity="high",
            pattern=r"fillna\s*\(\s*method\s*=\s*['\"]bfill['\"]",
            message="Backward fill can leak future observations into earlier rows.",
        )
    )
    return violations


def _no_future_index_access(code: str, contract: CausalContract) -> list[GuardViolation]:
    return _regex_violations(
        code,
        rule_id="future_index_access",
        severity="high",
        pattern=r"\.iloc\s*\[[^\]]*(\+\s*[1-9]\d*|-\s*-\s*[1-9]\d*)",
        message="Positive offset indexing can read future rows for current decisions.",
    )


def _no_global_quantile_for_online_signal(code: str, contract: CausalContract) -> list[GuardViolation]:
    violations: list[GuardViolation] = []
    if not contract.estimation_scope.forbid_full_sample_fit:
        return violations
    for line_no, line in enumerate(code.splitlines(), start=1):
        if ".quantile(" not in line:
            continue
        if ".rolling(" in line or ".expanding(" in line:
            continue
        violations.append(
            GuardViolation(
                rule_id="global_quantile_for_online_signal",
                severity="high",
                message="Full-sample quantile can leak future distribution information into online decisions.",
                line=line_no,
                evidence=line.strip(),
            )
        )
    return violations


def _no_global_fit_for_online_signal(code: str, contract: CausalContract) -> list[GuardViolation]:
    if not contract.estimation_scope.forbid_full_sample_fit:
        return []
    return _regex_violations(
        code,
        rule_id="global_fit_for_online_signal",
        severity="high",
        pattern=r"\.fit\s*\(",
        message="Model or scaler fitting must be scoped to a training prefix or rolling window for online decisions.",
    )


def _no_hardcoded_absolute_path(code: str, contract: CausalContract) -> list[GuardViolation]:
    return _regex_violations(
        code,
        rule_id="hardcoded_absolute_path",
        severity="critical",
        pattern=r"(read_csv|open)\s*\(\s*['\"](/|[A-Za-z]:\\)",
        message="Code must use provided data bindings rather than hardcoded absolute paths.",
    )


def _no_forbidden_optional_import(code: str, contract: CausalContract) -> list[GuardViolation]:
    return _regex_violations(
        code,
        rule_id="forbidden_optional_import",
        severity="high",
        pattern=r"^\s*(import|from)\s+(sklearn|scipy|cvxpy|statsmodels|talib)\b",
        message="Optional third-party packages are not allowed in the sandbox contract.",
        flags=re.MULTILINE,
    )


RULES: list[GuardrailRule] = [
    GuardrailRule(
        id="negative_shift",
        severity="critical",
        pillar="temporal_causality",
        description="Disallow negative shift in online decision code.",
        version="0.1",
        check=_no_negative_shift,
    ),
    GuardrailRule(
        id="backward_fill",
        severity="high",
        pillar="temporal_causality",
        description="Disallow backward fill in temporal feature code.",
        version="0.1",
        check=_no_backward_fill,
    ),
    GuardrailRule(
        id="future_index_access",
        severity="high",
        pillar="temporal_causality",
        description="Flag positive-offset row indexing.",
        version="0.1",
        check=_no_future_index_access,
    ),
    GuardrailRule(
        id="global_quantile_for_online_signal",
        severity="high",
        pillar="temporal_causality",
        description="Flag full-sample quantile in online decision code.",
        version="0.1",
        check=_no_global_quantile_for_online_signal,
    ),
    GuardrailRule(
        id="global_fit_for_online_signal",
        severity="high",
        pillar="temporal_causality",
        description="Flag unscoped model or scaler fitting in online decision code.",
        version="0.1",
        check=_no_global_fit_for_online_signal,
    ),
    GuardrailRule(
        id="hardcoded_absolute_path",
        severity="critical",
        pillar="execution_contract",
        description="Disallow hardcoded absolute file paths.",
        version="0.1",
        check=_no_hardcoded_absolute_path,
    ),
    GuardrailRule(
        id="forbidden_optional_import",
        severity="high",
        pillar="execution_contract",
        description="Disallow optional third-party imports outside the sandbox contract.",
        version="0.1",
        check=_no_forbidden_optional_import,
    ),
]
