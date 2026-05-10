import json
from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from eval.execution.code_extraction import extract_candidate_code
from eval.execution.execution_models import InputBinding
from eval.execution.local_subprocess_executor import (
    LocalSubprocessSandboxExecutor,
)
from eval.generation.data.fixture_models import UniverseFixture
from eval.recipes.recipe_models import MultiUniverseOutputRecipe


@dataclass(frozen=True)
class RecipeScore:
    total: Optional[float]
    status: str
    probe_results: List[Dict[str, Any]]
    outputs: Dict[str, Any]
    errors: Dict[str, str]
    extracted_code: str = ""
    decision: str = ""
    failure_type: Optional[str] = None
    vote_summary: Dict[str, int] = field(default_factory=dict)
    evidence_pack: Dict[str, Any] = field(default_factory=dict)
    judge_verdict: Optional[Dict[str, Any]] = None
    failure_origin: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class GenericRecipeScorer:
    """Small recipe-probe scorer for smoke controls.

    This intentionally starts narrow: enough output kinds and probe types to
    prove the generic path with smoke controls before replacing legacy scorers.
    """

    def __init__(self, timeout_seconds: int = 10, llm_judge: Any = None):
        self.executor = LocalSubprocessSandboxExecutor(timeout_seconds=timeout_seconds)
        self.llm_judge = llm_judge

    def score(
        self,
        recipe: MultiUniverseOutputRecipe,
        code: str,
        fixtures: Iterable[UniverseFixture],
    ) -> RecipeScore:
        fixture_map = {fixture.name: fixture for fixture in fixtures}
        outputs: Dict[str, Any] = {}
        errors: Dict[str, str] = self._static_contract_errors(code)

        if not errors:
            collector_code = self._with_output_collector(code, recipe.output.accepted_names)
            for universe in recipe.universes:
                fixture = fixture_map.get(universe.name)
                if fixture is None:
                    errors[universe.name] = "Missing universe fixture"
                    continue
                binding = self._fixture_binding(fixture)
                result = self.executor.run_with_bindings(collector_code, [binding])
                if not result.success or result.output_df is None:
                    errors[universe.name] = result.error or "Execution failed"
                    continue
                outputs[universe.name] = self._extract_output_value(
                    result.output_df,
                    recipe.output.accepted_names,
                    recipe.output.kind,
                )

        errors.update(self._output_contract_errors(recipe, outputs, fixture_map))
        probe_results = [] if errors else [self._run_probe(probe, outputs, fixture_map, code) for probe in recipe.probes]
        probe_results.extend(self._contract_lint_charges(recipe, code, outputs))
        probe_results = self._finalize_probe_results(probe_results)
        evidence_pack = self._build_evidence_pack(recipe, code, outputs, errors, probe_results, fixture_map)
        judge_verdict = None if errors else self._adjudicate(evidence_pack, probe_results)
        decision_record = self._decide(errors, probe_results, judge_verdict)
        return RecipeScore(
            total=decision_record["total"],
            status=decision_record["decision"],
            probe_results=probe_results,
            outputs=outputs,
            errors=errors,
            extracted_code=code,
            decision=decision_record["decision"],
            failure_type=decision_record["failure_type"],
            vote_summary=decision_record["vote_summary"],
            evidence_pack=evidence_pack,
            judge_verdict=judge_verdict,
            failure_origin=decision_record.get("failure_origin"),
            diagnostics=decision_record.get("diagnostics", {}),
        )

    def score_raw_response(
        self,
        recipe: MultiUniverseOutputRecipe,
        raw_response: str,
        fixtures: Iterable[UniverseFixture],
    ) -> RecipeScore:
        code = extract_candidate_code(raw_response)
        if not code:
            return RecipeScore(
                total=0.0,
                status="FAIL",
                probe_results=[],
                outputs={},
                errors={"candidate": "No executable Python code extracted from raw response"},
                extracted_code="",
                decision="FAIL",
                failure_type="contract_failure",
                failure_origin="contract_or_runtime_failure",
                vote_summary={"pass": 0, "fail": 0, "abstain": 0},
            )
        return self.score(recipe, code, fixtures)

    def _fixture_binding(self, fixture: UniverseFixture) -> InputBinding:
        if isinstance(fixture.data, dict):
            return InputBinding(name="DATA_PATH", kind="directory", files=fixture.data)
        return InputBinding(name="DATA_PATH", kind="file", data=fixture.data)

    def _with_output_collector(self, code: str, accepted_names: List[str]) -> str:
        names_repr = repr(accepted_names)
        return (
            code.rstrip()
            + "\n\n"
            + "# === Generic recipe smoke collector ===\n"
            + "import pandas as pd\n"
            + f"_accepted_output_names = {names_repr}\n"
            + "_row = {}\n"
            + "for _name in _accepted_output_names:\n"
            + "    if _name in globals():\n"
            + "        _value = globals()[_name]\n"
            + "        if isinstance(_value, pd.DataFrame):\n"
            + "            output_df = _value\n"
            + "            break\n"
            + "        if isinstance(_value, pd.Series):\n"
            + "            output_df = pd.DataFrame({_name: _value})\n"
            + "            break\n"
            + "        if isinstance(_value, dict):\n"
            + "            output_df = pd.DataFrame([_value])\n"
            + "            break\n"
            + "        _row[_name] = _value\n"
            + "        output_df = pd.DataFrame([_row])\n"
            + "        break\n"
        )

    def _extract_output_value(self, output_df: pd.DataFrame, accepted_names: List[str], output_kind: str) -> Any:
        if output_kind == "dataframe":
            return output_df
        if output_kind == "series":
            for name in accepted_names:
                if name in output_df.columns:
                    return output_df[name].reset_index(drop=True)
            if output_df.shape[1] == 1:
                return output_df.iloc[:, 0].reset_index(drop=True)
            return output_df.iloc[:, -1].reset_index(drop=True)
        if output_kind == "dict":
            if len(output_df) >= 1:
                return {col: self._coerce_scalar(output_df[col].iloc[0]) for col in output_df.columns}
            return {}
        for name in accepted_names:
            if name in output_df.columns:
                value = output_df[name].iloc[0]
                return self._coerce_scalar(value)
        if output_df.shape == (1, 1):
            return self._coerce_scalar(output_df.iloc[0, 0])
        if len(output_df) == 1:
            return {col: self._coerce_scalar(output_df[col].iloc[0]) for col in output_df.columns}
        return output_df

    def _coerce_scalar(self, value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _static_contract_errors(self, code: str) -> Dict[str, str]:
        errors: Dict[str, str] = {}
        if re.search(r"""(?m)^\s*(DATA_PATH|data_path)\s*=\s*['"][^'"]+['"]""", code):
            errors["contract:data_path_overwrite"] = (
                "Candidate assigns a string literal to DATA_PATH/data_path. The provided data binding must be used as-is."
            )
        if re.search(r"""pd\.read_csv\(\s*['"][^'"]+['"]""", code):
            errors["contract:hardcoded_read_csv_path"] = (
                "Candidate calls pd.read_csv with a string literal path instead of the provided DATA_PATH binding."
            )
        if re.search(r"""(?m)^\s*(DATA_DIR|data_dir)\s*=\s*['"][^'"]+['"]""", code):
            errors["contract:data_dir_overwrite"] = (
                "Candidate assigns a string literal to DATA_DIR/data_dir. Provided data bindings must be used as-is."
            )
        return errors

    def _output_contract_errors(
        self,
        recipe: MultiUniverseOutputRecipe,
        outputs: Mapping[str, Any],
        fixtures: Mapping[str, UniverseFixture],
    ) -> Dict[str, str]:
        errors: Dict[str, str] = {}
        for universe in recipe.universes:
            if universe.name not in outputs:
                continue
            value = outputs[universe.name]
            error = self._validate_output_value(recipe.output, value, fixtures.get(universe.name))
            if error:
                errors[f"contract:{universe.name}"] = error
        return errors

    def _validate_output_value(self, output: Any, value: Any, fixture: UniverseFixture | None) -> str | None:
        kind = output.kind
        if kind == "scalar":
            if isinstance(value, (dict, pd.DataFrame, pd.Series, list, tuple, np.ndarray)):
                return f"Expected scalar output for `{output.variable_name}`, got {type(value).__name__}."
            numeric = self._as_float(value)
            if output.valid_range is not None:
                if numeric is None:
                    return f"Expected numeric scalar in range {output.valid_range}, got {type(value).__name__}."
                low, high = output.valid_range
                if numeric < low or numeric > high:
                    return f"Scalar output {numeric} is outside valid range {output.valid_range}."
            return None
        if kind == "series":
            if not isinstance(value, (pd.Series, list, tuple, np.ndarray)):
                return f"Expected series-like output for `{output.variable_name}`, got {type(value).__name__}."
            series = pd.Series(value)
            if fixture is not None and len(series) != len(fixture.data):
                if isinstance(fixture.data, dict):
                    return None
                return f"Expected series output aligned to {len(fixture.data)} input rows, got {len(series)} rows."
            return None
        if kind == "dict":
            if not isinstance(value, dict):
                return f"Expected dict output for `{output.variable_name}`, got {type(value).__name__}."
            return None
        if kind == "dataframe":
            if not isinstance(value, pd.DataFrame):
                return f"Expected dataframe output for `{output.variable_name}`, got {type(value).__name__}."
            if output.shape is not None and tuple(value.shape) != tuple(output.shape):
                return f"Expected dataframe shape {output.shape}, got {tuple(value.shape)}."
            return None
        if kind == "vector":
            if self._as_numeric_array(value) is None:
                return f"Expected vector-like output for `{output.variable_name}`, got {type(value).__name__}."
            return None
        return None

    def _contract_lint_charges(
        self,
        recipe: MultiUniverseOutputRecipe,
        code: str,
        outputs: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        output = recipe.output
        if output.kind != "series" or not self._is_boolean_flag_output(output):
            return []

        messages = []
        if self._looks_like_inverse_flag_code(code, output.accepted_names):
            messages.append(
                "The code appears to assign or return an inverse safe-row mask, for example `~bad_tick` or `~outlier`, as the final boolean flags."
            )

        true_rates = []
        for value in outputs.values():
            series = self._as_bool_series(value)
            if series is not None and len(series) > 0:
                true_rates.append(float(series.mean()))
        if true_rates and min(true_rates) > 0.80:
            messages.append(
                "The boolean flag output is True for more than 80% of rows on validation input. For flag outputs, True must mean a row triggers the requested event, signal, or protection condition; normal rows should usually be False."
            )

        if not messages:
            return []
        return [
            {
                "name": "boolean_flag_polarity",
                "type": "contract_lint",
                "passed": False,
                "hard_fail": False,
                "severity": "soft",
                "metrics": {"messages": messages},
                "reason": "\n".join(messages),
                "limitations": [
                    "This is a semantic polarity lint, not a contract-gate failure; it requires adjudication."
                ],
            }
        ]

    def _finalize_probe_results(self, probe_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        finalized = []
        for idx, item in enumerate(probe_results, start=1):
            result = dict(item)
            result.setdefault("name", result.get("type", f"probe_{idx}"))
            result.setdefault("metrics", {})
            result.setdefault("evidence", [])
            result.setdefault("limitations", [])
            if "severity" not in result:
                result["severity"] = "hard" if result.get("hard_fail", True) else "soft"
            if "vote" not in result:
                if result.get("passed") is True:
                    result["vote"] = "pass"
                elif result.get("passed") is False:
                    result["vote"] = "fail"
                else:
                    result["vote"] = "abstain"
            result["requires_adjudication"] = result["vote"] == "fail"
            result["charge_id"] = result.get("charge_id") or f"{idx:03d}:{result.get('type')}:{result.get('name')}"
            finalized.append(result)
        return finalized

    def _build_evidence_pack(
        self,
        recipe: MultiUniverseOutputRecipe,
        code: str,
        outputs: Mapping[str, Any],
        errors: Mapping[str, str],
        probe_results: List[Dict[str, Any]],
        fixtures: Mapping[str, UniverseFixture],
    ) -> Dict[str, Any]:
        prompt_variant = recipe.prompt_variants.get(recipe.default_prompt_variant)
        deterministic_observations = [self._probe_to_observation(item) for item in probe_results]
        suspected_charges = [obs for obs in deterministic_observations if obs.get("requires_adjudication")]
        return {
            "recipe": {
                "behavior_key": recipe.behavior_key,
                "pillar": recipe.pillar,
                "difficulty": recipe.difficulty,
                "mechanism_variant": recipe.mechanism_variant,
                "causal_requirement": getattr(prompt_variant, "causal_requirement", None),
                "task_steps": list(recipe.task_steps),
                "output_contract": {
                    "variable_name": recipe.output.variable_name,
                    "accepted_names": list(recipe.output.accepted_names),
                    "kind": recipe.output.kind,
                    "semantic": recipe.output.semantic,
                    "valid_range": recipe.output.valid_range,
                    "shape": recipe.output.shape,
                },
                "schema": dict(recipe.schema_variants.get(recipe.default_schema_variant, {})),
                "known_traps": list(recipe.known_traps),
            },
            "candidate_code": code[:12000],
            "truncation": {"candidate_code_truncated": len(code) > 12000},
            "contract_errors": dict(errors),
            "runtime_diagnostics": self._runtime_diagnostics(errors),
            "fixture_metadata": {
                name: self._safe_value(getattr(fixture, "metadata", {}) or {})
                for name, fixture in fixtures.items()
            },
            "outputs": {name: self._summarize_output(value) for name, value in outputs.items()},
            "deterministic_observations": deterministic_observations,
            "suspected_charges": suspected_charges,
            "probe_results": [self._safe_value(item) for item in probe_results],
        }

    def _probe_to_observation(self, item: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "charge_id": item.get("charge_id"),
            "name": item.get("name"),
            "type": item.get("type"),
            "severity": item.get("severity"),
            "probe_passed": item.get("passed"),
            "requires_adjudication": bool(item.get("requires_adjudication")),
            "claim": self._probe_claim(item),
            "metrics": self._safe_value(item.get("metrics", {})),
            "evidence": self._safe_value(item.get("evidence", [])),
            "limitations": self._safe_value(item.get("limitations", [])),
        }

    def _probe_claim(self, item: Mapping[str, Any]) -> str:
        status = "passed" if item.get("passed") is True else "failed" if item.get("passed") is False else "abstained"
        return (
            f"Probe `{item.get('name')}` ({item.get('type')}) {status}. "
            "If failed, this is a suspected evaluation charge, not a final causal verdict."
        )

    def _runtime_diagnostics(self, errors: Mapping[str, str]) -> List[Dict[str, str]]:
        diagnostics = []
        for key, message in errors.items():
            diagnostics.append(
                {
                    "scope": str(key),
                    "origin": self._classify_error_origin(str(key), str(message)),
                    "message": str(message),
                }
            )
        return diagnostics

    def _classify_error_origin(self, key: str, message: str) -> str:
        text = f"{key}\n{message}".lower()
        if key.startswith("contract:"):
            return "contract_failure"
        if "modulenotfounderror" in text or "no module named" in text:
            return "candidate_code_runtime"
        if "traceback" in text or "attributeerror" in text or "valueerror" in text or "keyerror" in text:
            return "candidate_code_runtime"
        if "missing universe fixture" in text:
            return "scorer_mismatch"
        return "contract_or_runtime_failure"

    def _adjudicate(
        self,
        evidence_pack: Mapping[str, Any],
        probe_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        charges = [item for item in probe_results if item.get("requires_adjudication")]
        if not charges:
            return {
                "source": "deterministic",
                "causal_verdict": "no_causal_failure",
                "failure_origin": "none",
                "probe_assessment": [],
                "charge_verdicts": [],
                "holistic_vote": "pass",
                "failure_type": "none",
                "reason": "No deterministic charges require adjudication.",
            }
        if self.llm_judge is None:
            return {
                "source": "deterministic_fallback",
                "causal_verdict": "true_causal_failure",
                "failure_origin": "candidate_reasoning",
                "probe_assessment": [
                    {
                        "charge_id": item["charge_id"],
                        "verdict": "valid_charge",
                        "severity": item.get("severity", "hard"),
                        "reason": "No LLM judge configured; deterministic failed charge is treated as a valid suspected causal failure.",
                    }
                    for item in charges
                ],
                "charge_verdicts": [
                    {
                        "charge_id": item["charge_id"],
                        "verdict": "true_failure",
                        "severity": item.get("severity", "hard"),
                        "reason": "No LLM judge configured; deterministic failed charge is treated as true failure.",
                    }
                    for item in charges
                ],
                "holistic_vote": "abstain",
                "failure_type": "semantic_failure",
                "reason": "LLM adjudication is disabled.",
            }
        try:
            raw = self._invoke_llm_judge(evidence_pack)
            verdict = self._parse_judge_response(raw)
        except json.JSONDecodeError as exc:
            try:
                raw = self._repair_judge_json_response(raw, evidence_pack, exc)
                verdict = self._parse_judge_response(raw)
            except Exception as repair_exc:
                return self._judge_failure_verdict(charges, repair_exc, raw)
        except Exception as exc:
            return self._judge_failure_verdict(charges, exc, None)
        return self._validate_judge_verdict(verdict, charges)

    def _invoke_llm_judge(self, evidence_pack: Mapping[str, Any]) -> Any:
        prompt = self._build_judge_prompt(evidence_pack)
        return self._call_llm_judge(prompt)

    def _call_llm_judge(self, prompt: str) -> Any:
        if hasattr(self.llm_judge, "invoke"):
            response = self.llm_judge.invoke(prompt)
            return getattr(response, "content", response)
        if callable(self.llm_judge):
            return self.llm_judge(prompt)
        raise TypeError("llm_judge must be callable or expose .invoke(prompt).")

    def _repair_judge_json_response(
        self,
        raw_response: Any,
        evidence_pack: Mapping[str, Any],
        parse_error: Exception,
    ) -> Any:
        prompt = self._build_judge_json_repair_prompt(raw_response, evidence_pack, parse_error)
        return self._call_llm_judge(prompt)

    def _build_judge_json_repair_prompt(
        self,
        raw_response: Any,
        evidence_pack: Mapping[str, Any],
        parse_error: Exception,
    ) -> str:
        compact_evidence = {
            "recipe": evidence_pack.get("recipe", {}),
            "suspected_charges": evidence_pack.get("suspected_charges", []),
            "candidate_code": evidence_pack.get("candidate_code", ""),
        }
        payload = json.dumps(self._safe_value(compact_evidence), ensure_ascii=False, indent=2, default=str)
        return f"""Your previous causal judge response was not valid JSON.

Parse error:
{type(parse_error).__name__}: {parse_error}

Previous response:
{str(raw_response)[:4000]}

Return only valid JSON with this exact shape, using the evidence below:
{{
  "causal_verdict": "true_causal_failure|no_causal_failure|unclear",
  "failure_origin": "candidate_reasoning|candidate_code_runtime|scorer_mismatch|prompt_ambiguity|insufficient_evidence|none",
  "confidence": 0.0,
  "probe_assessment": [
    {{"charge_id": "...", "verdict": "valid_charge|invalid_charge|inconclusive_charge", "reason": "..."}}
  ],
  "required_evidence_missing": [],
  "failure_modes": [],
  "reason": "..."
}}

Evidence:
{payload}
"""

    def _build_judge_prompt(self, evidence_pack: Mapping[str, Any]) -> str:
        payload = json.dumps(self._safe_value(evidence_pack), ensure_ascii=False, indent=2, default=str)
        return f"""You are adjudicating a causal evaluation scorer.

The deterministic scorer has produced observations and suspected charges. Decide whether each suspected charge is a real causal candidate failure, a scorer mismatch, a runtime/contract issue, prompt ambiguity, or inconclusive.

Verdict definitions:
- true_causal_failure: the candidate's reasoning or output violates the recipe's causal requirement.
- no_causal_failure: the candidate appears causally correct, or the failed probe is not measuring the intended causal behavior.
- unclear: the evidence is insufficient.

Failure origins:
- candidate_reasoning: substantive causal logic failure.
- candidate_code_runtime: code/runtime issue prevents judging causal reasoning.
- scorer_mismatch: probe/harness is too narrow, checks the wrong field, or falsely accuses a correct solution.
- prompt_ambiguity: prompt or recipe does not specify enough to judge fairly.
- insufficient_evidence: evidence pack is not enough to decide.
- none: no issue.

Probe assessment verdicts:
- valid_charge: this suspected charge is causally meaningful and supported by evidence.
- invalid_charge: this suspected charge is a scorer mismatch or does not establish causal failure.
- inconclusive_charge: this suspected charge cannot be adjudicated from the evidence.

Important consistency rules:
- Deterministic probes are evidence collectors, not final truth.
- Mark scorer_mismatch only when you can explain why the candidate satisfies the causal requirement and why the probe is wrong or too narrow.
- If your reason says "candidate code does not...", "candidate fails...", "does not satisfy...", or similar, failure_origin should usually be candidate_reasoning and causal_verdict should usually be true_causal_failure.
- If all suspected charges are invalid_charge/scorer_mismatch, causal_verdict should be no_causal_failure or unclear, not true_causal_failure.
- Do not introduce new charge ids.

Return only JSON with this shape:
{{
  "causal_verdict": "true_causal_failure|no_causal_failure|unclear",
  "failure_origin": "candidate_reasoning|candidate_code_runtime|scorer_mismatch|prompt_ambiguity|insufficient_evidence|none",
  "confidence": 0.0,
  "probe_assessment": [
    {{"charge_id": "...", "verdict": "valid_charge|invalid_charge|inconclusive_charge", "reason": "..."}}
  ],
  "required_evidence_missing": [],
  "failure_modes": [],
  "reason": "..."
}}

Evidence pack:
{payload}
"""

    def _parse_judge_response(self, raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        text = str(raw).strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            extracted = self._extract_json_object(text)
            if extracted is None:
                raise
            return json.loads(extracted)

    def _extract_json_object(self, text: str) -> str | None:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return None

    def _judge_failure_verdict(
        self,
        charges: List[Dict[str, Any]],
        exc: Exception,
        raw_response: Any,
    ) -> Dict[str, Any]:
        reason = f"LLM judge failed: {type(exc).__name__}: {exc}"
        raw_excerpt = "" if raw_response is None else str(raw_response)[:1000]
        return {
            "source": "llm_judge",
            "causal_verdict": "unclear",
            "failure_origin": "insufficient_evidence",
            "probe_assessment": [
                {
                    "charge_id": item["charge_id"],
                    "verdict": "inconclusive_charge",
                    "severity": item.get("severity", "hard"),
                    "reason": reason,
                }
                for item in charges
            ],
            "charge_verdicts": [
                {
                    "charge_id": item["charge_id"],
                    "verdict": "inconclusive",
                    "severity": item.get("severity", "hard"),
                    "reason": reason,
                }
                for item in charges
            ],
            "holistic_vote": "abstain",
            "failure_type": "ambiguous",
            "required_evidence_missing": ["valid_json_judge_response"],
            "raw_response_excerpt": raw_excerpt,
            "reason": "LLM judge failed; charges remain unresolved.",
        }

    def _validate_judge_verdict(
        self,
        verdict: Mapping[str, Any],
        charges: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        charge_by_id = {item["charge_id"]: item for item in charges}
        allowed_charge_verdicts = {"true_failure", "scorer_mismatch", "inconclusive"}
        allowed_probe_verdicts = {"valid_charge", "invalid_charge", "inconclusive_charge"}
        allowed_causal_verdicts = {"true_causal_failure", "no_causal_failure", "unclear"}
        allowed_failure_origins = {
            "candidate_reasoning",
            "candidate_code_runtime",
            "scorer_mismatch",
            "prompt_ambiguity",
            "insufficient_evidence",
            "none",
        }
        allowed_votes = {"pass", "fail", "abstain"}
        raw_assessments = verdict.get("probe_assessment")
        if not isinstance(raw_assessments, list):
            raw_assessments = self._legacy_charge_verdicts_to_probe_assessment(verdict.get("charge_verdicts", []))

        valid_assessments = []
        legacy_verdicts = []
        validation_warnings = []
        seen = set()
        for item in raw_assessments if isinstance(raw_assessments, list) else []:
            charge_id = str(item.get("charge_id", ""))
            if charge_id not in charge_by_id:
                continue
            seen.add(charge_id)
            value = str(item.get("verdict", "inconclusive_charge"))
            if value not in allowed_probe_verdicts:
                value = "inconclusive_charge"
            reason = str(item.get("reason", ""))
            if self._probe_assessment_reason_contradicts(value, reason):
                validation_warnings.append(
                    {
                        "charge_id": charge_id,
                        "warning": "judge_probe_assessment_reason_contradicts_verdict",
                        "original_verdict": value,
                        "normalized_verdict": value,
                    }
                )
            normalized_value = self._normalize_probe_assessment(value, reason)
            if normalized_value != value:
                validation_warnings.append(
                    {
                        "charge_id": charge_id,
                        "warning": "judge_probe_assessment_reason_contradicts_verdict",
                        "original_verdict": value,
                        "normalized_verdict": normalized_value,
                    }
                )
            value = normalized_value
            legacy_value = self._probe_assessment_to_legacy_verdict(value)
            valid_assessments.append(
                {
                    "charge_id": charge_id,
                    "verdict": value,
                    "severity": charge_by_id[charge_id].get("severity", "hard"),
                    "reason": reason,
                }
            )
            legacy_verdicts.append(
                {
                    "charge_id": charge_id,
                    "verdict": legacy_value,
                    "severity": charge_by_id[charge_id].get("severity", "hard"),
                    "reason": reason,
                }
            )
        for charge_id, charge in charge_by_id.items():
            if charge_id not in seen:
                valid_assessments.append(
                    {
                        "charge_id": charge_id,
                        "verdict": "inconclusive_charge",
                        "severity": charge.get("severity", "hard"),
                        "reason": "LLM judge did not return a verdict for this charge.",
                    }
                )
                legacy_verdicts.append(
                    {
                        "charge_id": charge_id,
                        "verdict": "inconclusive",
                        "severity": charge.get("severity", "hard"),
                        "reason": "LLM judge did not return a verdict for this charge.",
                    }
                )

        causal_verdict = str(verdict.get("causal_verdict", "unclear"))
        if causal_verdict not in allowed_causal_verdicts:
            causal_verdict = self._legacy_holistic_to_causal_verdict(verdict)

        failure_origin = str(verdict.get("failure_origin", "insufficient_evidence"))
        if failure_origin not in allowed_failure_origins:
            failure_origin = self._infer_failure_origin(causal_verdict, valid_assessments)

        holistic_vote = str(verdict.get("holistic_vote", "abstain"))
        if holistic_vote not in allowed_votes:
            holistic_vote = self._causal_verdict_to_holistic_vote(causal_verdict)
        failure_type = verdict.get("failure_type") or self._failure_type_for_origin(causal_verdict, failure_origin)
        return {
            "source": "llm_judge",
            "causal_verdict": causal_verdict,
            "failure_origin": failure_origin,
            "confidence": self._coerce_confidence(verdict.get("confidence")),
            "probe_assessment": valid_assessments,
            "charge_verdicts": legacy_verdicts,
            "holistic_vote": holistic_vote,
            "failure_type": failure_type,
            "required_evidence_missing": list(verdict.get("required_evidence_missing", []) or []),
            "validation_warnings": validation_warnings,
            "failure_modes": verdict.get("failure_modes", []),
            "reason": str(verdict.get("reason", "")),
        }

    def _legacy_charge_verdicts_to_probe_assessment(self, raw_verdicts: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_verdicts, list):
            return []
        mapping = {
            "true_failure": "valid_charge",
            "scorer_mismatch": "invalid_charge",
            "inconclusive": "inconclusive_charge",
        }
        return [
            {
                "charge_id": item.get("charge_id"),
                "verdict": mapping.get(str(item.get("verdict")), "inconclusive_charge"),
                "reason": item.get("reason", ""),
            }
            for item in raw_verdicts
            if isinstance(item, Mapping)
        ]

    def _probe_assessment_to_legacy_verdict(self, verdict: str) -> str:
        if verdict == "valid_charge":
            return "true_failure"
        if verdict == "invalid_charge":
            return "scorer_mismatch"
        return "inconclusive"

    def _legacy_holistic_to_causal_verdict(self, verdict: Mapping[str, Any]) -> str:
        vote = str(verdict.get("holistic_vote", "abstain"))
        if vote == "fail":
            return "true_causal_failure"
        if vote == "pass":
            return "no_causal_failure"
        return "unclear"

    def _causal_verdict_to_holistic_vote(self, verdict: str) -> str:
        if verdict == "true_causal_failure":
            return "fail"
        if verdict == "no_causal_failure":
            return "pass"
        return "abstain"

    def _infer_failure_origin(self, causal_verdict: str, assessments: List[Dict[str, Any]]) -> str:
        if causal_verdict == "true_causal_failure":
            return "candidate_reasoning"
        if assessments and all(item.get("verdict") == "invalid_charge" for item in assessments):
            return "scorer_mismatch"
        if causal_verdict == "no_causal_failure":
            return "none"
        return "insufficient_evidence"

    def _failure_type_for_origin(self, causal_verdict: str, origin: str) -> str:
        if causal_verdict == "true_causal_failure" and origin == "candidate_reasoning":
            return "semantic_failure"
        if origin == "candidate_code_runtime":
            return "runtime_failure"
        if origin == "scorer_mismatch":
            return "scorer_mismatch"
        if origin == "prompt_ambiguity":
            return "prompt_ambiguity"
        if origin == "none":
            return "none"
        return "ambiguous"

    def _coerce_confidence(self, value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, numeric))

    def _normalize_charge_verdict(self, verdict: str, reason: str) -> str:
        if verdict != "scorer_mismatch":
            return verdict
        text = reason.lower()
        candidate_failure_patterns = [
            "candidate code does not",
            "candidate does not",
            "candidate fails",
            "code fails",
            "does not satisfy",
            "does not meet",
            "violates",
            "missing",
            "instead, it only",
        ]
        if any(pattern in text for pattern in candidate_failure_patterns):
            return "true_failure"
        return verdict

    def _normalize_probe_assessment(self, verdict: str, reason: str) -> str:
        text = reason.lower()
        candidate_success_patterns = [
            "satisfies the causal requirement",
            "satisfies the requirement",
            "candidate appears substantively correct",
            "candidate is correct",
            "causally correct",
            "uses point-in-time",
            "using information available at or before",
            "no causal failure",
        ]
        candidate_failure_patterns = [
            "candidate code does not",
            "candidate does not",
            "candidate fails",
            "code fails",
            "does not satisfy",
            "does not meet",
            "violates",
            "violate",
            "full-column",
            "future rows",
            "lookahead",
            "leakage",
            "missing",
            "instead, it only",
        ]
        has_success_language = any(pattern in text for pattern in candidate_success_patterns)
        has_failure_language = any(pattern in text for pattern in candidate_failure_patterns)

        if verdict == "valid_charge" and has_success_language and not has_failure_language:
            return verdict
        if verdict != "invalid_charge":
            return verdict
        if has_failure_language:
            return "valid_charge"
        return verdict

    def _probe_assessment_reason_contradicts(self, verdict: str, reason: str) -> bool:
        text = reason.lower()
        candidate_success_patterns = [
            "satisfies the causal requirement",
            "satisfies the requirement",
            "candidate appears substantively correct",
            "candidate is correct",
            "causally correct",
            "using information available at or before",
            "no causal failure",
        ]
        candidate_failure_patterns = [
            "candidate code does not",
            "candidate does not",
            "candidate fails",
            "code fails",
            "does not satisfy",
            "does not meet",
            "violates",
            "violate",
            "full-column",
            "future rows",
            "lookahead",
            "leakage",
        ]
        has_success_language = any(pattern in text for pattern in candidate_success_patterns)
        has_failure_language = any(pattern in text for pattern in candidate_failure_patterns)
        if verdict == "valid_charge" and has_success_language and not has_failure_language:
            return True
        if verdict == "invalid_charge" and has_failure_language and not has_success_language:
            return True
        return False

    def _decide(
        self,
        errors: Mapping[str, str],
        probe_results: List[Dict[str, Any]],
        judge_verdict: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if errors:
            origins = [self._classify_error_origin(key, message) for key, message in errors.items()]
            failure_origin = "scorer_mismatch" if origins and all(origin == "scorer_mismatch" for origin in origins) else "candidate_code_runtime"
            if any(origin == "contract_failure" for origin in origins):
                failure_type = "contract_failure"
                failure_origin = "contract_or_runtime_failure"
            else:
                failure_type = "runtime_failure"
            return {
                "decision": "FAIL",
                "total": 0.0,
                "failure_type": failure_type,
                "failure_origin": failure_origin,
                "vote_summary": {"pass": 0, "fail": 1, "abstain": 0},
                "diagnostics": {
                    "runtime_diagnostics": self._runtime_diagnostics(errors),
                    "consistency_warnings": [],
                },
            }

        pass_votes = sum(1 for item in probe_results if item.get("vote") == "pass")
        fail_votes = 0
        abstain_votes = sum(1 for item in probe_results if item.get("vote") == "abstain")
        suspected_charges = [item for item in probe_results if item.get("requires_adjudication")]
        unresolved_hard = False
        confirmed_hard_failure = False
        invalid_hard = False
        consistency_warnings = self._decision_consistency_warnings(probe_results, judge_verdict)
        consistency_warnings.extend(
            str(item.get("warning"))
            for item in (judge_verdict or {}).get("validation_warnings", [])
            if item.get("warning")
        )

        for item in (judge_verdict or {}).get("probe_assessment", []):
            verdict = item.get("verdict")
            severity = item.get("severity", "hard")
            if verdict == "valid_charge":
                fail_votes += 1
                if severity == "hard":
                    confirmed_hard_failure = True
            elif verdict == "invalid_charge":
                pass_votes += 1
                if severity == "hard":
                    invalid_hard = True
            elif verdict == "inconclusive_charge":
                abstain_votes += 1
                if severity == "hard":
                    unresolved_hard = True

        causal_verdict = (judge_verdict or {}).get("causal_verdict")
        failure_origin = (judge_verdict or {}).get("failure_origin")
        if causal_verdict == "no_causal_failure":
            pass_votes += 1
        elif causal_verdict == "true_causal_failure":
            fail_votes += 1
        elif causal_verdict == "unclear":
            abstain_votes += 1

        vote_summary = {"pass": pass_votes, "fail": fail_votes, "abstain": abstain_votes}
        diagnostics = {"consistency_warnings": consistency_warnings}
        if confirmed_hard_failure and causal_verdict == "true_causal_failure":
            return {
                "decision": "FAIL",
                "total": 0.0,
                "failure_type": "semantic_failure",
                "failure_origin": failure_origin or "candidate_reasoning",
                "vote_summary": vote_summary,
                "diagnostics": diagnostics,
            }
        if causal_verdict == "true_causal_failure" and not suspected_charges:
            return {
                "decision": "QUARANTINE",
                "total": None,
                "failure_type": "ambiguous",
                "failure_origin": "insufficient_evidence",
                "vote_summary": vote_summary,
                "diagnostics": diagnostics,
            }
        if unresolved_hard:
            return {
                "decision": "QUARANTINE",
                "total": None,
                "failure_type": "ambiguous",
                "failure_origin": failure_origin or "insufficient_evidence",
                "vote_summary": vote_summary,
                "diagnostics": diagnostics,
            }
        if suspected_charges and invalid_hard and not confirmed_hard_failure:
            return {
                "decision": "QUARANTINE",
                "total": None,
                "failure_type": "scorer_mismatch",
                "failure_origin": "scorer_mismatch",
                "vote_summary": vote_summary,
                "diagnostics": diagnostics,
            }
        if fail_votes > pass_votes:
            return {
                "decision": "QUARANTINE",
                "total": None,
                "failure_type": "ambiguous",
                "failure_origin": failure_origin or "insufficient_evidence",
                "vote_summary": vote_summary,
                "diagnostics": diagnostics,
            }
        if pass_votes > fail_votes and pass_votes > 0:
            return {
                "decision": "PASS",
                "total": 1.0,
                "failure_type": None,
                "failure_origin": "none",
                "vote_summary": vote_summary,
                "diagnostics": diagnostics,
            }
        return {
            "decision": "QUARANTINE",
            "total": None,
            "failure_type": "ambiguous",
            "failure_origin": failure_origin or "insufficient_evidence",
            "vote_summary": vote_summary,
            "diagnostics": diagnostics,
        }

    def _decision_consistency_warnings(
        self,
        probe_results: List[Dict[str, Any]],
        judge_verdict: Optional[Mapping[str, Any]],
    ) -> List[str]:
        warnings = []
        failed_charges = [item for item in probe_results if item.get("requires_adjudication")]
        if not failed_charges and (judge_verdict or {}).get("causal_verdict") == "true_causal_failure":
            warnings.append("judge_reported_causal_failure_without_suspected_charges")
        assessment_ids = {
            str(item.get("charge_id"))
            for item in (judge_verdict or {}).get("probe_assessment", [])
            if item.get("charge_id") is not None
        }
        missing = [item["charge_id"] for item in failed_charges if item.get("charge_id") not in assessment_ids]
        if missing:
            warnings.append(f"judge_missing_charge_assessments:{','.join(missing)}")
        if failed_charges and (judge_verdict or {}).get("causal_verdict") == "no_causal_failure":
            all_invalid = all(
                item.get("verdict") == "invalid_charge"
                for item in (judge_verdict or {}).get("probe_assessment", [])
            )
            if not all_invalid:
                warnings.append("judge_no_causal_failure_but_not_all_charges_invalid")
        return warnings

    def _summarize_output(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, pd.DataFrame):
            return {
                "type": "DataFrame",
                "shape": tuple(value.shape),
                "columns": list(value.columns[:40]),
                "dtypes": {col: str(dtype) for col, dtype in value.dtypes.items()},
                "head": value.head(5).to_dict(orient="records"),
            }
        if isinstance(value, pd.Series):
            return {
                "type": "Series",
                "length": len(value),
                "dtype": str(value.dtype),
                "head": value.head(10).tolist(),
            }
        if isinstance(value, dict):
            return {"type": "dict", "value": self._safe_value(value)}
        return {"type": type(value).__name__, "value": self._safe_value(value)}

    def _safe_value(self, value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, pd.DataFrame):
            return self._summarize_output(value)
        if isinstance(value, pd.Series):
            return self._summarize_output(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(key): self._safe_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._safe_value(item) for item in value]
        return value

    def _is_boolean_flag_output(self, output: Any) -> bool:
        text = " ".join(
            [
                str(getattr(output, "variable_name", "")),
                str(getattr(output, "semantic", "")),
                " ".join(str(name) for name in getattr(output, "accepted_names", []) or []),
            ]
        ).lower()
        return any(token in text for token in ("flag", "signal", "protection", "alert", "trigger"))

    def _looks_like_inverse_flag_code(self, code: str, accepted_names: List[str]) -> bool:
        lowered = code.lower()
        inverse_terms = ("bad", "outlier", "poison", "invalid", "error", "noise")
        for name in accepted_names:
            pattern = rf"{re.escape(name.lower())}\s*=\s*~[^\n]*(?:{'|'.join(inverse_terms)})"
            if re.search(pattern, lowered):
                return True
        return_pattern = rf"return\s+~[^\n]*(?:{'|'.join(inverse_terms)})"
        return bool(re.search(return_pattern, lowered))

    def _as_bool_series(self, value: Any) -> pd.Series | None:
        if not isinstance(value, (pd.Series, list, tuple, np.ndarray)):
            return None
        series = pd.Series(value).dropna()
        if series.empty:
            return None
        if series.dtype == bool:
            return series.astype(bool)
        normalized = series.astype(str).str.lower()
        if normalized.isin(["true", "false"]).all():
            return normalized.eq("true")
        return None

    def _run_probe(
        self,
        probe: Mapping[str, Any],
        outputs: Mapping[str, Any],
        fixtures: Mapping[str, UniverseFixture],
        code: str = "",
    ) -> Dict[str, Any]:
        probe_type = probe.get("type")
        if probe_type == "code_pattern_absent":
            return self._code_pattern_absent(probe, code)
        if probe_type == "code_pattern_present":
            return self._code_pattern_present(probe, code)
        if probe_type == "predicate_matrix":
            return self._predicate_matrix(probe, outputs, fixtures)
        if probe_type == "output_bounds":
            return self._output_bounds(probe, outputs, fixtures)
        if probe_type == "monotonic_response":
            return self._monotonic_response(probe, outputs)
        if probe_type == "time_scaling":
            return self._time_scaling(probe, outputs, fixtures)
        if probe_type == "output_stability":
            return self._output_stability(probe, outputs)
        if probe_type == "leverage_bounds":
            return self._leverage_bounds(probe, outputs)
        if probe_type == "field_bounds":
            return self._field_bounds(probe, outputs)
        if probe_type == "field_monotonic":
            return self._field_monotonic(probe, outputs)
        if probe_type == "derived_metric_monotonic":
            return self._derived_metric_monotonic(probe, outputs)
        if probe_type == "leakage_sentinel":
            return self._leakage_sentinel(probe, outputs)
        if probe_type == "prefix_invariance":
            return self._prefix_invariance(probe, outputs, fixtures)
        if probe_type == "required_columns":
            return self._required_columns(probe, outputs)
        if probe_type == "field_unique_count":
            return self._field_unique_count(probe, outputs)
        if probe_type == "timestamp_order":
            return self._timestamp_order(probe, outputs)
        if probe_type == "nonnegative_delta":
            return self._nonnegative_delta(probe, outputs)
        if probe_type == "dataframe_window_threshold":
            return self._dataframe_window_threshold(probe, outputs, fixtures)
        return {
            "name": probe.get("name", probe_type),
            "type": probe_type,
            "passed": None,
            "vote": "abstain",
            "hard_fail": bool(probe.get("hard_fail", True)),
            "reason": f"Unsupported probe type: {probe_type}",
            "limitations": [f"Unsupported probe type: {probe_type}"],
        }

    def _code_pattern_absent(self, probe: Mapping[str, Any], code: str) -> Dict[str, Any]:
        matches = []
        for pattern in probe.get("patterns", []):
            if re.search(str(pattern), code, flags=re.IGNORECASE | re.MULTILINE):
                matches.append(str(pattern))
        return {
            "name": probe.get("name"),
            "type": "code_pattern_absent",
            "passed": not matches,
            "hard_fail": bool(probe.get("hard_fail", True)),
            "metrics": {"matched_patterns": matches},
            "evidence": [
                {
                    "name": "forbidden_code_patterns",
                    "type": "code_patterns",
                    "patterns": matches,
                    "reason": probe.get("reason"),
                }
            ] if matches else [],
        }

    def _code_pattern_present(self, probe: Mapping[str, Any], code: str) -> Dict[str, Any]:
        matched = []
        patterns = [str(pattern) for pattern in probe.get("patterns", [])]
        for pattern in patterns:
            if re.search(pattern, code, flags=re.IGNORECASE | re.MULTILINE):
                matched.append(pattern)
        passed = bool(matched) if probe.get("mode", "any") == "any" else len(matched) == len(patterns)
        return {
            "name": probe.get("name"),
            "type": "code_pattern_present",
            "passed": passed,
            "hard_fail": bool(probe.get("hard_fail", True)),
            "metrics": {"matched_patterns": matched, "required_patterns": patterns, "mode": probe.get("mode", "any")},
            "evidence": [] if passed else [
                {
                    "name": "missing_required_code_patterns",
                    "type": "code_patterns",
                    "patterns": patterns,
                    "reason": probe.get("reason"),
                }
            ],
        }

    def _predicate_matrix(
        self,
        probe: Mapping[str, Any],
        outputs: Mapping[str, Any],
        fixtures: Mapping[str, UniverseFixture],
    ) -> Dict[str, Any]:
        failures = []
        metrics = {}
        evidence = []
        for universe, expectation in probe.get("expectations", {}).items():
            fixture = fixtures.get(universe)
            expectation = self._resolve_expectation(expectation, fixture)
            value = outputs.get(universe)
            ok = self._matches_expectation(value, expectation)
            metrics[universe] = {"value": value, "expectation": dict(expectation), "passed": ok}
            if not ok:
                failures.append(universe)
                evidence.extend(self._predicate_failure_evidence(universe, value, expectation, fixture))
        return {
            "name": probe.get("name"),
            "type": "predicate_matrix",
            "passed": not failures,
            "hard_fail": bool(probe.get("hard_fail", True)),
            "metrics": metrics,
            "failures": failures,
            "evidence": evidence[:20],
        }

    def _resolve_expectation(
        self,
        expectation: Mapping[str, Any],
        fixture: UniverseFixture | None,
    ) -> Mapping[str, Any]:
        metadata_path = expectation.get("from_metadata")
        if metadata_path and fixture is not None:
            resolved = self._metadata_path(getattr(fixture, "metadata", {}) or {}, str(metadata_path))
            if isinstance(resolved, Mapping):
                return resolved
        return expectation

    def _predicate_failure_evidence(
        self,
        universe: str,
        value: Any,
        expectation: Mapping[str, Any],
        fixture: UniverseFixture | None,
    ) -> List[Dict[str, Any]]:
        if not isinstance(value, (pd.Series, list, tuple, np.ndarray)):
            return []
        series = pd.Series(value).reset_index(drop=True)
        rows = []
        if "flag_at" in expectation:
            for raw_idx, expected in expectation["flag_at"].items():
                idx = int(raw_idx)
                observed = None if idx >= len(series) else self._safe_value(series.iloc[idx])
                row = {"row_idx": idx, "expected": bool(expected), "observed": observed}
                if fixture is not None and 0 <= idx < len(fixture.data):
                    row["input"] = self._safe_value(fixture.data.iloc[idx].to_dict())
                rows.append(row)
        if "sum_equals" in expectation:
            bool_series = self._as_bool_series(series)
            if bool_series is not None:
                true_indices = [int(idx) for idx in bool_series[bool_series].index[:10]]
                rows.append(
                    {
                        "expected_true_count": int(expectation["sum_equals"]),
                        "observed_true_count": int(bool_series.sum()),
                        "observed_true_indices_head": true_indices,
                    }
                )
        if not rows:
            return []
        return [
            {
                "name": f"{universe}_predicate_failure_rows",
                "type": "predicate_failure_rows",
                "universe": universe,
                "rows": rows,
            }
        ]

    def _matches_expectation(self, value: Any, expectation: Mapping[str, Any]) -> bool:
        if value is None:
            return False
        if "equals" in expectation:
            return value == expectation["equals"]
        if "flag_at" in expectation:
            return self._matches_flag_at(value, expectation["flag_at"])
        if "sum_equals" in expectation:
            total = self._series_sum(value)
            return total is not None and total == expectation["sum_equals"]
        if "not_contains" in expectation:
            spec = expectation["not_contains"]
            series = self._output_field_series(value, spec.get("field"))
            if series is None:
                return False
            return not series.astype(str).eq(str(spec.get("value"))).any()
        numeric = self._as_float(value)
        if numeric is None:
            return False
        if "min_value" in expectation and numeric < float(expectation["min_value"]):
            return False
        if "max_value" in expectation and numeric > float(expectation["max_value"]):
            return False
        if "abs_max" in expectation and abs(numeric) > float(expectation["abs_max"]):
            return False
        return True

    def _matches_flag_at(self, value: Any, expected_flags: Mapping[int, bool]) -> bool:
        if not isinstance(value, (pd.Series, list, tuple, np.ndarray)):
            return False
        series = pd.Series(value).reset_index(drop=True)
        for idx, expected in expected_flags.items():
            int_idx = int(idx)
            if int_idx >= len(series):
                return False
            observed = bool(series.iloc[int_idx])
            if observed != bool(expected):
                return False
        return True

    def _series_sum(self, value: Any) -> int | None:
        if not isinstance(value, (pd.Series, list, tuple, np.ndarray)):
            return None
        return int(pd.Series(value).fillna(False).astype(bool).sum())

    def _output_bounds(
        self,
        probe: Mapping[str, Any],
        outputs: Mapping[str, Any],
        fixtures: Mapping[str, UniverseFixture],
    ) -> Dict[str, Any]:
        universe = probe.get("universe")
        value = self._as_float(outputs.get(universe))
        fixture = fixtures.get(universe)
        min_value = self._resolve_probe_value(probe, "min_value", fixture)
        max_value = self._resolve_probe_value(probe, "max_value", fixture)
        passed = value is not None
        if min_value is not None and passed:
            passed = value >= float(min_value)
        if max_value is not None and passed:
            passed = value <= float(max_value)
        return {
            "name": probe.get("name"),
            "type": "output_bounds",
            "passed": bool(passed),
            "hard_fail": bool(probe.get("hard_fail", True)),
            "metrics": {"universe": universe, "value": value, "min_value": min_value, "max_value": max_value},
        }

    def _resolve_probe_value(
        self,
        probe: Mapping[str, Any],
        key: str,
        fixture: UniverseFixture | None,
    ) -> Any:
        metadata_key = f"{key}_from_metadata"
        metadata_path = probe.get(metadata_key)
        if metadata_path and fixture is not None:
            resolved = self._metadata_path(getattr(fixture, "metadata", {}) or {}, str(metadata_path))
            if resolved is not None:
                return resolved
        return probe.get(key)

    def _metadata_path(self, metadata: Mapping[str, Any], path: str) -> Any:
        current: Any = metadata
        for part in path.split("."):
            if isinstance(current, Mapping) and part in current:
                current = current[part]
            else:
                return None
        return current

    def _monotonic_response(self, probe: Mapping[str, Any], outputs: Mapping[str, Any]) -> Dict[str, Any]:
        baseline_name = probe.get("baseline_universe")
        treatment_name = probe.get("treatment_universe")
        baseline = self._as_float(outputs.get(baseline_name))
        treatment = self._as_float(outputs.get(treatment_name))
        direction = probe.get("direction")
        min_delta = probe.get("min_delta")
        ratio_range = probe.get("ratio_range")
        passed = baseline is not None and treatment is not None
        delta = None if not passed else treatment - baseline
        ratio = None
        if passed and baseline not in (0, 0.0):
            ratio = treatment / baseline
        if passed and direction == "increase":
            passed = treatment > baseline
        elif passed and direction == "decrease":
            passed = treatment < baseline
        if passed and min_delta is not None:
            passed = abs(delta) >= float(min_delta)
        if passed and ratio_range is not None:
            low, high = ratio_range
            if ratio is None:
                passed = False
            if passed and low is not None:
                passed = ratio >= float(low)
            if passed and high is not None:
                passed = ratio <= float(high)
        return {
            "name": probe.get("name"),
            "type": "monotonic_response",
            "passed": bool(passed),
            "hard_fail": bool(probe.get("hard_fail", True)),
            "metrics": {
                "baseline_universe": baseline_name,
                "treatment_universe": treatment_name,
                "baseline": baseline,
                "treatment": treatment,
                "delta": delta,
                "ratio": ratio,
            },
        }

    def _time_scaling(
        self,
        probe: Mapping[str, Any],
        outputs: Mapping[str, Any],
        fixtures: Mapping[str, UniverseFixture],
    ) -> Dict[str, Any]:
        universe = probe.get("universe")
        value = self._as_float(outputs.get(universe))
        fixture = fixtures.get(universe)
        expected = None
        rel_error = None
        passed = value is not None and value > 0 and fixture is not None
        if passed:
            expected = self._expected_stop_loss_pct(fixture.data, probe)
            if expected is None or expected == 0:
                passed = False
            else:
                rel_error = abs(value - expected) / abs(expected)
                passed = rel_error <= float(probe.get("max_relative_error", 0.05))
        return {
            "name": probe.get("name"),
            "type": "time_scaling",
            "passed": bool(passed),
            "hard_fail": bool(probe.get("hard_fail", True)),
            "metrics": {
                "universe": universe,
                "value": value,
                "expected": expected,
                "relative_error": rel_error,
                "max_relative_error": probe.get("max_relative_error", 0.05),
            },
        }

    def _expected_stop_loss_pct(self, df: pd.DataFrame, probe: Mapping[str, Any]) -> float | None:
        price_col = "price" if "price" in df.columns else "close" if "close" in df.columns else None
        if price_col is None:
            return None
        horizon = str(probe.get("target_horizon", "30min")).lower().replace("minutes", "min")
        if not horizon.endswith("min"):
            return None
        try:
            horizon_minutes = float(horizon[:-3])
        except ValueError:
            return None
        returns = pd.to_numeric(df[price_col], errors="coerce").pct_change().dropna()
        if returns.empty:
            return None
        return float(returns.std(ddof=1) * np.sqrt(horizon_minutes) * 2.0 * 100.0)

    def _output_stability(self, probe: Mapping[str, Any], outputs: Mapping[str, Any]) -> Dict[str, Any]:
        baseline_name = probe.get("baseline_universe")
        perturbed_name = probe.get("perturbed_universe")
        baseline = self._as_numeric_array(outputs.get(baseline_name))
        perturbed = self._as_numeric_array(outputs.get(perturbed_name))
        passed = baseline is not None and perturbed is not None and baseline.shape == perturbed.shape
        mean_abs_delta = None
        if passed:
            mean_abs_delta = float(np.mean(np.abs(baseline - perturbed)))
            passed = mean_abs_delta <= float(probe.get("max_mean_abs_delta", 0.0))
        return {
            "name": probe.get("name"),
            "type": "output_stability",
            "passed": bool(passed),
            "hard_fail": bool(probe.get("hard_fail", True)),
            "metrics": {
                "baseline_universe": baseline_name,
                "perturbed_universe": perturbed_name,
                "mean_abs_delta": mean_abs_delta,
                "max_mean_abs_delta": probe.get("max_mean_abs_delta"),
            },
        }

    def _leverage_bounds(self, probe: Mapping[str, Any], outputs: Mapping[str, Any]) -> Dict[str, Any]:
        universe = probe.get("universe")
        weights = self._as_numeric_array(outputs.get(universe))
        passed = weights is not None
        leverage = None
        weight_sum = None
        if passed:
            flat = weights.flatten()
            leverage = float(np.sum(np.abs(flat)))
            weight_sum = float(np.sum(flat))
            passed = leverage <= float(probe.get("max_leverage", np.inf))
            passed = passed and abs(weight_sum - 1.0) <= float(probe.get("fully_invested_tolerance", 0.0))
        return {
            "name": probe.get("name"),
            "type": "leverage_bounds",
            "passed": bool(passed),
            "hard_fail": bool(probe.get("hard_fail", True)),
            "metrics": {
                "universe": universe,
                "leverage": leverage,
                "weight_sum": weight_sum,
                "max_leverage": probe.get("max_leverage"),
                "fully_invested_tolerance": probe.get("fully_invested_tolerance"),
            },
        }

    def _field_bounds(self, probe: Mapping[str, Any], outputs: Mapping[str, Any]) -> Dict[str, Any]:
        universe = probe.get("universe")
        field = probe.get("field")
        value = self._as_float(self._field_value(outputs.get(universe), field))
        min_value = probe.get("min_value")
        max_value = probe.get("max_value")
        passed = value is not None
        if passed and min_value is not None:
            passed = value >= float(min_value)
        if passed and max_value is not None:
            passed = value <= float(max_value)
        return {
            "name": probe.get("name"),
            "type": "field_bounds",
            "passed": bool(passed),
            "hard_fail": bool(probe.get("hard_fail", True)),
            "metrics": {"universe": universe, "field": field, "value": value, "min_value": min_value, "max_value": max_value},
        }

    def _field_monotonic(self, probe: Mapping[str, Any], outputs: Mapping[str, Any]) -> Dict[str, Any]:
        baseline_name = probe.get("baseline_universe")
        treatment_name = probe.get("treatment_universe")
        field = probe.get("field")
        baseline = self._as_float(self._field_value(outputs.get(baseline_name), field))
        treatment = self._as_float(self._field_value(outputs.get(treatment_name), field))
        return self._compare_monotonic(
            probe,
            "field_monotonic",
            baseline_name,
            treatment_name,
            baseline,
            treatment,
            extra_metrics={"field": field},
        )

    def _derived_metric_monotonic(self, probe: Mapping[str, Any], outputs: Mapping[str, Any]) -> Dict[str, Any]:
        baseline_name = probe.get("baseline_universe")
        treatment_name = probe.get("treatment_universe")
        metric = probe.get("metric")
        baseline = self._derived_metric(outputs.get(baseline_name), metric)
        treatment = self._derived_metric(outputs.get(treatment_name), metric)
        return self._compare_monotonic(
            probe,
            "derived_metric_monotonic",
            baseline_name,
            treatment_name,
            baseline,
            treatment,
            extra_metrics={"metric": metric},
        )

    def _leakage_sentinel(self, probe: Mapping[str, Any], outputs: Mapping[str, Any]) -> Dict[str, Any]:
        clean_name = probe.get("clean_reference_universe")
        leakage_name = probe.get("leakage_universe")
        clean = self._as_float(outputs.get(clean_name))
        leakage = self._as_float(outputs.get(leakage_name))
        delta = None
        passed = clean is not None and leakage is not None
        if passed:
            delta = abs(clean - leakage)
            passed = delta <= float(probe.get("max_abs_delta", 0.0))
        return {
            "name": probe.get("name"),
            "type": "leakage_sentinel",
            "passed": bool(passed),
            "hard_fail": bool(probe.get("hard_fail", True)),
            "metrics": {
                "clean_reference_universe": clean_name,
                "leakage_universe": leakage_name,
                "clean": clean,
                "leakage": leakage,
                "abs_delta": delta,
                "max_abs_delta": probe.get("max_abs_delta"),
            },
        }

    def _prefix_invariance(
        self,
        probe: Mapping[str, Any],
        outputs: Mapping[str, Any],
        fixtures: Mapping[str, UniverseFixture],
    ) -> Dict[str, Any]:
        baseline_name = probe.get("baseline_universe")
        treatment_name = probe.get("treatment_universe")
        field = probe.get("field")
        baseline = self._output_field_series(outputs.get(baseline_name), field)
        treatment = self._output_field_series(outputs.get(treatment_name), field)
        raw_start_idx = self._resolve_probe_value(probe, "start_idx", fixtures.get(treatment_name))
        start_idx = int(raw_start_idx) if raw_start_idx is not None else 0
        end_idx = self._resolve_probe_value(probe, "end_idx", fixtures.get(treatment_name))
        passed = baseline is not None and treatment is not None
        mismatch_count = None
        mismatch_rate = None
        checked_rows = 0
        evidence = []
        if passed:
            end = min(
                len(baseline) - 1,
                len(treatment) - 1,
                int(end_idx) if end_idx is not None else min(len(baseline), len(treatment)) - 1,
            )
            if end < start_idx:
                passed = False
            else:
                left = baseline.iloc[start_idx : end + 1].reset_index(drop=True).astype(str)
                right = treatment.iloc[start_idx : end + 1].reset_index(drop=True).astype(str)
                mismatches = left.ne(right)
                mismatch_count = int(mismatches.sum())
                checked_rows = int(len(mismatches))
                mismatch_rate = mismatch_count / max(1, checked_rows)
                passed = mismatch_rate <= float(probe.get("max_mismatch_rate", 0.0))
                mismatch_positions = [start_idx + int(pos) for pos in mismatches[mismatches].index[:10]]
                if mismatch_positions:
                    evidence.append(
                        {
                            "name": "prefix_mismatch_rows",
                            "type": "output_rows",
                            "rows": [
                                {
                                    "row_idx": idx,
                                    "baseline": self._safe_value(baseline.iloc[idx]),
                                    "treatment": self._safe_value(treatment.iloc[idx]),
                                }
                                for idx in mismatch_positions
                            ],
                        }
                    )
        return {
            "name": probe.get("name"),
            "type": "prefix_invariance",
            "passed": bool(passed),
            "hard_fail": bool(probe.get("hard_fail", True)),
            "metrics": {
                "baseline_universe": baseline_name,
                "treatment_universe": treatment_name,
                "field": field,
                "checked_rows": checked_rows,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "mismatch_count": mismatch_count,
                "mismatch_rate": mismatch_rate,
                "max_mismatch_rate": probe.get("max_mismatch_rate", 0.0),
            },
            "evidence": evidence,
        }

    def _required_columns(self, probe: Mapping[str, Any], outputs: Mapping[str, Any]) -> Dict[str, Any]:
        universe = probe.get("universe")
        required = list(probe.get("columns", []))
        value = outputs.get(universe)
        columns = list(value.columns) if isinstance(value, pd.DataFrame) else []
        missing = [col for col in required if col not in columns]
        return {
            "name": probe.get("name"),
            "type": "required_columns",
            "passed": not missing,
            "hard_fail": bool(probe.get("hard_fail", True)),
            "metrics": {"universe": universe, "required": required, "columns": columns, "missing": missing},
        }

    def _field_unique_count(self, probe: Mapping[str, Any], outputs: Mapping[str, Any]) -> Dict[str, Any]:
        universe = probe.get("universe")
        field = probe.get("field")
        series = self._output_field_series(outputs.get(universe), field)
        unique_count = None if series is None else int(series.nunique(dropna=True))
        passed = unique_count is not None
        if passed and probe.get("min_unique") is not None:
            passed = unique_count >= int(probe["min_unique"])
        if passed and probe.get("max_unique") is not None:
            passed = unique_count <= int(probe["max_unique"])
        return {
            "name": probe.get("name"),
            "type": "field_unique_count",
            "passed": bool(passed),
            "hard_fail": bool(probe.get("hard_fail", True)),
            "metrics": {"universe": universe, "field": field, "unique_count": unique_count},
        }

    def _dataframe_window_threshold(
        self,
        probe: Mapping[str, Any],
        outputs: Mapping[str, Any],
        fixtures: Mapping[str, UniverseFixture],
    ) -> Dict[str, Any]:
        universe = probe.get("universe")
        fixture = fixtures.get(universe)
        field = probe.get("field")
        series = self._output_field_series(outputs.get(universe), field)
        start_idx = self._resolve_probe_value(probe, "start_idx", fixture)
        end_idx = self._resolve_probe_value(probe, "end_idx", fixture)
        passed = series is not None
        observed_abs_max = None
        checked_rows = 0
        evidence = []
        if passed:
            start = max(0, int(start_idx) if start_idx is not None else 0)
            end = min(len(series) - 1, int(end_idx) if end_idx is not None else len(series) - 1)
            if end < start:
                passed = False
            else:
                values = pd.to_numeric(series.iloc[start : end + 1], errors="coerce").dropna()
                checked_rows = int(len(values))
                passed = checked_rows > 0
                if passed:
                    abs_values = values.abs()
                    observed_abs_max = float(abs_values.max())
                    if probe.get("min_abs_max") is not None:
                        passed = observed_abs_max >= float(probe["min_abs_max"])
                    if passed and probe.get("max_abs_max") is not None:
                        passed = observed_abs_max <= float(probe["max_abs_max"])
                    idx = int(abs_values.idxmax())
                    evidence.append(
                        {
                            "name": "window_abs_extreme",
                            "type": "output_row",
                            "row_idx": idx,
                            "value": self._safe_value(series.iloc[idx]),
                        }
                    )
        return {
            "name": probe.get("name"),
            "type": "dataframe_window_threshold",
            "passed": bool(passed),
            "hard_fail": bool(probe.get("hard_fail", True)),
            "metrics": {
                "universe": universe,
                "field": field,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "checked_rows": checked_rows,
                "observed_abs_max": observed_abs_max,
                "min_abs_max": probe.get("min_abs_max"),
                "max_abs_max": probe.get("max_abs_max"),
            },
            "evidence": evidence,
        }

    def _timestamp_order(self, probe: Mapping[str, Any], outputs: Mapping[str, Any]) -> Dict[str, Any]:
        universe = probe.get("universe")
        left = self._output_field_series(outputs.get(universe), probe.get("left_field"))
        right = self._output_field_series(outputs.get(universe), probe.get("right_field"))
        passed = left is not None and right is not None
        invalid_count = None
        evidence = []
        if passed:
            left_time = pd.to_datetime(left, errors="coerce")
            right_time = pd.to_datetime(right, errors="coerce")
            mask = left_time.notna() & right_time.notna() & (left_time >= right_time)
            invalid_count = int(mask.sum())
            passed = invalid_count == 0
            positions = [int(idx) for idx in mask[mask].index[:10]]
            if positions:
                evidence.append(
                    {
                        "name": "timestamp_order_violations",
                        "type": "output_rows",
                        "rows": [
                            {
                                "row_idx": idx,
                                "left": self._safe_value(left.iloc[idx]),
                                "right": self._safe_value(right.iloc[idx]),
                            }
                            for idx in positions
                        ],
                    }
                )
        return {
            "name": probe.get("name"),
            "type": "timestamp_order",
            "passed": bool(passed),
            "hard_fail": bool(probe.get("hard_fail", True)),
            "metrics": {
                "universe": universe,
                "left_field": probe.get("left_field"),
                "right_field": probe.get("right_field"),
                "invalid_count": invalid_count,
            },
            "evidence": evidence,
        }

    def _nonnegative_delta(self, probe: Mapping[str, Any], outputs: Mapping[str, Any]) -> Dict[str, Any]:
        universe = probe.get("universe")
        field = probe.get("field")
        series = self._output_field_series(outputs.get(universe), field)
        passed = series is not None
        invalid_count = None
        if passed:
            values = pd.to_numeric(series, errors="coerce")
            mask = values.notna() & (values < 0)
            invalid_count = int(mask.sum())
            passed = invalid_count == 0
        return {
            "name": probe.get("name"),
            "type": "nonnegative_delta",
            "passed": bool(passed),
            "hard_fail": bool(probe.get("hard_fail", True)),
            "metrics": {"universe": universe, "field": field, "invalid_count": invalid_count},
        }

    def _output_field_series(self, output: Any, field: str | None) -> pd.Series | None:
        if field is None:
            return None
        if isinstance(output, pd.DataFrame) and field in output.columns:
            return output[field].reset_index(drop=True)
        if isinstance(output, pd.Series):
            if output.name == field or field in ("value", "output"):
                return output.reset_index(drop=True)
        return None

    def _compare_monotonic(
        self,
        probe: Mapping[str, Any],
        probe_type: str,
        baseline_name: str,
        treatment_name: str,
        baseline: float | None,
        treatment: float | None,
        extra_metrics: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        direction = probe.get("direction")
        min_delta = probe.get("min_delta")
        ratio_range = probe.get("ratio_range")
        passed = baseline is not None and treatment is not None
        delta = None if not passed else treatment - baseline
        ratio = None
        if passed and baseline not in (0, 0.0):
            ratio = treatment / baseline
        if passed and direction == "increase":
            passed = treatment > baseline
        elif passed and direction == "decrease":
            passed = treatment < baseline
        if passed and min_delta is not None:
            passed = abs(delta) >= float(min_delta)
        if passed and ratio_range is not None:
            low, high = ratio_range
            if ratio is None:
                passed = False
            if passed and low is not None:
                passed = ratio >= float(low)
            if passed and high is not None:
                passed = ratio <= float(high)
        metrics = {
            "baseline_universe": baseline_name,
            "treatment_universe": treatment_name,
            "baseline": baseline,
            "treatment": treatment,
            "delta": delta,
            "ratio": ratio,
        }
        metrics.update(extra_metrics or {})
        return {
            "name": probe.get("name"),
            "type": probe_type,
            "passed": bool(passed),
            "hard_fail": bool(probe.get("hard_fail", True)),
            "metrics": metrics,
        }

    def _field_value(self, output: Any, field: str) -> Any:
        if isinstance(output, dict):
            return output.get(field)
        if isinstance(output, pd.DataFrame) and field in output.columns and len(output) >= 1:
            return output[field].iloc[0]
        if isinstance(output, pd.Series) and field in output.index:
            return output[field]
        return None

    def _derived_metric(self, output: Any, metric: str) -> float | None:
        if metric == "mid_price_drop":
            bid = self._as_float(self._field_value(output, "bid_price"))
            ask = self._as_float(self._field_value(output, "ask_price"))
            if bid is None or ask is None:
                return None
            baseline_mid = 505.0
            return baseline_mid - ((bid + ask) / 2.0)
        return self._as_float(self._field_value(output, metric))

    def _as_numeric_array(self, value: Any) -> np.ndarray | None:
        if isinstance(value, pd.DataFrame):
            numeric = value.select_dtypes(include=[np.number])
            if numeric.empty:
                return None
            return numeric.to_numpy(dtype=float)
        if isinstance(value, pd.Series):
            numeric = pd.to_numeric(value, errors="coerce")
            if numeric.isna().all():
                return None
            return numeric.to_numpy(dtype=float)
        if isinstance(value, dict):
            values = [self._as_float(item) for item in value.values()]
            if any(item is None for item in values):
                return None
            return np.asarray(values, dtype=float)
        if isinstance(value, (list, tuple, np.ndarray)):
            try:
                return np.asarray(value, dtype=float)
            except (TypeError, ValueError):
                return None
        numeric = self._as_float(value)
        if numeric is None:
            return None
        return np.asarray([numeric], dtype=float)

    def _as_float(self, value: Any) -> float | None:
        try:
            if isinstance(value, bool):
                return float(int(value))
            return float(value)
        except (TypeError, ValueError):
            return None
