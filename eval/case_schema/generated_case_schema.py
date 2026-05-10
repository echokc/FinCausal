from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


PROTOCOL_VERSION = "0.1"
QUALITY_STATUSES = {"generated", "passed", "quarantine", "failed"}
DIVERSITY_REQUIRED = {
    "schema_variant",
    "prompt_variant",
    "distribution_variant",
    "mechanism_variant",
    "difficulty_level",
    "surface_domain",
}
TOP_LEVEL_REQUIRED = {
    "protocol_version",
    "case_id",
    "pillar",
    "behavior",
    "difficulty",
    "data_paths",
    "prompt",
    "causal_contract",
    "schema_manifest",
    "witness_map",
    "judge_config",
    "reference_behavior",
    "quality_report",
    "provenance",
}


@dataclass
class GeneratedCase:
    protocol_version: str
    case_id: str
    pillar: str
    behavior: str
    difficulty: str
    data_paths: Dict[str, str]
    prompt: Dict[str, Any]
    causal_contract: Dict[str, Any]
    schema_manifest: Dict[str, Any]
    witness_map: Dict[str, Any]
    judge_config: Dict[str, Any]
    reference_behavior: Dict[str, Any]
    quality_report: Dict[str, Any]
    provenance: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GeneratedCase":
        known = {field_name for field_name in cls.__dataclass_fields__}
        args = {key: value for key, value in payload.items() if key in known}
        return cls(**args)

    def validate(self) -> List[str]:
        return validate_generated_case(self.to_dict())


@dataclass
class DiversitySpec:
    schema_variant: str
    prompt_variant: str
    distribution_variant: str
    mechanism_variant: str
    difficulty_level: int
    surface_domain: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _missing_keys(payload: Dict[str, Any], required: set[str], prefix: str) -> List[str]:
    return [f"{prefix}.{key}" for key in sorted(required) if key not in payload]


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_generated_case(case: Dict[str, Any]) -> List[str]:
    """Return a list of schema errors. Empty list means the case is valid."""
    errors: List[str] = []
    errors.extend(_missing_keys(case, TOP_LEVEL_REQUIRED, "case"))

    if case.get("protocol_version") != PROTOCOL_VERSION:
        errors.append(
            f"case.protocol_version must be {PROTOCOL_VERSION!r}; got {case.get('protocol_version')!r}"
        )

    for key in ("case_id", "pillar", "behavior", "difficulty"):
        if key in case and not _is_nonempty_string(case[key]):
            errors.append(f"case.{key} must be a non-empty string")

    data_paths = case.get("data_paths", {})
    if not isinstance(data_paths, dict):
        errors.append("case.data_paths must be an object")
    else:
        errors.extend(_missing_keys(data_paths, {"universe_a", "universe_b"}, "case.data_paths"))
        for key in ("universe_a", "universe_b"):
            if key in data_paths and not _is_nonempty_string(data_paths[key]):
                errors.append(f"case.data_paths.{key} must be a non-empty string")

    prompt = case.get("prompt", {})
    if not isinstance(prompt, dict):
        errors.append("case.prompt must be an object")
    else:
        errors.extend(_missing_keys(prompt, {"text", "output_contract"}, "case.prompt"))
        if "text" in prompt and not _is_nonempty_string(prompt["text"]):
            errors.append("case.prompt.text must be a non-empty string")
        output_contract = prompt.get("output_contract", {})
        if not isinstance(output_contract, dict):
            errors.append("case.prompt.output_contract must be an object")
        else:
            errors.extend(
                _missing_keys(
                    output_contract,
                    {"type", "function_name", "signature"},
                    "case.prompt.output_contract",
                )
            )

    causal_contract = case.get("causal_contract", {})
    if not isinstance(causal_contract, dict):
        errors.append("case.causal_contract must be an object")
    else:
        errors.extend(
            _missing_keys(
                causal_contract,
                {
                    "allowed_information",
                    "forbidden_information",
                    "invariance_requirements",
                    "sensitivity_requirements",
                },
                "case.causal_contract",
            )
        )

    schema_manifest = case.get("schema_manifest", {})
    if not isinstance(schema_manifest, dict):
        errors.append("case.schema_manifest must be an object")
    elif not isinstance(schema_manifest.get("files"), list) or not schema_manifest.get("files"):
        errors.append("case.schema_manifest.files must be a non-empty list")

    witness_map = case.get("witness_map", {})
    if not isinstance(witness_map, dict):
        errors.append("case.witness_map must be an object")
    else:
        errors.extend(
            _missing_keys(
                witness_map,
                {"interventions", "must_match", "inspect_windows", "known_traps"},
                "case.witness_map",
            )
        )

    judge_config = case.get("judge_config", {})
    if not isinstance(judge_config, dict):
        errors.append("case.judge_config must be an object")
    else:
        errors.extend(
            _missing_keys(
                judge_config,
                {"required_output_semantics", "probes", "llm_judge"},
                "case.judge_config",
            )
        )
        if "probes" in judge_config and not isinstance(judge_config["probes"], list):
            errors.append("case.judge_config.probes must be a list")

    quality_report = case.get("quality_report", {})
    if not isinstance(quality_report, dict):
        errors.append("case.quality_report must be an object")
    else:
        status = quality_report.get("status")
        if status not in QUALITY_STATUSES:
            errors.append(
                f"case.quality_report.status must be one of {sorted(QUALITY_STATUSES)}; got {status!r}"
            )
        if not isinstance(quality_report.get("checks"), dict):
            errors.append("case.quality_report.checks must be an object")

    provenance = case.get("provenance", {})
    if not isinstance(provenance, dict):
        errors.append("case.provenance must be an object")
    else:
        errors.extend(
            _missing_keys(
                provenance,
                {"generator_name", "generator_version", "params", "created_at"},
                "case.provenance",
            )
        )

    metadata = case.get("metadata", {})
    if metadata:
        if not isinstance(metadata, dict):
            errors.append("case.metadata must be an object")
        else:
            diversity = metadata.get("diversity")
            if diversity is not None:
                if not isinstance(diversity, dict):
                    errors.append("case.metadata.diversity must be an object")
                else:
                    errors.extend(
                        _missing_keys(diversity, DIVERSITY_REQUIRED, "case.metadata.diversity")
                    )
                    if "difficulty_level" in diversity and not isinstance(
                        diversity["difficulty_level"], int
                    ):
                        errors.append("case.metadata.diversity.difficulty_level must be an int")

    return errors


def assert_valid_generated_case(case: Dict[str, Any]) -> None:
    errors = validate_generated_case(case)
    if errors:
        joined = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Invalid GeneratedCase:\n{joined}")


def make_prompt_contract(text: str, prompt_variant: str = "default") -> Dict[str, Any]:
    return {
        "text": text,
        "prompt_variant": prompt_variant,
        "output_contract": {
            "type": "python_function",
            "function_name": "run_analysis",
            "signature": "def run_analysis(data_path: str) -> pd.DataFrame",
        },
    }
