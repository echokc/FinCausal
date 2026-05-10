import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

from eval.generation.data.fixture_generation import (
    data_quality_report,
    generate_recipe_fixtures,
    write_recipe_fixtures,
)
from eval.generation.prompts.prompt_builders import build_prompt_from_recipe, prompt_quality_checks


RECIPE_CASE_PROTOCOL_VERSION = "recipe-case-0.1"


@dataclass(frozen=True)
class RecipeCaseBuildResult:
    manifest: Dict[str, Any]
    manifest_path: str | None
    data_paths: Dict[str, str]


def build_recipe_case_manifest(
    recipe: Any,
    *,
    data_paths: Dict[str, str],
    fixture_metadata: Dict[str, Dict[str, Any]] | None = None,
    prompt_text: str,
    seed: int,
    case_id: str | None = None,
) -> Dict[str, Any]:
    schema_variant = recipe.schema_variants[recipe.default_schema_variant]
    prompt_variant = recipe.prompt_variants[recipe.default_prompt_variant]
    output = recipe.output

    return {
        "protocol_version": RECIPE_CASE_PROTOCOL_VERSION,
        "case_id": case_id or f"recipe.{recipe.behavior_key}.seed_{seed}",
        "pillar": recipe.pillar,
        "behavior_key": recipe.behavior_key,
        "difficulty": recipe.difficulty,
        "mechanism_variant": recipe.mechanism_variant,
        "data_paths": dict(data_paths),
        "prompt": {
            "text": prompt_text,
            "prompt_variant": recipe.default_prompt_variant,
            "framing": prompt_variant.framing,
            "causal_requirement": prompt_variant.causal_requirement,
            "output_contract": {
                "kind": getattr(output, "kind", "scalar"),
                "variable_name": output.variable_name,
                "semantic": output.semantic,
                "accepted_names": list(output.accepted_names),
                "valid_range": getattr(output, "valid_range", None),
                "shape": getattr(output, "shape", None),
            },
        },
        "schema_manifest": {
            "schema_variant": recipe.default_schema_variant,
            "schema": dict(schema_variant),
            "universes": [
                {
                    "name": universe.name,
                    "role": universe.role,
                    "description": universe.description,
                    "path": data_paths.get(universe.name),
                    "interventions": universe.interventions,
                    "expected": universe.expected,
                    "metadata": (fixture_metadata or {}).get(universe.name, {}),
                }
                for universe in recipe.universes
            ],
        },
        "judge_config": {
            "output": {
                "kind": getattr(output, "kind", "scalar"),
                "semantic": output.semantic,
                "accepted_names": list(output.accepted_names),
            },
            "probes": recipe.probes,
        },
        "reference_behavior": {
            "known_traps": recipe.known_traps,
            "task_steps": recipe.task_steps,
            "load_snippet": recipe.load_snippet,
        },
        "quality_report": {
            "status": "generated",
            "checks": {
                "data_written": all(bool(path) and os.path.exists(path) for path in data_paths.values()),
                "prompt_quality": prompt_quality_checks(recipe, prompt_text),
            },
        },
        "provenance": {
            "generator_name": "recipe_case_manifest",
            "generator_version": RECIPE_CASE_PROTOCOL_VERSION,
            "seed": seed,
            "params": {"seed": seed},
            "created_at": datetime.now().isoformat(),
            "repo_commit": None,
        },
        "metadata": {
            "distribution_variants": list(recipe.distribution_variants),
            "default_schema_variant": recipe.default_schema_variant,
            "default_prompt_variant": recipe.default_prompt_variant,
            "surface_domains": list(prompt_variant.surface_domains),
            "difficulty_level": prompt_variant.difficulty_level,
            "fixture_metadata": fixture_metadata or {},
        },
    }


def validate_recipe_case_manifest(manifest: Dict[str, Any]) -> List[str]:
    errors = []
    required = {
        "protocol_version",
        "case_id",
        "pillar",
        "behavior_key",
        "difficulty",
        "data_paths",
        "prompt",
        "schema_manifest",
        "judge_config",
        "quality_report",
        "provenance",
    }
    for key in sorted(required):
        if key not in manifest:
            errors.append(f"manifest.{key} is required")
    if manifest.get("protocol_version") != RECIPE_CASE_PROTOCOL_VERSION:
        errors.append(f"manifest.protocol_version must be {RECIPE_CASE_PROTOCOL_VERSION!r}")
    data_paths = manifest.get("data_paths", {})
    if not isinstance(data_paths, dict) or not data_paths:
        errors.append("manifest.data_paths must be a non-empty object")
    else:
        missing_files = [name for name, path in data_paths.items() if not path or not os.path.exists(path)]
        if missing_files:
            errors.append(f"manifest.data_paths missing files for universes: {missing_files}")
    prompt = manifest.get("prompt", {})
    if not isinstance(prompt, dict) or not prompt.get("text"):
        errors.append("manifest.prompt.text must be present")
    output_contract = prompt.get("output_contract", {}) if isinstance(prompt, dict) else {}
    for key in ("kind", "variable_name", "semantic", "accepted_names"):
        if key not in output_contract:
            errors.append(f"manifest.prompt.output_contract.{key} is required")
    probes = manifest.get("judge_config", {}).get("probes")
    if not isinstance(probes, list) or not probes:
        errors.append("manifest.judge_config.probes must be a non-empty list")
    universes = manifest.get("schema_manifest", {}).get("universes")
    if not isinstance(universes, list) or not universes:
        errors.append("manifest.schema_manifest.universes must be a non-empty list")
    return errors


def build_and_write_recipe_case(
    recipe: Any,
    *,
    output_root: str,
    seed: int = 42,
    case_id: str | None = None,
) -> RecipeCaseBuildResult:
    case_id = case_id or f"recipe.{recipe.behavior_key}.seed_{seed}"
    case_dir = os.path.join(output_root, case_id)
    data_dir = os.path.join(case_dir, "data")
    os.makedirs(case_dir, exist_ok=True)

    fixtures = generate_recipe_fixtures(recipe.behavior_key, seed=seed)
    data_paths = write_recipe_fixtures(fixtures, data_dir)
    quality = data_quality_report(recipe, fixtures)
    prompt_text = build_prompt_from_recipe(recipe)
    manifest = build_recipe_case_manifest(
        recipe,
        data_paths=data_paths,
        fixture_metadata={fixture.name: fixture.metadata for fixture in fixtures},
        prompt_text=prompt_text,
        seed=seed,
        case_id=case_id,
    )
    manifest["quality_report"]["checks"]["data_quality"] = quality.__dict__
    manifest["quality_report"]["checks"]["data_quality_ok"] = quality.ok

    errors = validate_recipe_case_manifest(manifest)
    if errors:
        manifest["quality_report"]["status"] = "failed"
        manifest["quality_report"]["validation_errors"] = errors

    manifest_path = os.path.join(case_dir, "case_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, default=str)

    return RecipeCaseBuildResult(manifest=manifest, manifest_path=manifest_path, data_paths=data_paths)


