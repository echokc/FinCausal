import json
import os
from typing import Any, Callable, Dict, Iterable, List, Optional

from eval.evaluation.candidate_sources import (
    CandidateSource,
    RecipeEvalCandidate,
    build_llm_invoker,
    control_candidates,
    generate_llm_recipe_candidates,
)
from eval.evaluation.records import build_eval_record
from eval.evaluation.repair import repair_llm_recipe_candidate
from eval.generation.data.fixture_generation import generate_recipe_fixtures
from eval.generation.data.fixture_models import UniverseFixture
from eval.generation.prompts.prompt_builders import build_prompt_from_recipe
from eval.recipes.recipe_models import MultiUniverseOutputRecipe
from eval.evaluation.case_manifest import build_and_write_recipe_case
from eval.scoring.generic_recipe_scorer import GenericRecipeScorer, RecipeScore
from eval.smoketest.smoke_controls import SMOKE_CONTROLS


def score_candidate(
    scorer: GenericRecipeScorer,
    recipe: MultiUniverseOutputRecipe,
    fixtures: Iterable[UniverseFixture],
    candidate: RecipeEvalCandidate,
) -> RecipeScore:
    if candidate.generation_error:
        return RecipeScore(
            total=0.0,
            status="FAIL",
            probe_results=[],
            outputs={},
            errors={"generation": candidate.generation_error},
            extracted_code="",
            decision="FAIL",
            failure_type="generation_failure",
            failure_origin="candidate_code_runtime",
            vote_summary={"pass": 0, "fail": 1, "abstain": 0},
        )
    if candidate.raw_response:
        return scorer.score_raw_response(recipe, candidate.raw_response, fixtures)
    return scorer.score(recipe, candidate.code, fixtures)


def run_recipe_eval_pipeline(
    *,
    behavior_key: str | None = None,
    candidate_source: CandidateSource = "controls",
    llm_samples: int = 1,
    config_path: str = "config.yaml",
    timeout_seconds: int = 10,
    output_path: str | None = None,
    case_manifest_root: str | None = None,
    llm_invoke: Optional[Callable[[str], str]] = None,
    repair_attempts: int = 1,
) -> List[Dict[str, Any]]:
    if candidate_source == "llm" and llm_invoke is None:
        llm_invoke = build_llm_invoker(config_path)
    scorer = GenericRecipeScorer(
        timeout_seconds=timeout_seconds,
        llm_judge=llm_invoke if candidate_source == "llm" else None,
    )
    keys = [behavior_key] if behavior_key else list(SMOKE_CONTROLS)
    records: List[Dict[str, Any]] = []

    for key in keys:
        control = SMOKE_CONTROLS[key]
        recipe = control["recipe"]
        try:
            fixtures = generate_recipe_fixtures(recipe.behavior_key)
        except KeyError:
            fixtures = control["fixtures"]()
        prompt = build_prompt_from_recipe(recipe)
        case_manifest = None
        if case_manifest_root:
            build = build_and_write_recipe_case(recipe, output_root=case_manifest_root)
            case_manifest = {
                "case_id": build.manifest["case_id"],
                "manifest_path": build.manifest_path,
                "data_paths": build.data_paths,
                "protocol_version": build.manifest["protocol_version"],
            }

        if candidate_source == "controls":
            candidates = control_candidates(control)
        elif candidate_source == "llm_shaped_controls":
            candidates = control_candidates(control, include_llm_shaped=True)
        elif candidate_source == "llm":
            candidates = generate_llm_recipe_candidates(
                prompt,
                n=llm_samples,
                config_path=config_path,
                llm_invoke=llm_invoke,
            )
        else:
            raise ValueError(f"Unknown candidate_source: {candidate_source}")

        for candidate in candidates:
            score = score_candidate(scorer, recipe, fixtures, candidate)
            if (
                candidate_source == "llm"
                and llm_invoke is not None
                and repair_attempts > 0
                and _should_repair(score)
            ):
                for attempt in range(1, repair_attempts + 1):
                    try:
                        candidate = repair_llm_recipe_candidate(
                            prompt=prompt,
                            recipe=recipe,
                            candidate=candidate,
                            score=score,
                            llm_invoke=llm_invoke,
                            attempt=attempt,
                        )
                    except Exception as exc:
                        candidate = RecipeEvalCandidate(
                            candidate_id=candidate.candidate_id,
                            source=candidate.source,
                            raw_response=candidate.raw_response,
                            generation_error=f"Repair {attempt} failed: {type(exc).__name__}: {exc}",
                            metadata=candidate.metadata,
                        )
                        score = score_candidate(scorer, recipe, fixtures, candidate)
                        break
                    score = score_candidate(scorer, recipe, fixtures, candidate)
                    if not _should_repair(score):
                        break
            records.append(
                build_eval_record(
                    recipe=recipe,
                    behavior_key=key,
                    prompt=prompt,
                    candidate=candidate,
                    score=score,
                    case_manifest=case_manifest,
                )
            )

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    return records


def _should_repair(score: RecipeScore) -> bool:
    if score.status == "PASS":
        return False
    if score.errors:
        return True
    # Do not use hidden evaluation metrics to steer repair. A behavioral fail
    # without runtime/contract errors should be left for the scorer/judge layer.
    return False
