from datetime import datetime
from typing import Any, Dict

from eval.evaluation.candidate_sources import RecipeEvalCandidate
from eval.recipes.recipe_models import MultiUniverseOutputRecipe
from eval.scoring.generic_recipe_scorer import RecipeScore


def _score_to_record(score: RecipeScore) -> Dict[str, Any]:
    return {
        "total": score.total,
        "status": score.status,
        "decision": score.decision,
        "failure_type": score.failure_type,
        "vote_summary": score.vote_summary,
        "outputs": score.outputs,
        "errors": score.errors,
        "probe_results": score.probe_results,
        "judge_verdict": score.judge_verdict,
        "failure_origin": score.failure_origin,
        "diagnostics": score.diagnostics,
        "evidence_pack": score.evidence_pack,
        "extracted_code": score.extracted_code,
    }


def build_eval_record(
    *,
    recipe: MultiUniverseOutputRecipe,
    behavior_key: str,
    prompt: str,
    candidate: RecipeEvalCandidate,
    score: RecipeScore,
    case_manifest: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    decision = score.status.lower()
    if candidate.generation_error:
        decision = "quarantine"
    record = {
        "case_id": case_manifest.get("case_id") if case_manifest else f"recipe.{behavior_key}",
        "behavior_key": behavior_key,
        "pillar": recipe.pillar,
        "difficulty": recipe.difficulty,
        "mechanism_variant": recipe.mechanism_variant,
        "candidate_id": candidate.candidate_id,
        "candidate_source": candidate.source,
        "prompt": prompt,
        "generation": {
            "raw_response": candidate.raw_response,
            "extracted_code": score.extracted_code,
            "generation_error": candidate.generation_error,
            "metadata": candidate.metadata or {},
        },
        "decision": decision,
        "score": _score_to_record(score),
        "created_at": datetime.now().isoformat(),
    }
    if case_manifest:
        record["case_manifest"] = case_manifest
    return record
