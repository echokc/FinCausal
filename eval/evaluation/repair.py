from typing import Callable

from eval.execution.code_extraction import extract_candidate_code
from eval.generation.prompts.prompt_builders import build_repair_prompt
from eval.evaluation.candidate_sources import RecipeEvalCandidate
from eval.recipes.recipe_models import MultiUniverseOutputRecipe
from eval.scoring.generic_recipe_scorer import RecipeScore


def repair_llm_recipe_candidate(
    *,
    prompt: str,
    recipe: MultiUniverseOutputRecipe,
    candidate: RecipeEvalCandidate,
    score: RecipeScore,
    llm_invoke: Callable[[str], str],
    attempt: int,
) -> RecipeEvalCandidate:
    error_summary = _contract_error_summary(score)
    repair_prompt = build_repair_prompt(
        original_prompt=prompt,
        candidate_code=score.extracted_code or extract_candidate_code(candidate.raw_response) or candidate.code,
        recipe=recipe,
        error_summary=error_summary,
    )
    raw_response = llm_invoke(repair_prompt)
    metadata = dict(candidate.metadata or {})
    repair_history = list(metadata.get("repair_history", []))
    repair_history.append(
        {
            "attempt": attempt,
            "trigger": "contract_or_runtime_failure",
            "error_summary": error_summary,
            "repair_prompt": repair_prompt,
        }
    )
    metadata["repair_history"] = repair_history
    return RecipeEvalCandidate(
        candidate_id=candidate.candidate_id,
        source=candidate.source,
        raw_response=raw_response,
        generation_error=None,
        metadata=metadata,
    )


def _contract_error_summary(score: RecipeScore) -> str:
    if not score.errors:
        failed = [item for item in score.probe_results if not item.get("passed")]
        names = ", ".join(str(item.get("name", item.get("type", "unnamed_probe"))) for item in failed[:3])
        return f"The code executed, but behavioral validation failed: {names or 'one or more checks failed'}."

    messages = []
    for _, error in list(score.errors.items())[:2]:
        messages.append(_compact_traceback(str(error)))
    return "\n\n".join(messages)


def _compact_traceback(error: str) -> str:
    lines = [line.rstrip() for line in error.splitlines() if line.strip()]
    if len(lines) <= 10:
        return "\n".join(lines)
    head = lines[:3]
    tail = lines[-7:]
    return "\n".join(head + ["..."] + tail)
