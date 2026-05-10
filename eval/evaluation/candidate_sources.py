from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional


CandidateSource = Literal["controls", "llm_shaped_controls", "llm"]


@dataclass(frozen=True)
class RecipeEvalCandidate:
    candidate_id: str
    source: str
    code: str = ""
    raw_response: str = ""
    generation_error: Optional[str] = None
    metadata: Dict[str, Any] | None = None


def control_candidates(control: Dict[str, Any], include_llm_shaped: bool = False) -> List[RecipeEvalCandidate]:
    candidates = [
        RecipeEvalCandidate(candidate_id="positive_control", source="controls", code=control["positive"]),
        RecipeEvalCandidate(candidate_id="negative_control", source="controls", code=control["negative"]),
    ]
    if include_llm_shaped and "llm_shaped_positive" in control:
        candidates.append(
            RecipeEvalCandidate(
                candidate_id="llm_shaped_positive_control",
                source="llm_shaped_controls",
                raw_response=control["llm_shaped_positive"],
            )
        )
    return candidates


def generate_llm_recipe_candidates(
    prompt: str,
    *,
    n: int,
    config_path: str,
    llm_invoke: Optional[Callable[[str], str]] = None,
) -> List[RecipeEvalCandidate]:
    if llm_invoke is None:
        llm_invoke = build_llm_invoker(config_path)

    candidates = []
    for idx in range(n):
        candidate_id = f"llm_sample_{idx:02d}"
        try:
            raw_response = llm_invoke(prompt)
            candidates.append(
                RecipeEvalCandidate(
                    candidate_id=candidate_id,
                    source="llm",
                    raw_response=raw_response,
                    metadata={"config_path": config_path},
                )
            )
        except Exception as exc:
            candidates.append(
                RecipeEvalCandidate(
                    candidate_id=candidate_id,
                    source="llm",
                    raw_response="",
                    generation_error=f"{type(exc).__name__}: {exc}",
                    metadata={"config_path": config_path},
                )
            )
    return candidates


def build_llm_invoker(config_path: str) -> Callable[[str], str]:
    from eval.cli.llm_factory import build_llm, load_config

    llm = build_llm(load_config(config_path))

    def llm_invoke(prompt_text: str) -> str:
        response = llm.invoke(prompt_text)
        return response.content if hasattr(response, "content") else str(response)

    return llm_invoke
