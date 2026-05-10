from typing import List

from eval.recipes.components.prompt_models import PromptVariant


def prompt_variant(
    *,
    name: str,
    framing: str,
    causal_requirement: str,
    difficulty_level: int,
    surface_domains: List[str],
) -> PromptVariant:
    return PromptVariant(
        name=name,
        framing=framing,
        causal_requirement=causal_requirement,
        difficulty_level=difficulty_level,
        surface_domains=surface_domains,
    )
