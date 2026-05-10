from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class PromptVariant:
    name: str
    framing: str
    causal_requirement: str
    difficulty_level: int
    surface_domains: List[str]
