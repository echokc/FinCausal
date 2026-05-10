from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

from eval.recipes.components.behavior_models import TemporalBehaviorSpec
from eval.recipes.components.prompt_models import PromptVariant


@dataclass(frozen=True)
class UniverseRecipe:
    name: str
    role: Literal["baseline", "treatment", "leakage", "stress", "positive", "negative", "trap"]
    description: str
    interventions: List[Dict[str, Any]] = field(default_factory=list)
    expected: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScalarOutputRecipe:
    variable_name: str
    semantic: str
    valid_range: tuple[float, float] | None = None
    accepted_names: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class OutputRecipe:
    variable_name: str
    semantic: str
    kind: Literal["scalar", "vector", "dataframe", "series", "dict"]
    valid_range: tuple[float, float] | None = None
    shape: tuple[int, ...] | None = None
    accepted_names: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class MultiUniverseScalarRecipe:
    behavior_key: str
    pillar: str
    difficulty: str
    schema_variants: Dict[str, Dict[str, str]]
    default_schema_variant: str
    prompt_variants: Dict[str, PromptVariant]
    default_prompt_variant: str
    distribution_variants: List[str]
    mechanism_variant: str
    output: ScalarOutputRecipe
    universes: List[UniverseRecipe]
    probes: List[Dict[str, Any]]
    task_steps: List[str]
    load_snippet: str
    known_traps: List[Dict[str, str]] = field(default_factory=list)

    @property
    def behavior_spec(self) -> TemporalBehaviorSpec:
        return TemporalBehaviorSpec(
            behavior_key=self.behavior_key,
            pillar=self.pillar,
            difficulty=self.difficulty,
            default_schema_variant=self.default_schema_variant,
            default_prompt_variant=self.default_prompt_variant,
        )


@dataclass(frozen=True)
class MultiUniverseOutputRecipe:
    behavior_key: str
    pillar: str
    difficulty: str
    schema_variants: Dict[str, Dict[str, str]]
    default_schema_variant: str
    prompt_variants: Dict[str, PromptVariant]
    default_prompt_variant: str
    distribution_variants: List[str]
    mechanism_variant: str
    output: OutputRecipe
    universes: List[UniverseRecipe]
    probes: List[Dict[str, Any]]
    task_steps: List[str]
    load_snippet: str
    known_traps: List[Dict[str, str]] = field(default_factory=list)

    @property
    def behavior_spec(self) -> TemporalBehaviorSpec:
        return TemporalBehaviorSpec(
            behavior_key=self.behavior_key,
            pillar=self.pillar,
            difficulty=self.difficulty,
            default_schema_variant=self.default_schema_variant,
            default_prompt_variant=self.default_prompt_variant,
        )
