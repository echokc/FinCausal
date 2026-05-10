from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalBehaviorSpec:
    behavior_key: str
    pillar: str
    difficulty: str
    default_schema_variant: str
    default_prompt_variant: str
