from dataclasses import dataclass
from typing import Dict, List

from eval.generation.data.fixture_models import UniverseFixture


@dataclass(frozen=True)
class RecipeDataQuality:
    behavior_key: str
    expected_universes: List[str]
    generated_universes: List[str]
    missing_universes: List[str]
    extra_universes: List[str]
    row_counts: Dict[str, int]
    column_sets: Dict[str, List[str]]
    metadata: Dict[str, Dict[str, object]]

    @property
    def ok(self) -> bool:
        return not self.missing_universes and not self.extra_universes



def data_quality_report(recipe, fixtures: List[UniverseFixture]) -> RecipeDataQuality:
    expected = [universe.name for universe in recipe.universes]
    generated = [fixture.name for fixture in fixtures]
    return RecipeDataQuality(
        behavior_key=recipe.behavior_key,
        expected_universes=expected,
        generated_universes=generated,
        missing_universes=[name for name in expected if name not in generated],
        extra_universes=[name for name in generated if name not in expected],
        row_counts={
            fixture.name: (
                {filename: int(len(df)) for filename, df in fixture.data.items()}
                if isinstance(fixture.data, dict)
                else int(len(fixture.data))
            )
            for fixture in fixtures
        },
        column_sets={
            fixture.name: (
                {filename: list(df.columns) for filename, df in fixture.data.items()}
                if isinstance(fixture.data, dict)
                else list(fixture.data.columns)
            )
            for fixture in fixtures
        },
        metadata={fixture.name: fixture.metadata for fixture in fixtures},
    )
