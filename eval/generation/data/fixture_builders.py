from typing import Dict

from eval.generation.data.fixture_models import UniverseFixture

def fixture(
    name: str,
    data: object,
    *,
    variant: str,
    expected: Dict[str, object] | None = None,
    **generation: object,
) -> UniverseFixture:
    return UniverseFixture(
        name,
        data,
        {
            "generation": {"variant": variant, **generation},
            "expected": expected or {},
        },
    )
