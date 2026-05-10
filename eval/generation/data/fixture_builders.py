from typing import Dict

from eval.generation.data.fixture_models import UniverseFixture


# 所有 _sXXX 生成器都会用的公共包装函数，
# 负责把 name、data、generation metadata、expected metadata 打包成 UniverseFixture。
def _fixture(
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
