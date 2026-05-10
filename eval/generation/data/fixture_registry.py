from typing import Callable, Dict, List

from eval.generation.data.fixture_models import UniverseFixture
from eval.generation.data.generators.temporal_fixtures import TEMPORAL_GENERATORS
from eval.generation.data.generators.statistical_fixtures import STATISTICAL_GENERATORS
from eval.generation.data.generators.risk_fixtures import RISK_GENERATORS
from eval.generation.data.generators.regime_fixtures import REGIME_GENERATORS
from eval.generation.data.generators.system_feedback_fixtures import SYSTEM_FEEDBACK_GENERATORS


GeneratorFn = Callable[[int], List[UniverseFixture]]

RECIPE_DATA_GENERATORS: Dict[str, GeneratorFn] = {
    **TEMPORAL_GENERATORS,
    **STATISTICAL_GENERATORS,
    **RISK_GENERATORS,
    **REGIME_GENERATORS,
    **SYSTEM_FEEDBACK_GENERATORS,
}
