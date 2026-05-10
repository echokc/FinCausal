import argparse
import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from eval.generation.data.fixture_quality import data_quality_report
from eval.generation.data.fixture_io import write_recipe_fixtures
from eval.generation.data.fixture_models import UniverseFixture
from eval.generation.data.fixture_registry import RECIPE_DATA_GENERATORS



GeneratorFn = Callable[[int], List[UniverseFixture]]


# orchestration：生成 fixtures → 做 quality report → 写盘 → 返回 paths。它会依赖 registry.py、quality.py、io.py
def generate_and_write_recipe_data(recipe, output_root: str, seed: int = 42) -> Dict[str, object]:
    fixtures = generate_recipe_fixtures(recipe.behavior_key, seed=seed)
    quality = data_quality_report(recipe, fixtures)
    behavior_dir = os.path.join(output_root, recipe.behavior_key)
    paths = write_recipe_fixtures(fixtures, behavior_dir)
    return {
        "behavior_key": recipe.behavior_key,
        "quality": quality.__dict__,
        "paths": paths,
    }

def generate_recipe_fixtures(behavior_key: str, seed: int = 42) -> List[UniverseFixture]:
    if behavior_key not in RECIPE_DATA_GENERATORS:
        raise KeyError(f"No recipe data generator registered for {behavior_key!r}")
    return RECIPE_DATA_GENERATORS[behavior_key](seed)

