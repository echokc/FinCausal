import os
from typing import Dict, List

from eval.generation.data.fixture_models import UniverseFixture

def write_recipe_fixtures(fixtures: List[UniverseFixture], output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    for fixture in fixtures:
        if isinstance(fixture.data, dict):
            path = os.path.join(output_dir, fixture.name)
            os.makedirs(path, exist_ok=True)
            for filename, df in fixture.data.items():
                file_path = os.path.join(path, filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                df.to_csv(file_path, index=False)
        else:
            path = os.path.join(output_dir, f"{fixture.name}.csv")
            fixture.data.to_csv(path, index=False)
        paths[fixture.name] = path
    return paths
