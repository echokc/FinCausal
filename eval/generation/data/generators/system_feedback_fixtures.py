import os
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from eval.generation.data.fixture_builders import fixture
from eval.generation.data.fixture_models import UniverseFixture


GeneratorFn = Callable[[int], List[UniverseFixture]]

def _s006(seed: int) -> List[UniverseFixture]:
    rng = np.random.default_rng(seed)
    qty = 0.5
    low_vol = float(rng.uniform(0.0010, 0.0016))
    high_vol = float(low_vol * rng.uniform(2.4, 3.4))
    neutral = pd.DataFrame({
        "timestamp": pd.date_range("2026-02-01 10:00", periods=20, freq="min"),
        "side": ["BUY", "SELL"] * 10,
        "qty": [qty] * 20,
        "price": [505.0] * 20,
        "vol_1min": [0.0018] * 20,
    })
    long_low = pd.DataFrame({
        "timestamp": pd.date_range("2026-02-01 10:00", periods=25, freq="min"),
        "side": ["BUY"] * 25,
        "qty": [qty] * 25,
        "price": [505.0] * 25,
        "vol_1min": [low_vol] * 25,
    })
    long_high = long_low.copy()
    long_high["vol_1min"] = high_vol
    return [
        fixture("neutral_inventory", neutral, variant="neutral_inventory", expected={"net_position": 0.0}, qty=qty),
        fixture("long_low_vol", long_low, variant="long_inventory_low_vol", expected={"net_position": 25 * qty}, qty=qty, vol_1min=low_vol),
        fixture("long_high_vol", long_high, variant="long_inventory_high_vol", expected={"net_position": 25 * qty}, qty=qty, vol_1min=high_vol),
    ]


SYSTEM_FEEDBACK_GENERATORS: Dict[str, GeneratorFn] = {
    "s006_inventory_induced_skew": _s006,
}
