"""
tests.conftest
==============

PURPOSE
-------
Shared pytest fixtures. Ensures the project root is importable and provides a
small synthetic housing DataFrame so smoke tests run without any network or
downloaded data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture()
def synthetic_housing() -> pd.DataFrame:
    """
    A schema-conformant synthetic housing frame.

    Includes a spread of ``median_income`` so every income-category bin is
    populated (required for the stratified split), a few missing
    ``total_bedrooms`` values, and all schema columns with valid types/ranges.
    """

    rng = np.random.default_rng(42)
    n = 400

    # median_income spread across the [0, 1.5, 3.0, 4.5, 6.0, inf] bins.
    median_income = np.concatenate(
        [
            rng.uniform(0.5, 1.4, 80),
            rng.uniform(1.6, 2.9, 80),
            rng.uniform(3.1, 4.4, 80),
            rng.uniform(4.6, 5.9, 80),
            rng.uniform(6.1, 12.0, 80),
        ]
    )
    rng.shuffle(median_income)

    households = rng.integers(50, 600, n).astype(float)
    population = households * rng.uniform(1.5, 4.0, n)
    total_rooms = households * rng.uniform(4.0, 8.0, n)
    total_bedrooms = households * rng.uniform(0.8, 1.5, n)

    df = pd.DataFrame(
        {
            "longitude": rng.uniform(-124.0, -114.0, n),
            "latitude": rng.uniform(32.0, 42.0, n),
            "housing_median_age": rng.integers(1, 52, n).astype(float),
            "total_rooms": np.round(total_rooms),
            "total_bedrooms": np.round(total_bedrooms),
            "population": np.round(population),
            "households": households,
            "median_income": median_income,
            "median_house_value": rng.uniform(50_000, 500_000, n),
            "ocean_proximity": rng.choice(
                ["<1H OCEAN", "INLAND", "NEAR BAY", "NEAR OCEAN"], n
            ),
        }
    )

    # Inject a few missing total_bedrooms values (schema allows this).
    df.loc[df.sample(8, random_state=1).index, "total_bedrooms"] = np.nan
    return df
