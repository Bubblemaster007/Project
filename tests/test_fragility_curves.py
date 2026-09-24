import numpy as np
import pandas as pd

from src.chapter3.fragility_curves import (annual_probability_to_hourly_hazard,
                                           load_baseline, logistic_probability)


def test_baseline_table_loads_and_probability_is_monotone():
    frame = load_baseline("data/fragility/baseline_fragility.csv")
    row = frame.iloc[0]
    values = logistic_probability(np.array([0, row.median_intensity, 50]), row.median_intensity, row.scale_intensity, upper=row.exposure_probability)
    assert values[0] < values[1] < values[2] <= row.exposure_probability


def test_cumulative_probability_conversion_is_small_hourly_hazard():
    hazard = annual_probability_to_hourly_hazard(0.2)
    assert 0 < hazard < 1e-3
