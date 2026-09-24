"""Auditable literature-baseline fragility curves.

The values are engineering priors for an IEEE33 integration study, not
Alashankou observations.  They are intentionally stored outside the solver so
that local data can replace them without changing the simulation route.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def logistic_probability(intensity, median: float, scale: float,
                         lower: float = 0.0, upper: float = 0.98):
    if median <= 0 or scale <= 0 or not 0 <= lower < upper <= 1:
        raise ValueError("invalid fragility parameters")
    x = np.asarray(intensity, dtype=float)
    return np.clip(lower + (upper - lower) / (1.0 + np.exp(-(x - median) / scale)), lower, upper)


def annual_probability_to_hourly_hazard(probability, exposure_hours: float = 8760.0):
    """Convert a cumulative exposure probability to a constant hourly hazard."""
    p = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0 - 1e-12)
    if exposure_hours <= 0:
        raise ValueError("exposure_hours must be positive")
    return -np.log1p(-p) / float(exposure_hours)


def load_baseline(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"scenario", "hazard", "device_type", "median_intensity",
                "scale_intensity", "exposure_probability", "repair_hours"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"fragility table missing columns: {sorted(missing)}")
    if frame.empty or (frame.median_intensity <= 0).any() or (frame.scale_intensity <= 0).any():
        raise ValueError("invalid fragility table")
    if ((frame.exposure_probability <= 0) | (frame.exposure_probability >= 1)).any():
        raise ValueError("exposure_probability must be in (0, 1)")
    if (frame.repair_hours < 1).any():
        raise ValueError("repair_hours must be positive")
    return frame


def hourly_hazard_from_table(intensity, row: pd.Series, exposure_hours: float = 8760.0):
    cumulative = logistic_probability(intensity, float(row.median_intensity),
                                      float(row.scale_intensity),
                                      upper=float(row.exposure_probability))
    return annual_probability_to_hourly_hazard(cumulative, exposure_hours)
