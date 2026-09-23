"""Sequential repair states for explicitly supplied hourly equipment hazard rates.

Rates are hourly failure intensities, not cumulative fragility probabilities.
No resource outage is invented when an input rate column is absent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.chapter4.case33_network_balance import NetworkBalanceConfig


RESOURCE_NAMES = ("grid_channel", "transformer", "emergency_gen", "storage", "renewable")


def simulate_resource_states(source: pd.DataFrame, config: NetworkBalanceConfig,
                             seed: int, sample_id: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply failure/repair paths to capacity columns without changing source/load.

    A modeled resource requires both `failure_rate_per_hour_<name>` and
    `repair_hours_<name>` in every hour. A failure occupies the sampled hour
    and the next repair_hours-1 hours, even if the hazard later subsides.
    """
    result = source.copy()
    rng = np.random.default_rng(seed)
    failures = []
    ratings = {
        "grid_channel": config.grid_limit_kw,
        "transformer": config.grid_limit_kw,
        "emergency_gen": config.emergency_limit_kw,
        "storage": config.storage_power_kw,
    }
    for name in RESOURCE_NAMES:
        rate_col = f"failure_rate_per_hour_{name}"
        repair_col = f"repair_hours_{name}"
        capacity_col = f"available_kw_{name}"
        has_rate, has_repair = rate_col in result, repair_col in result
        if has_rate != has_repair:
            raise ValueError(f"Both {rate_col} and {repair_col} are required")
        if not has_rate:
            continue
        rate = pd.to_numeric(result[rate_col], errors="raise").to_numpy(float)
        repair = pd.to_numeric(result[repair_col], errors="raise").to_numpy(float)
        if (not np.isfinite(rate).all() or (rate < 0).any()
                or not np.isfinite(repair).all() or (repair < 1).any()
                or (repair != np.floor(repair)).any()):
            raise ValueError(f"Invalid failure rate or repair hours for {name}")
        if name == "renewable":
            rating = np.maximum(0.0, result.wind_kw.to_numpy(float) + result.pv_kw.to_numpy(float))
        else:
            rating = np.full(len(result), ratings[name], dtype=float)
        if capacity_col in result:
            baseline = pd.to_numeric(result[capacity_col], errors="raise").to_numpy(float)
            if not np.isfinite(baseline).all() or (baseline < 0).any():
                raise ValueError(f"Invalid baseline capacity for {name}")
            rating = np.minimum(rating, baseline)
        capacity = rating.copy()
        remaining = 0
        for hour, hazard_rate in enumerate(rate):
            if remaining > 0:
                capacity[hour] = 0.0
                remaining -= 1
            elif rng.random() < -np.expm1(-hazard_rate):
                duration = int(repair[hour])
                capacity[hour] = 0.0
                remaining = duration - 1
                failures.append({"timestamp": result.timestamp.iloc[hour], "sample_id": sample_id,
                                 "seed": seed, "resource": name, "repair_hours": duration,
                                 "failure_rate_per_hour": float(hazard_rate)})
        result[capacity_col] = capacity
    return result, pd.DataFrame(failures, columns=["timestamp", "sample_id", "seed", "resource",
                                                   "repair_hours", "failure_rate_per_hour"])
