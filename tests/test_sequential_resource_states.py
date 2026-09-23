import numpy as np
import pandas as pd
import pytest

from src.chapter3.sequential_resource_states import simulate_resource_states
from src.chapter4.case33_network_balance import NetworkBalanceConfig


def source():
    return pd.DataFrame({"timestamp": pd.date_range("2021-01-01", periods=6, freq="h"),
                         "load_kw": [1000.0] * 6, "wind_kw": [100.0] * 6,
                         "pv_kw": [50.0] * 6})


def test_failure_persists_for_repair_duration_then_recovers():
    data = source()
    data["failure_rate_per_hour_grid_channel"] = [1e6, 0, 0, 0, 0, 0]
    data["repair_hours_grid_channel"] = 3
    result, failures = simulate_resource_states(data, NetworkBalanceConfig(), 42)
    assert result.available_kw_grid_channel.tolist() == [0, 0, 0, 4000, 4000, 4000]
    assert failures.resource.tolist() == ["grid_channel"]
    assert failures.repair_hours.tolist() == [3]
    assert np.array_equal(result.load_kw, data.load_kw)


def test_incomplete_or_invalid_rate_inputs_fail():
    data = source()
    data["failure_rate_per_hour_storage"] = 0.1
    with pytest.raises(ValueError, match="Both"):
        simulate_resource_states(data, NetworkBalanceConfig(), 42)
    data["repair_hours_storage"] = 2
    data.loc[2, "failure_rate_per_hour_storage"] = np.nan
    with pytest.raises(ValueError, match="Invalid failure rate"):
        simulate_resource_states(data, NetworkBalanceConfig(), 42)
