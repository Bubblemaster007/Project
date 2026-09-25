import numpy as np
import pandas as pd

from scripts.run_paper_reference_case33 import select_wind_windows, simulate_line_states
from src.chapter3.fragility_curves import load_baseline


def test_weather_windows_are_distinct_and_not_selected_from_deficit():
    n = 8760
    wind = pd.Series(5 + 2 * np.sin(np.arange(n) / 25))
    wind.iloc[1000:1010] = 15
    wind.iloc[3000:3010] = 20
    wind.iloc[5000:5010] = 25
    timestamps = pd.Series(pd.date_range("2021-01-01", periods=n, freq="h"))
    events = select_wind_windows(wind, timestamps)
    starts = [event["start_index"] for event in events]
    assert len(events) == 3
    assert all(abs(a - b) >= 36 for i, a in enumerate(starts) for b in starts[i + 1:])
    assert events[0]["six_hour_rolling_wind_speed"] >= events[1]["six_hour_rolling_wind_speed"]


def test_line_state_is_seed_reproducible_and_repair_persists():
    n = 2000
    wind = np.linspace(5.0, 25.0, n)
    times = pd.Series(pd.date_range("2021-01-01", periods=n, freq="h"))
    events = np.full(n, -1)
    first, logs, _ = simulate_line_states(wind, list(range(1, 33)), 42, times, events)
    second, logs_again, _ = simulate_line_states(wind, list(range(1, 33)), 42, times, events)
    assert first == second
    assert logs == logs_again
    assert logs
    for item in logs:
        hour = int((item["timestamp"] - times.iloc[0]).total_seconds() / 3600)
        assert all(item["line"] in first[k] for k in range(hour, min(n, hour + item["repair_hours"])))


def test_literature_fragility_scenarios_change_hourly_hazard():
    table = load_baseline("data/fragility/baseline_fragility.csv")
    n = 2000
    wind = np.linspace(5.0, 35.0, n)
    times = pd.Series(pd.date_range("2021-01-01", periods=n, freq="h"))
    events = np.full(n, -1)
    _, base, rates_base = simulate_line_states(wind, list(range(1, 33)), 42, times, events, table, "base")
    _, conservative, rates_cons = simulate_line_states(wind, list(range(1, 33)), 42, times, events, table, "conservative")
    assert rates_cons.max() > rates_base.max()
    assert len(conservative) >= len(base)
