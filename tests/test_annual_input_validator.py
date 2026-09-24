import pandas as pd
import pytest

from scripts.validate_annual_input import validate_annual_input


def _frame(n=24):
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
        "load_kw": 100.0, "wind_kw": 20.0, "pv_kw": 10.0,
        "event_id": -1, "wind_speed": 4.0,
    })


def test_validator_accepts_continuous_input(tmp_path):
    path = tmp_path / "annual.csv"; _frame().to_csv(path, index=False)
    result = validate_annual_input(path, require_8760=False)
    assert result["rows"] == 24 and result["event_count"] == 0


def test_validator_rejects_unpaired_failure_columns(tmp_path):
    frame = _frame(); frame["failure_rate_per_hour_storage"] = 0.1
    path = tmp_path / "annual.csv"; frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="supplied together"):
        validate_annual_input(path, require_8760=False)


def test_validator_rejects_time_gap(tmp_path):
    frame = _frame(); frame.loc[5, "timestamp"] += pd.Timedelta(hours=1)
    path = tmp_path / "annual.csv"; frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="NaN or duplicates"):
        validate_annual_input(path, require_8760=False)
