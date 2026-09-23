import numpy as np
import pandas as pd
import json
import pytest

from src.chapter3.current_paper_adapter import run_paper_downstream
from run_full_pipeline import main


def fixture(hours=48, with_event=True):
    timestamp = pd.date_range("2021-01-01", periods=hours, freq="h")
    event = np.full(hours, -1, dtype=int)
    if with_event:
        event[6:42] = 7
    return pd.DataFrame({"timestamp": timestamp, "load_kw": 1800 + 100 * np.sin(np.arange(hours) / 5),
                         "wind_kw": 200 + 40 * np.sin(np.arange(hours) / 6),
                         "pv_kw": np.maximum(0, 100 * np.sin(np.arange(hours) * np.pi / 12)),
                         "wind_speed": 4 + 2 * np.sin(np.arange(hours) / 6),
                         "event_id": event})


def test_certified_downstream_uses_nodal_dispatch_and_same_event_trace(tmp_path):
    data = fixture()
    config = {"topology": {"type": "case33", "case33_dir": "data/case33"},
              "paper_method": {"seed": 42, "event_samples": 2}}
    summary = run_paper_downstream(data, config, tmp_path, {"requested": 2, "embedded": 1, "complete": False})
    hourly = pd.read_csv(tmp_path / "balance/hourly_balance.csv")
    phases = pd.read_csv(tmp_path / "event_phase_metrics.csv")
    status = pd.read_csv(tmp_path / "event_stage_status.csv")
    saved = pd.read_csv(tmp_path / "annual_random_production_sequence.csv")
    assert summary["embedding_complete"] is False
    assert summary["valid_event_count"] == 1
    assert len(hourly) == 48
    assert len(phases) == 4
    assert status.status.tolist() == ["analyzed"]
    assert np.allclose(saved[["load_kw", "wind_kw", "pv_kw"]], data[["load_kw", "wind_kw", "pv_kw"]])
    assert np.allclose(hourly.load_kw, hourly.load_served_kw + hourly.power_deficit_kw)
    assert (tmp_path / "events/event_0007/event_phase_conditional_metrics.csv").exists()


def test_no_event_keeps_empty_stage_output_explicit(tmp_path):
    summary = run_paper_downstream(fixture(with_event=False),
                                   {"topology": {"type": "case33", "case33_dir": "data/case33"}},
                                   tmp_path, {"requested": 0, "embedded": 0, "complete": True})
    assert summary["valid_event_count"] == 0
    assert pd.read_csv(tmp_path / "event_phase_metrics.csv").empty
    assert pd.read_csv(tmp_path / "event_stage_status.csv").empty


def test_missing_weather_fails_instead_of_synthesizing(tmp_path):
    data = fixture().drop(columns="wind_speed")
    try:
        run_paper_downstream(data, {}, tmp_path)
    except ValueError as exc:
        assert "wind_speed" in str(exc)
    else:
        raise AssertionError("Missing weather should fail")


def test_all_embedding_requests_failed_are_not_reported_as_complete(tmp_path):
    summary = run_paper_downstream(
        fixture(hours=12, with_event=False),
        {"topology": {"type": "case33", "case33_dir": "data/case33"}},
        tmp_path, {"requested": 3, "embedded": 0, "complete": False}
    )
    assert summary["requested"] == 3
    assert summary["embedded"] == 0
    assert summary["embedding_complete"] is False
    assert summary["valid_event_count"] == 0


def test_precomputed_paper_annual_cli_uses_same_network_downstream(tmp_path):
    annual_path = tmp_path / "annual.csv"
    fixture().to_csv(annual_path, index=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"output_dir": str(tmp_path / "out"),
                                       "topology": {"type": "case33", "case33_dir": "data/case33"},
                                       "paper_method": {"seed": 42, "event_samples": 2}}), encoding="utf-8")
    summary = main(["--config", str(config_path), "--paper-annual-csv", str(annual_path)])
    assert summary["valid_event_count"] == 1
    assert (tmp_path / "out/current_paper/imported_annual/balance/ac_screen.csv").exists()


def test_hourly_device_outage_reaches_nodal_balance(tmp_path):
    data = fixture(hours=12, with_event=False)
    for column, rating in (("available_kw_grid_channel", 4000.0),
                           ("available_kw_emergency_gen", 500.0),
                           ("available_kw_storage", 500.0),
                           ("available_kw_renewable", 1000.0)):
        data[column] = rating
        data.loc[3, column] = 0.0
    run_paper_downstream(data, {"topology": {"type": "case33", "case33_dir": "data/case33"}},
                         tmp_path)
    hourly = pd.read_csv(tmp_path / "balance/hourly_balance.csv")
    assert np.isclose(hourly.loc[3, "power_deficit_kw"], data.loc[3, "load_kw"])
    assert np.isclose(hourly.loc[3, "renewable_unavailable_kw"],
                      data.loc[3, "wind_kw"] + data.loc[3, "pv_kw"])
    assert np.isclose(hourly.loc[3, "curtailment_kw"], 0.0)
    assert np.isclose(hourly.loc[3, "ac_grid_dispatch_kw"], 0.0)


def test_missing_hourly_device_capacity_is_rejected(tmp_path):
    data = fixture(hours=12, with_event=False)
    data["available_kw_grid_channel"] = 4000.0
    data.loc[3, "available_kw_grid_channel"] = np.nan
    with pytest.raises(ValueError, match="Invalid grid capacity"):
        run_paper_downstream(data, {"topology": {"type": "case33", "case33_dir": "data/case33"}},
                             tmp_path)


def test_transformer_outage_caps_upstream_grid(tmp_path):
    data = fixture(hours=12, with_event=False)
    data["available_kw_transformer"] = 4000.0
    data.loc[3, "available_kw_transformer"] = 0.0
    run_paper_downstream(data, {"topology": {"type": "case33", "case33_dir": "data/case33"}},
                         tmp_path)
    hourly = pd.read_csv(tmp_path / "balance/hourly_balance.csv")
    assert hourly.loc[3, "grid_available_kw"] == 0.0
    assert hourly.loc[3, "firm_grid_dispatch_kw"] == 0.0
    assert hourly.loc[3, "ac_grid_dispatch_kw"] <= 1e-5
    assert hourly.loc[3, "power_deficit_kw"] > 0.0


def test_sequential_resource_outage_reaches_conditional_samples(tmp_path):
    data = fixture()
    data["failure_rate_per_hour_grid_channel"] = 0.0
    data.loc[10, "failure_rate_per_hour_grid_channel"] = 1e6
    data["repair_hours_grid_channel"] = 3
    config = {"topology": {"type": "case33", "case33_dir": "data/case33"},
              "paper_method": {"seed": 42, "event_samples": 2}}
    run_paper_downstream(data, config, tmp_path)
    failures = pd.read_csv(tmp_path / "resource_failures.csv")
    annual = pd.read_csv(tmp_path / "annual_random_production_sequence.csv")
    conditional = pd.read_csv(tmp_path / "events/event_0007/event_conditional_hourly.csv")
    assert failures.groupby("sample_id").size().to_dict() == {0: 1, 1: 1}
    assert annual.available_kw_grid_channel.iloc[10:13].eq(0).all()
    assert conditional.loc[conditional.hour_offset.eq(4), "grid_available_kw"].eq(0).all()
