from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.chapter4.case33_network_balance import Case33NetworkBalance, NetworkBalanceConfig, resource_limits_from_row


ROOT = Path(__file__).resolve().parents[1]


def network(config=None):
    nodes = pd.read_csv(ROOT / "data/case33/nodes.csv")
    lines = pd.read_csv(ROOT / "data/case33/lines.csv")
    return Case33NetworkBalance(nodes, lines, config or NetworkBalanceConfig())


def test_case33_full_load_conserved_with_voltage_limits():
    model = network()
    row, soc, state = model.dispatch_hour("2021-01-01", 3715.0, 0.0, 0.0, set(), 825.0)
    assert np.isclose(row["load_kw"], row["load_served_kw"] + row["power_deficit_kw"])
    assert row["min_linear_voltage_pu"] >= 0.95 - 1e-8
    assert row["max_abs_branch_flow_kw"] <= 5000.0 + 1e-8
    assert 150.0 <= soc <= 1425.0
    assert len(state["branch_flow_kw"]) == 32


def test_root_line_outage_disconnects_all_load_and_renewables():
    model = network()
    row, _, _ = model.dispatch_hour("2021-01-01", 3000.0, 500.0, 500.0, {1}, 825.0)
    assert np.isclose(row["power_deficit_kw"], 3000.0)
    assert np.isclose(row["renewable_used_kw"], 0.0)
    assert np.isclose(row["curtailment_kw"], 1000.0)


def test_tight_branch_limit_causes_shortage_without_overload():
    model = network(NetworkBalanceConfig(branch_limit_kw=100.0))
    row, _, _ = model.dispatch_hour("2021-01-01", 3000.0, 0.0, 0.0, set(), 825.0)
    assert row["power_deficit_kw"] > 0
    assert row["max_abs_branch_flow_kw"] <= 100.0 + 1e-8


def test_storage_state_passes_between_hours():
    model = network(NetworkBalanceConfig(grid_limit_kw=0, emergency_limit_kw=0))
    first, soc, _ = model.dispatch_hour("2021-01-01 00:00", 100.0, 0.0, 0.0, set(), 825.0)
    second, soc2, _ = model.dispatch_hour("2021-01-01 01:00", 100.0, 0.0, 0.0, set(), soc)
    assert first["storage_discharge_kw"] > 0
    assert second["storage_discharge_kw"] > 0
    assert soc2 < soc < 825.0


def test_ac_screen_converges_and_reports_unmodeled_losses():
    model = network()
    row, _, state = model.dispatch_hour("2021-01-01", 2500.0, 0.0, 0.0, set(), 825.0)
    screen = model.ac_screen(state, set())
    assert screen["ac_converged"]
    assert screen["ac_root_required_kw"] >= row["firm_grid_dispatch_kw"] - 1e-5
    assert 0.0 < screen["ac_min_voltage_pu"] <= 1.0


def test_explicit_zero_availability_overrides_normal_ratings():
    model = network()
    limits = resource_limits_from_row(pd.Series({
        "available_kw_grid_channel": 0.0,
        "available_kw_emergency_gen": 0.0,
        "available_kw_storage": 0.0,
        "available_kw_renewable": 0.0,
    }))
    row, _, state = model.dispatch_hour("2021-01-01", 100.0, 50.0, 50.0, set(), 825.0, **limits)
    assert np.isclose(row["power_deficit_kw"], 100.0)
    assert np.isclose(row["renewable_unavailable_kw"], 100.0)
    assert np.isclose(row["curtailment_kw"], 0.0)
    assert np.isclose(model.ac_screen(state, set())["ac_root_required_kw"], 0.0)


def test_invalid_explicit_availability_fails():
    model = network()
    for bad in (float("nan"), -1.0):
        with pytest.raises(ValueError, match="Invalid grid capacity"):
            model.dispatch_hour("2021-01-01", 100.0, 0.0, 0.0, set(), 825.0,
                                grid_available_kw=bad)


def test_root_transformer_caps_grid_and_aggregate_line_is_rejected():
    assert resource_limits_from_row(pd.Series({"available_kw_grid_channel": 4000.0,
                                               "available_kw_transformer": 0.0}))["grid_available_kw"] == 0
    with pytest.raises(ValueError, match="Aggregate available_kw_line"):
        resource_limits_from_row(pd.Series({"available_kw_line": 0.0}))


def test_ac_loss_reserve_converges_when_renewables_exceed_load():
    model = network()
    row, _, _, ac = model.dispatch_hour_ac_balanced(
        "2020-08-24 12:00", 489.5148048079973, 496.8995656507277,
        590.0382674390456, set(), 1425.0)
    assert abs(row["estimated_line_loss_kw"] - row["loss_reserve_kw"]) <= 1e-5
    assert ac["ac_root_required_kw"] <= 1e-5
