import unittest

import pandas as pd

from src.chapter4.power_energy_balance import analyze_power_energy_balance
from topic2_remaining_code.coupled_condition_builder import build_coupled_condition


class BalanceP0Tests(unittest.TestCase):
    def test_zero_network_capacity_blocks_grid_supply(self):
        seq = pd.DataFrame({"timestamp": ["2021-01-01"], "load_kw": [100.0],
                            "available_re_kw": [0.0], "available_kw_line": [0.0],
                            "is_extreme_condition": [1]})
        _, hourly = analyze_power_energy_balance(seq, {"grid_channel_kw": 200.0})
        self.assertEqual(float(hourly.power_deficit_kw.iloc[0]), 100.0)

    def test_unreachable_load_counts_in_original_demand(self):
        seq = pd.DataFrame({"timestamp": ["2021-01-01"], "load_kw": [100.0],
                            "reachable_load_kw": [40.0], "available_re_kw": [0.0]})
        metrics, hourly = analyze_power_energy_balance(seq, {"grid_channel_kw": 200.0})
        self.assertEqual(float(hourly.power_deficit_kw.iloc[0]), 60.0)
        self.assertEqual(metrics["critical_load_supply_ratio"], 1.0)
        self.assertEqual(metrics["important_load_restoration_ratio"], 20.0 / 35.0)

    def test_zero_renewable_capacity_means_zero_output(self):
        scene = pd.DataFrame({"timestamp": ["2021-01-01"], "load_kw": [100.0],
                              "wind_kw": [50.0], "pv_kw": [20.0]})
        cap = pd.DataFrame({"timestamp": ["2021-01-01"],
                            "available_kw_renewable": [0.0]})
        coupled = build_coupled_condition(scene, cap)
        self.assertEqual(float(coupled.available_re_kw.iloc[0]), 0.0)

    def test_missing_extreme_capacity_fails(self):
        seq = pd.DataFrame({"timestamp": ["2021-01-01"], "load_kw": [100.0],
                            "available_kw_storage": [float("nan")],
                            "is_extreme_condition": [1]})
        with self.assertRaises(ValueError):
            analyze_power_energy_balance(seq, {"storage_power_kw": 100.0})


if __name__ == "__main__":
    unittest.main()
