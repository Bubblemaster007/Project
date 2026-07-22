from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOPIC2_REMAINING = ROOT / "topic2_remaining_code"
if str(TOPIC2_REMAINING) not in sys.path:
    sys.path.insert(0, str(TOPIC2_REMAINING))

from fault_probability_model import FaultModelConfig, build_device_state_sequence
from coupled_condition_builder import build_coupled_condition
from src.chapter3.lankao_topology_adapter import build_device_params_from_lankao, build_load_reachability


class Chapter3CouplingTests(unittest.TestCase):
    def test_lankao_topology_can_drive_fault_coupling(self) -> None:
        ts = pd.date_range("2026-01-01", periods=36, freq="h")
        scene = pd.DataFrame(
            {
                "timestamp": ts,
                "load_kw": 42000.0,
                "wind_kw": 4000.0,
                "pv_kw": 2000.0,
                "event_core": [1] * 36,
            }
        )
        hazard = pd.DataFrame(
            {
                "timestamp": ts,
                "wind_speed": 18.0,
                "icing": 0.55,
                "rain": 20.0,
                "heat": 34.0,
            }
        )
        device_params, _, _, metadata = build_device_params_from_lankao(Path("兰考算例数据.xlsx"))
        states, capacity = build_device_state_sequence(hazard, device_params, FaultModelConfig(random_seed=9))
        coupled = build_coupled_condition(scene, capacity, build_load_reachability(hazard))
        self.assertGreater(metadata["line_count"], 0)
        self.assertTrue({"fault_prob", "state", "available_kw"}.issubset(states.columns))
        self.assertTrue({"reachable_load_kw", "available_re_kw", "condition_power_gap_kw_pre_balance"}.issubset(coupled.columns))
        self.assertEqual(len(coupled), 36)
