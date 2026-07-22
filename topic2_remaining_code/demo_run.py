"""
完整流程演示脚本。真实项目中，用已有极端风光荷生成结果替换 extreme，
用已有电力电量平衡分析代码输出替换 metrics。
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from fault_probability_model import FaultModelConfig, build_device_state_sequence
from coupled_condition_builder import build_coupled_condition
from annual_sequence_embedder import embed_extreme_condition
from strategy_trigger import trigger_strategies, strategy_summary_text
from resilience_reliability_planner import run_simple_planning, planning_result_to_dataframe


def make_demo_data():
    rng = np.random.default_rng(42)
    annual_ts = pd.date_range("2026-01-01 00:00:00", periods=8760, freq="h")
    h = np.arange(8760)
    load = 600 + 80 * np.sin(2 * np.pi * (h % 24) / 24 - np.pi / 2) + 60 * np.sin(2 * np.pi * h / 8760) + rng.normal(0, 20, 8760)
    wind = 180 + 60 * np.sin(2 * np.pi * h / 168) + rng.normal(0, 25, 8760)
    pv = np.maximum(0, 260 * np.sin(np.pi * ((h % 24) - 6) / 12)) + rng.normal(0, 10, 8760)
    annual = pd.DataFrame({"timestamp": annual_ts, "load_kw": np.clip(load, 200, None), "wind_kw": np.clip(wind, 0, None), "pv_kw": np.clip(pv, 0, None)})

    ex_ts = pd.date_range("2026-01-01 00:00:00", periods=36, freq="h")
    eh = np.arange(36)
    extreme = pd.DataFrame({
        "timestamp": ex_ts,
        "load_kw": 820 + 80 * np.sin(2 * np.pi * eh / 24) + rng.normal(0, 15, 36),
        "wind_kw": np.clip(40 + rng.normal(0, 10, 36), 0, None),
        "pv_kw": np.clip(60 * np.sin(np.pi * ((eh % 24) - 6) / 12), 0, None),
        "event_core": ((eh >= 8) & (eh <= 26)).astype(int),
    })

    hazard = pd.DataFrame({
        "timestamp": ex_ts,
        "wind_speed": 10 + 18 * np.exp(-((eh - 16) / 6) ** 2),
        "icing": 0.1 + 0.8 * np.exp(-((eh - 20) / 8) ** 2),
        "rain": 5 + 20 * np.exp(-((eh - 14) / 5) ** 2),
        "heat": 30 + 2 * rng.random(36),
    })

    device_params = pd.DataFrame([
        {"device_id": "line_01", "device_type": "line", "rated_kw": 1000, "a_wind": 0.30, "b_wind": 20, "a_icing": 1.8, "b_icing": 0.7, "a_rain": 0.08, "b_rain": 20, "normal_factor": 1, "derate_factor": 0.5, "recovery_factor": 0.8},
        {"device_id": "tr_01", "device_type": "transformer", "rated_kw": 800, "a_wind": 0.10, "b_wind": 25, "a_icing": 1.0, "b_icing": 0.8, "a_rain": 0.12, "b_rain": 18, "normal_factor": 1, "derate_factor": 0.6, "recovery_factor": 0.8},
        {"device_id": "ess_01", "device_type": "storage", "rated_kw": 300, "a_wind": 0.05, "b_wind": 30, "a_heat": 0.2, "b_heat": 38, "normal_factor": 1, "derate_factor": 0.7, "recovery_factor": 0.9},
        {"device_id": "eg_01", "device_type": "emergency_gen", "rated_kw": 250, "a_wind": 0.08, "b_wind": 28, "a_rain": 0.05, "b_rain": 25, "normal_factor": 1, "derate_factor": 0.7, "recovery_factor": 0.9},
        {"device_id": "grid_01", "device_type": "grid_channel", "rated_kw": 400, "a_wind": 0.25, "b_wind": 22, "a_icing": 1.4, "b_icing": 0.6, "normal_factor": 1, "derate_factor": 0.4, "recovery_factor": 0.7},
    ])
    return annual, extreme, hazard, device_params


def main():
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    annual, extreme, hazard, device_params = make_demo_data()

    states, capacity = build_device_state_sequence(hazard, device_params, config=FaultModelConfig(random_seed=7))
    coupled = build_coupled_condition(extreme, capacity)
    annual_seq = embed_extreme_condition(annual, coupled, embed_start_time="2026-01-15 00:00:00", sequence_id="seq_demo_001", sequence_weight=1.0)

    # 模拟第4章电力电量平衡分析代码输出
    metrics = {
        "loss_of_load_probability": 0.035,
        "curtailment_probability": 0.08,
        "max_power_deficit_kw": 260.0,
        "max_curtailment_kw": 180.0,
        "expected_energy_deficit_kwh": 760.0,
        "expected_curtailment_energy_kwh": 620.0,
        "critical_load_supply_ratio": 0.94,
        "important_load_restoration_ratio": 0.86,
        "mobile_reachability": 0.55,
        "island_required_hours": 8.0,
    }

    trigger_df = trigger_strategies(metrics)
    planning_result = run_simple_planning(metrics, trigger_df)
    planning_df = planning_result_to_dataframe(planning_result)

    states.to_csv(out_dir / "device_state_sequence_36h.csv", index=False, encoding="utf-8-sig")
    capacity.to_csv(out_dir / "available_capacity_36h.csv", index=False, encoding="utf-8-sig")
    coupled.to_csv(out_dir / "coupled_condition_36h.csv", index=False, encoding="utf-8-sig")
    annual_seq.to_csv(out_dir / "annual_random_production_sequence.csv", index=False, encoding="utf-8-sig")
    trigger_df.to_csv(out_dir / "strategy_trigger_result.csv", index=False, encoding="utf-8-sig")
    planning_df.to_csv(out_dir / "planning_result.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "planning_result.json", "w", encoding="utf-8") as f:
        json.dump(planning_result, f, ensure_ascii=False, indent=2)

    print("流程运行完成。输出目录：outputs/")
    print("\n策略触发结果：")
    print(trigger_df[["strategy", "triggered", "priority"]])
    print("\n策略说明：")
    print(strategy_summary_text(trigger_df))
    print("\n推荐配置：")
    print(json.dumps(planning_result["recommended_plan"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
