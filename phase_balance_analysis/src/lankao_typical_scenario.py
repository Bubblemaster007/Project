"""构造阶段逻辑连续的兰考拓扑驱动典型蒙特卡洛仿真数据。"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd


def _interp(profile: dict[str, float], key: str, progress: float) -> float:
    """在阶段起止参数之间线性插值。"""
    start = float(profile[f"{key}_start"])
    end = float(profile[f"{key}_end"])
    return start + (end - start) * progress


def _stage_for_hour(
    hour: int,
    stages: list[dict[str, Any]],
) -> tuple[dict[str, Any], float]:
    for stage in stages:
        start = int(stage["start_hour"])
        end = int(stage["end_hour"])
        if start <= hour <= end:
            duration = max(end - start, 1)
            return stage, (hour - start) / duration
    raise ValueError(f"小时 {hour} 未被四阶段配置覆盖")


def _ar_noise(
    rng: np.random.Generator,
    count: int,
    sigma: float,
    persistence: float = 0.72,
) -> np.ndarray:
    """生成轻量自相关扰动，避免逐小时完全独立跳变。"""
    values = np.zeros(count)
    for index in range(count):
        innovation = rng.normal(0.0, sigma)
        values[index] = (
            innovation
            if index == 0
            else persistence * values[index - 1] + innovation
        )
    return values


def build_lankao_typical_scenarios(
    project_root: Path,
    config: dict[str, Any],
    output_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """基于兰考拓扑规模生成36 h、四阶段连续的典型仿真序列。"""
    scenario_cfg = config.get("typical_scenario", {})
    phase_cfg = config.get("phase_segmentation", {})
    stages = sorted(phase_cfg.get("stages", []), key=lambda row: int(row["order"]))
    profiles = scenario_cfg.get("stage_profiles", {})
    if len(stages) != 4:
        raise ValueError("典型仿真要求四个阶段")
    missing_profiles = [row["name"] for row in stages if row["name"] not in profiles]
    if missing_profiles:
        raise ValueError(f"缺少阶段仿真参数：{missing_profiles}")

    # 复用兰考拓扑适配器获取负荷规模和各类资源额定容量。
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.chapter3.lankao_topology_adapter import build_device_params_from_lankao, build_device_params_from_case33

    topology_path = project_root / config.get("data", {}).get(
        "topology_file", "兰考算例数据.xlsx"
    )
    topology_cfg = scenario_cfg.get("topology_resource_scales", {})
    if scenario_cfg.get("topology_type") == "case33":
        device_params, _, _, metadata = build_device_params_from_case33(
            project_root / scenario_cfg.get("case33_dir", "data/case33"), topology_cfg
        )
    else:
        device_params, _, _, metadata = build_device_params_from_lankao(topology_path, topology_cfg)
    peak_load_kw = float(metadata["peak_load_kw"])
    rated_by_type = (
        device_params.groupby("device_type")["rated_kw"].sum().to_dict()
    )
    storage_rated_kw = float(rated_by_type.get("storage", peak_load_kw * 0.08))
    emergency_rated_kw = float(
        rated_by_type.get("emergency_gen", peak_load_kw * 0.10)
    )
    grid_rated_kw = float(
        rated_by_type.get("grid_channel", peak_load_kw * 0.25)
    )
    transformer_rated_kw = float(
        rated_by_type.get("transformer", peak_load_kw * 1.30)
    )
    line_reference_kw = max(peak_load_kw * 2.0, transformer_rated_kw * 1.35)

    simulation_count = int(scenario_cfg.get("simulation_count", 200))
    random_seed = int(config.get("random_seed", 2026))
    start_time = pd.Timestamp(
        scenario_cfg.get("event_start_time", "2026-07-15 00:00:00")
    )
    total_hours = max(int(row["end_hour"]) for row in stages) + 1
    wind_capacity_kw = peak_load_kw * float(
        scenario_cfg.get("wind_capacity_scale", 0.20)
    )
    pv_capacity_kw = peak_load_kw * float(
        scenario_cfg.get("pv_capacity_scale", 0.24)
    )
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []

    for simulation_index in range(simulation_count):
        scenario_severity = float(np.clip(rng.normal(1.0, 0.09), 0.78, 1.22))
        load_scale = float(np.clip(rng.normal(1.0, 0.035), 0.91, 1.09))
        wind_scale = float(np.clip(rng.normal(1.0, 0.10), 0.75, 1.25))
        pv_scale = float(np.clip(rng.normal(1.0, 0.08), 0.80, 1.20))
        load_noise = _ar_noise(rng, total_hours, sigma=0.012)
        wind_noise = _ar_noise(rng, total_hours, sigma=0.045)
        availability_noise = _ar_noise(rng, total_hours, sigma=0.018)

        for hour in range(total_hours):
            stage, progress = _stage_for_hour(hour, stages)
            stage_name = str(stage["name"])
            profile = profiles[stage_name]
            timestamp = start_time + pd.Timedelta(hours=hour)
            hour_of_day = timestamp.hour

            # 兰考节点总负荷规模上的典型日负荷曲线。
            base_load_ratio = (
                0.76
                + 0.095 * np.sin(2.0 * np.pi * (hour_of_day - 8) / 24.0)
                + 0.035 * np.sin(4.0 * np.pi * (hour_of_day - 17) / 24.0)
            )
            load_multiplier = _interp(profile, "load_multiplier", progress)
            load_kw = (
                peak_load_kw
                * base_load_ratio
                * load_multiplier
                * load_scale
                * (1.0 + load_noise[hour])
            )

            # 风光潜在出力保持日内形态，灾害影响通过可用系数连续施加。
            wind_cf = np.clip(
                0.48
                + 0.11 * np.sin(2.0 * np.pi * (hour_of_day + 2) / 24.0)
                + wind_noise[hour],
                0.18,
                0.78,
            )
            daylight = max(
                0.0,
                np.sin(np.pi * (hour_of_day - 6.0) / 12.0),
            )
            wind_potential_kw = wind_capacity_kw * wind_cf * wind_scale
            pv_potential_kw = pv_capacity_kw * daylight * pv_scale

            severity = np.clip(
                _interp(profile, "severity", progress) * scenario_severity,
                0.0,
                1.0,
            )
            common_noise = availability_noise[hour]
            renewable_factor = np.clip(
                _interp(profile, "renewable_availability", progress)
                - 0.10 * (scenario_severity - 1.0)
                + common_noise,
                0.05,
                1.0,
            )
            grid_factor = np.clip(
                _interp(profile, "grid_availability", progress)
                - 0.16 * (scenario_severity - 1.0)
                + common_noise,
                0.0,
                1.0,
            )
            network_factor = np.clip(
                _interp(profile, "network_availability", progress)
                - 0.12 * (scenario_severity - 1.0)
                + 0.6 * common_noise,
                0.30,
                1.0,
            )
            storage_factor = np.clip(
                _interp(profile, "storage_availability", progress)
                - 0.06 * (scenario_severity - 1.0),
                0.55,
                1.0,
            )
            emergency_factor = np.clip(
                _interp(profile, "emergency_availability", progress)
                - 0.08 * (scenario_severity - 1.0),
                0.55,
                1.0,
            )
            equipment_availability = float(
                np.mean([
                    renewable_factor,
                    grid_factor,
                    network_factor,
                    storage_factor,
                    emergency_factor,
                ])
            )
            simulation_id = f"lankao_mc_{simulation_index + 1:03d}"
            rows.append(
                {
                    "timestamp": timestamp,
                    "simulation_id": simulation_id,
                    "random_sequence_id": simulation_id,
                    "sequence_weight": 1.0 / simulation_count,
                    "event_id": "lankao_typical_event_001",
                    "stage": stage_name,
                    "event_hour": hour,
                    "load_kw": max(0.0, load_kw),
                    "reachable_load_kw": max(0.0, load_kw),
                    "wind_kw": max(0.0, wind_potential_kw),
                    "pv_kw": max(0.0, pv_potential_kw),
                    "available_wind_kw": max(
                        0.0, wind_potential_kw * renewable_factor
                    ),
                    "available_pv_kw": max(
                        0.0, pv_potential_kw * renewable_factor
                    ),
                    "available_re_kw": max(
                        0.0,
                        (wind_potential_kw + pv_potential_kw) * renewable_factor,
                    ),
                    "available_kw_storage": storage_rated_kw * storage_factor,
                    "available_kw_emergency_gen": (
                        emergency_rated_kw * emergency_factor
                    ),
                    "available_kw_grid_channel": grid_rated_kw * grid_factor,
                    "available_kw_line": line_reference_kw * network_factor,
                    "available_kw_transformer": (
                        transformer_rated_kw * network_factor
                    ),
                    "equipment_availability": equipment_availability,
                    "hazard_intensity": float(severity),
                    "event_core": int(stage_name in {"灾害冲击", "灾害持续"}),
                    "is_extreme_condition": 1,
                }
            )

    data = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False, encoding="utf-8-sig")
    assumptions = {
        "data_type": "兰考拓扑驱动典型仿真测试数据（非实测）",
        "random_seed": random_seed,
        "simulation_count": simulation_count,
        "event_hours": total_hours,
        "peak_load_mw": peak_load_kw / 1000.0,
        "wind_capacity_mw": wind_capacity_kw / 1000.0,
        "pv_capacity_mw": pv_capacity_kw / 1000.0,
        "storage_power_mw": storage_rated_kw / 1000.0,
        "emergency_power_mw": emergency_rated_kw / 1000.0,
        "grid_channel_mw": grid_rated_kw / 1000.0,
        "transformer_capacity_mw": transformer_rated_kw / 1000.0,
        "stage_profiles": profiles,
    }
    return data, assumptions
