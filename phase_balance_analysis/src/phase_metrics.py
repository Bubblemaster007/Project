"""四灾害阶段六项核心指标计算。"""

from __future__ import annotations

from typing import Any
import pandas as pd


MAIN_COLUMNS = [
    "stage",
    "stage_order",
    "electricity_shortage_probability_pct",
    "curtailment_probability_pct",
    "maximum_electricity_shortage_mw",
    "maximum_curtailment_power_mw",
    "expected_energy_not_served_mwh",
    "expected_curtailed_energy_mwh",
    "data_source",
    "simulation_count",
    "time_step_hours",
]


def compute_phase_metrics(
    hourly: pd.DataFrame,
    time_step_hours: float,
    epsilon_mw: float,
    data_source: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按全部仿真小时聚合概率、功率极值和期望电量。"""
    simulation_count = int(hourly["simulation_id"].nunique())
    if simulation_count <= 0:
        raise ValueError("simulation_count 必须大于 0")

    main_rows: list[dict[str, Any]] = []
    for (stage_order, stage), group in hourly.groupby(
        ["stage_order", "stage"], sort=True
    ):
        main_rows.append(
            {
                "stage": stage,
                "stage_order": int(stage_order),
                "electricity_shortage_probability_pct": float(
                    (group["power_loss_mw"] > epsilon_mw).mean() * 100.0
                ),
                "curtailment_probability_pct": float(
                    (group["curtailment_mw"] > epsilon_mw).mean() * 100.0
                ),
                "maximum_electricity_shortage_mw": float(
                    group["power_loss_mw"].max()
                ),
                "maximum_curtailment_power_mw": float(
                    group["curtailment_mw"].max()
                ),
                "expected_energy_not_served_mwh": float(
                    group["power_loss_mw"].sum()
                    * time_step_hours
                    / simulation_count
                ),
                "expected_curtailed_energy_mwh": float(
                    group["curtailment_mw"].sum()
                    * time_step_hours
                    / simulation_count
                ),
                "data_source": data_source,
                "simulation_count": simulation_count,
                "time_step_hours": float(time_step_hours),
            }
        )

    detail_rows: list[dict[str, Any]] = []
    group_cols = ["simulation_id", "event_id", "stage_order", "stage"]
    for keys, group in hourly.groupby(group_cols, sort=True):
        simulation_id, event_id, stage_order, stage = keys
        detail_rows.append(
            {
                "simulation_id": simulation_id,
                "event_id": event_id,
                "stage": stage,
                "stage_order": int(stage_order),
                "stage_time_steps": int(len(group)),
                "electricity_shortage_probability_pct": float(
                    (group["power_loss_mw"] > epsilon_mw).mean() * 100.0
                ),
                "curtailment_probability_pct": float(
                    (group["curtailment_mw"] > epsilon_mw).mean() * 100.0
                ),
                "maximum_electricity_shortage_mw": float(
                    group["power_loss_mw"].max()
                ),
                "maximum_curtailment_power_mw": float(
                    group["curtailment_mw"].max()
                ),
                "energy_not_served_mwh": float(
                    group["power_loss_mw"].sum() * time_step_hours
                ),
                "curtailed_energy_mwh": float(
                    group["curtailment_mw"].sum() * time_step_hours
                ),
            }
        )

    main = pd.DataFrame(main_rows).sort_values("stage_order").reset_index(drop=True)
    detail = pd.DataFrame(detail_rows).sort_values(
        ["simulation_id", "event_id", "stage_order"]
    ).reset_index(drop=True)
    return main[MAIN_COLUMNS], detail


def total_expected_energy(
    hourly: pd.DataFrame,
    time_step_hours: float,
) -> tuple[float, float]:
    """计算完整事件窗口的期望缺电量和期望弃电量。"""
    simulation_count = int(hourly["simulation_id"].nunique())
    return (
        float(hourly["power_loss_mw"].sum() * time_step_hours / simulation_count),
        float(hourly["curtailment_mw"].sum() * time_step_hours / simulation_count),
    )
