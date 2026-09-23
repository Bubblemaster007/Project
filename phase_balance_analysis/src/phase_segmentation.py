"""极端事件窗口识别与四阶段划分。"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


STAGE_ALIASES = {
    "灾前准备": "灾前准备",
    "灾前准备阶段": "灾前准备",
    "preparation": "灾前准备",
    "pre_event": "灾前准备",
    "灾害冲击": "灾害冲击",
    "灾害冲击阶段": "灾害冲击",
    "impact": "灾害冲击",
    "灾害持续": "灾害持续",
    "灾害持续阶段": "灾害持续",
    "sustain": "灾害持续",
    "sustained": "灾害持续",
    "灾后恢复": "灾后恢复",
    "灾后恢复阶段": "灾后恢复",
    "recovery": "灾后恢复",
}


def _normalize_existing_stages(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map(
        {key.lower(): value for key, value in STAGE_ALIASES.items()}
    )


def _assign_event_ids(data: pd.DataFrame, time_step_hours: float) -> pd.DataFrame:
    """按相邻时间间隔识别同一仿真中的连续极端事件。"""
    out = data.copy()
    if "event_id" in out.columns and out["event_id"].notna().all():
        out["event_id"] = out["event_id"].astype(str)
        return out

    event_ids = pd.Series(index=out.index, dtype=object)
    for simulation_id, group in out.groupby("simulation_id", sort=False):
        group = group.sort_values("timestamp")
        gaps = group["timestamp"].diff().dt.total_seconds().div(3600.0)
        event_no = ((gaps.isna()) | (gaps > time_step_hours * 1.5)).cumsum()
        event_ids.loc[group.index] = [
            f"{simulation_id}_event_{int(number):03d}" for number in event_no
        ]
    out["event_id"] = event_ids
    return out


def segment_event_phases(
    data: pd.DataFrame,
    phase_config: dict[str, Any],
    time_step_hours: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """优先使用已有四阶段标签，否则使用配置中的测试边界。"""
    out = _assign_event_ids(data, time_step_hours)
    stages = sorted(phase_config.get("stages", []), key=lambda row: int(row["order"]))
    if len(stages) != 4:
        raise ValueError("phase_segmentation.stages 必须恰好包含四个灾害阶段")

    used_existing = False
    if "stage" in out.columns:
        normalized = _normalize_existing_stages(out["stage"])
        if normalized.notna().all() and set(normalized.unique()) == {
            row["name"] for row in stages
        }:
            out["stage"] = normalized
            used_existing = True

    stage_order_map = {str(row["name"]): int(row["order"]) for row in stages}
    if not used_existing:
        out["event_hour"] = (
            out.groupby(["simulation_id", "event_id"], sort=False).cumcount()
            * float(time_step_hours)
        )
        out["stage"] = pd.Series(index=out.index, dtype=object)
        for row in stages:
            start = float(row["start_hour"])
            end = float(row["end_hour"])
            mask = (out["event_hour"] >= start) & (out["event_hour"] <= end + 1e-9)
            out.loc[mask, "stage"] = str(row["name"])
        if out["stage"].isna().any():
            bad = out.loc[out["stage"].isna(), "event_hour"].tolist()[:8]
            raise ValueError(f"存在未被配置阶段边界覆盖的事件小时：{bad}")
    else:
        out["event_hour"] = (
            out.groupby(["simulation_id", "event_id"], sort=False).cumcount()
            * float(time_step_hours)
        )

    out["stage_order"] = out["stage"].map(stage_order_map).astype(int)
    expected_hours = sum(
        int(round((float(row["end_hour"]) - float(row["start_hour"])) / time_step_hours)) + 1
        for row in stages
    )
    event_counts = out.groupby(["simulation_id", "event_id"]).size()
    if bool(phase_config.get("require_exact_window_hours", False)):
        if not (event_counts == expected_hours).all():
            raise ValueError(
                f"测试阶段划分要求每个事件 {expected_hours} 个时间步，"
                f"实际范围为 {int(event_counts.min())}～{int(event_counts.max())}"
            )

    audit = {
        "method": "existing_stage_labels" if used_existing else "configured_test_boundaries",
        "label": str(phase_config.get("label", "阶段划分")),
        "expected_event_steps": expected_hours,
        "event_count": int(event_counts.size),
        "stage_definitions": stages,
    }
    return out.sort_values(
        ["simulation_id", "event_id", "event_hour"]
    ).reset_index(drop=True), audit
