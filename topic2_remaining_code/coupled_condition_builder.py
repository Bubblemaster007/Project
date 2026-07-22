"""
源荷-故障耦合极端工况构建模块。

场景：只描述负荷、风电、光伏时序。
工况：源荷场景 + 设备状态 + 资源可用容量 + 负荷可达性。
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd


def _ensure_timestamp(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError(f"{name} 必须包含 timestamp 字段")
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out


def build_coupled_condition(
    extreme_scene_df: pd.DataFrame,
    capacity_df: pd.DataFrame,
    load_reachability_df: Optional[pd.DataFrame] = None,
    external_grid_available_kw: float = 0.0,
) -> pd.DataFrame:
    """构建36 h源荷-故障耦合极端工况。"""
    scene = _ensure_timestamp(extreme_scene_df, "extreme_scene_df")
    cap = _ensure_timestamp(capacity_df, "capacity_df")
    for col in ["load_kw", "wind_kw", "pv_kw"]:
        if col not in scene.columns:
            raise ValueError(f"extreme_scene_df 缺少字段：{col}")

    coupled = scene.merge(cap, on="timestamp", how="left").fillna(0.0)

    if load_reachability_df is not None:
        reach = _ensure_timestamp(load_reachability_df, "load_reachability_df")
        coupled = coupled.merge(reach, on="timestamp", how="left")
    if "load_reachability_factor" not in coupled.columns:
        coupled["load_reachability_factor"] = 1.0
    coupled["load_reachability_factor"] = coupled["load_reachability_factor"].clip(0.0, 1.0)
    coupled["reachable_load_kw"] = coupled["load_kw"] * coupled["load_reachability_factor"]

    for col in [
        "available_kw_storage",
        "available_kw_emergency_gen",
        "available_kw_grid_channel",
        "available_kw_line",
        "available_kw_transformer",
        "available_kw_renewable",
    ]:
        if col not in coupled.columns:
            coupled[col] = 0.0

    if coupled["available_kw_grid_channel"].sum() == 0.0:
        coupled["available_kw_grid_channel"] = external_grid_available_kw

    renewable_factor = np.ones(len(coupled))
    if coupled["available_kw_renewable"].max() > 0:
        renewable_factor = np.minimum(1.0, coupled["available_kw_renewable"] / (coupled["wind_kw"] + coupled["pv_kw"] + 1e-6))

    coupled["available_wind_kw"] = coupled["wind_kw"] * renewable_factor
    coupled["available_pv_kw"] = coupled["pv_kw"] * renewable_factor
    coupled["available_re_kw"] = coupled["available_wind_kw"] + coupled["available_pv_kw"]
    coupled["available_supply_kw_no_storage"] = (
        coupled["available_re_kw"] + coupled["available_kw_emergency_gen"] + coupled["available_kw_grid_channel"]
    )
    coupled["net_load_kw"] = coupled["reachable_load_kw"] - coupled["available_re_kw"]
    coupled["condition_power_gap_kw_pre_balance"] = (
        coupled["reachable_load_kw"] - coupled["available_supply_kw_no_storage"]
    ).clip(lower=0.0)
    return coupled
