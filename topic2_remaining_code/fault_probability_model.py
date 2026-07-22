"""
设备故障概率与设备状态序列生成模块。

技术路线：灾害强度输入 → 设备脆弱性分析 → 单灾害故障概率映射
→ 多灾害故障概率叠加 → 设备状态抽样 → 可用容量修正。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple
import numpy as np
import pandas as pd

HAZARD_NAMES = ["wind", "icing", "rain", "heat"]


@dataclass
class FaultModelConfig:
    random_seed: int = 42
    derate_probability_band: Tuple[float, float] = (0.20, 0.55)
    recovery_probability_band: Tuple[float, float] = (0.08, 0.20)
    max_failure_probability: float = 0.98


def sigmoid_fragility(intensity: pd.Series, a: float, b: float) -> pd.Series:
    """单灾害脆弱性曲线。"""
    x = a * (intensity.astype(float) - b)
    return 1.0 / (1.0 + np.exp(-x))


def calculate_fault_probabilities(
    hazard_df: pd.DataFrame,
    device_params: pd.DataFrame,
    hazard_names: Iterable[str] = HAZARD_NAMES,
    max_failure_probability: float = 0.98,
) -> pd.DataFrame:
    """计算设备逐时综合故障概率。"""
    if "timestamp" not in hazard_df.columns:
        raise ValueError("hazard_df 必须包含 timestamp 字段")
    if "device_id" not in device_params.columns:
        raise ValueError("device_params 必须包含 device_id 字段")

    hazard_alias = {
        "wind": ["wind", "wind_speed"],
        "icing": ["icing", "ice", "ice_thickness"],
        "rain": ["rain", "rainfall"],
        "heat": ["heat", "temperature"],
    }
    rows = []

    for _, dev in device_params.iterrows():
        combined_success = pd.Series(1.0, index=hazard_df.index)
        for hz in hazard_names:
            a_col, b_col = f"a_{hz}", f"b_{hz}"
            if a_col not in device_params.columns or b_col not in device_params.columns:
                continue
            if pd.isna(dev.get(a_col)) or pd.isna(dev.get(b_col)):
                continue
            intensity_col = None
            for candidate in hazard_alias.get(hz, [hz]):
                if candidate in hazard_df.columns:
                    intensity_col = candidate
                    break
            if intensity_col is None:
                continue
            p_single = sigmoid_fragility(hazard_df[intensity_col], float(dev[a_col]), float(dev[b_col]))
            combined_success *= (1.0 - p_single)

        p_fault = (1.0 - combined_success).clip(lower=0.0, upper=max_failure_probability)
        for idx, ts in enumerate(hazard_df["timestamp"].values):
            rows.append({
                "timestamp": ts,
                "device_id": dev["device_id"],
                "device_type": dev.get("device_type", "unknown"),
                "rated_kw": float(dev.get("rated_kw", 0.0)),
                "fault_prob": float(p_fault.iloc[idx]),
            })
    return pd.DataFrame(rows)


def sample_device_states(
    fault_prob_df: pd.DataFrame,
    device_params: pd.DataFrame,
    config: Optional[FaultModelConfig] = None,
) -> pd.DataFrame:
    """根据故障概率抽样设备状态：normal、derated、failed、recovery。"""
    config = config or FaultModelConfig()
    rng = np.random.default_rng(config.random_seed)
    param_map = device_params.set_index("device_id").to_dict(orient="index")
    low_derate, high_derate = config.derate_probability_band
    low_recovery, high_recovery = config.recovery_probability_band

    rows = []
    for _, row in fault_prob_df.iterrows():
        p = float(row["fault_prob"])
        u = rng.random()
        if u < p:
            state = "failed"
        elif low_derate <= p < high_derate:
            state = "derated"
        elif low_recovery <= p < high_recovery:
            state = "recovery"
        else:
            state = "normal"

        param = param_map.get(row["device_id"], {})
        normal_factor = float(param.get("normal_factor", 1.0))
        derate_factor = float(param.get("derate_factor", 0.5))
        recovery_factor = float(param.get("recovery_factor", 0.8))
        factor = {"normal": normal_factor, "derated": derate_factor, "recovery": recovery_factor, "failed": 0.0}[state]
        rated_kw = float(row.get("rated_kw", param.get("rated_kw", 0.0)))
        rows.append({**row.to_dict(), "state": state, "available_factor": factor, "available_kw": rated_kw * factor})
    return pd.DataFrame(rows)


def summarize_available_capacity(state_df: pd.DataFrame) -> pd.DataFrame:
    """按设备类型汇总逐时可用容量。"""
    if state_df.empty:
        return pd.DataFrame()
    pivot = state_df.pivot_table(
        index="timestamp",
        columns="device_type",
        values="available_kw",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    pivot.columns = ["timestamp" if c == "timestamp" else f"available_kw_{c}" for c in pivot.columns]
    return pivot


def build_device_state_sequence(hazard_df: pd.DataFrame, device_params: pd.DataFrame, config: Optional[FaultModelConfig] = None):
    """生成设备状态序列和设备类型可用容量汇总。"""
    config = config or FaultModelConfig()
    fault_prob = calculate_fault_probabilities(hazard_df, device_params, max_failure_probability=config.max_failure_probability)
    states = sample_device_states(fault_prob, device_params, config=config)
    capacity = summarize_available_capacity(states)
    return states, capacity
