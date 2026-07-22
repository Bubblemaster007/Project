"""
极端工况嵌入8760 h年度常规源荷背景模块。
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
import pandas as pd


def _to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out


def smooth_boundary(annual: pd.DataFrame, start_idx: int, end_idx: int, columns: List[str], ramp_hours: int = 2) -> pd.DataFrame:
    """对嵌入区间前后做线性边界平滑。"""
    out = annual.copy()
    n = len(out)
    for col in columns:
        if col not in out.columns:
            continue
        left0, left1 = max(0, start_idx - ramp_hours), start_idx
        if left1 > left0 and start_idx < n:
            out.loc[left0:left1, col] = np.linspace(out.loc[left0, col], out.loc[start_idx, col], left1 - left0 + 1)
        right0, right1 = end_idx, min(n - 1, end_idx + ramp_hours)
        if right1 > right0 and end_idx < n:
            out.loc[right0:right1, col] = np.linspace(out.loc[end_idx, col], out.loc[right1, col], right1 - right0 + 1)
    return out


def embed_extreme_condition(
    annual_background_df: pd.DataFrame,
    coupled_condition_df: pd.DataFrame,
    embed_start_time: str,
    sequence_id: str = "seq_001",
    sequence_weight: float = 1.0,
    smooth: bool = True,
    ramp_hours: int = 2,
) -> pd.DataFrame:
    """将36 h源荷-故障耦合极端工况嵌入8760 h年度背景序列。"""
    annual = _to_datetime(annual_background_df).sort_values("timestamp").reset_index(drop=True)
    condition = _to_datetime(coupled_condition_df).sort_values("timestamp").reset_index(drop=True)
    start_time = pd.to_datetime(embed_start_time)
    candidates = annual.index[annual["timestamp"] == start_time].tolist()
    start_idx = candidates[0] if candidates else int((annual["timestamp"] - start_time).abs().idxmin())
    end_idx = start_idx + len(condition) - 1
    if end_idx >= len(annual):
        raise ValueError("嵌入区间超出年度序列长度")

    out = annual.copy()
    copy_cols = [
        "load_kw", "wind_kw", "pv_kw", "reachable_load_kw", "available_wind_kw", "available_pv_kw",
        "available_re_kw", "available_kw_storage", "available_kw_emergency_gen", "available_kw_grid_channel",
        "available_kw_line", "available_kw_transformer", "load_reachability_factor", "net_load_kw",
        "condition_power_gap_kw_pre_balance", "event_core"
    ]
    for col in copy_cols:
        if col in condition.columns:
            if col not in out.columns:
                out[col] = np.nan
            out.loc[start_idx:end_idx, col] = condition[col].values

    # 常规时段默认值，保证第4章平衡代码可读
    if "reachable_load_kw" not in out.columns:
        out["reachable_load_kw"] = out["load_kw"]
    else:
        out["reachable_load_kw"] = out["reachable_load_kw"].fillna(out["load_kw"])
    if "available_wind_kw" not in out.columns:
        out["available_wind_kw"] = out["wind_kw"]
    else:
        out["available_wind_kw"] = out["available_wind_kw"].fillna(out["wind_kw"])
    if "available_pv_kw" not in out.columns:
        out["available_pv_kw"] = out["pv_kw"]
    else:
        out["available_pv_kw"] = out["available_pv_kw"].fillna(out["pv_kw"])
    out["available_re_kw"] = out.get("available_re_kw", out["available_wind_kw"] + out["available_pv_kw"]).fillna(out["available_wind_kw"] + out["available_pv_kw"])
    for col, default in {
        "available_kw_storage": 0.0,
        "available_kw_emergency_gen": 0.0,
        "available_kw_grid_channel": 0.0,
        "available_kw_line": np.inf,
        "available_kw_transformer": np.inf,
        "load_reachability_factor": 1.0,
        "event_core": 0,
    }.items():
        if col not in out.columns:
            out[col] = default
        else:
            out[col] = out[col].fillna(default)

    out["random_sequence_id"] = sequence_id
    out["sequence_weight"] = sequence_weight
    out["is_extreme_condition"] = 0
    out.loc[start_idx:end_idx, "is_extreme_condition"] = 1

    if smooth:
        out = smooth_boundary(out, start_idx, end_idx, ["load_kw", "wind_kw", "pv_kw", "reachable_load_kw"], ramp_hours)
    for col in ["load_kw", "wind_kw", "pv_kw", "reachable_load_kw", "available_wind_kw", "available_pv_kw"]:
        out[col] = out[col].clip(lower=0.0)
    return out


def batch_embed_conditions(annual_background_df: pd.DataFrame, conditions: List[pd.DataFrame], embed_start_times: List[str], weights: Optional[List[float]] = None) -> pd.DataFrame:
    if weights is None:
        weights = [1.0 / len(conditions)] * len(conditions)
    seqs = []
    for idx, (cond, st, w) in enumerate(zip(conditions, embed_start_times, weights), start=1):
        seqs.append(embed_extreme_condition(annual_background_df, cond, st, f"seq_{idx:03d}", w))
    return pd.concat(seqs, ignore_index=True)
