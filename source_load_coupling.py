"""
源荷耦合不确定性表征模块。

对应第2章：
- 净负荷 N_t；
- 源荷偏差 B_t；
- 源荷匹配度 M_t；
- 源荷错配标识；
- 36 h极端窗口候选。

这些结果进入第3章极端风光荷场景生成。
"""

from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd

from data_schema import Chapter2Config
from load_uncertainty_model import prepare_time_index


def compute_source_load_features(
    df: pd.DataFrame,
    config: Chapter2Config,
    load_col: str = "load_kw",
    wind_col: str = "wind_kw",
    pv_col: str = "pv_kw",
) -> pd.DataFrame:
    """
    计算源荷耦合特征。

    N_t = L_t - P_{w,t} - P_{pv,t}
    B_t = L_t - P_{RE,t}
    M_t = P_{RE,t} / L_t
    """
    out = prepare_time_index(df, config)

    for col in [load_col, wind_col, pv_col]:
        if col not in out.columns:
            raise ValueError(f"输入数据缺少字段：{col}")

    out["renewable_kw"] = out[wind_col].astype(float) + out[pv_col].astype(float)
    out["net_load_kw"] = out[load_col].astype(float) - out["renewable_kw"]
    out["source_load_deviation_kw"] = out[load_col].astype(float) - out["renewable_kw"]
    out["source_load_match_rate"] = out["renewable_kw"] / (out[load_col].astype(float) + 1e-6)

    out["net_load_3h_ramp_kw"] = out["net_load_kw"].diff(3).fillna(0.0)
    out["net_load_6h_ramp_kw"] = out["net_load_kw"].diff(6).fillna(0.0)

    high_load_th = out.groupby(["month", "hour"])[load_col].transform(lambda x: x.quantile(config.high_load_quantile))
    low_re_th = out.groupby(["month", "hour"])["renewable_kw"].transform(lambda x: x.quantile(config.low_re_quantile))

    out["high_load_flag"] = (out[load_col] >= high_load_th).astype(int)
    out["low_renewable_flag"] = (out["renewable_kw"] <= low_re_th).astype(int)
    out["source_load_mismatch_flag"] = ((out["high_load_flag"] == 1) & (out["low_renewable_flag"] == 1)).astype(int)

    return out


def score_extreme_windows(
    feature_df: pd.DataFrame,
    config: Chapter2Config,
    window_hours: int | None = None,
) -> pd.DataFrame:
    """
    对所有滑动窗口计算极端候选得分。

    得分由四类因素组成：
    - 累计净负荷缺额压力；
    - 最大3 h净负荷爬坡；
    - 源荷错配小时数；
    - 连续低新能源小时数。
    """
    if window_hours is None:
        window_hours = config.extreme_window_hours

    df = feature_df.sort_values(config.timestamp_col).reset_index(drop=True)
    if len(df) < window_hours:
        raise ValueError("数据长度小于窗口长度")

    rows = []
    net = df["net_load_kw"].values
    ramp = np.abs(df["net_load_3h_ramp_kw"].values)
    mismatch = df["source_load_mismatch_flag"].values
    low_re = df["low_renewable_flag"].values

    for start in range(0, len(df) - window_hours + 1):
        end = start + window_hours
        seg_net = net[start:end]
        seg_ramp = ramp[start:end]
        seg_mismatch = mismatch[start:end]
        seg_low_re = low_re[start:end]

        positive_net = np.clip(seg_net, 0, None)
        cum_net_pressure = float(positive_net.sum())
        max_ramp = float(seg_ramp.max())
        mismatch_hours = int(seg_mismatch.sum())

        # 最长连续低出力
        run = 0
        max_run = 0
        for v in seg_low_re:
            run = run + 1 if v == 1 else 0
            max_run = max(max_run, run)

        score = (
            0.50 * cum_net_pressure
            + 0.30 * max_ramp * window_hours
            + 80.0 * mismatch_hours
            + 50.0 * max_run
        )

        center_idx = start + window_hours // 2
        rows.append({
            "window_start": df.loc[start, config.timestamp_col],
            "window_end": df.loc[end - 1, config.timestamp_col],
            "window_center": df.loc[center_idx, config.timestamp_col],
            "cum_net_pressure_kwh_like": cum_net_pressure,
            "max_3h_net_load_ramp_kw": max_ramp,
            "mismatch_hours": mismatch_hours,
            "max_low_re_continuous_hours": max_run,
            "extreme_window_score": score,
        })

    result = pd.DataFrame(rows)
    return result.sort_values("extreme_window_score", ascending=False).reset_index(drop=True)


def extract_top_extreme_windows(
    feature_df: pd.DataFrame,
    config: Chapter2Config,
) -> pd.DataFrame:
    """
    提取Top-K极端源荷窗口候选。
    """
    scored = score_extreme_windows(feature_df, config)
    return scored.head(config.extreme_top_k).copy()


def slice_window(df: pd.DataFrame, start_time: str, end_time: str, config: Chapter2Config) -> pd.DataFrame:
    """按窗口起止时间截取源荷片段。"""
    data = prepare_time_index(df, config)
    st = pd.to_datetime(start_time)
    et = pd.to_datetime(end_time)
    return data[(data[config.timestamp_col] >= st) & (data[config.timestamp_col] <= et)].copy()
