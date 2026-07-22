"""
8760 h年度常规源荷背景场景生成模块。

技术含义：
把第2章源荷动态预测结果从“概率边界/样本”转化为第3章可嵌入极端工况的全年时序基底。

简化实现：
按“月份-小时”从历史样本重采样，并叠加近中远期增长系数和随机扰动。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_schema import Chapter2Config, StageScenarioConfig
from load_uncertainty_model import prepare_time_index


def generate_annual_timestamps(year: int = 2026, freq: str = "h") -> pd.DatetimeIndex:
    """生成一年8760 h时间戳。"""
    ts = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:00:00", freq=freq)
    if len(ts) > 8760:
        ts = ts[:8760]
    return ts


def build_month_hour_sampler(historical_df: pd.DataFrame, cols: list[str]) -> tuple[dict[str, dict[tuple[int, int], np.ndarray]], dict[str, np.ndarray]]:
    """Pre-group historical samples so annual generation stays fast."""
    grouped: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    fallback: dict[str, np.ndarray] = {}
    for col in cols:
        grouped[col] = {
            key: values.dropna().astype(float).to_numpy()
            for key, values in historical_df.groupby(["month", "hour"])[col]
        }
        fallback_values = historical_df[col].dropna().astype(float).to_numpy()
        fallback[col] = fallback_values if len(fallback_values) else np.array([0.0])
    return grouped, fallback


def sample_by_month_hour(
    grouped: dict[str, dict[tuple[int, int], np.ndarray]],
    fallback: dict[str, np.ndarray],
    month: int,
    hour: int,
    rng: np.random.Generator,
    col: str,
) -> float:
    sample = grouped[col].get((month, hour))
    if sample is None or len(sample) == 0:
        sample = fallback[col]
    return float(rng.choice(sample))


def generate_annual_background(
    historical_df: pd.DataFrame,
    stage_config: StageScenarioConfig,
    config: Chapter2Config,
    year: int = 2026,
) -> pd.DataFrame:
    """
    生成指定水平年的8760 h常规源荷背景场景。

    输出字段：
    timestamp, load_kw, wind_kw, pv_kw, stage
    """
    rng = np.random.default_rng(config.random_seed)
    hist = prepare_time_index(historical_df, config)
    sample_cols = [config.load_col, config.wind_col, config.pv_col]
    grouped, fallback = build_month_hour_sampler(hist, sample_cols)

    ts = generate_annual_timestamps(year, config.freq)
    rows = []

    for t in ts:
        month = t.month
        hour = t.hour

        load = sample_by_month_hour(grouped, fallback, month, hour, rng, config.load_col)
        wind = sample_by_month_hour(grouped, fallback, month, hour, rng, config.wind_col)
        pv = sample_by_month_hour(grouped, fallback, month, hour, rng, config.pv_col)

        # 近中远期修正
        load *= (1.0 + stage_config.load_growth_rate)
        wind *= stage_config.wind_capacity_scale
        pv *= stage_config.pv_capacity_scale

        # 扰动
        load *= max(0.0, 1.0 + rng.normal(0.0, stage_config.noise_scale))
        wind *= max(0.0, 1.0 + rng.normal(0.0, stage_config.noise_scale))
        pv *= max(0.0, 1.0 + rng.normal(0.0, stage_config.noise_scale))

        rows.append({
            "timestamp": t,
            "load_kw": max(0.0, load),
            "wind_kw": max(0.0, wind),
            "pv_kw": max(0.0, pv),
            "stage": stage_config.name,
        })

    out = pd.DataFrame(rows)
    out["renewable_kw"] = out["wind_kw"] + out["pv_kw"]
    out["net_load_kw"] = out["load_kw"] - out["renewable_kw"]
    return out
