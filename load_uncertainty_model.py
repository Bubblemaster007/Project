"""
负荷不确定性建模模块。

技术含义：
1. 通过多尺度分解近似实现“趋势项—周期项—高频波动项”分离；
2. 通过增长系数把历史负荷映射到近中远期水平年；
3. 通过“月份-小时”分组分位数得到逐时概率边界；
4. 根据负荷分级比例形成关键/重要/一般负荷时序边界。

说明：
报告中可写VMD、Res-GRU、WGAN-GP、KDE；这里给出阶段性可运行代码版本。
后续若已有深度学习模型，可替换 forecast_load_stage()。
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple
import numpy as np
import pandas as pd

from data_schema import Chapter2Config, StageScenarioConfig


def prepare_time_index(df: pd.DataFrame, config: Chapter2Config) -> pd.DataFrame:
    """统一时间字段并补充月份、小时、日期等标签。"""
    out = df.copy()
    out[config.timestamp_col] = pd.to_datetime(out[config.timestamp_col])
    out = out.sort_values(config.timestamp_col).reset_index(drop=True)
    out["month"] = out[config.timestamp_col].dt.month
    out["hour"] = out[config.timestamp_col].dt.hour
    out["dayofyear"] = out[config.timestamp_col].dt.dayofyear
    out["weekday"] = out[config.timestamp_col].dt.weekday
    return out


def multi_scale_load_decompose(
    df: pd.DataFrame,
    config: Chapter2Config,
    short_window: int = 6,
    long_window: int = 24 * 7,
) -> pd.DataFrame:
    """
    负荷多尺度分解。

    用滚动均值近似替代VMD的工程实现：
    - trend_kw：低频趋势项；
    - daily_component_kw：日内与周内波动项；
    - high_frequency_kw：高频扰动项。

    这段代码不是严格VMD，只用于阶段性流程跑通。
    正式算法可替换为VMD/Res-GRU/WGAN-GP。
    """
    out = prepare_time_index(df, config)
    load = out[config.load_col].astype(float)

    out["load_trend_kw"] = load.rolling(long_window, center=True, min_periods=1).mean()
    short_mean = load.rolling(short_window, center=True, min_periods=1).mean()
    out["load_daily_component_kw"] = short_mean - out["load_trend_kw"]
    out["load_high_frequency_kw"] = load - short_mean
    out["load_ramp_kw"] = load.diff().fillna(0.0)
    out["load_3h_ramp_kw"] = load.diff(3).fillna(0.0)

    return out


def forecast_load_stage(
    df: pd.DataFrame,
    stage_config: StageScenarioConfig,
    config: Chapter2Config,
) -> pd.DataFrame:
    """
    近中远期负荷动态预测的简化版本。

    逻辑：
    - 趋势项按增长率放大；
    - 波动项保留历史形态；
    - 高频项按增长后负荷略微放大。
    """
    decomp = multi_scale_load_decompose(df, config)
    g = stage_config.load_growth_rate

    out = decomp.copy()
    out["forecast_load_kw"] = (
        out["load_trend_kw"] * (1.0 + g)
        + out["load_daily_component_kw"] * (1.0 + 0.5 * g)
        + out["load_high_frequency_kw"] * (1.0 + 0.3 * g)
    )
    out["forecast_load_kw"] = out["forecast_load_kw"].clip(lower=0.0)
    out["stage"] = stage_config.name
    return out


def hourly_probability_boundary(
    df: pd.DataFrame,
    value_col: str,
    config: Chapter2Config,
    prefix: str,
) -> pd.DataFrame:
    """
    按“月份-小时”计算概率边界。

    输出字段：
    prefix_q05, prefix_q10, prefix_q50, prefix_q90, prefix_q95 等。
    """
    data = prepare_time_index(df, config)
    grouped = data.groupby(["month", "hour"])[value_col]

    rows = []
    for (month, hour), series in grouped:
        row = {"month": month, "hour": hour}
        clean = series.dropna().astype(float)
        if clean.empty:
            for q in config.quantiles:
                row[f"{prefix}_q{int(q*100):02d}"] = np.nan
        else:
            for q in config.quantiles:
                row[f"{prefix}_q{int(q*100):02d}"] = float(clean.quantile(q))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["month", "hour"]).reset_index(drop=True)


def build_load_probability_boundary(
    historical_df: pd.DataFrame,
    stage_config: StageScenarioConfig,
    config: Chapter2Config,
) -> pd.DataFrame:
    """
    生成指定水平年的负荷概率边界。
    """
    forecast = forecast_load_stage(historical_df, stage_config, config)
    boundary = hourly_probability_boundary(
        forecast,
        value_col="forecast_load_kw",
        config=config,
        prefix=f"load_{stage_config.name}",
    )
    boundary["stage"] = stage_config.name
    return boundary


def build_classified_load_boundary(
    forecast_df: pd.DataFrame,
    critical_ratio: float = 0.20,
    important_ratio: float = 0.35,
    general_ratio: float = 0.45,
) -> pd.DataFrame:
    """
    分级负荷时序边界。

    Parameters
    ----------
    critical_ratio:
        关键负荷比例。
    important_ratio:
        重要负荷比例。
    general_ratio:
        一般负荷比例。
    """
    total = critical_ratio + important_ratio + general_ratio
    if not np.isclose(total, 1.0):
        critical_ratio /= total
        important_ratio /= total
        general_ratio /= total

    out = forecast_df.copy()
    load_col = "forecast_load_kw" if "forecast_load_kw" in out.columns else "load_kw"

    out["critical_load_kw"] = out[load_col] * critical_ratio
    out["important_load_kw"] = out[load_col] * important_ratio
    out["general_load_kw"] = out[load_col] * general_ratio
    return out
