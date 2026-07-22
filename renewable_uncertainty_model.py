"""
新能源出力不确定性建模模块。

技术含义：
- 风电：识别低风持续、强风切出、短时波动。
- 光伏：识别低辐照、夜间、积雪/沙尘遮挡、短时骤降。
- 输出：风光出力概率边界和低出力风险标识。

这里给出可运行工程版本，正式模型可替换为SVMD + Attention-TCN + KDE。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_schema import Chapter2Config, StageScenarioConfig
from load_uncertainty_model import prepare_time_index, hourly_probability_boundary


def scale_renewable_stage(
    df: pd.DataFrame,
    stage_config: StageScenarioConfig,
    config: Chapter2Config,
) -> pd.DataFrame:
    """
    根据水平年装机增长系数修正风光出力。
    """
    out = prepare_time_index(df, config)
    out["forecast_wind_kw"] = (out[config.wind_col].astype(float) * stage_config.wind_capacity_scale).clip(lower=0.0)
    out["forecast_pv_kw"] = (out[config.pv_col].astype(float) * stage_config.pv_capacity_scale).clip(lower=0.0)
    out["forecast_re_kw"] = out["forecast_wind_kw"] + out["forecast_pv_kw"]
    out["stage"] = stage_config.name
    return out


def build_renewable_probability_boundary(
    historical_df: pd.DataFrame,
    stage_config: StageScenarioConfig,
    config: Chapter2Config,
) -> pd.DataFrame:
    """
    构建风电、光伏、新能源总出力概率边界。
    """
    scaled = scale_renewable_stage(historical_df, stage_config, config)
    wind_b = hourly_probability_boundary(scaled, "forecast_wind_kw", config, f"wind_{stage_config.name}")
    pv_b = hourly_probability_boundary(scaled, "forecast_pv_kw", config, f"pv_{stage_config.name}")
    re_b = hourly_probability_boundary(scaled, "forecast_re_kw", config, f"re_{stage_config.name}")

    out = wind_b.merge(pv_b, on=["month", "hour"], how="outer").merge(re_b, on=["month", "hour"], how="outer")
    out["stage"] = stage_config.name
    return out.sort_values(["month", "hour"]).reset_index(drop=True)


def identify_low_renewable_risk(
    df: pd.DataFrame,
    config: Chapter2Config,
    wind_low_quantile: float = 0.20,
    pv_low_quantile: float = 0.20,
    re_low_quantile: float = 0.20,
) -> pd.DataFrame:
    """
    识别低风、低光、风光同时低出力风险。
    """
    out = prepare_time_index(df, config)

    wind_th = out.groupby(["month", "hour"])[config.wind_col].transform(lambda x: x.quantile(wind_low_quantile))
    pv_th = out.groupby(["month", "hour"])[config.pv_col].transform(lambda x: x.quantile(pv_low_quantile))
    re = out[config.wind_col].astype(float) + out[config.pv_col].astype(float)
    re_th = re.groupby([out["month"], out["hour"]]).transform(lambda x: x.quantile(re_low_quantile))

    out["low_wind_flag"] = (out[config.wind_col] <= wind_th).astype(int)
    out["low_pv_flag"] = (out[config.pv_col] <= pv_th).astype(int)
    out["low_re_flag"] = (re <= re_th).astype(int)
    out["wind_pv_simultaneous_low_flag"] = ((out["low_wind_flag"] == 1) & (out["low_pv_flag"] == 1)).astype(int)

    # 连续低出力时长，按连续段累计
    flag = out["low_re_flag"].values
    duration = np.zeros(len(flag), dtype=int)
    run = 0
    for i, v in enumerate(flag):
        run = run + 1 if v == 1 else 0
        duration[i] = run
    out["low_re_continuous_hours"] = duration

    return out
