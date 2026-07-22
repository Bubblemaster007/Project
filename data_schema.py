"""
第2章数据结构与配置。

对应报告逻辑：
多元不确定性因素识别 → 负荷不确定性建模 → 新能源出力不确定性建模
→ 源荷耦合不确定性表征 → 源荷动态增长预测。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class StageScenarioConfig:
    """
    水平年配置。

    load_growth_rate:
        负荷增长比例。例如0.10表示较历史基准增长10%。
    wind_capacity_scale:
        风电装机或可用出力缩放系数。
    pv_capacity_scale:
        光伏装机或可用出力缩放系数。
    noise_scale:
        生成年度背景场景时的随机扰动强度。
    """
    name: str
    load_growth_rate: float
    wind_capacity_scale: float
    pv_capacity_scale: float
    noise_scale: float = 0.03


@dataclass
class Chapter2Config:
    """第2章流程配置。"""
    timestamp_col: str = "timestamp"
    load_col: str = "load_kw"
    wind_col: str = "wind_kw"
    pv_col: str = "pv_kw"
    freq: str = "h"
    quantiles: Tuple[float, ...] = (0.05, 0.10, 0.50, 0.90, 0.95)
    high_load_quantile: float = 0.90
    low_re_quantile: float = 0.20
    extreme_window_hours: int = 36
    extreme_top_k: int = 20
    random_seed: int = 42


DEFAULT_STAGE_CONFIGS: Dict[str, StageScenarioConfig] = {
    "near": StageScenarioConfig(
        name="near",
        load_growth_rate=0.05,
        wind_capacity_scale=1.05,
        pv_capacity_scale=1.05,
        noise_scale=0.03,
    ),
    "mid": StageScenarioConfig(
        name="mid",
        load_growth_rate=0.18,
        wind_capacity_scale=1.25,
        pv_capacity_scale=1.30,
        noise_scale=0.04,
    ),
    "far": StageScenarioConfig(
        name="far",
        load_growth_rate=0.35,
        wind_capacity_scale=1.55,
        pv_capacity_scale=1.70,
        noise_scale=0.05,
    ),
}
