"""
八项保供策略触发模块。

根据第4章六项核心指标及辅助变量，输出策略触发结果、触发原因和配置建议。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd


@dataclass
class StrategyThresholds:
    loss_of_load_probability: float = 0.02
    curtailment_probability: float = 0.05
    max_power_deficit_kw: float = 200.0
    max_curtailment_kw: float = 200.0
    expected_energy_deficit_kwh: float = 500.0
    expected_curtailment_energy_kwh: float = 500.0
    critical_load_supply_ratio_min: float = 0.98
    important_load_restoration_ratio_min: float = 0.90
    mobile_reachability_min: float = 0.60
    island_hours_min: float = 2.0


def _get(metrics: Dict[str, float], key: str, default: float = 0.0) -> float:
    try:
        return float(metrics.get(key, default))
    except Exception:
        return default


def trigger_strategies(metrics: Dict[str, float], thresholds: Optional[StrategyThresholds] = None) -> pd.DataFrame:
    """触发八项保供策略。"""
    th = thresholds or StrategyThresholds()
    rows: List[Dict[str, object]] = []

    def add(strategy: str, triggered: bool, priority: int, reason: str, config_hint: str):
        rows.append({"strategy": strategy, "triggered": bool(triggered), "priority": priority if triggered else 0, "reason": reason, "config_hint": config_hint})

    lolp = _get(metrics, "loss_of_load_probability")
    curp = _get(metrics, "curtailment_probability")
    pdef = _get(metrics, "max_power_deficit_kw")
    pcur = _get(metrics, "max_curtailment_kw")
    edef = _get(metrics, "expected_energy_deficit_kwh")
    ecur = _get(metrics, "expected_curtailment_energy_kwh")
    crit_ratio = _get(metrics, "critical_load_supply_ratio", 1.0)
    imp_ratio = _get(metrics, "important_load_restoration_ratio", 1.0)
    mobile_reach = _get(metrics, "mobile_reachability", 1.0)
    island_hours = _get(metrics, "island_required_hours", 0.0)

    add("关键负荷刚性保供策略", crit_ratio < th.critical_load_supply_ratio_min or pdef > th.max_power_deficit_kw, 1,
        "关键负荷供电比例不足或最大功率缺口偏高。", "提高关键负荷邻近储能、应急电源和冗余通道配置。")
    add("重要负荷有序保障策略", imp_ratio < th.important_load_restoration_ratio_min or lolp > th.loss_of_load_probability, 2,
        "重要负荷恢复率不足或缺电事件出现频繁。", "按负荷权重、恢复优先级和可用路径组织分阶段恢复。")
    add("一般负荷弹性调节策略", pdef > th.max_power_deficit_kw or edef > th.expected_energy_deficit_kwh or ecur > th.expected_curtailment_energy_kwh, 3,
        "存在短时功率缺口、长时电量缺额或弃电压力。", "建立可削减、可转移、可延后恢复负荷资源池。")
    add("储能快速功率支撑策略", pdef > th.max_power_deficit_kw, 1,
        "电力不足极大值偏高，短时功率支撑不足。", "提高储能功率配置和SOC预留比例。")
    add("长时电量保障策略", edef > th.expected_energy_deficit_kwh, 2,
        "电量不足期望值偏高，长时连续保供压力突出。", "配置长时储能、燃料保障和可持续电量型电源。")
    add("应急发电持续保供策略", edef > th.expected_energy_deficit_kwh or crit_ratio < th.critical_load_supply_ratio_min, 2,
        "长时电量不足或关键负荷连续保供压力高。", "配置固定或移动应急发电设备，明确启动时间和燃料储备。")
    add("移动资源应急支援策略", mobile_reach < th.mobile_reachability_min or imp_ratio < th.important_load_restoration_ratio_min, 3,
        "局部资源空间错配或固定资源难以覆盖薄弱节点。", "配置移动储能、移动发电车和标准化临时接入点。")
    add("源储荷孤岛协同运行策略", island_hours > th.island_hours_min or curp > th.curtailment_probability or ecur > th.expected_curtailment_energy_kwh, 2,
        "存在外部通道受阻、局部孤岛运行需求或新能源消纳压力。", "划定孤岛边界，配置构网型储能、分布式电源和分级负荷控制。")

    return pd.DataFrame(rows).sort_values(["triggered", "priority"], ascending=[False, True]).reset_index(drop=True)


def strategy_summary_text(trigger_df: pd.DataFrame) -> str:
    active = trigger_df[trigger_df["triggered"]]
    if active.empty:
        return "当前指标未触发强制性保供策略，可保持既有配置并开展常规校核。"
    return "\n".join(f"{r.strategy}：{r.reason}建议：{r.config_hint}" for r in active.itertuples())
