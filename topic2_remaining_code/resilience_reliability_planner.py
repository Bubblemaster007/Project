"""
韧性-可靠性协同规划简化模块。

根据策略触发结果、六项指标和成本参数，形成储能功率/容量、应急发电容量、
移动资源数量、SOC预留比例等阶段性配置建议。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import math
import pandas as pd


@dataclass
class PlanningConfig:
    storage_power_unit_kw: float = 100.0
    storage_energy_unit_kwh: float = 200.0
    emergency_gen_unit_kw: float = 100.0
    mobile_unit_kw: float = 100.0
    max_storage_units: int = 10
    max_emergency_units: int = 10
    max_mobile_units: int = 5
    storage_power_cost: float = 600.0
    storage_energy_cost: float = 900.0
    emergency_gen_cost: float = 450.0
    mobile_resource_cost: float = 200000.0
    penalty_energy_deficit: float = 20.0
    penalty_power_deficit: float = 100.0
    penalty_curtailment: float = 2.0
    default_soc_reserve_ratio: float = 0.35


def estimate_required_resources(metrics: Dict[str, float], config: PlanningConfig) -> Dict[str, float]:
    p_def = float(metrics.get("max_power_deficit_kw", 0.0))
    e_def = float(metrics.get("expected_energy_deficit_kwh", 0.0))
    e_cur = float(metrics.get("expected_curtailment_energy_kwh", 0.0))
    return {
        "required_storage_power_kw": max(0.0, 0.7 * p_def),
        "required_storage_energy_kwh": max(0.0, 0.5 * e_def + 0.3 * e_cur),
        "required_emergency_kw": max(0.0, 0.4 * p_def),
        "required_mobile_units": 1 if metrics.get("mobile_reachability", 1.0) < 0.6 else 0,
    }


def evaluate_plan(metrics: Dict[str, float], plan: Dict[str, float], config: PlanningConfig) -> Dict[str, float]:
    p_def = float(metrics.get("max_power_deficit_kw", 0.0))
    e_def = float(metrics.get("expected_energy_deficit_kwh", 0.0))
    e_cur = float(metrics.get("expected_curtailment_energy_kwh", 0.0))
    storage_p = plan["storage_power_kw"]
    storage_e = plan["storage_energy_kwh"]
    emergency_p = plan["emergency_gen_kw"]
    mobile_p = plan["mobile_units"] * config.mobile_unit_kw

    residual_p_def = max(0.0, p_def - 0.85 * storage_p - 0.65 * emergency_p - 0.50 * mobile_p)
    residual_e_def = max(0.0, e_def - 0.80 * storage_e - 4.0 * 0.70 * emergency_p)
    residual_e_cur = max(0.0, e_cur - 0.50 * storage_e)

    investment_cost = (
        storage_p * config.storage_power_cost
        + storage_e * config.storage_energy_cost
        + emergency_p * config.emergency_gen_cost
        + plan["mobile_units"] * config.mobile_resource_cost
    )
    risk_cost = residual_p_def * config.penalty_power_deficit + residual_e_def * config.penalty_energy_deficit + residual_e_cur * config.penalty_curtailment
    return {
        "residual_max_power_deficit_kw": residual_p_def,
        "residual_expected_energy_deficit_kwh": residual_e_def,
        "residual_expected_curtailment_energy_kwh": residual_e_cur,
        "investment_cost": investment_cost,
        "risk_cost": risk_cost,
        "total_cost": investment_cost + risk_cost,
    }


def run_simple_planning(metrics: Dict[str, float], trigger_df: pd.DataFrame, config: Optional[PlanningConfig] = None) -> Dict[str, object]:
    config = config or PlanningConfig()
    active_strategies = trigger_df.loc[trigger_df["triggered"], "strategy"].tolist()
    demand = estimate_required_resources(metrics, config)
    min_spu = min(math.floor(demand["required_storage_power_kw"] / config.storage_power_unit_kw), config.max_storage_units)
    min_seu = min(math.floor(demand["required_storage_energy_kwh"] / config.storage_energy_unit_kwh), config.max_storage_units)
    min_egu = min(math.floor(demand["required_emergency_kw"] / config.emergency_gen_unit_kw), config.max_emergency_units)
    min_mu = min(int(demand["required_mobile_units"]), config.max_mobile_units)

    best = None
    count = 0
    for spu in range(min_spu, config.max_storage_units + 1):
        for seu in range(min_seu, config.max_storage_units + 1):
            for egu in range(min_egu, config.max_emergency_units + 1):
                for mu in range(min_mu, config.max_mobile_units + 1):
                    plan = {
                        "storage_power_kw": spu * config.storage_power_unit_kw,
                        "storage_energy_kwh": seu * config.storage_energy_unit_kwh,
                        "emergency_gen_kw": egu * config.emergency_gen_unit_kw,
                        "mobile_units": mu,
                    }
                    score = evaluate_plan(metrics, plan, config)
                    result = {**plan, **score}
                    count += 1
                    if best is None or result["total_cost"] < best["total_cost"]:
                        best = result
    if best is None:
        raise RuntimeError("未找到可行配置方案")

    crit_ratio = float(metrics.get("critical_load_supply_ratio", 1.0))
    if crit_ratio < 0.95:
        soc_reserve = max(config.default_soc_reserve_ratio, 0.50)
    elif float(metrics.get("expected_energy_deficit_kwh", 0.0)) > 0:
        soc_reserve = max(config.default_soc_reserve_ratio, 0.40)
    else:
        soc_reserve = config.default_soc_reserve_ratio

    return {
        "active_strategies": active_strategies,
        "recommended_plan": {
            "storage_power_kw": best["storage_power_kw"],
            "storage_energy_kwh": best["storage_energy_kwh"],
            "emergency_gen_kw": best["emergency_gen_kw"],
            "mobile_units": int(best["mobile_units"]),
            "soc_reserve_ratio": soc_reserve,
        },
        "evaluation": {k: best[k] for k in ["residual_max_power_deficit_kw", "residual_expected_energy_deficit_kwh", "residual_expected_curtailment_energy_kwh", "investment_cost", "risk_cost", "total_cost"]},
        "candidate_count": count,
    }


def planning_result_to_dataframe(result: Dict[str, object]) -> pd.DataFrame:
    row = {**result["recommended_plan"], **result["evaluation"]}
    row["active_strategies"] = "；".join(result["active_strategies"])
    return pd.DataFrame([row])
