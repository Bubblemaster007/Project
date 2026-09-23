"""逐时电力电量平衡计算与物理一致性检查。"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


SUPPLY_COLUMNS = [
    "wind_available_mw",
    "pv_available_mw",
    "grid_dispatch_mw",
    "storage_discharge_mw",
    "emergency_mw",
    "other_generation_mw",
]


def _numeric_or_zero(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data.columns:
        return pd.Series(0.0, index=data.index)
    return pd.to_numeric(data[column], errors="coerce").fillna(0.0)


def calculate_hourly_balance(
    data: pd.DataFrame,
    epsilon_mw: float = 1e-6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """复用既有缺额/弃电结果；缺失时按统一供需表达式计算。"""
    out = data.copy()
    precomputed = (
        "power_loss_mw" in out
        and "curtailment_mw" in out
        and out["power_loss_mw"].notna().all()
        and out["curtailment_mw"].notna().all()
    )

    if precomputed:
        out["power_loss_mw"] = pd.to_numeric(
            out["power_loss_mw"], errors="coerce"
        ).clip(lower=0.0)
        out["curtailment_mw"] = pd.to_numeric(
            out["curtailment_mw"], errors="coerce"
        ).clip(lower=0.0)
        out["net_supply_mw"] = (
            pd.to_numeric(out["load_mw"], errors="coerce")
            - out["power_loss_mw"]
            + out["curtailment_mw"]
        )
        method = "reused_upstream_constrained_dispatch"
    else:
        supply = sum((_numeric_or_zero(out, col) for col in SUPPLY_COLUMNS), start=0)
        net_supply = (
            supply
            - _numeric_or_zero(out, "storage_charge_mw")
            - _numeric_or_zero(out, "export_mw")
        )
        load = pd.to_numeric(out["load_mw"], errors="coerce")
        if load.isna().any():
            raise ValueError("负荷数据存在缺失，不能静默填补")
        out["net_supply_mw"] = net_supply
        out["power_loss_mw"] = (load - net_supply).clip(lower=0.0)
        out["curtailment_mw"] = (net_supply - load).clip(lower=0.0)
        method = "unified_supply_demand_equation"

    simultaneous = (
        (out["power_loss_mw"] > epsilon_mw)
        & (out["curtailment_mw"] > epsilon_mw)
    )
    diagnostics = {
        "method": method,
        "row_count": int(len(out)),
        "negative_loss_count": int((out["power_loss_mw"] < 0).sum()),
        "negative_curtailment_count": int((out["curtailment_mw"] < 0).sum()),
        "simultaneous_shortage_curtailment_count": int(simultaneous.sum()),
        "epsilon_mw": float(epsilon_mw),
    }
    if diagnostics["negative_loss_count"] or diagnostics["negative_curtailment_count"]:
        raise AssertionError("逐时缺额或弃电出现负值")
    return out, diagnostics
