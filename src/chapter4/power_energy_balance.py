from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.common.validators import require_columns


@dataclass
class BalanceResourceConfig:
    firm_supply_kw: float = 0.0
    storage_power_kw: float = 0.0
    storage_energy_kwh: float = 0.0
    emergency_gen_kw: float = 0.0
    grid_channel_kw: float = 0.0
    soc_initial_ratio: float = 0.50
    soc_min_ratio: float = 0.10
    soc_max_ratio: float = 0.95
    eta_charge: float = 0.92
    eta_discharge: float = 0.90
    critical_load_ratio: float = 0.20
    important_load_ratio: float = 0.35


def _series_or_default(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def _resource_limit(df: pd.DataFrame, col: str, default: float, is_extreme: pd.Series) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float).clip(lower=0.0)
    raw = _series_or_default(df, col, np.nan)
    out = raw.copy()
    out.loc[~is_extreme & out.isna()] = default
    if out.isna().any():
        raise ValueError(f"{col} has missing capacity during an extreme condition")
    return out.clip(lower=0.0)


def _weighted_groups(sequence_df: pd.DataFrame) -> list[tuple[str, float, pd.DataFrame]]:
    if "random_sequence_id" not in sequence_df.columns:
        sequence_df = sequence_df.copy()
        sequence_df["random_sequence_id"] = "seq_001"
    if "sequence_weight" not in sequence_df.columns:
        sequence_df = sequence_df.copy()
        sequence_df["sequence_weight"] = 1.0
    groups = []
    for seq_id, group in sequence_df.groupby("random_sequence_id", sort=False):
        weight = float(pd.to_numeric(group["sequence_weight"], errors="coerce").dropna().iloc[0])
        groups.append((str(seq_id), weight, group))
    total_weight = sum(weight for _, weight, _ in groups) or 1.0
    return [(seq_id, weight / total_weight, group) for seq_id, weight, group in groups]


def analyze_power_energy_balance(
    sequence_df: pd.DataFrame,
    resource_config: dict[str, Any] | BalanceResourceConfig | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Analyze 8760 h random production sequences and return metrics plus hourly dispatch."""
    cfg = (
        resource_config
        if isinstance(resource_config, BalanceResourceConfig)
        else BalanceResourceConfig(**(resource_config or {}))
    )
    seq = sequence_df.copy()
    require_columns(seq, ["timestamp", "load_kw"], "annual_random_production_sequence")
    seq["timestamp"] = pd.to_datetime(seq["timestamp"])
    if "reachable_load_kw" not in seq.columns:
        seq["reachable_load_kw"] = seq["load_kw"]
    if "available_re_kw" not in seq.columns:
        wind = _series_or_default(seq, "wind_kw", 0.0)
        pv = _series_or_default(seq, "pv_kw", 0.0)
        seq["available_re_kw"] = wind + pv

    hourly_parts = []
    scenario_metrics = []

    for seq_id, weight, group in _weighted_groups(seq):
        group = group.sort_values("timestamp").reset_index(drop=True)
        is_extreme = _series_or_default(group, "is_extreme_condition", 0.0).fillna(0.0) > 0.5
        load = _series_or_default(group, "reachable_load_kw", 0.0).clip(lower=0.0)
        raw_load = _series_or_default(group, "load_kw", 0.0).clip(lower=0.0)
        renewable = _series_or_default(group, "available_re_kw", 0.0).clip(lower=0.0)
        storage_limit = _resource_limit(group, "available_kw_storage", cfg.storage_power_kw, is_extreme)
        emergency_limit = _resource_limit(group, "available_kw_emergency_gen", cfg.emergency_gen_kw, is_extreme)
        grid_limit = _resource_limit(group, "available_kw_grid_channel", cfg.grid_channel_kw, is_extreme)
        line_limit = _series_or_default(group, "available_kw_line", np.inf)
        transformer_limit = _series_or_default(group, "available_kw_transformer", np.inf)
        if line_limit.isna().any() or transformer_limit.isna().any():
            raise ValueError("network capacity contains missing values")
        network_limit = pd.concat([line_limit, transformer_limit], axis=1).min(axis=1).replace(np.inf, cfg.firm_supply_kw + cfg.grid_channel_kw)

        energy_capacity = max(float(cfg.storage_energy_kwh), 0.0)
        soc_min = cfg.soc_min_ratio * energy_capacity
        soc_max = cfg.soc_max_ratio * energy_capacity
        soc = min(max(cfg.soc_initial_ratio * energy_capacity, soc_min), soc_max) if energy_capacity else 0.0

        rows = []
        for i, row in group.iterrows():
            demand = float(load.iloc[i])
            re_kw = float(renewable.iloc[i])
            firm_grid_limit = min(float(cfg.firm_supply_kw) + float(grid_limit.iloc[i]), float(network_limit.iloc[i]))

            renewable_used = min(demand, re_kw)
            remaining = demand - renewable_used
            firm_grid_dispatch = min(remaining, firm_grid_limit)
            remaining -= firm_grid_dispatch
            emergency_dispatch = min(remaining, float(emergency_limit.iloc[i]))
            remaining -= emergency_dispatch

            discharge = 0.0
            charge = 0.0
            if remaining > 0 and energy_capacity > 0:
                energy_available_as_power = max(0.0, soc - soc_min) * cfg.eta_discharge
                discharge = min(remaining, float(storage_limit.iloc[i]), energy_available_as_power)
                soc -= discharge / max(cfg.eta_discharge, 1e-6)
                remaining -= discharge
            elif re_kw > demand and energy_capacity > 0:
                surplus = re_kw - demand
                energy_room_as_power = max(0.0, soc_max - soc) / max(cfg.eta_charge, 1e-6)
                charge = min(surplus, float(storage_limit.iloc[i]), energy_room_as_power)
                soc += charge * cfg.eta_charge

            deficit = max(remaining, 0.0) + max(float(raw_load.iloc[i]) - demand, 0.0)
            curtailment = max(re_kw - demand - charge, 0.0)
            load_served = max(demand - max(remaining, 0.0), 0.0)
            critical_load = float(raw_load.iloc[i]) * cfg.critical_load_ratio
            important_load = float(raw_load.iloc[i]) * cfg.important_load_ratio
            critical_served = min(load_served, critical_load)
            important_served = min(max(load_served - critical_load, 0.0), important_load)

            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "random_sequence_id": seq_id,
                    "sequence_weight": weight,
                    "load_kw": float(raw_load.iloc[i]),
                    "reachable_load_kw": demand,
                    "available_re_kw": re_kw,
                    "renewable_used_kw": renewable_used,
                    "firm_grid_dispatch_kw": firm_grid_dispatch,
                    "emergency_gen_kw": emergency_dispatch,
                    "storage_charge_kw": charge,
                    "storage_discharge_kw": discharge,
                    "power_deficit_kw": deficit,
                    "curtailment_kw": curtailment,
                    "soc": soc / energy_capacity if energy_capacity else 0.0,
                    "load_served_kw": load_served,
                    "critical_load_kw": critical_load,
                    "critical_load_served_kw": critical_served,
                    "important_load_kw": important_load,
                    "important_load_served_kw": important_served,
                    "is_extreme_condition": int(bool(is_extreme.iloc[i])),
                    "load_reachability_factor": float(_series_or_default(group, "load_reachability_factor", 1.0).iloc[i]),
                }
            )

        hourly = pd.DataFrame(rows)
        hourly_parts.append(hourly)
        scenario_metrics.append(
            {
                "weight": weight,
                "loss_of_load_probability": float((hourly["power_deficit_kw"] > 1e-6).mean()),
                "curtailment_probability": float((hourly["curtailment_kw"] > 1e-6).mean()),
                "expected_energy_deficit_kwh": float(hourly["power_deficit_kw"].sum()),
                "expected_curtailment_energy_kwh": float(hourly["curtailment_kw"].sum()),
                "critical_load_supply_ratio": _safe_ratio(hourly["critical_load_served_kw"].sum(), hourly["critical_load_kw"].sum()),
                "important_load_restoration_ratio": _safe_ratio(hourly["important_load_served_kw"].sum(), hourly["important_load_kw"].sum()),
                "mobile_reachability": float(hourly["load_reachability_factor"].mean()),
                "island_required_hours": float(((hourly["is_extreme_condition"] == 1) & (group["available_kw_grid_channel"].fillna(0.0).to_numpy() <= 1e-6)).sum()) if "available_kw_grid_channel" in group.columns else 0.0,
            }
        )

    hourly_balance = pd.concat(hourly_parts, ignore_index=True)
    metrics = {
        "loss_of_load_probability": _weighted_sum(scenario_metrics, "loss_of_load_probability"),
        "curtailment_probability": _weighted_sum(scenario_metrics, "curtailment_probability"),
        "max_power_deficit_kw": float(hourly_balance["power_deficit_kw"].max()),
        "max_curtailment_kw": float(hourly_balance["curtailment_kw"].max()),
        "expected_energy_deficit_kwh": _weighted_sum(scenario_metrics, "expected_energy_deficit_kwh"),
        "expected_curtailment_energy_kwh": _weighted_sum(scenario_metrics, "expected_curtailment_energy_kwh"),
        "critical_load_supply_ratio": _weighted_sum(scenario_metrics, "critical_load_supply_ratio"),
        "important_load_restoration_ratio": _weighted_sum(scenario_metrics, "important_load_restoration_ratio"),
        "mobile_reachability": _weighted_sum(scenario_metrics, "mobile_reachability"),
        "island_required_hours": _weighted_sum(scenario_metrics, "island_required_hours"),
    }
    metrics["balance_probability"] = 1.0 - metrics["loss_of_load_probability"]
    return metrics, hourly_balance


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 1.0


def _weighted_sum(rows: list[dict[str, float]], key: str) -> float:
    return float(sum(row["weight"] * row[key] for row in rows))
