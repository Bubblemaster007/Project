from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_lankao_topology(excel_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Read the Lankao workbook and return cleaned line/node tables."""
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到兰考算例数据：{path}")

    lines = pd.read_excel(path, sheet_name="线路")
    raw_nodes = pd.read_excel(path, sheet_name="节点", header=None)
    nodes = pd.read_excel(path, sheet_name="节点", header=2)

    for col in ["支路", "起点", "终点", "线路容量(kW)", "长度(km)"]:
        if col in lines.columns:
            lines[col] = pd.to_numeric(lines[col], errors="coerce")
    for col in ["节点", "有功", "无功", "馈线容量"]:
        if col in nodes.columns:
            nodes[col] = pd.to_numeric(nodes[col], errors="coerce")

    transformer_loads = pd.to_numeric(raw_nodes.iloc[:2, 1], errors="coerce").dropna()
    total_transformer_load_kw = float(transformer_loads.sum()) if not transformer_loads.empty else 0.0
    total_node_load_kw = float(nodes.get("有功", pd.Series(dtype=float)).fillna(0.0).sum())
    peak_load_kw = total_transformer_load_kw or total_node_load_kw or 42000.0
    metadata = {
        "total_transformer_load_kw": total_transformer_load_kw,
        "total_node_load_kw": total_node_load_kw,
        "peak_load_kw": peak_load_kw,
        "line_count": float(len(lines)),
        "node_count": float(nodes["节点"].notna().sum()) if "节点" in nodes.columns else float(len(nodes)),
        "total_line_capacity_kw": float(lines.get("线路容量(kW)", pd.Series(dtype=float)).fillna(0.0).sum()),
    }
    return lines, nodes, metadata


def _line_fragility(length_km: float) -> dict[str, float]:
    length = max(float(length_km or 0.0), 0.1)
    return {
        "a_wind": min(0.45, 0.08 + 0.018 * length),
        "b_wind": max(15.0, 23.0 - 0.25 * length),
        "a_icing": min(1.8, 0.75 + 0.045 * length),
        "b_icing": 0.62,
        "a_rain": min(0.13, 0.035 + 0.004 * length),
        "b_rain": max(15.0, 24.0 - 0.15 * length),
        "a_heat": 0.055,
        "b_heat": 39.0,
    }


def build_device_params_from_lankao(excel_path: str | Path, config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Build fault-model device parameters from the Lankao topology workbook."""
    cfg = config or {}
    lines, nodes, metadata = read_lankao_topology(excel_path)
    peak_load_kw = metadata["peak_load_kw"]
    devices: list[dict[str, Any]] = []

    for _, row in lines.dropna(subset=["支路", "起点", "终点"]).iterrows():
        branch_id = int(row["支路"])
        rated_kw = float(row.get("线路容量(kW)", 0.0) or 0.0)
        if rated_kw <= 0:
            rated_kw = max(1000.0, peak_load_kw / max(len(lines), 1))
        fragility = _line_fragility(float(row.get("长度(km)", 0.0) or 0.0))
        devices.append(
            {
                "device_id": f"line_{branch_id:03d}",
                "device_type": "line",
                "from_node": int(row["起点"]),
                "to_node": int(row["终点"]),
                "rated_kw": rated_kw,
                **fragility,
                "normal_factor": 1.0,
                "derate_factor": 0.55,
                "recovery_factor": 0.80,
            }
        )

    transformer_capacity = max(peak_load_kw * 0.65, 1000.0)
    devices.extend(
        [
            {
                "device_id": "transformer_leiji",
                "device_type": "transformer",
                "rated_kw": transformer_capacity,
                "a_wind": 0.08,
                "b_wind": 25.0,
                "a_icing": 0.85,
                "b_icing": 0.78,
                "a_rain": 0.08,
                "b_rain": 22.0,
                "a_heat": 0.12,
                "b_heat": 38.0,
                "normal_factor": 1.0,
                "derate_factor": 0.65,
                "recovery_factor": 0.85,
            },
            {
                "device_id": "transformer_zhuaying",
                "device_type": "transformer",
                "rated_kw": transformer_capacity,
                "a_wind": 0.08,
                "b_wind": 25.0,
                "a_icing": 0.85,
                "b_icing": 0.78,
                "a_rain": 0.08,
                "b_rain": 22.0,
                "a_heat": 0.12,
                "b_heat": 38.0,
                "normal_factor": 1.0,
                "derate_factor": 0.65,
                "recovery_factor": 0.85,
            },
        ]
    )

    scale_rows = [
        ("storage_lankao", "storage", float(cfg.get("storage_power_scale", 0.08)), 0.04, 30.0, 0.50, 0.85, 0.06, 30.0, 0.18, 38.0, 0.70, 0.90),
        ("emergency_gen_lankao", "emergency_gen", float(cfg.get("emergency_gen_scale", 0.10)), 0.07, 28.0, 0.45, 0.85, 0.05, 26.0, 0.16, 40.0, 0.75, 0.90),
        ("grid_channel_lankao", "grid_channel", float(cfg.get("grid_channel_scale", 0.25)), 0.22, 22.0, 1.10, 0.62, 0.08, 18.0, 0.08, 40.0, 0.40, 0.75),
        ("renewable_collection_lankao", "renewable", float(cfg.get("renewable_capacity_scale", 0.35)), 0.16, 24.0, 0.95, 0.70, 0.06, 20.0, 0.12, 38.0, 0.55, 0.80),
    ]
    for device_id, device_type, scale, a_wind, b_wind, a_icing, b_icing, a_rain, b_rain, a_heat, b_heat, derate, recovery in scale_rows:
        devices.append(
            {
                "device_id": device_id,
                "device_type": device_type,
                "rated_kw": max(500.0, peak_load_kw * scale),
                "a_wind": a_wind,
                "b_wind": b_wind,
                "a_icing": a_icing,
                "b_icing": b_icing,
                "a_rain": a_rain,
                "b_rain": b_rain,
                "a_heat": a_heat,
                "b_heat": b_heat,
                "normal_factor": 1.0,
                "derate_factor": derate,
                "recovery_factor": recovery,
            }
        )

    device_params = pd.DataFrame(devices)
    return device_params, lines, nodes, metadata


def build_device_params_from_case33(data_dir: str | Path, config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Build the same device tables from the downloaded IEEE/Baran-Wu 33-bus CSVs."""
    root = Path(data_dir)
    lines = pd.read_csv(root / "lines.csv")
    nodes = pd.read_csv(root / "nodes.csv")
    peak_load_kw = float(nodes["pd_kw"].sum())
    devices = []
    for _, row in lines.iterrows():
        devices.append({"device_id": f"line_{int(row.line):03d}", "device_type": "line",
                        "from_node": int(row.from_node), "to_node": int(row.to_node),
                        "rated_kw": max(500.0, peak_load_kw / 8.0), **_line_fragility(1.0),
                        "normal_factor": 1.0, "derate_factor": 0.55, "recovery_factor": 0.80})
    for device_id, device_type, scale in [("transformer_case33", "transformer", .65),
                                          ("storage_case33", "storage", .08),
                                          ("emergency_gen_case33", "emergency_gen", .10),
                                          ("grid_channel_case33", "grid_channel", .25),
                                          ("renewable_collection_case33", "renewable", .35)]:
        devices.append({"device_id": device_id, "device_type": device_type,
                        "rated_kw": max(500.0, peak_load_kw * scale),
                        **_line_fragility(1.0), "normal_factor": 1.0,
                        "derate_factor": .65 if device_type == "transformer" else .55,
                        "recovery_factor": .85})
    metadata = {"peak_load_kw": peak_load_kw, "node_count": float(len(nodes)),
                "line_count": float(len(lines)), "source": "MATPOWER case33bw / Baran-Wu"}
    return pd.DataFrame(devices), lines, nodes, metadata


def build_load_reachability(hazard_df: pd.DataFrame) -> pd.DataFrame:
    """Create a lightweight load reachability sequence from hazard intensity."""
    hazard = hazard_df.copy()
    hazard["timestamp"] = pd.to_datetime(hazard["timestamp"])
    wind = pd.to_numeric(hazard.get("wind_speed", 0.0), errors="coerce").fillna(0.0)
    rain = pd.to_numeric(hazard.get("rain", hazard.get("rainfall", 0.0)), errors="coerce").fillna(0.0)
    icing = pd.to_numeric(hazard.get("icing", 0.0), errors="coerce").fillna(0.0)
    heat = pd.to_numeric(hazard.get("heat", hazard.get("temperature", 25.0)), errors="coerce").fillna(25.0)

    factor = (
        1.0
        - 0.12 * np.clip((wind - 18.0) / 12.0, 0.0, 1.0)
        - 0.10 * np.clip((rain - 15.0) / 25.0, 0.0, 1.0)
        - 0.12 * np.clip((icing - 0.45) / 0.55, 0.0, 1.0)
        - 0.04 * np.clip((heat - 36.0) / 8.0, 0.0, 1.0)
    )
    return pd.DataFrame({"timestamp": hazard["timestamp"], "load_reachability_factor": np.clip(factor, 0.70, 1.0)})
