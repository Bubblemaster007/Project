"""项目数据自动扫描、读取、字段统一和单位换算。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd


SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls", ".npy", ".mat"}

FIELD_ALIASES: dict[str, list[str]] = {
    "timestamp": ["timestamp", "time", "datetime", "date_time", "时刻", "时间戳", "小时"],
    "simulation_id": [
        "simulation_id", "random_sequence_id", "scenario_id", "monte_carlo_id",
        "run_id", "场景编号", "仿真编号",
    ],
    "event_id": ["event_id", "extreme_event_id", "事件编号", "极端事件编号"],
    "stage": ["disaster_stage", "phase", "event_stage", "stage", "灾害阶段", "阶段"],
    "event_flag": ["is_extreme_condition", "extreme_event_flag", "event_flag", "是否极端工况"],
    "load": ["reachable_load_mw", "reachable_load_kw", "load_mw", "load_kw", "load", "负荷功率"],
    "wind": [
        "wind_available_mw", "available_wind_mw", "available_wind_kw",
        "wind_mw", "wind_kw", "wind_power", "风电实际可用出力",
    ],
    "pv": [
        "pv_available_mw", "available_pv_mw", "available_pv_kw",
        "pv_mw", "pv_kw", "solar_power", "光伏实际可用出力",
    ],
    "storage_charge": ["storage_charge_mw", "storage_charge_kw", "储能充电功率"],
    "storage_discharge": ["storage_discharge_mw", "storage_discharge_kw", "储能放电功率"],
    "emergency": ["emergency_mw", "emergency_gen_mw", "emergency_gen_kw", "应急电源出力"],
    "grid_import_available": [
        "grid_import_available_mw", "available_kw_grid_channel",
        "grid_channel_mw", "外部电网可用受电功率",
    ],
    "grid_dispatch": ["grid_import_mw", "firm_grid_dispatch_mw", "firm_grid_dispatch_kw"],
    "other_generation": ["other_generation_mw", "other_generation_kw", "其他可调电源出力"],
    "export": ["export_mw", "export_kw", "grid_export_mw", "外送功率"],
    "equipment_availability": [
        "equipment_availability", "available_factor", "device_availability", "设备可用系数",
    ],
    "hazard_intensity": [
        "hazard_intensity", "disaster_intensity", "weather_severity", "灾害强度",
    ],
    "power_loss": ["power_loss_mw", "power_deficit_mw", "power_deficit_kw", "电力不足功率"],
    "curtailment": ["curtailment_mw", "curtailment_kw", "弃电功率"],
}


@dataclass
class DataLoadResult:
    """数据读取结果及可审计元数据。"""

    data: pd.DataFrame
    source_files: list[str]
    scanned_files: list[str]
    field_mapping: dict[str, str]
    unit_conversions: list[str]
    missing_optional_fields: list[str]
    data_source: str
    time_step_hours: float
    upstream_balance_reused: bool
    lineage_notes: list[str]


def scan_project_data_files(project_root: Path) -> list[Path]:
    """扫描项目中的 CSV、Excel、NPY 和 MAT 数据文件。"""
    files: list[Path] = []
    excluded_parts = {".git", "__pycache__", "node_modules", ".codex_tmp_lankao"}
    for path in project_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if any(part in excluded_parts for part in path.parts):
            continue
        if "phase_balance_analysis/results" in path.as_posix():
            continue
        files.append(path.resolve())
    return sorted(files)


def _read_table(path: Path) -> pd.DataFrame:
    """读取 CSV 或 Excel 表；NPY/MAT 仅在扫描清单中记录。"""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"当前统一表接口不能直接读取：{path.name}")


def _resolve_config_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def _find_alias(columns: list[str], aliases: list[str]) -> str | None:
    lower_map = {str(col).strip().lower(): str(col) for col in columns}
    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def _power_to_mw(series: pd.Series, source_col: str, configured_unit: str) -> tuple[pd.Series, str]:
    """依据列名或配置把功率统一为 MW。"""
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    unit = configured_unit.lower()
    lower = source_col.lower()
    if unit == "kw" or (unit == "auto" and ("_kw" in lower or "(kw)" in lower)):
        return numeric / 1000.0, f"{source_col}: kW ÷ 1000 → MW"
    if unit == "mw" or (unit == "auto" and ("_mw" in lower or "(mw)" in lower)):
        return numeric, f"{source_col}: MW → MW（无需换算）"
    # 无单位列按数量级保守推断，并在报告中明确记录。
    median_abs = float(numeric.abs().median()) if numeric.notna().any() else 0.0
    if unit == "auto" and median_abs > 1000.0:
        return numeric / 1000.0, f"{source_col}: 无显式单位，按数量级推断为 kW，÷ 1000 → MW"
    return numeric, f"{source_col}: 无显式单位，按 MW 读取"


def _load_or_build_dispatch(
    raw: pd.DataFrame,
    dispatch_path: Path,
    project_root: Path,
    balance_config: dict[str, Any],
) -> tuple[pd.DataFrame | None, bool, list[str]]:
    """优先读取第四章调度结果，缺失时调用项目现有完整平衡模型。"""
    notes: list[str] = []
    if dispatch_path.exists():
        notes.append(
            f"复用第四章约束调度模型生成的逐时结果：{dispatch_path}。"
        )
        return _read_table(dispatch_path), True, notes

    required = {"timestamp", "load_kw"}
    if required.issubset(raw.columns):
        try:
            from src.chapter4.power_energy_balance import analyze_power_energy_balance

            _, dispatch = analyze_power_energy_balance(
                raw,
                balance_config.get("upstream_resource_config", {}),
            )
            notes.append("未找到既有逐时调度文件，已调用 src/chapter4/power_energy_balance.py 完整模型。")
            return dispatch, True, notes
        except Exception as exc:  # pragma: no cover - 只在外部数据异常时触发
            notes.append(f"调用项目第四章完整模型失败：{type(exc).__name__}: {exc}")
    return None, False, notes


def _merge_dispatch(raw: pd.DataFrame, dispatch: pd.DataFrame | None) -> pd.DataFrame:
    if dispatch is None:
        return raw.copy()
    left = raw.copy()
    right = dispatch.copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"])
    right["timestamp"] = pd.to_datetime(right["timestamp"])
    left_sim = _find_alias(list(left.columns), FIELD_ALIASES["simulation_id"])
    right_sim = _find_alias(list(right.columns), FIELD_ALIASES["simulation_id"])
    join_cols = ["timestamp"]
    if left_sim and right_sim:
        if left_sim != "simulation_id":
            left = left.rename(columns={left_sim: "simulation_id"})
        if right_sim != "simulation_id":
            right = right.rename(columns={right_sim: "simulation_id"})
        join_cols.append("simulation_id")
    keep = join_cols.copy()
    for key in [
        "storage_charge", "storage_discharge", "emergency", "grid_dispatch",
        "power_loss", "curtailment",
    ]:
        col = _find_alias(list(right.columns), FIELD_ALIASES[key])
        if col and col not in keep:
            keep.append(col)
    return left.merge(right[keep], on=join_cols, how="left", suffixes=("", "_dispatch"))


def _infer_time_step_hours(data: pd.DataFrame) -> float:
    values: list[float] = []
    for _, group in data.groupby("simulation_id", sort=False):
        delta = group["timestamp"].sort_values().diff().dt.total_seconds().div(3600.0)
        values.extend(delta[(delta > 0) & np.isfinite(delta)].tolist())
    return float(np.median(values)) if values else 1.0


def _lineage_notes(project_root: Path, topology_path: Path) -> list[str]:
    notes: list[str] = []
    if topology_path.exists():
        notes.append(f"兰考拓扑文件：{topology_path}")
    topology_summary = project_root / "outputs/chapter3/topology_summary.json"
    if topology_summary.exists():
        try:
            payload = json.loads(topology_summary.read_text(encoding="utf-8"))
            metadata = payload.get("metadata", {})
            notes.append(
                "上游拓扑派生规模："
                f"{int(metadata.get('node_count', 0))} 个节点、"
                f"{int(metadata.get('line_count', 0))} 条线路、"
                f"峰值负荷 {float(metadata.get('peak_load_kw', 0)) / 1000:.3f} MW。"
            )
        except (OSError, ValueError, TypeError):
            notes.append("存在 topology_summary.json，但未能解析其元数据。")
    return notes


def load_and_standardize(
    package_root: Path,
    config: dict[str, Any],
) -> DataLoadResult:
    """读取项目年度序列并建立统一 MW 数据接口。"""
    data_cfg = config.get("data", {})
    balance_cfg = config.get("balance", {})
    project_root = _resolve_config_path(data_cfg.get("project_root", ".."), package_root)
    scanned = scan_project_data_files(project_root)

    input_path = _resolve_config_path(
        data_cfg.get("input_file", "outputs/chapter3/annual_random_production_sequence.csv"),
        project_root,
    )
    if not input_path.exists():
        ranked = [
            path for path in scanned
            if "annual_random_production_sequence" in path.name.lower()
        ]
        if not ranked:
            raise FileNotFoundError(
                "未找到 8760 h 随机生产模拟序列。请先运行现有 run_full_pipeline.py，"
                "或在配置 data.input_file 中指定项目数据。"
            )
        input_path = ranked[0]

    raw = _read_table(input_path)
    timestamp_col = _find_alias(list(raw.columns), FIELD_ALIASES["timestamp"])
    if timestamp_col is None:
        raise ValueError(f"{input_path.name} 缺少时间戳或小时序号字段")
    if timestamp_col != "timestamp":
        raw = raw.rename(columns={timestamp_col: "timestamp"})
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    if raw["timestamp"].isna().any():
        raise ValueError("时间戳中存在无法解析的值")

    dispatch_path = _resolve_config_path(
        data_cfg.get("dispatch_file", "outputs/chapter4/hourly_balance.csv"),
        project_root,
    )
    dispatch, reused, balance_notes = _load_or_build_dispatch(
        raw, dispatch_path, project_root, balance_cfg
    )
    merged = _merge_dispatch(raw, dispatch)

    mapping: dict[str, str] = {}
    conversions: list[str] = []
    missing: list[str] = []
    out = pd.DataFrame(index=merged.index)
    out["timestamp"] = pd.to_datetime(merged["timestamp"])
    mapping["timestamp"] = "timestamp"

    sim_col = _find_alias(list(merged.columns), FIELD_ALIASES["simulation_id"])
    if sim_col:
        out["simulation_id"] = merged[sim_col].astype(str)
        mapping["simulation_id"] = sim_col
    else:
        out["simulation_id"] = str(data_cfg.get("default_simulation_id", "sim_001"))
        mapping["simulation_id"] = "配置默认值"

    for key in ["event_id", "stage", "event_flag"]:
        col = _find_alias(list(merged.columns), FIELD_ALIASES[key])
        if col:
            out[key] = merged[col]
            mapping[key] = col

    unit = str(data_cfg.get("power_unit", "auto"))
    standardized_names = {
        "load": "load_mw",
        "wind": "wind_available_mw",
        "pv": "pv_available_mw",
        "storage_charge": "storage_charge_mw",
        "storage_discharge": "storage_discharge_mw",
        "emergency": "emergency_mw",
        "grid_import_available": "grid_import_available_mw",
        "grid_dispatch": "grid_dispatch_mw",
        "other_generation": "other_generation_mw",
        "export": "export_mw",
        "power_loss": "power_loss_mw",
        "curtailment": "curtailment_mw",
    }
    required = {"load"}
    for key, target in standardized_names.items():
        col = _find_alias(list(merged.columns), FIELD_ALIASES[key])
        if col:
            out[target], note = _power_to_mw(merged[col], col, unit)
            mapping[target] = col
            conversions.append(note)
        elif key in required:
            raise ValueError(f"输入数据缺少必需字段：{key}")
        else:
            out[target] = np.nan
            missing.append(target)

    availability_col = _find_alias(
        list(merged.columns), FIELD_ALIASES["equipment_availability"]
    )
    if availability_col:
        out["equipment_availability"] = pd.to_numeric(
            merged[availability_col], errors="coerce"
        )
        mapping["equipment_availability"] = availability_col
    else:
        out["equipment_availability"] = np.nan
        missing.append("equipment_availability")

    hazard_col = _find_alias(list(merged.columns), FIELD_ALIASES["hazard_intensity"])
    if hazard_col:
        out["hazard_intensity"] = pd.to_numeric(
            merged[hazard_col], errors="coerce"
        )
        mapping["hazard_intensity"] = hazard_col
    else:
        out["hazard_intensity"] = np.nan
        missing.append("hazard_intensity")

    if bool(data_cfg.get("restrict_to_extreme_event", True)) and "event_flag" in out:
        flag = pd.to_numeric(out["event_flag"], errors="coerce").fillna(0.0)
        if (flag > 0.5).any():
            out = out.loc[flag > 0.5].copy()

    out = out.sort_values(["simulation_id", "timestamp"]).reset_index(drop=True)
    time_step = _infer_time_step_hours(out)
    source_files = [str(input_path)]
    if dispatch_path.exists():
        source_files.append(str(dispatch_path))
    topology_path = _resolve_config_path(
        data_cfg.get("topology_file", "兰考算例数据.xlsx"), project_root
    )
    lineage = balance_notes + _lineage_notes(project_root, topology_path)

    return DataLoadResult(
        data=out,
        source_files=source_files,
        scanned_files=[str(path) for path in scanned],
        field_mapping=mapping,
        unit_conversions=conversions,
        missing_optional_fields=sorted(set(missing)),
        data_source=str(data_cfg.get("data_source_label", "项目数据")),
        time_step_hours=time_step,
        upstream_balance_reused=reused,
        lineage_notes=lineage,
    )
