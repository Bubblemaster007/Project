from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.validators import require_columns


def _normalize_existing_extreme(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    rename = {
        "time": "timestamp",
        "load": "load_kw",
        "wind_power": "wind_kw",
        "solar_power": "pv_kw",
        "solar_kw": "pv_kw",
        "extreme_prob": "extreme_probability",
        "extreme_degree": "extreme_probability",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    require_columns(df, ["timestamp", "load_kw", "wind_kw", "pv_kw"], "extreme_36h")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "event_core" not in df.columns:
        df["event_core"] = 0
        if len(df) >= 12:
            df.loc[len(df) // 4 : len(df) * 3 // 4, "event_core"] = 1
    if "event_type" not in df.columns:
        df["event_type"] = "external_extreme_scene"
    if "resource_state" not in df.columns:
        df["resource_state"] = "external"
    if "extreme_probability" not in df.columns:
        df["extreme_probability"] = 1.0
    return df


def _longest_run(values: np.ndarray) -> int:
    run = 0
    best = 0
    for value in values:
        run = run + 1 if bool(value) else 0
        best = max(best, run)
    return best


def build_extreme_scene(
    source_load_features: pd.DataFrame,
    extreme_window_candidates: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return a unified 36 h extreme scene and matching hazard sequence."""
    cfg = config or {}
    external_output_file = str(cfg.get("external_output_file", "") or "")
    if external_output_file and Path(external_output_file).exists():
        scene = _normalize_existing_extreme(external_output_file)
        hazard = build_hazard_from_scene(scene, random_seed=int(cfg.get("random_seed", 42)))
        return scene, hazard, {"source": "external_output_file", "path": external_output_file}

    features = source_load_features.copy()
    features["timestamp"] = pd.to_datetime(features["timestamp"])
    features = features.sort_values("timestamp").reset_index(drop=True)
    require_columns(features, ["timestamp", "load_kw", "wind_kw", "pv_kw", "net_load_kw"], "source_load_features")

    window_hours = int(cfg.get("window_hours", 36))
    if extreme_window_candidates.empty:
        center_idx = int(features["net_load_kw"].idxmax())
        start_idx = max(0, min(center_idx - window_hours // 2, len(features) - window_hours))
        candidate = {}
    else:
        candidate = extreme_window_candidates.iloc[0].to_dict()
        start_time = pd.to_datetime(candidate.get("window_start"))
        start_matches = features.index[features["timestamp"] == start_time].tolist()
        start_idx = start_matches[0] if start_matches else int((features["timestamp"] - start_time).abs().idxmin())
        start_idx = max(0, min(start_idx, len(features) - window_hours))

    window = features.iloc[start_idx : start_idx + window_hours].copy()
    if len(window) < window_hours:
        raise ValueError("源荷特征长度不足，无法构造36 h极端场景")

    net_load = window["net_load_kw"].astype(float)
    threshold = float(net_load.quantile(0.70))
    event_core = (net_load >= threshold).astype(int).to_numpy()
    if event_core.sum() == 0:
        event_core[window_hours // 3 : window_hours * 2 // 3] = 1

    scene = window[["timestamp", "load_kw", "wind_kw", "pv_kw"]].copy()
    scene["event_core"] = event_core
    scene["event_type"] = _infer_event_type(window)
    scene["resource_state"] = np.where(scene["event_core"] == 1, "low_re_high_load", "buffer")
    pressure = np.clip(net_load.to_numpy(), 0, None)
    scene["cum_deficit_kwh"] = float(pressure.sum())
    scene["core_cum_deficit_kwh"] = float((pressure * event_core).sum())
    scene["net_load_3h_ramp_kw"] = float(np.abs(net_load.diff(3).fillna(0.0)).max())
    imbalance = net_load.to_numpy() > threshold
    scene["imbalance_duration_h"] = int(imbalance.sum())
    scene["max_imbalance_run_h"] = _longest_run(imbalance)
    score = float(candidate.get("extreme_window_score", pressure.sum()))
    scene["extreme_probability"] = min(0.999, max(0.001, score / (score + pressure.mean() * window_hours + 1e-6)))
    hazard = build_hazard_from_scene(window, random_seed=int(cfg.get("random_seed", 42)))
    return scene, hazard, {"source": "source_load_window_adapter", "candidate": candidate}


def _infer_event_type(window: pd.DataFrame) -> str:
    temp = pd.to_numeric(window.get("temperature", pd.Series([25.0])), errors="coerce").mean()
    wind_speed = pd.to_numeric(window.get("wind_speed", pd.Series([0.0])), errors="coerce").max()
    rainfall = pd.to_numeric(window.get("rainfall", pd.Series([0.0])), errors="coerce").max()
    if temp >= 34:
        return "高温"
    if wind_speed >= 12:
        return "大风/沙尘暴"
    if rainfall >= 8:
        return "暴雨/强降水"
    return "源荷错配"


def build_hazard_from_scene(scene_df: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    scene = scene_df.copy()
    scene["timestamp"] = pd.to_datetime(scene["timestamp"])
    n = len(scene)
    core = pd.to_numeric(scene.get("event_core", pd.Series(np.zeros(n))), errors="coerce").fillna(0).to_numpy()
    x = np.arange(n)
    pulse = np.exp(-((x - n * 0.55) / max(n * 0.18, 1.0)) ** 2)

    wind_speed = pd.to_numeric(scene.get("wind_speed", pd.Series(np.nan, index=scene.index)), errors="coerce")
    if wind_speed.isna().all():
        wind_speed = pd.Series(7.0 + 12.0 * pulse + 4.0 * core + rng.normal(0, 0.8, n), index=scene.index)

    rain = pd.to_numeric(scene.get("rainfall", pd.Series(np.nan, index=scene.index)), errors="coerce")
    if rain.isna().all():
        rain = pd.Series(np.clip(2.0 + 18.0 * pulse + rng.normal(0, 1.2, n), 0, None), index=scene.index)

    heat = pd.to_numeric(scene.get("temperature", pd.Series(np.nan, index=scene.index)), errors="coerce")
    if heat.isna().all():
        heat = pd.Series(30.0 + 7.0 * pulse + rng.normal(0, 0.6, n), index=scene.index)

    icing = pd.to_numeric(scene.get("icing", pd.Series(np.nan, index=scene.index)), errors="coerce")
    if icing.isna().all():
        icing = pd.Series(np.clip(0.08 + 0.55 * pulse + rng.normal(0, 0.03, n), 0, 1), index=scene.index)

    dust = pd.to_numeric(scene.get("dust", pd.Series(np.nan, index=scene.index)), errors="coerce")
    if dust.isna().all():
        dust = pd.Series(np.clip(0.10 + 0.35 * (wind_speed.to_numpy() > 12) + rng.normal(0, 0.04, n), 0, 1), index=scene.index)

    return pd.DataFrame(
        {
            "timestamp": scene["timestamp"],
            "wind_speed": wind_speed.astype(float).clip(lower=0),
            "icing": icing.astype(float).clip(lower=0, upper=1),
            "rain": rain.astype(float).clip(lower=0),
            "heat": heat.astype(float),
            "dust": dust.astype(float).clip(lower=0, upper=1),
            "temperature": heat.astype(float),
        }
    )
