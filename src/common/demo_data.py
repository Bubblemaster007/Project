from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io_utils import ensure_dir, write_csv


def estimate_lankao_peak_load_kw(excel_path: str | Path) -> float:
    path = Path(excel_path)
    if not path.exists():
        return 42000.0
    raw = pd.read_excel(path, sheet_name="节点", header=None)
    transformer_load = pd.to_numeric(raw.iloc[:2, 1], errors="coerce").dropna()
    if not transformer_load.empty and float(transformer_load.sum()) > 0:
        return float(transformer_load.sum())
    nodes = pd.read_excel(path, sheet_name="节点", header=2)
    if "有功" in nodes.columns:
        active = pd.to_numeric(nodes["有功"], errors="coerce").fillna(0.0)
        if float(active.sum()) > 0:
            return float(active.sum())
    return 42000.0


def make_demo_history(
    year: int = 2025,
    peak_load_kw: float = 42000.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    ts = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:00:00", freq="h")
    h = np.arange(len(ts))

    daily = 0.12 * np.sin(2 * np.pi * (h % 24) / 24 - np.pi / 2)
    evening = 0.06 * np.sin(2 * np.pi * ((h % 24) - 17) / 24)
    seasonal = 0.10 * np.sin(2 * np.pi * (h - 1200) / len(ts))
    temperature = 18 + 13 * np.sin(2 * np.pi * (h - 1200) / len(ts)) + rng.normal(0, 2.5, len(ts))

    base_load = 0.78 * peak_load_kw
    heat_load = 0.008 * peak_load_kw * np.maximum(temperature - 31, 0)
    load_kw = base_load * (1 + daily + evening + seasonal) + heat_load + rng.normal(0, 0.025 * peak_load_kw, len(ts))

    wind_capacity = 0.28 * peak_load_kw
    pv_capacity = 0.34 * peak_load_kw
    wind_kw = wind_capacity * (0.42 + 0.18 * np.sin(2 * np.pi * h / 168) + rng.normal(0, 0.09, len(ts)))
    pv_shape = np.maximum(0, np.sin(np.pi * ((h % 24) - 6) / 12))
    pv_kw = pv_capacity * pv_shape * (0.78 + 0.18 * np.sin(2 * np.pi * (h - 1000) / len(ts))) + rng.normal(0, 0.015 * peak_load_kw, len(ts))

    wind_speed = np.clip(5.5 + 2.8 * np.sin(2 * np.pi * h / 168) + rng.normal(0, 1.1, len(ts)), 0, None)
    irradiance = np.clip(pv_shape * 900 + rng.normal(0, 60, len(ts)), 0, None)
    rainfall = np.clip(rng.gamma(shape=0.8, scale=1.8, size=len(ts)) - 1.1, 0, None)
    icing = np.clip((2 - temperature) / 8 + rng.normal(0, 0.04, len(ts)), 0, 1)
    dust = np.clip(rng.normal(0.15, 0.08, len(ts)) + (wind_speed > 9) * 0.25, 0, 1)

    return pd.DataFrame(
        {
            "timestamp": ts,
            "load_kw": np.clip(load_kw, 0.35 * peak_load_kw, None),
            "wind_kw": np.clip(wind_kw, 0, None),
            "pv_kw": np.clip(pv_kw, 0, None),
            "temperature": temperature,
            "wind_speed": wind_speed,
            "irradiance": irradiance,
            "rainfall": rainfall,
            "icing": icing,
            "dust": dust,
        }
    )


def save_demo_history(output_dir: str | Path, excel_path: str | Path, year: int, random_seed: int) -> Path:
    out_dir = ensure_dir(output_dir)
    peak_load_kw = estimate_lankao_peak_load_kw(excel_path)
    history = make_demo_history(year=year, peak_load_kw=peak_load_kw, random_seed=random_seed)
    return write_csv(history, out_dir / "historical_source_load.csv")
