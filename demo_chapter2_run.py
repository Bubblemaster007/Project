"""
第2章代码演示。

运行：
python demo_chapter2_run.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

from data_schema import Chapter2Config
from chapter2_pipeline import run_chapter2_pipeline


def make_demo_history(year: int = 2025) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    ts = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:00:00", freq="h")
    h = np.arange(len(ts))

    daily = 100 * np.sin(2 * np.pi * (h % 24) / 24 - np.pi / 2)
    seasonal = 80 * np.sin(2 * np.pi * h / len(ts))
    temperature = 18 + 12 * np.sin(2 * np.pi * h / len(ts)) + rng.normal(0, 2, len(ts))

    load_kw = 700 + daily + seasonal + 5 * np.maximum(temperature - 28, 0) + rng.normal(0, 25, len(ts))
    wind_kw = 200 + 70 * np.sin(2 * np.pi * h / 168) + rng.normal(0, 35, len(ts))
    pv_shape = np.maximum(0, np.sin(np.pi * ((h % 24) - 6) / 12))
    pv_kw = 300 * pv_shape * (0.8 + 0.2 * np.sin(2 * np.pi * h / len(ts))) + rng.normal(0, 12, len(ts))

    df = pd.DataFrame({
        "timestamp": ts,
        "load_kw": np.clip(load_kw, 150, None),
        "wind_kw": np.clip(wind_kw, 0, None),
        "pv_kw": np.clip(pv_kw, 0, None),
        "temperature": temperature,
        "wind_speed": np.clip(6 + 2 * np.sin(2 * np.pi * h / 168) + rng.normal(0, 1, len(ts)), 0, None),
        "irradiance": np.clip(pv_shape * 900 + rng.normal(0, 50, len(ts)), 0, None),
    })
    return df


def main():
    hist = make_demo_history()
    out_dir = Path("outputs_ch2")
    paths = run_chapter2_pipeline(hist, out_dir, config=Chapter2Config())

    print("第2章流程运行完成，输出文件：")
    for k, v in paths.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
