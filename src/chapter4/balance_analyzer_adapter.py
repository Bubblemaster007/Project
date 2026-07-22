from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.common.io_utils import write_csv, write_json
from .power_energy_balance import analyze_power_energy_balance


def run_balance_analysis(
    sequence_df_or_path: pd.DataFrame | str | Path,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, float], pd.DataFrame, dict[str, str]]:
    if isinstance(sequence_df_or_path, pd.DataFrame):
        sequence_df = sequence_df_or_path
    else:
        sequence_df = pd.read_csv(sequence_df_or_path, encoding="utf-8-sig")
    metrics, hourly_balance = analyze_power_energy_balance(sequence_df, config or {})
    out_dir = Path(output_dir)
    metrics_path = write_json(out_dir / "metrics.json", metrics)
    hourly_path = write_csv(hourly_balance, out_dir / "hourly_balance.csv")
    return metrics, hourly_balance, {"metrics": str(metrics_path), "hourly_balance": str(hourly_path)}
