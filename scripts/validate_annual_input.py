"""Validate the annual source/load/weather contract before downstream dispatch."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


REQUIRED = ("timestamp", "load_kw", "wind_kw", "pv_kw", "event_id", "wind_speed")
RESOURCES = ("grid_channel", "transformer", "emergency_gen", "storage", "renewable")


def validate_annual_input(path: str | Path, require_8760: bool = True) -> dict:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if frame.empty:
        raise ValueError("annual input is empty")
    if frame.timestamp.isna().any() or frame.timestamp.duplicated().any():
        raise ValueError("timestamp contains NaN or duplicates")
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    if not ordered.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("timestamp must be continuous hourly data")
    if require_8760 and len(ordered) != 8760:
        raise ValueError(f"expected 8760 hourly rows, got {len(ordered)}")
    numeric = ["load_kw", "wind_kw", "pv_kw", "wind_speed"]
    for col in numeric:
        values = pd.to_numeric(ordered[col], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(f"{col} contains non-numeric or non-finite values")
        if col != "wind_speed" and (values < 0).any():
            raise ValueError(f"{col} contains negative values")
    event = pd.to_numeric(ordered.event_id, errors="coerce")
    if event.isna().any() or not np.isfinite(event).all():
        raise ValueError("event_id contains invalid values")
    for resource in RESOURCES:
        rate = f"failure_rate_per_hour_{resource}"
        repair = f"repair_hours_{resource}"
        if (rate in ordered) != (repair in ordered):
            raise ValueError(f"{rate} and {repair} must be supplied together")
        if rate in ordered:
            r = pd.to_numeric(ordered[rate], errors="coerce")
            d = pd.to_numeric(ordered[repair], errors="coerce")
            if r.isna().any() or d.isna().any() or (r < 0).any() or (d < 1).any() or (d != d.round()).any():
                raise ValueError(f"invalid failure/repair values for {resource}")
    for col in [c for c in ordered.columns if c.startswith("available_kw_")]:
        values = pd.to_numeric(ordered[col], errors="coerce")
        if values.isna().any() or (values < 0).any() or not np.isfinite(values).all():
            raise ValueError(f"invalid availability values in {col}")
    extreme = event.ge(0)
    return {"path": str(path), "rows": len(ordered), "start": str(ordered.timestamp.iloc[0]),
            "end": str(ordered.timestamp.iloc[-1]), "event_hours": int(extreme.sum()),
            "event_count": int(event[extreme].nunique()),
            "failure_rate_resources": [r for r in RESOURCES if f"failure_rate_per_hour_{r}" in ordered]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--allow-non-8760", action="store_true")
    args = parser.parse_args()
    print(validate_annual_input(args.csv, require_8760=not args.allow_non_8760))


if __name__ == "__main__":
    main()
