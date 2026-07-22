from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat


CHANNELS = ["load", "wind_power", "solar_power", "net_load"]
LABELS = {
    "load": "Load",
    "wind_power": "Wind",
    "solar_power": "Solar",
    "net_load": "Net load",
}
COLORS = {
    "load": "#1f77b4",
    "wind_power": "#2ca02c",
    "solar_power": "#ffb000",
    "net_load": "#d62728",
}


def _find_file(root: Path, file_name: str) -> Path:
    for path in root.rglob(file_name):
        if ".venv" not in path.parts:
            return path
    raise FileNotFoundError(f"Cannot find {file_name} under {root}")


def _as_year_day_hour(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr.shape}")
    hour_axes = [idx for idx, size in enumerate(arr.shape) if size == 24]
    day_axes = [idx for idx, size in enumerate(arr.shape) if size == 365]
    if not hour_axes or not day_axes:
        raise ValueError(f"Cannot infer hour/day axes from shape {arr.shape}")
    hour_axis = hour_axes[0]
    day_axis = next((idx for idx in day_axes if idx != hour_axis), day_axes[0])
    year_axis = [idx for idx in range(3) if idx not in (hour_axis, day_axis)]
    if len(year_axis) != 1:
        raise ValueError(f"Cannot infer year axis from shape {arr.shape}")
    out = np.transpose(arr, (year_axis[0], day_axis, hour_axis))
    return out.astype(float)


def _load_mat_array(root: Path, file_name: str, preferred_key: str) -> np.ndarray:
    path = _find_file(root, file_name)
    data = loadmat(path)
    keys = [key for key in data.keys() if not key.startswith("__")]
    if preferred_key in data:
        keys = [preferred_key] + [key for key in keys if key != preferred_key]
    for key in keys:
        arr = np.asarray(data[key])
        if np.issubdtype(arr.dtype, np.number) and arr.ndim == 3:
            return _as_year_day_hour(arr)
    raise ValueError(f"No 3D numeric array found in {path}")


def load_historical_profiles(root: Path) -> dict[str, np.ndarray]:
    load = _load_mat_array(root, "Load_data_8760.mat", "load_data")
    wind_on = _load_mat_array(root, "Onshore_wind_data_8760.mat", "wind_data")
    wind_off = _load_mat_array(root, "Offshore_wind_data_8760.mat", "wind_data_ofs")
    solar_u = _load_mat_array(root, "Utility_PV_data_8760.mat", "solar_data")
    solar_d = _load_mat_array(root, "Distributed_PV_data_8760.mat", "solar_data_dis")
    solar_csp = _load_mat_array(root, "CSP_data_8760.mat", "solar_data")
    wind = wind_on + 0.6 * wind_off
    solar = solar_u + 0.7 * solar_d + 0.0 * solar_csp
    net = load - wind - solar
    return {
        "load": load.reshape(load.shape[0], -1),
        "wind_power": wind.reshape(wind.shape[0], -1),
        "solar_power": solar.reshape(solar.shape[0], -1),
        "net_load": net.reshape(net.shape[0], -1),
    }


def load_embedded(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["time", *CHANNELS]:
        if col not in df.columns:
            raise ValueError(f"Embedded CSV is missing column: {col}")
    df["time"] = pd.to_datetime(df["time"])
    return df


def month_tick_positions(time: pd.Series) -> tuple[list[int], list[str]]:
    month_start = time.dt.to_period("M").drop_duplicates().dt.to_timestamp()
    labels = [ts.strftime("%b") for ts in month_start]
    positions = [int(time[time.dt.to_period("M") == ts.to_period("M")].index[0]) for ts in month_start]
    return positions, labels


def _history_quantiles(history: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "p05": np.quantile(history, 0.05, axis=0),
        "p10": np.quantile(history, 0.10, axis=0),
        "p50": np.quantile(history, 0.50, axis=0),
        "p90": np.quantile(history, 0.90, axis=0),
        "p95": np.quantile(history, 0.95, axis=0),
    }


def plot_annual_envelope(embedded: pd.DataFrame, history: dict[str, np.ndarray], out_path: Path) -> None:
    x = np.arange(len(embedded))
    xticks, xticklabels = month_tick_positions(embedded["time"])
    extreme_mask = embedded.get("is_extreme", pd.Series(False, index=embedded.index)).fillna(False).astype(bool).to_numpy()
    spans = []
    if extreme_mask.any():
        idx = np.flatnonzero(extreme_mask)
        breaks = np.where(np.diff(idx) > 1)[0]
        starts = np.r_[idx[0], idx[breaks + 1]]
        ends = np.r_[idx[breaks], idx[-1]]
        spans = list(zip(starts, ends))

    fig, axes = plt.subplots(4, 1, figsize=(16, 10.5), sharex=True)
    fig.suptitle("8760-hour Embedded Annual Scenario vs Historical Real Samples", fontsize=17, fontweight="bold")
    for ax, channel in zip(axes, CHANNELS):
        q = _history_quantiles(history[channel])
        ax.fill_between(x, q["p05"], q["p95"], color="#d9d9d9", alpha=0.45, linewidth=0, label="Historical P05-P95")
        ax.fill_between(x, q["p10"], q["p90"], color="#bdbdbd", alpha=0.55, linewidth=0, label="Historical P10-P90")
        ax.plot(x, q["p50"], color="#555555", linewidth=0.9, linestyle="--", label="Historical median")
        ax.plot(x, embedded[channel].to_numpy(dtype=float), color=COLORS[channel], linewidth=0.75, label="Embedded 8760")
        for start, end in spans:
            ax.axvspan(start, end, color="#f03b20", alpha=0.07)
        ax.set_ylabel(LABELS[channel])
        ax.grid(True, color="#eeeeee", linewidth=0.7)
        ax.margins(x=0)
    axes[0].legend(loc="upper right", ncol=4, fontsize=9, frameon=False)
    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels(xticklabels)
    axes[-1].set_xlabel("Month")
    fig.text(0.012, 0.01, "Red bands mark embedded extreme-event windows.", fontsize=9, color="#666666")
    fig.tight_layout(rect=[0, 0.02, 1, 0.965])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _monthly_sum(series: np.ndarray, months: np.ndarray) -> np.ndarray:
    return np.asarray([series[months == month].sum() for month in range(1, 13)], dtype=float)


def plot_summary_dashboard(embedded: pd.DataFrame, history: dict[str, np.ndarray], out_path: Path) -> pd.DataFrame:
    months = embedded["time"].dt.month.to_numpy()
    x_month = np.arange(1, 13)
    hist_net = history["net_load"]
    emb_net = embedded["net_load"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5))
    fig.suptitle("Historical Consistency Diagnostics for Embedded 8760 Scenario", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    hist_sorted = np.sort(hist_net, axis=1)[:, ::-1]
    q = _history_quantiles(hist_sorted)
    ax.fill_between(np.arange(hist_sorted.shape[1]), q["p10"], q["p90"], color="#bdbdbd", alpha=0.55, linewidth=0)
    ax.plot(q["p50"], color="#555555", linestyle="--", linewidth=1.1, label="Historical median")
    ax.plot(np.sort(emb_net)[::-1], color=COLORS["net_load"], linewidth=1.1, label="Embedded")
    ax.set_title("Net-load Duration Curve")
    ax.set_xlabel("Ranked hour")
    ax.set_ylabel("Net load")
    ax.grid(True, color="#eeeeee")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    energy_channels = ["load", "wind_power", "solar_power"]
    width = 0.24
    offsets = [-width, 0.0, width]
    for offset, channel in zip(offsets, energy_channels):
        hist_month = np.vstack([_monthly_sum(year, months) for year in history[channel]])
        hist_mean = hist_month.mean(axis=0)
        emb_month = _monthly_sum(embedded[channel].to_numpy(dtype=float), months)
        ratio = emb_month / np.maximum(hist_mean, 1e-12)
        ax.bar(x_month + offset, ratio, width=width, color=COLORS[channel], alpha=0.82, label=LABELS[channel])
    ax.axhline(1.0, color="#333333", linewidth=0.9)
    ax.set_title("Monthly Energy Ratio: Embedded / Historical Mean")
    ax.set_xlabel("Month")
    ax.set_ylabel("Ratio")
    ax.set_xticks(x_month)
    ax.set_ylim(0.0, max(1.6, ax.get_ylim()[1]))
    ax.grid(True, axis="y", color="#eeeeee")
    ax.legend(ncol=3, fontsize=9, frameon=False)

    ax = axes[1, 0]
    hist_flat = hist_net.reshape(-1)
    bins = np.linspace(min(hist_flat.min(), emb_net.min()), max(hist_flat.max(), emb_net.max()), 70)
    ax.hist(hist_flat, bins=bins, density=True, color="#bdbdbd", alpha=0.65, label="Historical hours")
    ax.hist(emb_net, bins=bins, density=True, color=COLORS["net_load"], alpha=0.45, label="Embedded")
    ax.set_title("Net-load Distribution")
    ax.set_xlabel("Net load")
    ax.set_ylabel("Density")
    ax.grid(True, color="#eeeeee")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    log_cols = ["event_id", "start_time", "end_time", "event_type", "severity_level"]
    event_table = embedded.loc[embedded.get("is_extreme", False).fillna(False).astype(bool), ["time", "net_load", "event_id"]]
    if not event_table.empty:
        first_event = event_table["event_id"].iloc[0]
        event_idx = event_table.index[event_table["event_id"] == first_event].to_numpy()
        start = max(0, int(event_idx[0]) - 24)
        end = min(len(embedded), int(event_idx[-1]) + 25)
        xs = np.arange(start, end)
        q = _history_quantiles(hist_net)
        ax.fill_between(xs, q["p10"][start:end], q["p90"][start:end], color="#bdbdbd", alpha=0.55, linewidth=0)
        ax.plot(xs, q["p50"][start:end], color="#555555", linestyle="--", linewidth=1.0, label="Historical median")
        ax.plot(xs, emb_net[start:end], color=COLORS["net_load"], linewidth=1.5, label="Embedded")
        ax.axvspan(event_idx[0], event_idx[-1], color="#f03b20", alpha=0.10)
        ax.set_title(f"Zoom Around {first_event}")
        ax.set_xlabel("Hour index")
        ax.set_ylabel("Net load")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No embedded events", ha="center", va="center", transform=ax.transAxes)
    ax.grid(True, color="#eeeeee")

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for channel in CHANNELS:
        hist = history[channel]
        emb = embedded[channel].to_numpy(dtype=float)
        rows.append(
            {
                "channel": channel,
                "embedded_mean": float(emb.mean()),
                "historical_mean": float(hist.mean()),
                "embedded_q05": float(np.quantile(emb, 0.05)),
                "historical_q05": float(np.quantile(hist, 0.05)),
                "embedded_q50": float(np.quantile(emb, 0.50)),
                "historical_q50": float(np.quantile(hist, 0.50)),
                "embedded_q95": float(np.quantile(emb, 0.95)),
                "historical_q95": float(np.quantile(hist, 0.95)),
                "embedded_max": float(emb.max()),
                "historical_max": float(hist.max()),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot embedded 8760 scenario against historical real 8760 samples.")
    parser.add_argument(
        "--embedded",
        default="ExtremeScene/outputs/annual_embedded_8760_final/annual_embedded_8760.csv",
        help="Path to annual_embedded_8760.csv.",
    )
    parser.add_argument("--history-root", default=".", help="Project root used to locate historical .mat files.")
    parser.add_argument(
        "--out-dir",
        default="ExtremeScene/outputs/annual_embedded_8760_final/figures",
        help="Directory for comparison figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    embedded = load_embedded(Path(args.embedded))
    history = load_historical_profiles(Path(args.history_root))

    annual_path = out_dir / "annual_8760_history_comparison.png"
    dashboard_path = out_dir / "annual_8760_history_dashboard.png"
    metrics_path = out_dir / "annual_8760_history_metrics.csv"
    plot_annual_envelope(embedded, history, annual_path)
    metrics = plot_summary_dashboard(embedded, history, dashboard_path)
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"saved {annual_path}")
    print(f"saved {dashboard_path}")
    print(f"saved {metrics_path}")


if __name__ == "__main__":
    main()
