from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


CHANNELS = ["load", "wind_power", "solar_power"]
ID_COLUMNS = ["event_id", "scenario_id", "generated_id", "sample_id"]
STEP_COLUMNS = ["t", "hour", "step", "time_idx", "time_index"]
REQUIRED_META_COLUMNS = ["month", "event_type", "severity_level"]
OPTIONAL_META_COLUMNS = [
    "extreme_prob",
    "cum_deficit",
    "core_cum_deficit",
    "netload_ramp_max",
    "imbalance_duration",
    "max_imbalance_run",
]
LOG_COLUMNS = [
    "event_id",
    "start_time",
    "end_time",
    "month",
    "event_type",
    "severity_level",
    "extreme_prob",
    "cum_deficit",
    "core_cum_deficit",
    "netload_ramp_max",
    "imbalance_duration",
    "max_imbalance_run",
]
BOUNDARY_COLUMNS = [
    "event_id",
    "start_jump_before",
    "start_jump_after",
    "end_jump_before",
    "end_jump_after",
    "max_ramp_before",
    "max_ramp_after",
]


@dataclass
class AnnualEmbeddingConfig:
    background_file: str
    extreme_file: str
    out_dir: str
    condition_file: Optional[str] = None
    num_events: int = 8
    segment_hours: int = 36
    smooth_hours: int = 4
    random_seed: int = 42
    min_candidates: int = 1
    max_attempts: Optional[int] = None
    imbalance_quantile: float = 0.75
    position_temperature: float = 0.2
    allow_partial: bool = False


@dataclass
class SegmentLibrary:
    meta: pd.DataFrame
    arrays: dict[str, np.ndarray]
    core_masks: dict[str, np.ndarray]


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _read_csv(path: str | Path, label: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{label} not found: {csv_path}")
    return _clean_columns(pd.read_csv(csv_path))


def _require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _nonempty_series(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def _choose_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns and _nonempty_series(df[col]).any():
            return col
    return None


def _choose_common_id(left: pd.DataFrame, right: pd.DataFrame, preferred: Optional[str]) -> Optional[str]:
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend([c for c in ID_COLUMNS if c != preferred])
    for col in candidates:
        if col in left.columns and col in right.columns:
            if _nonempty_series(left[col]).any() and _nonempty_series(right[col]).any():
                return col
    return None


def _coerce_numeric(df: pd.DataFrame, columns: list[str], label: str) -> None:
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            bad = int(df[col].isna().sum())
            raise ValueError(f"{label}.{col} contains {bad} non-numeric or missing values.")


def load_background(path: str | Path) -> pd.DataFrame:
    df = _read_csv(path, "background")
    _require_columns(df, ["time", *CHANNELS], "background")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if df["time"].isna().any():
        raise ValueError("background.time contains values that cannot be parsed as datetimes.")
    _coerce_numeric(df, CHANNELS, "background")
    if "month" not in df.columns:
        df["month"] = df["time"].dt.month
    else:
        df["month"] = pd.to_numeric(df["month"], errors="coerce")
        if df["month"].isna().any():
            raise ValueError("background.month contains non-numeric or missing values.")
        df["month"] = df["month"].astype(int)
    if "net_load" not in df.columns:
        df["net_load"] = df["load"] - df["wind_power"] - df["solar_power"]
    else:
        df["net_load"] = pd.to_numeric(df["net_load"], errors="coerce")
        if df["net_load"].isna().any():
            df["net_load"] = df["load"] - df["wind_power"] - df["solar_power"]
    return df


def _normalize_long_table(raw: pd.DataFrame, segment_hours: int) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    _require_columns(raw, CHANNELS, "extreme long table")
    df = raw.copy()
    id_col = _choose_column(df, ID_COLUMNS)
    if id_col is None:
        if len(df) % segment_hours != 0:
            raise ValueError(
                "Extreme long table has no scenario_id/event_id/generated_id/sample_id column "
                f"and row count {len(df)} is not divisible by segment_hours={segment_hours}."
            )
        df["segment_id"] = [f"SEG{idx // segment_hours:05d}" for idx in range(len(df))]
        id_col = "segment_id"
    else:
        df["segment_id"] = df[id_col].astype(str)

    step_col = _choose_column(df, STEP_COLUMNS)
    if step_col is None:
        df["_step"] = df.groupby("segment_id", sort=False).cumcount()
        step_col = "_step"
    else:
        df["_step"] = pd.to_numeric(df[step_col], errors="coerce")
        if df["_step"].isna().any():
            raise ValueError(f"Extreme long table step column '{step_col}' contains non-numeric values.")

    _coerce_numeric(df, CHANNELS, "extreme long table")
    df = df.sort_values(["segment_id", "_step"]).reset_index(drop=True)

    arrays: dict[str, np.ndarray] = {}
    core_masks: dict[str, np.ndarray] = {}
    meta_rows = []
    for segment_id, group in df.groupby("segment_id", sort=False):
        if len(group) != segment_hours:
            raise ValueError(
                f"Segment {segment_id!r} has {len(group)} rows; expected segment_hours={segment_hours}."
            )
        arrays[str(segment_id)] = group[CHANNELS].to_numpy(dtype=float)
        if "is_core" in group.columns:
            core_masks[str(segment_id)] = group["is_core"].fillna(False).astype(bool).to_numpy()
        first = group.iloc[0].drop(labels=["_step"], errors="ignore").to_dict()
        first["segment_id"] = str(segment_id)
        meta_rows.append(first)

    meta = pd.DataFrame(meta_rows)
    for segment_id in arrays:
        if segment_id not in core_masks:
            core_masks[segment_id] = np.zeros(segment_hours, dtype=bool)
    return meta, arrays, core_masks


def _parse_wide_column(column: str) -> Optional[tuple[str, int]]:
    name = column.strip().lower()
    patterns = [
        r"^(?P<channel>load|wind_power|wind|solar_power|solar)[_\-\s]?(?:t|h|hour|step)?[_\-\s]?(?P<step>\d+)$",
        r"^(?:t|h|hour|step)?[_\-\s]?(?P<step>\d+)[_\-\s]?(?P<channel>load|wind_power|wind|solar_power|solar)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, name)
        if not match:
            continue
        channel = match.group("channel")
        step = int(match.group("step"))
        if channel == "wind":
            channel = "wind_power"
        elif channel == "solar":
            channel = "solar_power"
        return channel, step
    return None


def _wide_column_map(columns: list[str]) -> dict[str, dict[int, str]]:
    mapping: dict[str, dict[int, str]] = {channel: {} for channel in CHANNELS}
    for col in columns:
        parsed = _parse_wide_column(col)
        if parsed is None:
            continue
        channel, step = parsed
        if channel in mapping and step not in mapping[channel]:
            mapping[channel][step] = col
    return mapping


def _normalize_wide_table(raw: pd.DataFrame, segment_hours: int) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    mapping = _wide_column_map(list(raw.columns))
    missing = {
        channel: [step for step in range(segment_hours) if step not in mapping[channel]]
        for channel in CHANNELS
    }
    missing = {channel: steps for channel, steps in missing.items() if steps}
    if missing:
        raise ValueError(
            "Extreme file is not a recognized long table and is missing wide columns for "
            f"segment_hours={segment_hours}: {missing}"
        )

    id_col = _choose_column(raw, ID_COLUMNS)
    arrays: dict[str, np.ndarray] = {}
    core_masks: dict[str, np.ndarray] = {}
    meta_rows = []
    wide_cols = {col for by_step in mapping.values() for col in by_step.values()}

    for row_idx, row in raw.iterrows():
        segment_id = str(row[id_col]) if id_col else f"SEG{row_idx:05d}"
        segment = np.zeros((segment_hours, len(CHANNELS)), dtype=float)
        for channel_idx, channel in enumerate(CHANNELS):
            cols = [mapping[channel][step] for step in range(segment_hours)]
            values = pd.to_numeric(row[cols], errors="coerce").to_numpy(dtype=float)
            if np.isnan(values).any():
                raise ValueError(f"Wide segment {segment_id!r} has non-numeric values in channel {channel}.")
            segment[:, channel_idx] = values
        arrays[segment_id] = segment
        core_masks[segment_id] = np.zeros(segment_hours, dtype=bool)
        meta = row.drop(labels=list(wide_cols), errors="ignore").to_dict()
        meta["segment_id"] = segment_id
        meta_rows.append(meta)

    meta_df = pd.DataFrame(meta_rows)
    return meta_df, arrays, core_masks


def _merge_condition_file(meta: pd.DataFrame, condition_file: str | Path, preferred_id: Optional[str]) -> pd.DataFrame:
    cond = _read_csv(condition_file, "condition_file")
    merge_col = _choose_common_id(meta, cond, preferred_id)
    if merge_col is None:
        raise ValueError(
            "condition_file was provided, but no usable common id column was found. "
            f"Tried: {ID_COLUMNS}."
        )
    cond = cond.drop_duplicates(subset=[merge_col], keep="first")
    merged = meta.merge(cond, on=merge_col, how="left", suffixes=("", "_cond"))
    for col in list(cond.columns):
        cond_col = f"{col}_cond"
        if cond_col in merged.columns:
            if col not in merged.columns:
                merged[col] = merged[cond_col]
            else:
                merged[col] = merged[col].where(_nonempty_series(merged[col]), merged[cond_col])
            merged = merged.drop(columns=[cond_col])
    return merged


def _finalize_meta(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    _require_columns(out, REQUIRED_META_COLUMNS, "extreme metadata")
    out["month"] = pd.to_numeric(out["month"], errors="coerce")
    if out["month"].isna().any():
        raise ValueError("extreme metadata month contains non-numeric or missing values.")
    out["month"] = out["month"].astype(int)
    if (~out["month"].between(1, 12)).any():
        bad = sorted(out.loc[~out["month"].between(1, 12), "month"].unique().tolist())
        raise ValueError(f"extreme metadata month must be in 1..12; found {bad}.")
    out["event_type"] = out["event_type"].astype(str)
    out["severity_level"] = out["severity_level"].astype(str)
    for col in OPTIONAL_META_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _default_core_mask(segment_hours: int, smooth_hours: int) -> np.ndarray:
    mask = np.zeros(segment_hours, dtype=bool)
    start = min(max(smooth_hours, 0), segment_hours)
    end = max(start, segment_hours - max(smooth_hours, 0))
    mask[start:end] = True
    return mask


def load_extreme_library(
    extreme_file: str | Path,
    condition_file: Optional[str | Path],
    segment_hours: int,
    smooth_hours: int,
) -> SegmentLibrary:
    raw = _read_csv(extreme_file, "extreme_file")
    preferred_id = _choose_column(raw, ID_COLUMNS)
    if all(channel in raw.columns for channel in CHANNELS):
        meta, arrays, core_masks = _normalize_long_table(raw, segment_hours)
    else:
        meta, arrays, core_masks = _normalize_wide_table(raw, segment_hours)

    if condition_file:
        meta = _merge_condition_file(meta, condition_file, preferred_id)
    meta = _finalize_meta(meta)
    default_core = _default_core_mask(segment_hours, smooth_hours)
    core_masks = {
        str(row["segment_id"]): core_masks.get(str(row["segment_id"]), default_core).astype(bool)
        for _, row in meta.iterrows()
    }
    for segment_id, mask in list(core_masks.items()):
        if mask.size != segment_hours or not mask.any():
            core_masks[segment_id] = default_core.copy()

    return SegmentLibrary(meta=meta.reset_index(drop=True), arrays=arrays, core_masks=core_masks)


def compute_probabilities(meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    month_event = (
        meta.groupby(["month", "event_type"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["month", "event_type"])
        .reset_index(drop=True)
    )
    month_event["probability"] = month_event["count"] / month_event["count"].sum()

    severity = (
        meta.groupby(["month", "event_type", "severity_level"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["month", "event_type", "severity_level"])
        .reset_index(drop=True)
    )
    severity["probability"] = severity["count"] / severity.groupby(["month", "event_type"])["count"].transform("sum")
    return month_event, severity


def _sample_from_prob_table(
    table: pd.DataFrame,
    rng: np.random.Generator,
    value_columns: list[str],
) -> tuple:
    if table.empty:
        raise ValueError("Cannot sample from an empty probability table.")
    probs = table["probability"].to_numpy(dtype=float)
    probs = probs / probs.sum()
    idx = int(rng.choice(np.arange(len(table)), p=probs))
    row = table.iloc[idx]
    return tuple(row[col] for col in value_columns)


def sample_target(
    month_event_prob: pd.DataFrame,
    severity_prob: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[int, str, str]:
    month, event_type = _sample_from_prob_table(month_event_prob, rng, ["month", "event_type"])
    sub = severity_prob[(severity_prob["month"] == month) & (severity_prob["event_type"] == event_type)]
    severity_level = _sample_from_prob_table(sub, rng, ["severity_level"])[0]
    return int(month), str(event_type), str(severity_level)


def select_candidate_segments(
    meta: pd.DataFrame,
    month: int,
    event_type: str,
    severity_level: str,
    min_candidates: int,
) -> tuple[pd.DataFrame, str]:
    same_month = meta["month"].astype(int) == int(month)
    exact = meta[same_month & (meta["event_type"].astype(str) == event_type) & (meta["severity_level"].astype(str) == severity_level)]
    if len(exact) >= min_candidates:
        return exact, "exact"

    relaxed_severity = meta[same_month & (meta["event_type"].astype(str) == event_type)]
    if len(relaxed_severity) >= min_candidates or (len(relaxed_severity) > 0 and len(exact) == 0):
        return relaxed_severity, "relaxed_severity"

    relaxed_event = meta[same_month]
    if len(relaxed_event) > 0:
        return relaxed_event, "relaxed_event_type"

    return meta.iloc[0:0], "no_candidate_in_month"


def net_load(values: np.ndarray) -> np.ndarray:
    return values[:, 0] - values[:, 1] - values[:, 2]


def candidate_positions(background: pd.DataFrame, month: int, length: int, occupied: np.ndarray) -> np.ndarray:
    month_values = background["month"].to_numpy(dtype=int)
    starts = []
    for start in np.flatnonzero(month_values == int(month)):
        end = start + length
        if end > len(background):
            continue
        if occupied[start:end].any():
            continue
        if np.all(month_values[start:end] == int(month)):
            starts.append(start)
    return np.asarray(starts, dtype=int)


def max_3h_ramp(series: np.ndarray) -> float:
    if len(series) <= 3:
        return float(np.abs(np.diff(series)).max()) if len(series) > 1 else 0.0
    return float(np.abs(series[3:] - series[:-3]).max())


def vulnerability_score(segment_net_load: np.ndarray, imbalance_tau: float) -> float:
    imbalance_duration = float((segment_net_load > imbalance_tau).sum())
    return (
        0.5 * float(segment_net_load.mean())
        + 0.3 * max_3h_ramp(segment_net_load)
        + 0.2 * imbalance_duration
    )


def score_positions(values: np.ndarray, starts: np.ndarray, length: int, imbalance_tau: float) -> np.ndarray:
    net = net_load(values)
    return np.asarray([vulnerability_score(net[start : start + length], imbalance_tau) for start in starts], dtype=float)


def choose_position(
    starts: np.ndarray,
    scores: np.ndarray,
    rng: np.random.Generator,
    temperature: float,
) -> tuple[int, float]:
    if len(starts) == 0:
        raise ValueError("No candidate positions are available.")
    if len(starts) == 1:
        return int(starts[0]), float(scores[0])
    if temperature <= 0:
        idx = int(np.nanargmax(scores))
        return int(starts[idx]), float(scores[idx])
    clean = np.nan_to_num(scores, nan=np.nanmin(scores), posinf=np.nanmax(scores), neginf=np.nanmin(scores))
    spread = float(clean.max() - clean.min())
    if spread <= 1e-12:
        probs = np.full(len(clean), 1.0 / len(clean))
    else:
        z = (clean - clean.min()) / spread
        logits = z / max(temperature, 1e-9)
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()
    idx = int(rng.choice(np.arange(len(starts)), p=probs))
    return int(starts[idx]), float(scores[idx])


def boundary_align(segment: np.ndarray, background_window: np.ndarray) -> np.ndarray:
    start_offset = background_window[0] - segment[0]
    end_offset = background_window[-1] - segment[-1]
    weights = np.linspace(0.0, 1.0, len(segment), dtype=float)[:, None]
    offsets = (1.0 - weights) * start_offset + weights * end_offset
    return np.clip(segment + offsets, 0.0, 1.0)


def _cosine_weights(width: int) -> np.ndarray:
    if width <= 0:
        return np.asarray([], dtype=float)
    if width == 1:
        return np.asarray([0.5], dtype=float)
    return 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, width)))


def cosine_boundary_smooth(
    background_window: np.ndarray,
    aligned_segment: np.ndarray,
    smooth_hours: int,
    core_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    out = aligned_segment.copy()
    width = min(max(int(smooth_hours), 0), len(out) // 2)
    if width <= 0:
        return out
    if core_mask is None or len(core_mask) != len(out):
        core_mask = np.zeros(len(out), dtype=bool)
    in_weights = _cosine_weights(width)
    out_weights = in_weights[::-1]
    for i, weight in enumerate(in_weights):
        if not core_mask[i]:
            out[i] = (1.0 - weight) * background_window[i] + weight * aligned_segment[i]
    for j, weight in enumerate(out_weights):
        idx = len(out) - width + j
        if not core_mask[idx]:
            out[idx] = (1.0 - weight) * background_window[idx] + weight * aligned_segment[idx]
    return np.clip(out, 0.0, 1.0)


def _channel_jump(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(right - left).mean())


def _max_local_net_ramp(left: Optional[np.ndarray], segment: np.ndarray, right: Optional[np.ndarray]) -> float:
    parts = []
    if left is not None:
        parts.append(left.reshape(1, -1))
    parts.append(segment)
    if right is not None:
        parts.append(right.reshape(1, -1))
    local = np.vstack(parts)
    local_net = net_load(local)
    return float(np.abs(np.diff(local_net)).max()) if len(local_net) > 1 else 0.0


def boundary_metrics(
    values: np.ndarray,
    start: int,
    raw_segment: np.ndarray,
    aligned_segment: np.ndarray,
    smoothed_segment: np.ndarray,
) -> dict[str, float]:
    end = start + len(aligned_segment)
    left = values[start - 1] if start > 0 else None
    right = values[end] if end < len(values) else None
    return {
        "start_jump_raw": _channel_jump(left, raw_segment[0]) if left is not None else np.nan,
        "start_jump_before": _channel_jump(left, aligned_segment[0]) if left is not None else np.nan,
        "start_jump_after": _channel_jump(left, smoothed_segment[0]) if left is not None else np.nan,
        "end_jump_raw": _channel_jump(raw_segment[-1], right) if right is not None else np.nan,
        "end_jump_before": _channel_jump(aligned_segment[-1], right) if right is not None else np.nan,
        "end_jump_after": _channel_jump(smoothed_segment[-1], right) if right is not None else np.nan,
        "max_ramp_raw": _max_local_net_ramp(left, raw_segment, right),
        "max_ramp_before": _max_local_net_ramp(left, aligned_segment, right),
        "max_ramp_after": _max_local_net_ramp(left, smoothed_segment, right),
    }


def _safe_value(row: pd.Series, col: str):
    return row[col] if col in row.index else np.nan


def _format_event_id(index: int) -> str:
    return f"EMB{index:04d}"


def embed_extreme_events(
    background: pd.DataFrame,
    library: SegmentLibrary,
    cfg: AnnualEmbeddingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.random_seed)
    values = background[CHANNELS].to_numpy(dtype=float, copy=True)
    occupied = np.zeros(len(background), dtype=bool)
    annual = background.copy()
    annual["is_extreme"] = False
    annual["event_id"] = ""
    annual["event_type"] = ""
    annual["severity_level"] = pd.NA
    annual["extreme_prob"] = np.nan
    annual["is_core"] = False

    month_event_prob, severity_prob = compute_probabilities(library.meta)
    imbalance_tau = float(np.quantile(net_load(values), cfg.imbalance_quantile))
    max_attempts = cfg.max_attempts or max(cfg.num_events * 50, 50)

    log_rows = []
    boundary_rows = []
    inserted = 0
    attempts = 0
    warnings_seen: list[str] = []

    while inserted < cfg.num_events and attempts < max_attempts:
        attempts += 1
        month, event_type, severity_level = sample_target(month_event_prob, severity_prob, rng)
        candidates, fallback_level = select_candidate_segments(
            library.meta, month, event_type, severity_level, cfg.min_candidates
        )
        if candidates.empty:
            warnings_seen.append(f"attempt {attempts}: no segment candidate for month={month}.")
            continue

        starts = candidate_positions(background, month, cfg.segment_hours, occupied)
        if starts.size == 0:
            warnings_seen.append(f"attempt {attempts}: no non-overlapping {cfg.segment_hours}h position in month={month}.")
            continue

        scores = score_positions(values, starts, cfg.segment_hours, imbalance_tau)
        start, chosen_score = choose_position(starts, scores, rng, cfg.position_temperature)
        chosen = candidates.iloc[int(rng.integers(0, len(candidates)))]
        segment_id = str(chosen["segment_id"])
        raw_segment = library.arrays[segment_id].astype(float, copy=True)
        end = start + cfg.segment_hours
        background_window = values[start:end].copy()
        aligned_segment = boundary_align(raw_segment, background_window)
        core_mask = library.core_masks.get(segment_id, _default_core_mask(cfg.segment_hours, cfg.smooth_hours))
        smoothed_segment = cosine_boundary_smooth(background_window, aligned_segment, cfg.smooth_hours, core_mask)
        event_id = _format_event_id(inserted + 1)

        metrics = boundary_metrics(values, start, np.clip(raw_segment, 0.0, 1.0), aligned_segment, smoothed_segment)
        values[start:end] = smoothed_segment
        occupied[start:end] = True

        annual.loc[start : end - 1, "is_extreme"] = True
        annual.loc[start : end - 1, "event_id"] = event_id
        annual.loc[start : end - 1, "event_type"] = str(chosen["event_type"])
        annual.loc[start : end - 1, "severity_level"] = chosen["severity_level"]
        annual.loc[start : end - 1, "extreme_prob"] = _safe_value(chosen, "extreme_prob")
        annual.loc[start : end - 1, "is_core"] = core_mask.astype(bool)

        log_row = {
            "event_id": event_id,
            "start_time": annual.loc[start, "time"],
            "end_time": annual.loc[end - 1, "time"],
            "month": int(chosen["month"]),
            "event_type": str(chosen["event_type"]),
            "severity_level": chosen["severity_level"],
            "extreme_prob": _safe_value(chosen, "extreme_prob"),
            "cum_deficit": _safe_value(chosen, "cum_deficit"),
            "core_cum_deficit": _safe_value(chosen, "core_cum_deficit"),
            "netload_ramp_max": _safe_value(chosen, "netload_ramp_max"),
            "imbalance_duration": _safe_value(chosen, "imbalance_duration"),
            "max_imbalance_run": _safe_value(chosen, "max_imbalance_run"),
            "source_segment_id": segment_id,
            "start_idx": start,
            "end_idx": end - 1,
            "requested_event_type": event_type,
            "requested_severity_level": severity_level,
            "fallback_level": fallback_level,
            "vulnerability": chosen_score,
        }
        boundary_row = {"event_id": event_id, **metrics, "source_segment_id": segment_id, "start_idx": start, "end_idx": end - 1}
        log_rows.append(log_row)
        boundary_rows.append(boundary_row)
        inserted += 1

    if inserted < cfg.num_events:
        message = (
            f"Inserted {inserted}/{cfg.num_events} events after {attempts} attempts. "
            "This usually means segment candidates or non-overlapping monthly positions are insufficient."
        )
        if warnings_seen:
            message += " Recent issues: " + " | ".join(warnings_seen[-5:])
        if not cfg.allow_partial:
            raise RuntimeError(message + " Re-run with --allow-partial to write partial results.")
        print(f"WARNING: {message}", file=sys.stderr)

    annual[CHANNELS] = values
    annual["net_load"] = annual["load"] - annual["wind_power"] - annual["solar_power"]

    log_df = pd.DataFrame(log_rows)
    boundary_df = pd.DataFrame(boundary_rows)
    for col in LOG_COLUMNS:
        if col not in log_df.columns:
            log_df[col] = np.nan
    for col in BOUNDARY_COLUMNS:
        if col not in boundary_df.columns:
            boundary_df[col] = np.nan
    log_order = LOG_COLUMNS + [c for c in log_df.columns if c not in LOG_COLUMNS]
    boundary_order = BOUNDARY_COLUMNS + [c for c in boundary_df.columns if c not in BOUNDARY_COLUMNS]
    return annual, log_df[log_order], boundary_df[boundary_order]


def validate_config(cfg: AnnualEmbeddingConfig) -> None:
    if cfg.num_events < 0:
        raise ValueError("num_events must be non-negative.")
    if cfg.segment_hours <= 0:
        raise ValueError("segment_hours must be positive.")
    if cfg.smooth_hours < 0:
        raise ValueError("smooth_hours must be non-negative.")
    if cfg.smooth_hours * 2 >= cfg.segment_hours:
        raise ValueError("smooth_hours must be less than half of segment_hours.")
    if cfg.min_candidates <= 0:
        raise ValueError("min_candidates must be positive.")
    if not 0.0 <= cfg.imbalance_quantile <= 1.0:
        raise ValueError("imbalance_quantile must be in [0, 1].")
    if cfg.position_temperature < 0:
        raise ValueError("position_temperature must be non-negative.")


def run_annual_embedding(cfg: AnnualEmbeddingConfig) -> dict[str, int | str]:
    validate_config(cfg)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    background = load_background(cfg.background_file)
    if len(background) < cfg.segment_hours:
        raise ValueError(
            f"background has only {len(background)} rows; segment_hours={cfg.segment_hours} cannot be inserted."
        )
    library = load_extreme_library(cfg.extreme_file, cfg.condition_file, cfg.segment_hours, cfg.smooth_hours)
    if library.meta.empty:
        raise ValueError("No usable extreme segments were loaded.")

    annual, log_df, boundary_df = embed_extreme_events(background, library, cfg)
    annual.to_csv(out_dir / "annual_embedded_8760.csv", index=False, encoding="utf-8-sig")
    log_df.to_csv(out_dir / "embedding_log.csv", index=False, encoding="utf-8-sig")
    boundary_df.to_csv(out_dir / "boundary_metrics.csv", index=False, encoding="utf-8-sig")
    return {
        "inserted_events": int(len(log_df)),
        "annual_rows": int(len(annual)),
        "out_dir": str(out_dir),
    }


def parse_args() -> AnnualEmbeddingConfig:
    parser = argparse.ArgumentParser(
        description="Embed generated 36h extreme wind-load-solar segments into an annual background scenario."
    )
    parser.add_argument("--background-file", "--background", required=True, help="Path to background_8760.csv.")
    parser.add_argument("--extreme-file", "--segments", required=True, help="Path to generated extreme CSV.")
    parser.add_argument("--condition-file", "--cond", default=None, help="Optional per-segment label/condition CSV.")
    parser.add_argument("--out-dir", default=".", help="Directory for annual_embedded_8760.csv and logs.")
    parser.add_argument("--num-events", type=int, default=8, help="Number of annual extreme events to insert.")
    parser.add_argument("--segment-hours", type=int, default=36, help="Extreme segment length in hours.")
    parser.add_argument("--smooth-hours", type=int, default=4, help="Cosine smoothing hours at each boundary.")
    parser.add_argument("--smooth-width", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--random-seed", type=int, default=42, help="Seed for reproducible sampling.")
    parser.add_argument("--min-candidates", type=int, default=1, help="Minimum exact candidates before fallback.")
    parser.add_argument("--max-attempts", type=int, default=None, help="Maximum sampling attempts.")
    parser.add_argument(
        "--imbalance-quantile",
        type=float,
        default=0.75,
        help="Annual net-load quantile used to count vulnerable imbalance hours.",
    )
    parser.add_argument(
        "--position-temperature",
        type=float,
        default=0.2,
        help="Softmax temperature for vulnerability-weighted position sampling; 0 picks the maximum.",
    )
    parser.add_argument("--allow-partial", action="store_true", help="Write outputs even if fewer than num_events are inserted.")
    args = parser.parse_args()
    smooth_hours = args.smooth_width if args.smooth_width is not None else args.smooth_hours
    return AnnualEmbeddingConfig(
        background_file=args.background_file,
        extreme_file=args.extreme_file,
        condition_file=args.condition_file,
        out_dir=args.out_dir,
        num_events=args.num_events,
        segment_hours=args.segment_hours,
        smooth_hours=smooth_hours,
        random_seed=args.random_seed,
        min_candidates=args.min_candidates,
        max_attempts=args.max_attempts,
        imbalance_quantile=args.imbalance_quantile,
        position_temperature=args.position_temperature,
        allow_partial=args.allow_partial,
    )


if __name__ == "__main__":
    result = run_annual_embedding(parse_args())
    print(
        "annual embedding complete: "
        f"inserted_events={result['inserted_events']}, "
        f"annual_rows={result['annual_rows']}, "
        f"out_dir={result['out_dir']}"
    )
