from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from annual_embedding import (
    CHANNELS,
    LOG_COLUMNS,
    AnnualEmbeddingConfig,
    SegmentLibrary,
    _default_core_mask,
    _format_event_id,
    _safe_value,
    boundary_align,
    candidate_positions,
    choose_position,
    compute_probabilities,
    cosine_boundary_smooth,
    load_background,
    load_extreme_library,
    max_3h_ramp,
    net_load,
    score_positions,
    select_candidate_segments,
    sample_target,
    validate_config,
)


@dataclass(frozen=True)
class AblationVariant:
    name: str
    boundary_mode: str
    position_strategy: str


@dataclass
class PlannedEvent:
    target_month: int
    target_event_type: str
    target_severity_level: str
    fallback_level: str
    source_segment_id: str
    meta: dict


DEFAULT_VARIANTS = [
    AblationVariant("raw_vulnerability", "raw", "vulnerability"),
    AblationVariant("offset_vulnerability", "offset", "vulnerability"),
    AblationVariant("smooth_vulnerability", "smooth", "vulnerability"),
    AblationVariant("smooth_random", "smooth", "random"),
]


def parse_variant(value: str) -> AblationVariant:
    parts = value.strip().lower().split("_")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid variant {value!r}; expected '<raw|offset|smooth>_<random|vulnerability>'."
        )
    boundary_mode, position_strategy = parts
    if boundary_mode not in {"raw", "offset", "smooth"}:
        raise ValueError(f"Invalid boundary mode in {value!r}.")
    if position_strategy not in {"random", "vulnerability"}:
        raise ValueError(f"Invalid position strategy in {value!r}.")
    return AblationVariant(value.strip(), boundary_mode, position_strategy)


def make_event_plan(
    library: SegmentLibrary,
    cfg: AnnualEmbeddingConfig,
    plan_size: int,
) -> list[PlannedEvent]:
    rng = np.random.default_rng(cfg.random_seed)
    month_event_prob, severity_prob = compute_probabilities(library.meta)
    plans: list[PlannedEvent] = []
    attempts = 0
    max_attempts = max(plan_size * 20, cfg.num_events * 50, 100)

    while len(plans) < plan_size and attempts < max_attempts:
        attempts += 1
        month, event_type, severity_level = sample_target(month_event_prob, severity_prob, rng)
        candidates, fallback_level = select_candidate_segments(
            library.meta, month, event_type, severity_level, cfg.min_candidates
        )
        if candidates.empty:
            continue
        chosen = candidates.iloc[int(rng.integers(0, len(candidates)))]
        plans.append(
            PlannedEvent(
                target_month=month,
                target_event_type=event_type,
                target_severity_level=severity_level,
                fallback_level=fallback_level,
                source_segment_id=str(chosen["segment_id"]),
                meta=chosen.to_dict(),
            )
        )

    if len(plans) < cfg.num_events and not cfg.allow_partial:
        raise RuntimeError(
            f"Only planned {len(plans)} candidate events for requested num_events={cfg.num_events}."
        )
    return plans


def _channel_jump(left: Optional[np.ndarray], right: Optional[np.ndarray]) -> float:
    if left is None or right is None:
        return np.nan
    return float(np.abs(right - left).mean())


def _local_net_ramp(local_values: np.ndarray) -> float:
    local_net = net_load(local_values)
    return float(np.abs(np.diff(local_net)).max()) if len(local_net) > 1 else 0.0


def _whole_local_ramp(
    pre_values: np.ndarray,
    start: int,
    segment: np.ndarray,
) -> float:
    end = start + len(segment)
    parts = []
    if start > 0:
        parts.append(pre_values[start - 1].reshape(1, -1))
    parts.append(segment)
    if end < len(pre_values):
        parts.append(pre_values[end].reshape(1, -1))
    return _local_net_ramp(np.vstack(parts))


def _boundary_window_ramp(
    pre_values: np.ndarray,
    start: int,
    segment: np.ndarray,
    smooth_hours: int,
) -> float:
    end = start + len(segment)
    width = min(max(int(smooth_hours), 1), len(segment))
    ramps = []

    left_parts = []
    if start > 0:
        left_parts.append(pre_values[start - 1].reshape(1, -1))
    left_parts.append(segment[:width])
    ramps.append(_local_net_ramp(np.vstack(left_parts)))

    right_parts = [segment[-width:]]
    if end < len(pre_values):
        right_parts.append(pre_values[end].reshape(1, -1))
    ramps.append(_local_net_ramp(np.vstack(right_parts)))
    return float(max(ramps))


def _masked_net_mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return np.nan
    return float(np.abs(net_load(a[mask]) - net_load(b[mask])).mean())


def _masked_net_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return np.nan
    return float(net_load(values[mask]).mean())


def _masked_net_max(values: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return np.nan
    return float(net_load(values[mask]).max())


def apply_boundary_mode(
    raw_segment: np.ndarray,
    background_window: np.ndarray,
    core_mask: np.ndarray,
    cfg: AnnualEmbeddingConfig,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_clipped = np.clip(raw_segment, 0.0, 1.0)
    aligned = boundary_align(raw_segment, background_window)
    if mode == "raw":
        final = raw_clipped
    elif mode == "offset":
        final = aligned
    elif mode == "smooth":
        final = cosine_boundary_smooth(background_window, aligned, cfg.smooth_hours, core_mask)
    else:
        raise ValueError(f"Unknown boundary mode: {mode}")
    return raw_clipped, aligned, final


def event_metrics(
    pre_values: np.ndarray,
    start: int,
    raw_segment: np.ndarray,
    aligned_segment: np.ndarray,
    final_segment: np.ndarray,
    core_mask: np.ndarray,
    cfg: AnnualEmbeddingConfig,
) -> dict[str, float]:
    end = start + len(final_segment)
    left = pre_values[start - 1] if start > 0 else None
    right = pre_values[end] if end < len(pre_values) else None
    edge_mask = ~core_mask

    return {
        "start_jump_raw": _channel_jump(left, raw_segment[0]),
        "start_jump_aligned": _channel_jump(left, aligned_segment[0]),
        "start_jump_final": _channel_jump(left, final_segment[0]),
        "end_jump_raw": _channel_jump(raw_segment[-1], right),
        "end_jump_aligned": _channel_jump(aligned_segment[-1], right),
        "end_jump_final": _channel_jump(final_segment[-1], right),
        "whole_net_ramp_raw": _whole_local_ramp(pre_values, start, raw_segment),
        "whole_net_ramp_aligned": _whole_local_ramp(pre_values, start, aligned_segment),
        "whole_net_ramp_final": _whole_local_ramp(pre_values, start, final_segment),
        "boundary_net_ramp_raw": _boundary_window_ramp(pre_values, start, raw_segment, cfg.smooth_hours),
        "boundary_net_ramp_aligned": _boundary_window_ramp(pre_values, start, aligned_segment, cfg.smooth_hours),
        "boundary_net_ramp_final": _boundary_window_ramp(pre_values, start, final_segment, cfg.smooth_hours),
        "edge_net_mae_final_vs_aligned": _masked_net_mae(final_segment, aligned_segment, edge_mask),
        "core_net_mae_final_vs_aligned": _masked_net_mae(final_segment, aligned_segment, core_mask),
        "edge_net_mae_final_vs_raw": _masked_net_mae(final_segment, raw_segment, edge_mask),
        "core_net_mae_final_vs_raw": _masked_net_mae(final_segment, raw_segment, core_mask),
        "final_core_net_mean": _masked_net_mean(final_segment, core_mask),
        "final_core_net_max": _masked_net_max(final_segment, core_mask),
        "final_edge_net_mean": _masked_net_mean(final_segment, edge_mask),
        "final_edge_net_max": _masked_net_max(final_segment, edge_mask),
    }


def _init_annual(background: pd.DataFrame) -> pd.DataFrame:
    annual = background.copy()
    annual["is_extreme"] = False
    annual["event_id"] = ""
    annual["event_type"] = ""
    annual["severity_level"] = pd.NA
    annual["extreme_prob"] = np.nan
    annual["is_core"] = False
    return annual


def _choose_variant_position(
    starts: np.ndarray,
    scores: np.ndarray,
    rng: np.random.Generator,
    cfg: AnnualEmbeddingConfig,
    strategy: str,
) -> tuple[int, float]:
    if strategy == "random":
        idx = int(rng.integers(0, len(starts)))
        return int(starts[idx]), float(scores[idx])
    if strategy == "vulnerability":
        return choose_position(starts, scores, rng, cfg.position_temperature)
    raise ValueError(f"Unknown position strategy: {strategy}")


def run_variant(
    background: pd.DataFrame,
    library: SegmentLibrary,
    plans: list[PlannedEvent],
    cfg: AnnualEmbeddingConfig,
    variant: AblationVariant,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_offset = 0 if variant.position_strategy == "vulnerability" else 100_000
    rng = np.random.default_rng(cfg.random_seed + seed_offset)
    values = background[CHANNELS].to_numpy(dtype=float, copy=True)
    annual = _init_annual(background)
    occupied = np.zeros(len(background), dtype=bool)
    imbalance_tau = float(np.quantile(net_load(values), cfg.imbalance_quantile))

    log_rows = []
    metric_rows = []
    inserted = 0
    skipped = 0
    for plan in plans:
        if inserted >= cfg.num_events:
            break
        chosen = pd.Series(plan.meta)
        month = int(chosen["month"])
        starts = candidate_positions(background, month, cfg.segment_hours, occupied)
        if starts.size == 0:
            skipped += 1
            continue
        scores = score_positions(values, starts, cfg.segment_hours, imbalance_tau)
        start, chosen_score = _choose_variant_position(starts, scores, rng, cfg, variant.position_strategy)
        end = start + cfg.segment_hours
        segment_id = plan.source_segment_id
        raw_segment = library.arrays[segment_id].astype(float, copy=True)
        background_window = values[start:end].copy()
        core_mask = library.core_masks.get(segment_id, _default_core_mask(cfg.segment_hours, cfg.smooth_hours)).astype(bool)
        raw_clipped, aligned_segment, final_segment = apply_boundary_mode(
            raw_segment, background_window, core_mask, cfg, variant.boundary_mode
        )
        event_id = _format_event_id(inserted + 1)
        metrics = event_metrics(values, start, raw_clipped, aligned_segment, final_segment, core_mask, cfg)

        values[start:end] = final_segment
        occupied[start:end] = True
        annual.loc[start : end - 1, "is_extreme"] = True
        annual.loc[start : end - 1, "event_id"] = event_id
        annual.loc[start : end - 1, "event_type"] = str(chosen["event_type"])
        annual.loc[start : end - 1, "severity_level"] = chosen["severity_level"]
        annual.loc[start : end - 1, "extreme_prob"] = _safe_value(chosen, "extreme_prob")
        annual.loc[start : end - 1, "is_core"] = core_mask

        log_rows.append(
            {
                "event_id": event_id,
                "start_time": annual.loc[start, "time"],
                "end_time": annual.loc[end - 1, "time"],
                "month": month,
                "event_type": str(chosen["event_type"]),
                "severity_level": chosen["severity_level"],
                "extreme_prob": _safe_value(chosen, "extreme_prob"),
                "cum_deficit": _safe_value(chosen, "cum_deficit"),
                "core_cum_deficit": _safe_value(chosen, "core_cum_deficit"),
                "netload_ramp_max": _safe_value(chosen, "netload_ramp_max"),
                "imbalance_duration": _safe_value(chosen, "imbalance_duration"),
                "max_imbalance_run": _safe_value(chosen, "max_imbalance_run"),
                "variant": variant.name,
                "boundary_mode": variant.boundary_mode,
                "position_strategy": variant.position_strategy,
                "source_segment_id": segment_id,
                "start_idx": start,
                "end_idx": end - 1,
                "requested_event_type": plan.target_event_type,
                "requested_severity_level": plan.target_severity_level,
                "fallback_level": plan.fallback_level,
                "vulnerability": chosen_score,
            }
        )
        metric_rows.append(
            {
                "event_id": event_id,
                "variant": variant.name,
                "boundary_mode": variant.boundary_mode,
                "position_strategy": variant.position_strategy,
                "source_segment_id": segment_id,
                "start_idx": start,
                "end_idx": end - 1,
                "month": month,
                "event_type": str(chosen["event_type"]),
                "severity_level": chosen["severity_level"],
                "vulnerability": chosen_score,
                **metrics,
            }
        )
        inserted += 1

    if inserted < cfg.num_events and not cfg.allow_partial:
        raise RuntimeError(
            f"Variant {variant.name} inserted {inserted}/{cfg.num_events}; skipped={skipped}. "
            "Re-run with --allow-partial to keep partial results."
        )

    annual[CHANNELS] = values
    annual["net_load"] = annual["load"] - annual["wind_power"] - annual["solar_power"]

    log_df = pd.DataFrame(log_rows)
    metric_df = pd.DataFrame(metric_rows)
    for col in LOG_COLUMNS:
        if col not in log_df.columns:
            log_df[col] = np.nan
    log_order = LOG_COLUMNS + [c for c in log_df.columns if c not in LOG_COLUMNS]
    return annual, log_df[log_order], metric_df


def summarize_variant(
    annual: pd.DataFrame,
    log_df: pd.DataFrame,
    event_metrics_df: pd.DataFrame,
    variant: AblationVariant,
) -> dict[str, float | int | str]:
    annual_net = annual["net_load"].to_numpy(dtype=float)
    is_extreme = annual["is_extreme"].fillna(False).astype(bool).to_numpy()
    is_core = annual["is_core"].fillna(False).astype(bool).to_numpy()
    extreme_net = annual_net[is_extreme]
    core_net = annual_net[is_core]
    edge_net = annual_net[is_extreme & ~is_core]

    def mean_col(col: str) -> float:
        return float(pd.to_numeric(event_metrics_df.get(col, pd.Series(dtype=float)), errors="coerce").mean())

    row = {
        "variant": variant.name,
        "boundary_mode": variant.boundary_mode,
        "position_strategy": variant.position_strategy,
        "inserted_events": int(len(log_df)),
        "extreme_hours": int(is_extreme.sum()),
        "core_hours": int(is_core.sum()),
        "annual_net_mean": float(np.mean(annual_net)),
        "annual_net_q95": float(np.quantile(annual_net, 0.95)),
        "annual_net_q99": float(np.quantile(annual_net, 0.99)),
        "annual_net_max": float(np.max(annual_net)),
        "annual_max_3h_ramp": max_3h_ramp(annual_net),
        "extreme_net_mean": float(np.mean(extreme_net)) if len(extreme_net) else np.nan,
        "extreme_net_max": float(np.max(extreme_net)) if len(extreme_net) else np.nan,
        "core_net_mean": float(np.mean(core_net)) if len(core_net) else np.nan,
        "core_net_max": float(np.max(core_net)) if len(core_net) else np.nan,
        "edge_net_mean": float(np.mean(edge_net)) if len(edge_net) else np.nan,
        "edge_net_max": float(np.max(edge_net)) if len(edge_net) else np.nan,
        "mean_position_vulnerability": float(pd.to_numeric(log_df.get("vulnerability", pd.Series(dtype=float)), errors="coerce").mean()),
        "mean_start_jump_raw": mean_col("start_jump_raw"),
        "mean_start_jump_final": mean_col("start_jump_final"),
        "mean_end_jump_raw": mean_col("end_jump_raw"),
        "mean_end_jump_final": mean_col("end_jump_final"),
        "mean_boundary_net_ramp_raw": mean_col("boundary_net_ramp_raw"),
        "mean_boundary_net_ramp_aligned": mean_col("boundary_net_ramp_aligned"),
        "mean_boundary_net_ramp_final": mean_col("boundary_net_ramp_final"),
        "mean_whole_net_ramp_raw": mean_col("whole_net_ramp_raw"),
        "mean_whole_net_ramp_final": mean_col("whole_net_ramp_final"),
        "mean_edge_net_mae_final_vs_aligned": mean_col("edge_net_mae_final_vs_aligned"),
        "mean_core_net_mae_final_vs_aligned": mean_col("core_net_mae_final_vs_aligned"),
        "mean_edge_net_mae_final_vs_raw": mean_col("edge_net_mae_final_vs_raw"),
        "mean_core_net_mae_final_vs_raw": mean_col("core_net_mae_final_vs_raw"),
    }
    raw_ramp = row["mean_boundary_net_ramp_raw"]
    aligned_ramp = row["mean_boundary_net_ramp_aligned"]
    final_ramp = row["mean_boundary_net_ramp_final"]
    row["boundary_ramp_reduction_pct_vs_raw"] = (
        100.0 * (raw_ramp - final_ramp) / max(abs(raw_ramp), 1e-12)
        if pd.notna(raw_ramp) and pd.notna(final_ramp)
        else np.nan
    )
    row["boundary_ramp_reduction_pct_vs_aligned"] = (
        100.0 * (aligned_ramp - final_ramp) / max(abs(aligned_ramp), 1e-12)
        if variant.boundary_mode != "raw" and pd.notna(aligned_ramp) and pd.notna(final_ramp)
        else np.nan
    )
    raw_jump = row["mean_start_jump_raw"] + row["mean_end_jump_raw"]
    final_jump = row["mean_start_jump_final"] + row["mean_end_jump_final"]
    row["boundary_jump_reduction_pct_vs_raw"] = (
        100.0 * (raw_jump - final_jump) / max(abs(raw_jump), 1e-12)
        if pd.notna(raw_jump) and pd.notna(final_jump)
        else np.nan
    )
    return row


def pairwise_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add_delta(name: str, left: str, right: str, cols: list[str]) -> None:
        if left not in set(summary["variant"]) or right not in set(summary["variant"]):
            return
        lrow = summary.set_index("variant").loc[left]
        rrow = summary.set_index("variant").loc[right]
        row = {"comparison": name, "left": left, "right": right}
        for col in cols:
            row[f"delta_{col}"] = float(lrow[col]) - float(rrow[col])
        rows.append(row)

    metric_cols = [
        "mean_boundary_net_ramp_final",
        "mean_start_jump_final",
        "mean_end_jump_final",
        "mean_core_net_mae_final_vs_aligned",
        "mean_position_vulnerability",
        "boundary_jump_reduction_pct_vs_raw",
        "boundary_ramp_reduction_pct_vs_aligned",
        "annual_net_q99",
        "annual_max_3h_ramp",
    ]
    add_delta("offset_minus_raw", "offset_vulnerability", "raw_vulnerability", metric_cols)
    add_delta("smooth_minus_offset", "smooth_vulnerability", "offset_vulnerability", metric_cols)
    add_delta("vulnerability_minus_random", "smooth_vulnerability", "smooth_random", metric_cols)
    return pd.DataFrame(rows)


def run_ablation(
    cfg: AnnualEmbeddingConfig,
    variants: list[AblationVariant],
    plan_multiplier: int,
) -> dict[str, str | int]:
    validate_config(cfg)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    background = load_background(cfg.background_file)
    library = load_extreme_library(cfg.extreme_file, cfg.condition_file, cfg.segment_hours, cfg.smooth_hours)
    if library.meta.empty:
        raise ValueError("No usable extreme segments were loaded.")

    plan_size = max(cfg.num_events * max(plan_multiplier, 1), cfg.num_events)
    plans = make_event_plan(library, cfg, plan_size)
    pd.DataFrame([event.__dict__ | {"meta": str(event.meta)} for event in plans]).to_csv(
        out_dir / "ablation_event_plan.csv", index=False, encoding="utf-8-sig"
    )

    summary_rows = []
    all_event_metrics = []
    for variant in variants:
        variant_dir = out_dir / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        annual, log_df, metric_df = run_variant(background, library, plans, cfg, variant)
        annual.to_csv(variant_dir / "annual_embedded_8760.csv", index=False, encoding="utf-8-sig")
        log_df.to_csv(variant_dir / "embedding_log.csv", index=False, encoding="utf-8-sig")
        metric_df.to_csv(variant_dir / "boundary_metrics.csv", index=False, encoding="utf-8-sig")
        all_event_metrics.append(metric_df)
        summary_rows.append(summarize_variant(annual, log_df, metric_df, variant))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    if all_event_metrics:
        pd.concat(all_event_metrics, ignore_index=True).to_csv(
            out_dir / "ablation_event_metrics.csv", index=False, encoding="utf-8-sig"
        )
    pairwise_deltas(summary).to_csv(out_dir / "ablation_pairwise_deltas.csv", index=False, encoding="utf-8-sig")
    return {"out_dir": str(out_dir), "variants": len(variants), "planned_events": len(plans)}


def parse_args() -> tuple[AnnualEmbeddingConfig, list[AblationVariant], int]:
    parser = argparse.ArgumentParser(description="Run annual embedding ablations for boundary handling and position selection.")
    parser.add_argument("--background-file", "--background", required=True, help="Path to background_8760.csv.")
    parser.add_argument("--extreme-file", "--segments", required=True, help="Path to generated extreme CSV.")
    parser.add_argument("--condition-file", "--cond", default=None, help="Optional per-segment label/condition CSV.")
    parser.add_argument("--out-dir", required=True, help="Output directory for ablation results.")
    parser.add_argument("--num-events", type=int, default=8, help="Number of events to insert per variant.")
    parser.add_argument("--segment-hours", type=int, default=36, help="Extreme segment length in hours.")
    parser.add_argument("--smooth-hours", type=int, default=4, help="Cosine smoothing hours at each boundary.")
    parser.add_argument("--random-seed", type=int, default=42, help="Seed for reproducible sampling.")
    parser.add_argument("--min-candidates", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--imbalance-quantile", type=float, default=0.75)
    parser.add_argument("--position-temperature", type=float, default=0.2)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--variants",
        default=",".join(variant.name for variant in DEFAULT_VARIANTS),
        help="Comma-separated variants such as raw_vulnerability,offset_vulnerability,smooth_vulnerability,smooth_random.",
    )
    parser.add_argument(
        "--plan-multiplier",
        type=int,
        default=5,
        help="Planned candidate events per requested event; larger values help when some monthly positions are unavailable.",
    )
    args = parser.parse_args()
    variants = [parse_variant(item) for item in args.variants.split(",") if item.strip()]
    if not variants:
        raise ValueError("At least one ablation variant is required.")
    cfg = AnnualEmbeddingConfig(
        background_file=args.background_file,
        extreme_file=args.extreme_file,
        condition_file=args.condition_file,
        out_dir=args.out_dir,
        num_events=args.num_events,
        segment_hours=args.segment_hours,
        smooth_hours=args.smooth_hours,
        random_seed=args.random_seed,
        min_candidates=args.min_candidates,
        max_attempts=args.max_attempts,
        imbalance_quantile=args.imbalance_quantile,
        position_temperature=args.position_temperature,
        allow_partial=args.allow_partial,
    )
    return cfg, variants, args.plan_multiplier


if __name__ == "__main__":
    config, ablation_variants, multiplier = parse_args()
    result = run_ablation(config, ablation_variants, multiplier)
    print(
        "annual embedding ablation complete: "
        f"variants={result['variants']}, "
        f"planned_events={result['planned_events']}, "
        f"out_dir={result['out_dir']}"
    )
