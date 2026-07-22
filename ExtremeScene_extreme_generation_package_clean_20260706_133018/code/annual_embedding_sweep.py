from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from annual_embedding import AnnualEmbeddingConfig, load_background, load_extreme_library, validate_config
from annual_embedding_ablation import AblationVariant, make_event_plan, run_variant, summarize_variant


def _parse_number_grid(value: str, cast):
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(cast(item))
    if not out:
        raise ValueError(f"Empty grid: {value!r}")
    return out


def _score_row(row: pd.Series) -> float:
    jump = float(row.get("boundary_jump_reduction_pct_vs_raw", 0.0))
    ramp_aligned = float(row.get("boundary_ramp_reduction_pct_vs_aligned", 0.0))
    ramp_raw = float(row.get("boundary_ramp_reduction_pct_vs_raw", 0.0))
    vulnerability = float(row.get("mean_position_vulnerability", 0.0))
    annual_ramp = float(row.get("annual_max_3h_ramp", 0.0))
    core_mae = float(row.get("mean_core_net_mae_final_vs_aligned", 0.0))

    # Prefer strong jump reduction, offset-induced ramp repair, and vulnerable positions.
    # Penalize settings that still worsen ramp against raw too much or perturb core hours.
    return (
        0.35 * jump
        + 0.30 * ramp_aligned
        + 5.0 * vulnerability
        - 0.20 * max(0.0, -ramp_raw)
        - 3.0 * annual_ramp
        - 20.0 * core_mae
    )


def run_sweep(
    base_cfg: AnnualEmbeddingConfig,
    smooth_grid: list[int],
    temperature_grid: list[float],
    out_dir: Path,
    variant_name: str,
    plan_multiplier: int,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    background = load_background(base_cfg.background_file)

    for smooth_hours in smooth_grid:
        cfg_for_library = AnnualEmbeddingConfig(**{**base_cfg.__dict__, "smooth_hours": smooth_hours})
        validate_config(cfg_for_library)
        library = load_extreme_library(
            cfg_for_library.extreme_file,
            cfg_for_library.condition_file,
            cfg_for_library.segment_hours,
            cfg_for_library.smooth_hours,
        )
        plan_size = max(cfg_for_library.num_events * max(plan_multiplier, 1), cfg_for_library.num_events)
        plans = make_event_plan(library, cfg_for_library, plan_size)

        for temperature in temperature_grid:
            cfg = AnnualEmbeddingConfig(
                **{
                    **cfg_for_library.__dict__,
                    "position_temperature": temperature,
                    "out_dir": str(out_dir),
                }
            )
            variant = AblationVariant(variant_name, "smooth", "vulnerability")
            annual, log_df, metric_df = run_variant(background, library, plans, cfg, variant)
            row = summarize_variant(annual, log_df, metric_df, variant)
            row["smooth_hours"] = smooth_hours
            row["position_temperature"] = temperature
            row["score"] = _score_row(pd.Series(row))
            rows.append(row)

    sweep = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    sweep.to_csv(out_dir / "sweep_summary.csv", index=False, encoding="utf-8-sig")
    return sweep


def parse_args() -> tuple[AnnualEmbeddingConfig, list[int], list[float], Path, str, int]:
    parser = argparse.ArgumentParser(description="Sweep annual embedding smooth_hours and position_temperature.")
    parser.add_argument("--background-file", "--background", required=True)
    parser.add_argument("--extreme-file", "--segments", required=True)
    parser.add_argument("--condition-file", "--cond", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-events", type=int, default=8)
    parser.add_argument("--segment-hours", type=int, default=36)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--min-candidates", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--imbalance-quantile", type=float, default=0.75)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--smooth-grid", default="4,6,8,10,12,14")
    parser.add_argument("--temperature-grid", default="0,0.2,0.35,0.6,1.0")
    parser.add_argument("--variant-name", default="smooth_vulnerability")
    parser.add_argument("--plan-multiplier", type=int, default=5)
    args = parser.parse_args()

    cfg = AnnualEmbeddingConfig(
        background_file=args.background_file,
        extreme_file=args.extreme_file,
        condition_file=args.condition_file,
        out_dir=args.out_dir,
        num_events=args.num_events,
        segment_hours=args.segment_hours,
        smooth_hours=4,
        random_seed=args.random_seed,
        min_candidates=args.min_candidates,
        max_attempts=args.max_attempts,
        imbalance_quantile=args.imbalance_quantile,
        position_temperature=0.2,
        allow_partial=args.allow_partial,
    )
    return (
        cfg,
        _parse_number_grid(args.smooth_grid, int),
        _parse_number_grid(args.temperature_grid, float),
        Path(args.out_dir),
        args.variant_name,
        args.plan_multiplier,
    )


if __name__ == "__main__":
    config, smooth_values, temperature_values, output_dir, name, multiplier = parse_args()
    result = run_sweep(config, smooth_values, temperature_values, output_dir, name, multiplier)
    best = result.iloc[0]
    print(
        "annual embedding sweep complete: "
        f"best_smooth_hours={int(best['smooth_hours'])}, "
        f"best_position_temperature={float(best['position_temperature']):.3g}, "
        f"score={float(best['score']):.3f}, "
        f"out_dir={output_dir}"
    )
