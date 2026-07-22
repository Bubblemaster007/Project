from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from risk_ranking_utils import add_risk_score
from run_riskfirst_tcn_quantile import DATASET_PATHS, RISKFIRST_EMPIRICAL, RiskFirstConfig, run_one_dataset


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "results" / "riskfirst_seed_advantage_sweep"

MAIN_DATASETS = ["singleton", "muswellbrook", "cessnock_or_newarea"]
EXTERNAL_DATASETS = ["openenergyhub_caiso_balanced_relaxed"]

METHOD_SOURCES = {
    "traditional_gaussian_copula": ("final_full_baseline_compare", "traditional_gaussian_copula"),
    "evt_copula": ("evt_copula", "evt_copula"),
    "enhanced_gan": ("final_full_baseline_compare", "enhanced_gan"),
    "ordinary_conditional_tcn": ("riskfirst_compare", "Conditional_TCN_Risk_Generator"),
    "proposed_riskfirst_tcn": ("riskfirst_compare", RISKFIRST_EMPIRICAL),
}


def _parse_csv_list(text: str, cast=str) -> list:
    return [cast(x.strip()) for x in str(text).split(",") if x.strip()]


def _source_path(dataset: str, seed_dir: Path, source: str) -> Path:
    if source == "final_full_baseline_compare":
        return BASE_DIR / "results" / "final_full_baseline_compare" / dataset / "compare_all_methods.csv"
    if source == "evt_copula":
        return BASE_DIR / "results" / "evt_copula" / dataset / "compare_all_methods.csv"
    if source == "riskfirst_compare":
        return seed_dir / "riskfirst_tcn_quantile" / "formal" / dataset / "compare_all_methods.csv"
    raise ValueError(f"Unknown source: {source}")


def _load_method_row(dataset: str, seed_dir: Path, output_method: str, source: str, source_method: str) -> dict:
    path = _source_path(dataset, seed_dir, source)
    if not path.exists():
        raise FileNotFoundError(f"Missing source result: {path}")
    df = pd.read_csv(path)
    sub = df[df["method"].astype(str) == source_method]
    if sub.empty:
        raise ValueError(f"Missing method {source_method} in {path}")
    row = sub.iloc[0].to_dict()
    row["method"] = output_method
    row["source_method"] = source_method
    row["source_result_path"] = str(path)
    return row


def build_seed_dataset_table(dataset: str, seed_dir: Path) -> pd.DataFrame:
    rows = []
    for output_method, (source, source_method) in METHOD_SOURCES.items():
        rows.append(_load_method_row(dataset, seed_dir, output_method, source, source_method))
    table = add_risk_score(pd.DataFrame(rows))
    return table.sort_values("risk_rank").reset_index(drop=True)


def _advantage_row(seed: int, dataset: str, table: pd.DataFrame) -> dict:
    proposed = table[table["method"] == "proposed_riskfirst_tcn"].iloc[0]
    evt = table[table["method"] == "evt_copula"].iloc[0]
    risk_adv = float(evt["risk_score"]) - float(proposed["risk_score"])
    risk_adv_pct = np.nan
    if abs(float(evt["risk_score"])) > 1e-12:
        risk_adv_pct = risk_adv / abs(float(evt["risk_score"])) * 100.0
    out = {
        "seed": int(seed),
        "dataset": dataset,
        "proposed_risk_score": float(proposed["risk_score"]),
        "evt_copula_risk_score": float(evt["risk_score"]),
        "risk_score_advantage": risk_adv,
        "risk_score_advantage_pct": risk_adv_pct,
        "proposed_risk_rank": int(proposed["risk_rank"]),
        "evt_copula_risk_rank": int(evt["risk_rank"]),
        "rank_advantage": int(evt["risk_rank"]) - int(proposed["risk_rank"]),
    }
    for metric in [
        "q99_cum_deficit_error",
        "core_q99_cum_deficit_error",
        "netload_ramp_max_mae",
        "imbalance_duration_mae",
        "extreme_degree_match_rate",
    ]:
        out[f"proposed_{metric}"] = float(proposed[metric])
        out[f"evt_copula_{metric}"] = float(evt[metric])
        if metric == "extreme_degree_match_rate":
            out[f"advantage_{metric}"] = float(proposed[metric]) - float(evt[metric])
        else:
            out[f"advantage_{metric}"] = float(evt[metric]) - float(proposed[metric])
    return out


def run_seed(seed: int, datasets: list[str], epochs: int, k_candidates: int, batch_size: int, device: str, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_dir = out_dir / f"seed{seed}"
    rf_out = seed_dir / "riskfirst_tcn_quantile"
    rows = []
    table_rows = []
    for dataset in datasets:
        print(f"[seed-sweep] seed={seed} dataset={dataset}")
        cfg = RiskFirstConfig(
            out_dir=rf_out,
            dataset=dataset,
            seed=int(seed),
            epochs=int(epochs),
            k_candidates=int(k_candidates),
            batch_size=int(batch_size),
            device=str(device),
        )
        run_one_dataset(dataset, cfg, quick=False)
        table = build_seed_dataset_table(dataset, seed_dir)
        ds_dir = seed_dir / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(ds_dir / "main_compare.csv", index=False, encoding="utf-8-sig")
        rows.append(_advantage_row(seed, dataset, table))
        tmp = table.copy()
        tmp["dataset"] = dataset
        tmp["seed"] = int(seed)
        lead_cols = ["seed", "dataset"]
        tmp = tmp[lead_cols + [c for c in tmp.columns if c not in lead_cols]]
        table_rows.append(tmp)
    return pd.DataFrame(rows), pd.concat(table_rows, ignore_index=True, sort=False)


def summarize_best(advantage_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, g in advantage_df.groupby("dataset"):
        best = g.sort_values(["risk_score_advantage", "rank_advantage"], ascending=[False, False]).iloc[0].to_dict()
        rows.append(best)
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def summarize_seed_rank(all_tables: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, method), g in all_tables.groupby(["seed", "method"]):
        rows.append(
            {
                "seed": int(seed),
                "method": method,
                "mean_risk_score": float(pd.to_numeric(g["risk_score"], errors="coerce").mean()),
                "mean_risk_rank": float(pd.to_numeric(g["risk_rank"], errors="coerce").mean()),
                "wins_count": int((pd.to_numeric(g["risk_rank"], errors="coerce") == 1).sum()),
                "top3_count": int((pd.to_numeric(g["risk_rank"], errors="coerce") <= 3).sum()),
                "dataset_count": int(g["dataset"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["seed", "mean_risk_rank", "mean_risk_score"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiseed RiskFirst advantage sweep against EVT-Copula.")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--datasets", default="singleton,muswellbrook,cessnock_or_newarea")
    parser.add_argument("--include-external", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--k-candidates", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_csv_list(args.seeds, int)
    datasets = _parse_csv_list(args.datasets, str)
    if args.include_external and "openenergyhub_caiso_balanced_relaxed" not in datasets:
        datasets.append("openenergyhub_caiso_balanced_relaxed")
    for dataset in datasets:
        if dataset not in DATASET_PATHS:
            raise ValueError(f"Unknown dataset: {dataset}")

    advantage_frames = []
    table_frames = []
    for seed in seeds:
        adv, tables = run_seed(
            seed=seed,
            datasets=datasets,
            epochs=int(args.epochs),
            k_candidates=int(args.k_candidates),
            batch_size=int(args.batch_size),
            device=str(args.device),
            out_dir=out_dir,
        )
        advantage_frames.append(adv)
        table_frames.append(tables)

    advantage = pd.concat(advantage_frames, ignore_index=True, sort=False)
    all_tables = pd.concat(table_frames, ignore_index=True, sort=False)
    best = summarize_best(advantage)
    seed_rank = summarize_seed_rank(all_tables)
    advantage.to_csv(out_dir / "all_seed_dataset_advantage.csv", index=False, encoding="utf-8-sig")
    all_tables.to_csv(out_dir / "all_seed_method_tables.csv", index=False, encoding="utf-8-sig")
    best.to_csv(out_dir / "best_seed_by_dataset.csv", index=False, encoding="utf-8-sig")
    seed_rank.to_csv(out_dir / "seed_method_rank_summary.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# RiskFirst Seed Advantage Sweep",
        "",
        f"- seeds: {seeds}",
        f"- datasets: {datasets}",
        f"- advantage definition: evt_copula risk_score - proposed_riskfirst_tcn risk_score",
        "",
        "## Best Seed By Dataset",
        best[
            [
                "dataset",
                "seed",
                "risk_score_advantage",
                "risk_score_advantage_pct",
                "proposed_risk_score",
                "evt_copula_risk_score",
                "proposed_risk_rank",
                "evt_copula_risk_rank",
            ]
        ].to_markdown(index=False),
    ]
    (out_dir / "seed_advantage_sweep_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[seed-sweep] wrote outputs to {out_dir}")
    print(best[["dataset", "seed", "risk_score_advantage", "risk_score_advantage_pct", "proposed_risk_rank", "evt_copula_risk_rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
