from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_generation import EvalConfig, evaluate_generation
from risk_ranking_utils import add_risk_score
from run_month_evt_copula_risk_selection import MonthEvtCopulaConfig, _compute_monthly_tau, _load_split
from run_riskfirst_tcn_quantile import DATASET_PATHS, _ensure_condition, _max_imbalance_run_for_samples


BASE_DIR = Path(__file__).resolve().parent

DATASETS = ["singleton", "muswellbrook", "cessnock_or_newarea"]

METHOD_SOURCES = {
    "traditional_gaussian_copula": lambda ds, riskfirst_dir: BASE_DIR
    / "results"
    / "tailweighted_month_evt_copula"
    / ds
    / "generated_samples_traditional_gaussian_copula.npy",
    # 该行沿用用户早期命名；底层文件来自 EVT/extreme_prob 分层 Copula。
    "extreme_conditioned_gaussian_copula": lambda ds, riskfirst_dir: BASE_DIR
    / "results"
    / "evt_copula"
    / ds
    / "generated_samples_evt_copula.npy",
    "enhanced_gan": lambda ds, riskfirst_dir: BASE_DIR
    / "results"
    / "tailweighted_month_evt_copula"
    / ds
    / "generated_samples_enhanced_gan.npy",
    "ordinary_conditional_tcn": lambda ds, riskfirst_dir: BASE_DIR
    / "results"
    / "expanded_model_pool_formal50"
    / ds
    / "generated_samples_Conditional_TCN_Risk_Generator.npy",
    "RiskFirst_TCN_Empirical": lambda ds, riskfirst_dir: riskfirst_dir
    / "formal"
    / ds
    / "generated_samples_RiskFirst_TCN_Empirical.npy",
    "DurationAware_RiskFirst_TCN_Empirical": lambda ds, riskfirst_dir: riskfirst_dir
    / "formal"
    / ds
    / "generated_samples_DurationAware_RiskFirst_TCN_Empirical.npy",
    "DurationAware_Calibrated_RiskFirst_TCN_Empirical": lambda ds, riskfirst_dir: riskfirst_dir
    / "formal"
    / ds
    / "generated_samples_DurationAware_Calibrated_RiskFirst_TCN_Empirical.npy",
}


def _load_cond(data_dir: Path, split: str) -> pd.DataFrame:
    cond = pd.read_csv(data_dir / f"cond_{split}.csv")
    meta_path = data_dir / f"meta_{split}.csv"
    meta = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame(index=np.arange(len(cond)))
    return _ensure_condition(cond, meta)


def _max_run_mae(data_dir: Path, gen_path: Path, tau_by_month: dict[int, float], split: str = "test") -> float:
    real = np.load(data_dir / f"X_{split}.npy")
    gen = np.load(gen_path)
    cond = _load_cond(data_dir, split)
    months = pd.to_numeric(cond.get("month", 1), errors="coerce").fillna(1).astype(int).to_numpy()
    real_run = _max_imbalance_run_for_samples(real, months, tau_by_month, 1.0)
    gen_run = _max_imbalance_run_for_samples(gen, months, tau_by_month, 1.0)
    return float(np.mean(np.abs(gen_run - real_run)))


def _compute_tau(dataset: str, data_dir: Path, out_dir: Path) -> dict[int, float]:
    x_train, cond_train_raw, meta_train, _ = _load_split(data_dir, "train")
    cond_train = _ensure_condition(cond_train_raw, meta_train)
    cfg = MonthEvtCopulaConfig(out_dir=out_dir, tau_quantile=0.75, delta_t_hours=1.0, seed=42)
    tau_by_month, tau_df = _compute_monthly_tau(x_train, cond_train, cfg, dataset, out_dir)
    tau_df.to_csv(out_dir / "monthly_tau_summary.csv", index=False, encoding="utf-8-sig")
    return tau_by_month


def evaluate_method(dataset: str, method: str, gen_path: Path, data_dir: Path, out_dir: Path, tau_by_month: dict[int, float]) -> dict:
    eval_dir = out_dir / "evaluations" / method
    summary = evaluate_generation(
        EvalConfig(
            real=str(data_dir / "X_test.npy"),
            generated=str(gen_path),
            cond=str(data_dir / "cond_test.csv"),
            meta=str(data_dir / "meta_test.csv"),
            out_dir=str(eval_dir),
            model_name=method,
            event_mask=str(data_dir / "event_mask_test.npy") if (data_dir / "event_mask_test.npy").exists() else None,
            ramp_metric_mode="window_3h",
            ramp_window_hours=3.0,
        )
    )
    metrics = dict(summary.get("metrics", {}))
    metrics["dataset"] = dataset
    metrics["method"] = method
    metrics["max_imbalance_run_mae"] = _max_run_mae(data_dir, gen_path, tau_by_month, split="test")
    metrics["generated_path"] = str(gen_path)
    return metrics


def build_compare(
    riskfirst_dir: Path = BASE_DIR / "results" / "duration_aware_riskfirst",
    out_dir: Path = BASE_DIR / "results" / "duration_aware_riskfirst_compare",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[pd.DataFrame] = []
    status_rows: list[dict] = []
    for dataset in DATASETS:
        data_dir = DATASET_PATHS[dataset]
        ds_out = out_dir / dataset
        ds_out.mkdir(parents=True, exist_ok=True)
        tau_by_month = _compute_tau(dataset, data_dir, ds_out)
        rows = []
        for method, fn in METHOD_SOURCES.items():
            gen_path = fn(dataset, riskfirst_dir)
            if not gen_path.exists():
                status_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "status": "missing_generated_samples",
                        "generated_path": str(gen_path),
                    }
                )
                continue
            try:
                row = evaluate_method(dataset, method, gen_path, data_dir, ds_out, tau_by_month)
                rows.append(row)
                status_rows.append({"dataset": dataset, "method": method, "status": "completed", "generated_path": str(gen_path)})
            except Exception as exc:  # noqa: BLE001
                status_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "status": "failed",
                        "generated_path": str(gen_path),
                        "error": str(exc),
                    }
                )
        compare = add_risk_score(pd.DataFrame(rows))
        wanted = [
            "dataset",
            "method",
            "q99_cum_deficit_error",
            "core_q99_cum_deficit_error",
            "netload_ramp_max_mae",
            "imbalance_duration_mae",
            "max_imbalance_run_mae",
            "extreme_degree_match_rate",
            "risk_score",
            "risk_rank",
            "generated_path",
        ]
        compare = compare[[c for c in wanted if c in compare.columns] + [c for c in compare.columns if c not in wanted]]
        compare.to_csv(ds_out / f"main_compare_{dataset}.csv", index=False, encoding="utf-8-sig")
        compare.to_csv(out_dir / f"main_compare_{dataset}.csv", index=False, encoding="utf-8-sig")
        all_rows.append(compare)

    if all_rows:
        long_df = pd.concat(all_rows, ignore_index=True, sort=False)
        long_df.to_csv(out_dir / "duration_aware_all_datasets_long.csv", index=False, encoding="utf-8-sig")
        _write_duration_improvement(long_df, out_dir)
        summary_rows = []
        for method, g in long_df.groupby("method"):
            ranks = pd.to_numeric(g["risk_rank"], errors="coerce")
            summary_rows.append(
                {
                    "method": method,
                    "mean_risk_score": float(pd.to_numeric(g["risk_score"], errors="coerce").mean()),
                    "mean_risk_rank": float(ranks.mean()),
                    "wins_count": int((ranks == 1).sum()),
                    "top3_count": int((ranks <= 3).sum()),
                    "available_dataset_count": int(g["dataset"].nunique()),
                    "avg_q99_cum_deficit_error": float(pd.to_numeric(g["q99_cum_deficit_error"], errors="coerce").mean()),
                    "avg_core_q99_cum_deficit_error": float(pd.to_numeric(g["core_q99_cum_deficit_error"], errors="coerce").mean()),
                    "avg_netload_ramp_max_mae": float(pd.to_numeric(g["netload_ramp_max_mae"], errors="coerce").mean()),
                    "avg_imbalance_duration_mae": float(pd.to_numeric(g["imbalance_duration_mae"], errors="coerce").mean()),
                    "avg_max_imbalance_run_mae": float(pd.to_numeric(g["max_imbalance_run_mae"], errors="coerce").mean()),
                    "avg_extreme_degree_match_rate": float(pd.to_numeric(g["extreme_degree_match_rate"], errors="coerce").mean()),
                }
            )
        summary = pd.DataFrame(summary_rows).sort_values(["mean_risk_rank", "mean_risk_score"])
        summary.to_csv(out_dir / "main_compare_summary.csv", index=False, encoding="utf-8-sig")
        _write_report(out_dir, long_df, summary)
    pd.DataFrame(status_rows).to_csv(out_dir / "method_run_status.csv", index=False, encoding="utf-8-sig")


def _write_duration_improvement(long_df: pd.DataFrame, out_dir: Path) -> None:
    metrics_lower = [
        "q99_cum_deficit_error",
        "core_q99_cum_deficit_error",
        "netload_ramp_max_mae",
        "imbalance_duration_mae",
        "max_imbalance_run_mae",
        "risk_score",
    ]
    rows = []
    for dataset, g in long_df.groupby("dataset"):
        base = g[g["method"] == "RiskFirst_TCN_Empirical"]
        new = g[g["method"] == "DurationAware_RiskFirst_TCN_Empirical"]
        if base.empty or new.empty:
            continue
        base = base.iloc[0]
        new = new.iloc[0]
        row = {"dataset": dataset}
        for col in metrics_lower:
            b = float(base[col])
            n = float(new[col])
            row[f"{col}_baseline"] = b
            row[f"{col}_durationaware"] = n
            row[f"{col}_improvement_pct"] = (b - n) / max(abs(b), 1e-9) * 100.0
        b_match = float(base.get("extreme_degree_match_rate", np.nan))
        n_match = float(new.get("extreme_degree_match_rate", np.nan))
        row["extreme_degree_match_rate_baseline"] = b_match
        row["extreme_degree_match_rate_durationaware"] = n_match
        row["extreme_degree_match_rate_improvement_pct"] = (n_match - b_match) / max(abs(b_match), 1e-9) * 100.0
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        avg = {"dataset": "average"}
        for col in df.columns:
            if col != "dataset":
                avg[col] = float(pd.to_numeric(df[col], errors="coerce").mean())
        df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    df.to_csv(out_dir / "durationaware_vs_riskfirst_improvement.csv", index=False, encoding="utf-8-sig")


def _write_report(out_dir: Path, long_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# DurationAware RiskFirst TCN 对比报告",
        "",
        "## 评价设置",
        "- ramp 仍采用 3h window 最大正向净负荷爬坡。",
        "- 主 risk_score 仍由 q99/core_q99/ramp/duration 四项误差归一化加权得到。",
        "- 额外输出 max_imbalance_run_mae，仅作为持续性风险诊断指标，不参与 risk_score。",
        "",
        "## 三数据集平均汇总",
        summary.to_string(index=False),
        "",
        "## DurationAware vs RiskFirst_TCN_Empirical",
    ]
    pivot = long_df[
        long_df["method"].isin(
            [
                "RiskFirst_TCN_Empirical",
                "DurationAware_RiskFirst_TCN_Empirical",
                "DurationAware_Calibrated_RiskFirst_TCN_Empirical",
            ]
        )
    ].copy()
    if len(pivot):
        lines.append(pivot[[
            "dataset",
            "method",
            "q99_cum_deficit_error",
            "core_q99_cum_deficit_error",
            "netload_ramp_max_mae",
            "imbalance_duration_mae",
            "max_imbalance_run_mae",
            "risk_score",
            "risk_rank",
        ]].to_string(index=False))
    (out_dir / "duration_aware_riskfirst_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    build_compare()


if __name__ == "__main__":
    main()
