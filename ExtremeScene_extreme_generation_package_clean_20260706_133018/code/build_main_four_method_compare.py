from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from risk_ranking_utils import add_risk_score


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "results" / "main_four_method_compare"

# 论文正文主表使用三组 Ausgrid/AEMO/NASA 数据集；CAISO/OpenEnergyHub 单独作为外部验证表输出。
DATASETS = ["singleton", "muswellbrook", "cessnock_or_newarea"]
EXTERNAL_DATASETS = ["openenergyhub_caiso_balanced_relaxed"]

MAIN_METRICS = [
    "q99_cum_deficit_error",
    "core_q99_cum_deficit_error",
    "netload_ramp_max_mae",
    "imbalance_duration_mae",
]

OUTPUT_COLUMNS = [
    "method",
    *MAIN_METRICS,
    "extreme_degree_match_rate",
    "risk_score",
    "risk_rank",
    "source_method",
    "source_result_path",
]


METHOD_MAP = {
    # 传统统计相关性基线。
    "traditional_gaussian_copula": {
        "source": "traditional_gaussian_copula",
        "source_dir": "final_full_baseline_compare",
    },
    # 新增中间基线：传统 Copula + EVT extreme_prob 极端程度分层。
    "evt_copula": {
        "source": "evt_copula",
        "source_dir": "evt_copula",
    },
    # 已有深度生成对照。
    "enhanced_gan": {
        "source": "enhanced_gan",
        "source_dir": "final_full_baseline_compare",
    },
    # 普通条件 TCN：只使用 month/event_type/extreme_prob 等条件，不使用 R_target。
    "ordinary_conditional_tcn": {
        "source": "Conditional_TCN_Risk_Generator",
        "source_dir": "riskfirst_tcn_quantile",
    },
    # Proposed：当前综合最稳的 RiskFirst_TCN_Empirical。
    "proposed_riskfirst_tcn": {
        "source": "RiskFirst_TCN_Empirical",
        "source_dir": "riskfirst_tcn_quantile",
    },
}


def _source_path(dataset: str, source_dir: str) -> Path:
    if source_dir == "final_full_baseline_compare":
        return BASE_DIR / "results" / source_dir / dataset / "compare_all_methods.csv"
    if source_dir == "evt_copula":
        return BASE_DIR / "results" / source_dir / dataset / "compare_all_methods.csv"
    if source_dir == "riskfirst_tcn_quantile":
        return BASE_DIR / "results" / source_dir / "formal" / dataset / "risk_main_compare.csv"
    raise ValueError(f"Unknown source_dir: {source_dir}")


def _load_source_row(dataset: str, method_name: str, source_dir: str) -> dict:
    path = _source_path(dataset, source_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing source result: {path}")
    df = pd.read_csv(path)
    if "method" not in df.columns:
        raise ValueError(f"Missing method column in {path}")
    sub = df[df["method"].astype(str) == method_name]
    if sub.empty:
        raise ValueError(f"Missing method {method_name} in {path}")
    row = sub.iloc[0].to_dict()
    row["source_method"] = method_name
    row["source_result_path"] = str(path)
    return row


def build_dataset_table(dataset: str) -> pd.DataFrame:
    rows = []
    for output_method, spec in METHOD_MAP.items():
        row = _load_source_row(dataset, spec["source"], spec["source_dir"])
        row["method"] = output_method
        rows.append(row)
    df = pd.DataFrame(rows)
    # 在当前五方法集合内重新计算 risk_score/risk_rank，避免沿用旧大表的归一化范围。
    df = add_risk_score(df)
    cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    return df[cols].sort_values("risk_rank").reset_index(drop=True)


def build_summary(dataset_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for method in METHOD_MAP:
        method_rows = []
        for dataset, table in dataset_tables.items():
            row = table[table["method"] == method].iloc[0].to_dict()
            row["dataset"] = dataset
            method_rows.append(row)
        g = pd.DataFrame(method_rows)
        rows.append(
            {
                "method": method,
                "mean_risk_score": float(pd.to_numeric(g["risk_score"], errors="coerce").mean()),
                "mean_risk_rank": float(pd.to_numeric(g["risk_rank"], errors="coerce").mean()),
                "wins_count": int((pd.to_numeric(g["risk_rank"], errors="coerce") == 1).sum()),
                "top3_count": int((pd.to_numeric(g["risk_rank"], errors="coerce") <= 3).sum()),
                "avg_q99_cum_deficit_error": float(pd.to_numeric(g["q99_cum_deficit_error"], errors="coerce").mean()),
                "avg_core_q99_cum_deficit_error": float(pd.to_numeric(g["core_q99_cum_deficit_error"], errors="coerce").mean()),
                "avg_netload_ramp_max_mae": float(pd.to_numeric(g["netload_ramp_max_mae"], errors="coerce").mean()),
                "avg_imbalance_duration_mae": float(pd.to_numeric(g["imbalance_duration_mae"], errors="coerce").mean()),
                "avg_extreme_degree_match_rate": float(pd.to_numeric(g["extreme_degree_match_rate"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_risk_rank", "mean_risk_score"]).reset_index(drop=True)


def _improvement_pct(baseline: float, proposed: float, smaller_is_better: bool = True) -> float:
    if not np.isfinite(baseline) or abs(baseline) < 1e-12:
        return np.nan
    if smaller_is_better:
        return float((baseline - proposed) / abs(baseline) * 100.0)
    return float((proposed - baseline) / abs(baseline) * 100.0)


def build_pairwise_improvement(
    dataset_tables: dict[str, pd.DataFrame],
    baseline_method: str,
    proposed_method: str,
) -> pd.DataFrame:
    """计算 proposed_method 相对 baseline_method 的百分比提升。

    q99/core/ramp/duration/risk_score 为误差，越小越好；
    extreme_degree_match_rate 为匹配率，越大越好。
    """

    rows = []
    for dataset, table in dataset_tables.items():
        baseline = table[table["method"] == baseline_method].iloc[0]
        proposed = table[table["method"] == proposed_method].iloc[0]
        row = {
            "dataset": dataset,
            "baseline_method": baseline_method,
            "proposed_method": proposed_method,
        }
        for metric in MAIN_METRICS:
            base = float(baseline[metric])
            prop = float(proposed[metric])
            row[f"baseline_{metric}"] = base
            row[f"proposed_{metric}"] = prop
            row[f"improvement_pct_{metric}"] = _improvement_pct(base, prop, smaller_is_better=True)
        row["baseline_extreme_degree_match_rate"] = float(baseline["extreme_degree_match_rate"])
        row["proposed_extreme_degree_match_rate"] = float(proposed["extreme_degree_match_rate"])
        row["improvement_pct_extreme_degree_match_rate"] = _improvement_pct(
            float(baseline["extreme_degree_match_rate"]),
            float(proposed["extreme_degree_match_rate"]),
            smaller_is_better=False,
        )
        row["baseline_risk_score"] = float(baseline["risk_score"])
        row["proposed_risk_score"] = float(proposed["risk_score"])
        row["improvement_pct_risk_score"] = _improvement_pct(float(baseline["risk_score"]), float(proposed["risk_score"]), True)
        rows.append(row)

    df = pd.DataFrame(rows)
    avg_row = {"dataset": "average", "baseline_method": baseline_method, "proposed_method": proposed_method}
    for col in df.columns:
        if col.startswith("improvement_pct_"):
            avg_row[col] = float(pd.to_numeric(df[col], errors="coerce").mean())
    return pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_tables: dict[str, pd.DataFrame] = {}
    external_tables: dict[str, pd.DataFrame] = {}
    for dataset in [*DATASETS, *EXTERNAL_DATASETS]:
        table = build_dataset_table(dataset)
        table.to_csv(OUT_DIR / f"main_compare_{dataset}.csv", index=False, encoding="utf-8-sig")
        if dataset in DATASETS:
            dataset_tables[dataset] = table
        else:
            external_tables[dataset] = table

    summary = build_summary(dataset_tables)
    summary.to_csv(OUT_DIR / "main_compare_summary.csv", index=False, encoding="utf-8-sig")
    if external_tables:
        build_summary({**dataset_tables, **external_tables}).to_csv(
            OUT_DIR / "main_compare_summary_with_external.csv",
            index=False,
            encoding="utf-8-sig",
        )
    build_pairwise_improvement(
        dataset_tables,
        "ordinary_conditional_tcn",
        "proposed_riskfirst_tcn",
    ).to_csv(OUT_DIR / "proposed_vs_ordinary_tcn_improvement.csv", index=False, encoding="utf-8-sig")
    build_pairwise_improvement(
        dataset_tables,
        "traditional_gaussian_copula",
        "evt_copula",
    ).to_csv(OUT_DIR / "evt_copula_vs_traditional_improvement.csv", index=False, encoding="utf-8-sig")
    build_pairwise_improvement(
        dataset_tables,
        "evt_copula",
        "proposed_riskfirst_tcn",
    ).to_csv(OUT_DIR / "proposed_vs_evt_copula_improvement.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote outputs to {OUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
