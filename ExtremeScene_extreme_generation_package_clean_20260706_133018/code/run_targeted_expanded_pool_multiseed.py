from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from risk_ranking_utils import add_risk_score
from run_expanded_model_pool_experiments import (
    ExpandedModelPoolConfig,
    RISK_KNN_METHOD,
    TCN_METHOD,
    run_all as run_expanded_pool,
)
from run_gan_augmented_tailweighted_copula import METHOD_NAME as GAN_METHOD
from run_model_pool_experiments import (
    FLOW_METHOD,
    TRANSFORMER_METHOD,
    VAE_METHOD,
    ModelPoolConfig,
    run_all as run_original_pool,
)
from run_tailweighted_month_evt_copula import TAIL_FIXED_METHOD


BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_MARGIN_METHOD = "ValSelected_Model_Ensemble_Margin_0.05"
EXPANDED_MARGIN_METHOD = "Expanded_ValSelected_Model_Ensemble_Margin_0.05"
TARGET_METHODS = [
    TAIL_FIXED_METHOD,
    ORIGINAL_MARGIN_METHOD,
    TCN_METHOD,
    RISK_KNN_METHOD,
    EXPANDED_MARGIN_METHOD,
]


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _risk_rows_for_seed(seed_dir: Path) -> pd.DataFrame:
    original = _read_csv_if_exists(seed_dir / "original_pool" / "all_datasets_risk_summary.csv")
    expanded = _read_csv_if_exists(seed_dir / "expanded_pool" / "all_datasets_risk_summary.csv")
    rows = []
    if not original.empty:
        sub = original[original["method"].astype(str).eq(ORIGINAL_MARGIN_METHOD)].copy()
        sub["source_pool"] = "original_pool"
        rows.append(sub)
    if not expanded.empty:
        keep = [TAIL_FIXED_METHOD, TCN_METHOD, RISK_KNN_METHOD, EXPANDED_MARGIN_METHOD]
        sub = expanded[expanded["method"].astype(str).isin(keep)].copy()
        sub["source_pool"] = "expanded_pool"
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _aux_rows_for_seed(seed_dir: Path) -> pd.DataFrame:
    original = _read_csv_if_exists(seed_dir / "original_pool" / "all_datasets_auxiliary_summary.csv")
    expanded = _read_csv_if_exists(seed_dir / "expanded_pool" / "all_datasets_auxiliary_summary.csv")
    rows = []
    if not original.empty:
        sub = original[original["method"].astype(str).eq(ORIGINAL_MARGIN_METHOD)].copy()
        sub["source_pool"] = "original_pool"
        rows.append(sub)
    if not expanded.empty:
        keep = [TAIL_FIXED_METHOD, TCN_METHOD, RISK_KNN_METHOD, EXPANDED_MARGIN_METHOD]
        sub = expanded[expanded["method"].astype(str).isin(keep)].copy()
        sub["source_pool"] = "expanded_pool"
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _rank_summary(risk_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    datasets = risk_df["dataset"].astype(str).unique().tolist()
    for method in risk_df["method"].astype(str).unique().tolist():
        row = {"method": method}
        ranks, scores = [], []
        for dataset in datasets:
            sub = risk_df[(risk_df["dataset"].astype(str) == dataset) & (risk_df["method"].astype(str) == method)]
            if sub.empty:
                row[f"{dataset}_risk_rank"] = np.nan
                row[f"{dataset}_risk_score"] = np.nan
                continue
            rank = float(sub["risk_rank"].iloc[0])
            score = float(sub["risk_score"].iloc[0])
            row[f"{dataset}_risk_rank"] = rank
            row[f"{dataset}_risk_score"] = score
            ranks.append(rank)
            scores.append(score)
        row["mean_risk_rank"] = float(np.mean(ranks)) if ranks else np.nan
        row["mean_risk_score"] = float(np.mean(scores)) if scores else np.nan
        row["wins_count"] = int(sum(rank == 1.0 for rank in ranks))
        row["top3_count"] = int(sum(rank <= 3.0 for rank in ranks))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mean_risk_rank", "mean_risk_score"], na_position="last")


def _recompute_seed_tables(seed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_risk = _risk_rows_for_seed(seed_dir)
    if raw_risk.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    ranked_parts = []
    for dataset, group in raw_risk.groupby("dataset", sort=False):
        ranked = add_risk_score(group.drop(columns=["risk_score", "risk_rank"], errors="ignore"))
        if "dataset" in ranked.columns:
            # 中文注释：部分上游汇总已经保留数据集列，这里只校正取值，避免重复插入导致中断。
            ranked["dataset"] = dataset
        else:
            ranked.insert(0, "dataset", dataset)
        ranked_parts.append(ranked)
    risk = pd.concat(ranked_parts, ignore_index=True)
    risk.to_csv(seed_dir / "all_datasets_risk_summary.csv", index=False, encoding="utf-8-sig")

    aux = _aux_rows_for_seed(seed_dir)
    aux.to_csv(seed_dir / "all_datasets_auxiliary_summary.csv", index=False, encoding="utf-8-sig")

    rank = _rank_summary(risk)
    rank.to_csv(seed_dir / "all_datasets_rank_summary.csv", index=False, encoding="utf-8-sig")

    selection_rows = []
    original_margin = _read_csv_if_exists(seed_dir / "original_pool" / "all_margins_selection_summary.csv")
    if not original_margin.empty:
        sub = original_margin[(original_margin.get("row_type", "") == "dataset") & np.isclose(original_margin["margin"].astype(float), 0.05)].copy()
        if not sub.empty:
            sub["strategy"] = ORIGINAL_MARGIN_METHOD
            selection_rows.append(sub)
    expanded_sel = _read_csv_if_exists(seed_dir / "expanded_pool" / "expanded_val_selection_summary.csv")
    if not expanded_sel.empty:
        sub = expanded_sel[np.isclose(expanded_sel["margin"].astype(float), 0.05)].copy()
        sub["strategy"] = EXPANDED_MARGIN_METHOD
        selection_rows.append(sub)
    selection = pd.concat(selection_rows, ignore_index=True) if selection_rows else pd.DataFrame()
    selection.to_csv(seed_dir / "val_selection_summary.csv", index=False, encoding="utf-8-sig")

    lines = [
        f"# Targeted Expanded Pool Seed Report - {seed_dir.name}",
        "",
        "## Main Risk Summary",
        "",
        risk.to_markdown(index=False),
        "",
        "## Rank Summary",
        "",
        rank.to_markdown(index=False),
        "",
        "## Validation Selection",
        "",
        selection.to_markdown(index=False) if not selection.empty else "- no selection records",
    ]
    (seed_dir / "method_report.md").write_text("\n".join(lines), encoding="utf-8-sig")
    return risk, aux, rank, selection


def _run_one_seed(seed: int, root: Path, datasets: list[str], epochs: int, k_candidates: int, candidate_count: int) -> None:
    seed_dir = root / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("MPLCONFIGDIR", str(seed_dir / "tmp_mplconfig"))

    original_cfg = ModelPoolConfig(
        out_dir=seed_dir / "original_pool",
        augmented_data_root=BASE_DIR / "outputs" / "targeted_expanded_pool_multiseed" / f"seed{seed}" / "original_augmented",
        seed=int(seed),
        model_epochs=int(epochs),
        vae_epochs=int(epochs),
        transformer_epochs=int(epochs),
        flow_epochs=int(epochs),
        k_candidates=int(k_candidates),
        direct_k_candidates=int(k_candidates),
        candidate_count=int(candidate_count),
        alpha_tail=1.0,
        fixed_weights="0.35,0.25,0.20,0.20",
        selection_margins=(0.05,),
    )
    run_original_pool(original_cfg, datasets)

    expanded_cfg = ExpandedModelPoolConfig(
        out_dir=seed_dir / "expanded_pool",
        augmented_data_root=BASE_DIR / "outputs" / "targeted_expanded_pool_multiseed" / f"seed{seed}" / "expanded_augmented",
        seed=int(seed),
        model_epochs=int(epochs),
        vae_epochs=int(epochs),
        transformer_epochs=int(epochs),
        flow_epochs=int(epochs),
        tcn_epochs=int(epochs),
        k_candidates=int(k_candidates),
        direct_k_candidates=int(k_candidates),
        candidate_count=int(candidate_count),
        alpha_tail=1.0,
        fixed_weights="0.35,0.25,0.20,0.20",
        selection_margins=(0.05,),
        run_margins=(0.05,),
        stacked_candidate_methods=(),
    )
    methods = [
        TAIL_FIXED_METHOD,
        TRANSFORMER_METHOD,
        FLOW_METHOD,
        VAE_METHOD,
        GAN_METHOD,
        TCN_METHOD,
        RISK_KNN_METHOD,
    ]
    run_expanded_pool(expanded_cfg, datasets, methods)
    _recompute_seed_tables(seed_dir)


def _write_multiseed_summary(root: Path, seeds: list[int]) -> None:
    risk_rows, aux_rows, rank_rows, selection_rows = [], [], [], []
    for seed in seeds:
        seed_dir = root / f"seed{seed}"
        risk = _read_csv_if_exists(seed_dir / "all_datasets_risk_summary.csv")
        if not risk.empty:
            risk.insert(0, "seed", seed)
            risk_rows.append(risk)
        aux = _read_csv_if_exists(seed_dir / "all_datasets_auxiliary_summary.csv")
        if not aux.empty:
            aux.insert(0, "seed", seed)
            aux_rows.append(aux)
        rank = _read_csv_if_exists(seed_dir / "all_datasets_rank_summary.csv")
        if not rank.empty:
            rank.insert(0, "seed", seed)
            rank_rows.append(rank)
        sel = _read_csv_if_exists(seed_dir / "val_selection_summary.csv")
        if not sel.empty:
            sel.insert(0, "seed", seed)
            selection_rows.append(sel)

    all_rank = pd.concat(rank_rows, ignore_index=True)
    method_summary = (
        all_rank.groupby("method", as_index=False)
        .agg(
            avg_mean_risk_rank=("mean_risk_rank", "mean"),
            std_mean_risk_rank=("mean_risk_rank", "std"),
            avg_mean_risk_score=("mean_risk_score", "mean"),
            std_mean_risk_score=("mean_risk_score", "std"),
            avg_wins_count=("wins_count", "mean"),
            avg_top3_count=("top3_count", "mean"),
        )
        .sort_values(["avg_mean_risk_rank", "avg_mean_risk_score"])
    )
    method_summary.to_csv(root / "targeted_multiseed_method_rank_summary.csv", index=False, encoding="utf-8-sig")

    all_risk = pd.concat(risk_rows, ignore_index=True)
    dataset_summary = (
        all_risk.groupby(["method", "dataset"], as_index=False)
        .agg(
            avg_risk_rank=("risk_rank", "mean"),
            std_risk_rank=("risk_rank", "std"),
            avg_risk_score=("risk_score", "mean"),
            std_risk_score=("risk_score", "std"),
            avg_q99_cum_deficit_error=("q99_cum_deficit_error", "mean"),
            avg_core_q99_cum_deficit_error=("core_q99_cum_deficit_error", "mean"),
            avg_netload_ramp_max_mae=("netload_ramp_max_mae", "mean"),
            avg_imbalance_duration_mae=("imbalance_duration_mae", "mean"),
            avg_extreme_degree_match_rate=("extreme_degree_match_rate", "mean"),
        )
        .sort_values(["dataset", "avg_risk_rank", "avg_risk_score"])
    )
    dataset_summary.to_csv(root / "targeted_multiseed_dataset_summary.csv", index=False, encoding="utf-8-sig")

    score_summary = (
        all_risk.groupby("method", as_index=False)
        .agg(
            avg_q99_cum_deficit_error=("q99_cum_deficit_error", "mean"),
            avg_core_q99_cum_deficit_error=("core_q99_cum_deficit_error", "mean"),
            avg_netload_ramp_max_mae=("netload_ramp_max_mae", "mean"),
            avg_imbalance_duration_mae=("imbalance_duration_mae", "mean"),
            avg_extreme_degree_match_rate=("extreme_degree_match_rate", "mean"),
        )
        .merge(method_summary, on="method", how="left")
        .sort_values(["avg_mean_risk_rank", "avg_mean_risk_score"])
    )
    score_summary.to_csv(root / "targeted_multiseed_score_summary.csv", index=False, encoding="utf-8-sig")

    all_sel = pd.concat(selection_rows, ignore_index=True) if selection_rows else pd.DataFrame()
    if not all_sel.empty:
        expanded_sel = all_sel[all_sel["strategy"].astype(str).eq(EXPANDED_MARGIN_METHOD)].copy()
        freq = (
            expanded_sel.groupby(["dataset", "selected_method"], as_index=False)
            .agg(selected_count=("selected_method", "size"))
        )
        total = expanded_sel.groupby("dataset").size().rename("total").reset_index()
        freq = freq.merge(total, on="dataset", how="left")
        freq["selection_frequency"] = freq["selected_count"] / freq["total"].clip(lower=1)
    else:
        freq = pd.DataFrame(columns=["dataset", "selected_method", "selected_count", "total", "selection_frequency"])
    freq.to_csv(root / "targeted_multiseed_selection_frequency.csv", index=False, encoding="utf-8-sig")

    tcn = method_summary[method_summary["method"] == TCN_METHOD]
    knn = method_summary[method_summary["method"] == RISK_KNN_METHOD]
    expanded = method_summary[method_summary["method"] == EXPANDED_MARGIN_METHOD]
    original = method_summary[method_summary["method"] == ORIGINAL_MARGIN_METHOD]
    tail = method_summary[method_summary["method"] == TAIL_FIXED_METHOD]
    best = method_summary.iloc[0]

    lines = [
        "# Targeted Expanded Pool Multiseed Report",
        "",
        "## Overall Method Summary",
        "",
        method_summary.to_markdown(index=False),
        "",
        "## Dataset Summary",
        "",
        dataset_summary.to_markdown(index=False),
        "",
        "## Expanded Selection Frequency",
        "",
        freq.to_markdown(index=False) if not freq.empty else "- no expanded selection records",
        "",
        "## Final Judgment",
        "",
        "【实验结论】",
        f"- 是否有新增方法稳定超过当前主线：{'是' if not original.empty and float(best['avg_mean_risk_rank']) < float(original['avg_mean_risk_rank'].iloc[0]) else '否'}",
        f"- 是否建议升级主方法：{'是' if not expanded.empty and not original.empty and float(expanded['avg_mean_risk_rank'].iloc[0]) < float(original['avg_mean_risk_rank'].iloc[0]) and float(expanded['avg_mean_risk_score'].iloc[0]) <= float(original['avg_mean_risk_score'].iloc[0]) * 1.05 else '否'}",
        f"- 推荐主方法：{EXPANDED_MARGIN_METHOD if not expanded.empty and not original.empty and float(expanded['avg_mean_risk_rank'].iloc[0]) < float(original['avg_mean_risk_rank'].iloc[0]) and float(expanded['avg_mean_risk_score'].iloc[0]) <= float(original['avg_mean_risk_score'].iloc[0]) * 1.05 else ORIGINAL_MARGIN_METHOD}",
        f"- 推荐强基准：{TAIL_FIXED_METHOD}",
        "",
        "【总体多种子结果】",
        f"- 最优 avg_mean_risk_rank：{float(best['avg_mean_risk_rank']):.4f}",
        f"- 最优 avg_mean_risk_score：{float(best['avg_mean_risk_score']):.4f}",
        f"- 最优方法：{best['method']}",
        f"- top3_count 最稳方法：{method_summary.sort_values(['avg_top3_count', 'avg_mean_risk_rank'], ascending=[False, True]).iloc[0]['method']}",
        "",
        "【TCN 判断】",
        f"- 是否稳定：{'是' if not tcn.empty and float(tcn['std_mean_risk_rank'].fillna(0).iloc[0]) <= 1.0 else '否'}",
        f"- 平均排名：{float(tcn['avg_mean_risk_rank'].iloc[0]):.4f}" if not tcn.empty else "- 平均排名：NA",
        f"- 平均 risk_score：{float(tcn['avg_mean_risk_score'].iloc[0]):.4f}" if not tcn.empty else "- 平均 risk_score：NA",
        "- 哪些数据集有效：见 targeted_multiseed_dataset_summary.csv 中 TCN 各数据集 avg_risk_rank",
        "- 哪些数据集退化：同上",
        f"- 是否建议进入最终候选池：{'是' if not tcn.empty and float(tcn['avg_mean_risk_rank'].iloc[0]) <= float(tail['avg_mean_risk_rank'].iloc[0]) else '否'}",
        "",
        "【RiskKNN 判断】",
        f"- 是否稳定：{'是' if not knn.empty and float(knn['std_mean_risk_rank'].fillna(0).iloc[0]) <= 1.0 else '否'}",
        f"- 平均排名：{float(knn['avg_mean_risk_rank'].iloc[0]):.4f}" if not knn.empty else "- 平均排名：NA",
        f"- 平均 risk_score：{float(knn['avg_mean_risk_score'].iloc[0]):.4f}" if not knn.empty else "- 平均 risk_score：NA",
        "- 是否只在 muswellbrook 有效：请结合 dataset summary 判断",
        f"- 是否建议进入最终候选池：{'是' if not knn.empty and float(knn['avg_mean_risk_rank'].iloc[0]) <= float(tail['avg_mean_risk_rank'].iloc[0]) else '否'}",
        "",
        "【Expanded ValSelected 判断】",
        f"- 是否超过原 ValSelected_Margin_0.05：{'是' if not expanded.empty and not original.empty and float(expanded['avg_mean_risk_rank'].iloc[0]) < float(original['avg_mean_risk_rank'].iloc[0]) else '否'}",
        f"- 是否超过 TailWeighted Fixed：{'是' if not expanded.empty and not tail.empty and float(expanded['avg_mean_risk_rank'].iloc[0]) < float(tail['avg_mean_risk_rank'].iloc[0]) else '否'}",
        "- 选择频率如何：见 targeted_multiseed_selection_frequency.csv",
        f"- 是否建议替换当前主线：{'是' if not expanded.empty and not original.empty and float(expanded['avg_mean_risk_rank'].iloc[0]) < float(original['avg_mean_risk_rank'].iloc[0]) and float(expanded['avg_mean_risk_score'].iloc[0]) <= float(original['avg_mean_risk_score'].iloc[0]) * 1.05 else '否'}",
        "",
        "【最终建议】",
        f"- 主方法：{EXPANDED_MARGIN_METHOD if not expanded.empty and not original.empty and float(expanded['avg_mean_risk_rank'].iloc[0]) < float(original['avg_mean_risk_rank'].iloc[0]) and float(expanded['avg_mean_risk_score'].iloc[0]) <= float(original['avg_mean_risk_score'].iloc[0]) * 1.05 else ORIGINAL_MARGIN_METHOD}",
        f"- 强基准：{TAIL_FIXED_METHOD}",
        f"- 需要保留的候选生成器：{', '.join([m for m in [TCN_METHOD, RISK_KNN_METHOD] if m in set(method_summary['method'])])}",
        "- 可以停止的候选生成器：StudentT、QuantileRisk、Stacked、Lightweight U-Net Diffusion、FlowMatching、Vine/Pair Copula",
        "- 是否还需要继续多 seed：如果要写最终论文主表，建议再固定最终候选池补到 5 seeds。",
    ]
    (root / "final_targeted_expanded_pool_multiseed_report.md").write_text("\n".join(lines), encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Targeted multiseed check for expanded model pool candidates.")
    parser.add_argument("--seeds", type=str, default="45,46,47")
    parser.add_argument("--datasets", type=str, default="singleton,muswellbrook,cessnock_or_newarea")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "results" / "targeted_expanded_pool_multiseed")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--k-candidates", type=int, default=20)
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    for seed in seeds:
        seed_dir = args.out_dir / f"seed{seed}"
        if args.skip_existing and (seed_dir / "all_datasets_rank_summary.csv").exists():
            completed.append(seed)
            continue
        print(f"\n=== Targeted seed {seed} ===")
        _run_one_seed(seed, args.out_dir, datasets, args.epochs, args.k_candidates, args.candidate_count)
        completed.append(seed)
    if completed:
        _write_multiseed_summary(args.out_dir, completed)


if __name__ == "__main__":
    main()
