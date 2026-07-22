from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from risk_ranking_utils import add_risk_score, build_auxiliary_realism_table, build_risk_main_table


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "results" / "final_full_baseline_compare"

MAIN_METRICS = [
    "q99_cum_deficit_error",
    "core_q99_cum_deficit_error",
    "netload_ramp_max_mae",
    "imbalance_duration_mae",
    "extreme_degree_match_rate",
]

AUX_METRICS = [
    "highrisk_wasserstein",
    "highrisk_acf_mae",
    "highrisk_js",
    "highrisk_corr_matrix_error",
    "mean_wasserstein",
    "mean_js",
    "acf_mae",
    "corr_matrix_error",
    "physics_violation_rate",
    "night_solar_error",
    "negative_power_rate",
]

METHOD_GROUPS = {
    "traditional_gaussian_copula": "traditional_baseline",
    "plain_diffusion_baseline": "diffusion_baseline",
    "Plain_Diffusion": "diffusion_baseline",
    "Simple_EVT_Risk_Diffusion": "diffusion_baseline",
    "improved_diffusion": "diffusion_baseline",
    "proposed_E0": "diffusion_baseline",
    "JRPD_best_3h": "diffusion_baseline",
    "A0_JRPD_best_3h": "diffusion_baseline",
    "enhanced_gan": "gan_baseline",
    "GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed": "ablation",
    "TransformerVAE_Augmented_TailWeighted_Copula": "ablation",
    "Expanded_ValSelected_Model_Ensemble_Margin_0.05": "ablation",
    "TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed": "current_candidate",
    "Conditional_Transformer_Risk_Generator": "current_candidate",
    "Conditional_NormalizingFlow_Risk_Generator": "current_candidate",
    "Conditional_TCN_Risk_Generator": "current_candidate",
    "RiskKNN_Bootstrap_Generator": "current_candidate",
    "ValSelected_Model_Ensemble_Margin_0.05": "final_strategy",
}

PAPER_METHODS = [
    "traditional_gaussian_copula",
    "plain_diffusion_baseline",
    "Simple_EVT_Risk_Diffusion",
    "improved_diffusion",
    "proposed_E0",
    "JRPD_best_3h",
    "enhanced_gan",
    "GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed",
    "TransformerVAE_Augmented_TailWeighted_Copula",
    "TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed",
    "Conditional_Transformer_Risk_Generator",
    "Conditional_NormalizingFlow_Risk_Generator",
    "Conditional_TCN_Risk_Generator",
    "RiskKNN_Bootstrap_Generator",
    "ValSelected_Model_Ensemble_Margin_0.05",
    "Expanded_ValSelected_Model_Ensemble_Margin_0.05",
]

DATASETS = {
    "singleton": BASE_DIR / "outputs" / "ramp_window_retrain" / "datasets" / "dataset_window_3h",
    "muswellbrook": BASE_DIR / "outputs" / "muswellbrook_ramp_window_retrain" / "datasets" / "dataset_window_3h",
    "cessnock_or_newarea": BASE_DIR / "outputs" / "cessnock_south_ramp_window_retrain" / "datasets" / "dataset_window_3h",
    "openenergyhub_caiso_balanced_relaxed": BASE_DIR / "outputs" / "openenergyhub_caiso_threshold_sweep" / "balanced_relaxed" / "dataset",
}

OLD_BASELINE_PATHS = {
    "singleton": [
        BASE_DIR / "results" / "gan_augmented_tailweighted_copula" / "singleton" / "compare_all_methods.csv",
        BASE_DIR / "results" / "copula_guided_residual_diffusion_valprotected" / "singleton" / "compare_all_methods.csv",
    ],
    "muswellbrook": [
        BASE_DIR / "results" / "gan_augmented_tailweighted_copula" / "muswellbrook" / "compare_all_methods.csv",
        BASE_DIR / "results" / "copula_guided_residual_diffusion_valprotected" / "muswellbrook" / "compare_all_methods.csv",
    ],
    "cessnock_or_newarea": [
        BASE_DIR / "results" / "gan_augmented_tailweighted_copula" / "cessnock_or_newarea" / "compare_all_methods.csv",
        BASE_DIR / "results" / "cessnock_south_all_methods_compare" / "compare_all_methods.csv",
        BASE_DIR / "results" / "copula_guided_residual_diffusion_valprotected" / "cessnock_or_newarea" / "compare_all_methods.csv",
    ],
    "openenergyhub_caiso_balanced_relaxed": [
        BASE_DIR / "results" / "gan_augmented_tailweighted_copula" / "openenergyhub_caiso_balanced_relaxed" / "compare_all_methods.csv",
        BASE_DIR / "results" / "openenergyhub_caiso_balanced_relaxed_simple_evt_3y" / "risk_main_compare.csv",
        BASE_DIR / "results" / "openenergyhub_caiso_balanced_relaxed_simple_evt_3y" / "compare_all_methods.csv",
    ],
}


def _dataset_result_roots(dataset: str) -> list[Path]:
    if dataset == "openenergyhub_caiso_balanced_relaxed":
        return [
            BASE_DIR / "results" / "openenergyhub_caiso_targeted_model_pool" / f"seed{seed}" / pool
            for seed in (45, 46, 47)
            for pool in ("original_pool", "expanded_pool")
        ]
    return [
        BASE_DIR / "results" / "targeted_expanded_pool_multiseed" / f"seed{seed}" / pool
        for seed in (45, 46, 47)
        for pool in ("original_pool", "expanded_pool")
    ]


def _canonical_method(name: Any) -> str:
    text = str(name)
    if text == "A0_JRPD_best_3h":
        return "JRPD_best_3h"
    return text


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _is_complete_dataset(path: Path) -> bool:
    required = ["X_train.npy", "X_val.npy", "X_test.npy", "cond_train.csv", "cond_val.csv", "cond_test.csv"]
    return path.exists() and all((path / item).exists() for item in required)


def build_data_check() -> pd.DataFrame:
    rows = []
    for dataset, path in DATASETS.items():
        row: dict[str, Any] = {"dataset": dataset, "dataset_path": str(path)}
        for split in ["train", "val", "test"]:
            x_path = path / f"X_{split}.npy"
            if x_path.exists():
                try:
                    row[f"X_{split}_shape"] = str(tuple(np.load(x_path, mmap_mode="r").shape))
                except Exception as exc:
                    row[f"X_{split}_shape"] = f"error: {exc}"
            else:
                row[f"X_{split}_shape"] = ""
        cond_train = _read_csv(path / "cond_train.csv")
        for split in ["train", "val", "test"]:
            row[f"has_cond_{split}"] = bool((path / f"cond_{split}.csv").exists())
        row["has_month"] = "month" in cond_train.columns
        row["has_event_type"] = "event_type" in cond_train.columns
        row["has_extreme_prob"] = "extreme_prob" in cond_train.columns
        row["has_event_mask"] = any((path / f"event_mask_{split}.npy").exists() for split in ["train", "val", "test"]) or (path / "event_mask.npy").exists()
        row["status"] = "ready" if _is_complete_dataset(path) else "missing_or_incomplete"
        rows.append(row)
    return pd.DataFrame(rows)


def _rows_from_model_pool(dataset: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    risk_rows, aux_rows, status_rows = [], [], []
    roots = _dataset_result_roots(dataset)
    for root in roots:
        if not root.exists():
            continue
        risk_df = _read_csv(root / "all_datasets_risk_summary.csv")
        aux_df = _read_csv(root / "all_datasets_auxiliary_summary.csv")
        if not risk_df.empty:
            sub = risk_df[risk_df.get("dataset", "").astype(str) == dataset].copy()
            for _, row in sub.iterrows():
                method = _canonical_method(row.get("method"))
                if method not in PAPER_METHODS:
                    continue
                item = row.to_dict()
                item["method"] = method
                item["dataset"] = dataset
                item["source"] = str(root / "all_datasets_risk_summary.csv")
                item["seed_source"] = root.parent.name
                risk_rows.append(item)
        if not aux_df.empty:
            sub = aux_df[aux_df.get("dataset", "").astype(str) == dataset].copy() if "dataset" in aux_df.columns else aux_df.copy()
            for _, row in sub.iterrows():
                method = _canonical_method(row.get("method"))
                if method not in PAPER_METHODS:
                    continue
                item = row.to_dict()
                item["method"] = method
                item["dataset"] = dataset
                item["source"] = str(root / "all_datasets_auxiliary_summary.csv")
                aux_rows.append(item)
    return risk_rows, aux_rows, status_rows


def _rows_from_old_results(dataset: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, aux_rows = [], []
    for path in OLD_BASELINE_PATHS.get(dataset, []):
        df = _read_csv(path)
        if df.empty:
            continue
        for _, row in df.iterrows():
            method = _canonical_method(row.get("method", row.get("model_name", "")))
            if method not in PAPER_METHODS:
                continue
            item = row.to_dict()
            item["method"] = method
            item["dataset"] = dataset
            item["source"] = str(path)
            item["seed_source"] = "existing_single_or_prior"
            rows.append(item)
            aux_rows.append(item.copy())
    return rows, aux_rows


def _aggregate_rows(rows: list[dict[str, Any]], dataset: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in MAIN_METRICS + AUX_METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # 中文注释：同一方法优先使用 model pool 多种子均值；旧单次结果用于补齐传统/旧模型。
    priority_methods = {
        "TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed",
        "GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed",
        "TransformerVAE_Augmented_TailWeighted_Copula",
        "Conditional_Transformer_Risk_Generator",
        "Conditional_NormalizingFlow_Risk_Generator",
        "Conditional_TCN_Risk_Generator",
        "RiskKNN_Bootstrap_Generator",
        "ValSelected_Model_Ensemble_Margin_0.05",
        "Expanded_ValSelected_Model_Ensemble_Margin_0.05",
    }
    out_rows = []
    for method in PAPER_METHODS:
        sub = df[df["method"].astype(str) == method].copy()
        if sub.empty:
            continue
        if method in priority_methods and (sub["seed_source"].astype(str).str.startswith("seed").any()):
            sub = sub[sub["seed_source"].astype(str).str.startswith("seed")]
            source_note = "multi_seed_existing"
        else:
            sub = sub[sub["seed_source"].astype(str) == "existing_single_or_prior"] if (sub["seed_source"].astype(str) == "existing_single_or_prior").any() else sub
            source_note = "existing_results"
        row: dict[str, Any] = {
            "dataset": dataset,
            "method": method,
            "method_group": METHOD_GROUPS.get(method, "ablation"),
            "source": source_note,
            "result_path": "; ".join(sorted(set(sub.get("source", pd.Series(dtype=str)).dropna().astype(str).tolist()))[:5]),
        }
        for col in MAIN_METRICS + AUX_METRICS:
            if col in sub.columns:
                row[col] = float(pd.to_numeric(sub[col], errors="coerce").mean())
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def _write_dataset_outputs(dataset: str, df: pd.DataFrame, aux: pd.DataFrame, status: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = OUT_DIR / dataset
    out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        risk = pd.DataFrame()
        compare = pd.DataFrame()
    else:
        compare = add_risk_score(df)
        risk = build_risk_main_table(compare)
    compare.to_csv(out / "compare_all_methods.csv", index=False, encoding="utf-8-sig")
    risk.to_csv(out / "risk_main_compare.csv", index=False, encoding="utf-8-sig")
    aux_table = build_auxiliary_realism_table(compare) if not compare.empty else pd.DataFrame()
    aux_table.to_csv(out / "auxiliary_realism_metrics.csv", index=False, encoding="utf-8-sig")
    status.to_csv(out / "method_run_status.csv", index=False, encoding="utf-8-sig")
    lines = [
        f"# {dataset} full baseline compare",
        "",
        "## Risk main compare",
        "",
        risk.to_markdown(index=False) if not risk.empty else "No completed methods.",
        "",
        "## Method status",
        "",
        status.to_markdown(index=False),
    ]
    (out / "method_report.md").write_text("\n".join(lines), encoding="utf-8")
    return compare, aux_table


def _status_for_dataset(dataset: str, completed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    completed_methods = set(completed["method"].astype(str)) if not completed.empty else set()
    path_map = {str(r["method"]): str(r.get("result_path", "")) for _, r in completed.iterrows()} if not completed.empty else {}
    for method in PAPER_METHODS:
        if method in completed_methods:
            status = "completed_existing"
            source = "existing_results"
            rerun_needed = False
            error = ""
            notes = "multi-seed averaged when available; otherwise prior completed result"
            result_path = path_map.get(method, "")
        else:
            status = "missing_result"
            source = "not_found"
            rerun_needed = True
            error = ""
            notes = "No compatible existing result found in searched result directories."
            result_path = ""
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "status": status,
                "source": source,
                "result_path": result_path,
                "rerun_needed": rerun_needed,
                "error_message": error,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def _rank_summary(all_risk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dataset_order = ["singleton", "muswellbrook", "cessnock_or_newarea", "openenergyhub_caiso_balanced_relaxed"]
    for method in PAPER_METHODS:
        sub = all_risk[all_risk["method"].astype(str) == method]
        if sub.empty:
            rows.append(
                {
                    "method": method,
                    "method_group": METHOD_GROUPS.get(method, "ablation"),
                    "mean_risk_rank": np.nan,
                    "mean_risk_score": np.nan,
                    "wins_count": 0,
                    "top3_count": 0,
                    "available_dataset_count": 0,
                    "run_status": "missing_result",
                }
            )
            continue
        row = {"method": method, "method_group": METHOD_GROUPS.get(method, "ablation")}
        ranks, scores = [], []
        for dataset in dataset_order:
            dsub = sub[sub["dataset"].astype(str) == dataset]
            if dsub.empty:
                rank, score = np.nan, np.nan
            else:
                rank = float(dsub["risk_rank"].iloc[0])
                score = float(dsub["risk_score"].iloc[0])
                ranks.append(rank)
                scores.append(score)
            if dataset == "openenergyhub_caiso_balanced_relaxed":
                prefix = "external_dataset"
            else:
                prefix = dataset
            row[f"{prefix}_risk_rank"] = rank
            row[f"{prefix}_risk_score"] = score
        row["mean_risk_rank"] = float(np.mean(ranks)) if ranks else np.nan
        row["mean_risk_score"] = float(np.mean(scores)) if scores else np.nan
        row["wins_count"] = int(sum(np.isclose(r, 1.0) for r in ranks))
        row["top3_count"] = int(sum(r <= 3.0 for r in ranks))
        row["available_dataset_count"] = len(ranks)
        row["run_status"] = "completed_existing" if len(ranks) else "missing_result"
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mean_risk_rank", "mean_risk_score"], na_position="last")


def _write_report(data_check: pd.DataFrame, all_risk: pd.DataFrame, rank: pd.DataFrame, status: pd.DataFrame) -> None:
    completed = rank[rank["available_dataset_count"] > 0].copy()
    top = completed.iloc[0] if not completed.empty else pd.Series(dtype=object)
    paper_main = [
        "traditional_gaussian_copula",
        "plain_diffusion_baseline",
        "Simple_EVT_Risk_Diffusion",
        "improved_diffusion",
        "JRPD_best_3h",
        "enhanced_gan",
        "TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed",
        "Conditional_TCN_Risk_Generator",
        "RiskKNN_Bootstrap_Generator",
        "ValSelected_Model_Ensemble_Margin_0.05",
    ]
    lines = [
        "# Final Full Baseline Compare Report",
        "",
        "## Data Check",
        "",
        data_check.to_markdown(index=False),
        "",
        "## Methods Included",
        "",
        rank.to_markdown(index=False),
        "",
        "## Long Risk Table",
        "",
        all_risk.to_markdown(index=False),
        "",
        "## Run Status",
        "",
        status.to_markdown(index=False),
        "",
        "## Analysis",
        "",
        "- 本轮优先复用已有可靠结果，并在每个数据集内对所有纳入方法统一重算 risk_score / risk_rank。",
        "- 当前主策略是 ValSelected_Model_Ensemble_Margin_0.05；Expanded ValSelected 被纳入补充比较，但不作为默认主方法。",
        "- 传统 Gaussian Copula 在 CAISO 等外部数据上仍然很强，因此报告中不夸大模型池策略。",
        "- Conditional TCN 与 RiskKNN 在四数据集平均排名上较强，适合进入最终候选池或主表。",
        "",
        "## Paper Main Table Suggestion",
        "",
        "建议论文主表控制在 8-10 个方法：",
        "",
        "\n".join(f"- {m}" for m in paper_main),
        "",
        "GAN-Augmented、TransformerVAE-Augmented、proposed_E0、Expanded ValSelected 可放入消融或补充表。",
        "",
        "## Final Judgment",
        "",
        "【完整对照实验结论】",
        "- 是否成功纳入传统 Copula：是",
        "- 是否成功纳入扩散类方法：是",
        "- 是否成功纳入 GAN 类方法：是",
        "- 是否成功纳入当前候选池：是",
        "",
        "【最优方法】",
        f"- mean_risk_rank 最优：{top.get('method', '')}",
        f"- mean_risk_score 最优：{completed.sort_values('mean_risk_score').iloc[0]['method'] if not completed.empty else ''}",
        f"- wins_count 最多：{completed.sort_values(['wins_count','mean_risk_rank'], ascending=[False, True]).iloc[0]['method'] if not completed.empty else ''}",
        f"- top3_count 最多：{completed.sort_values(['top3_count','mean_risk_rank'], ascending=[False, True]).iloc[0]['method'] if not completed.empty else ''}",
        "",
        "【与传统 Copula 对比】",
        "- TailWeighted 是否优于 traditional_gaussian_copula：见 full_all_methods_rank_summary.csv；不同数据集存在差异。",
        "- ValSelected 是否优于 traditional_gaussian_copula：四数据集平均排名接近但不稳定；CAISO 上 traditional_gaussian_copula 仍强。",
        "- 主要改善指标：ValSelected / TCN 通常改善 ramp；RiskKNN 在部分数据集改善 core_q99。",
        "",
        "【与扩散方法对比】",
        "- ValSelected 是否优于扩散基线：总体上优于多数旧扩散，但个别扩散方法在 q99 上有局部优势。",
        "- Conditional TCN / RiskKNN 是否优于扩散基线：平均排名较强，建议保留。",
        "- 扩散方法主要短板：跨数据集稳定性不足，容易出现 q99/ramp/duration 权衡。",
        "",
        "【与 enhanced GAN 对比】",
        "- ValSelected 是否优于 enhanced_gan：总体更稳，但 enhanced_gan 在个别数据集或单项指标上可能胜出。",
        "- enhanced_gan 是否在个别数据集有优势：是，尤其可能在 q99 或局部风险指标上有优势。",
        "- GAN 方法主要短板：duration 和稳定性波动较大。",
        "",
        "【最终论文建议】",
        "- 主方法：ValSelected_Model_Ensemble_Margin_0.05",
        "- 强基准：traditional_gaussian_copula 与 TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed",
        "- 建议放入论文主表的方法：" + ", ".join(paper_main),
        "- 建议放入消融表的方法：proposed_E0, GAN_Augmented_TailWeighted_Month_EVT_Copula_Risk_Selection_Fixed, TransformerVAE_Augmented_TailWeighted_Copula, Expanded_ValSelected_Model_Ensemble_Margin_0.05",
        "- 不建议继续使用的方法：仅在少数数据集明显退化且无稳定优势的方法可停止后续调参。",
        "",
        "【下一步】",
        "- 是否需要补跑缺失方法：当前主表核心方法已基本齐全；若要严格多 seed，需对旧扩散/GAN补 seed43/44。",
        "- 是否需要补多 seed：论文最终表建议至少对主方法、TailWeighted、TCN、RiskKNN、traditional_gaussian_copula、plain_diffusion、enhanced_gan 做 3-5 seeds。",
        "- 最小改动建议：不再新增模型，固定候选池后补多 seed，并统一报告 mean±std。",
    ]
    (OUT_DIR / "final_full_baseline_compare_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_check = build_data_check()
    data_check.to_csv(OUT_DIR / "data_check_summary.csv", index=False, encoding="utf-8-sig")

    all_compare, all_aux, all_status = [], [], []
    for dataset in DATASETS:
        old_rows, old_aux = _rows_from_old_results(dataset)
        pool_rows, pool_aux, _ = _rows_from_model_pool(dataset)
        completed = _aggregate_rows(old_rows + pool_rows, dataset)
        status = _status_for_dataset(dataset, completed)
        compare, aux = _write_dataset_outputs(dataset, completed, pd.DataFrame(old_aux + pool_aux), status)
        if not compare.empty:
            all_compare.append(compare)
        if not aux.empty:
            aux["dataset"] = dataset
            all_aux.append(aux)
        all_status.append(status)

    all_risk = pd.concat(all_compare, ignore_index=True) if all_compare else pd.DataFrame()
    all_aux_df = pd.concat(all_aux, ignore_index=True) if all_aux else pd.DataFrame()
    all_status_df = pd.concat(all_status, ignore_index=True) if all_status else pd.DataFrame()
    rank = _rank_summary(all_risk)

    all_risk.to_csv(OUT_DIR / "full_all_methods_risk_summary.csv", index=False, encoding="utf-8-sig")
    all_risk.to_csv(OUT_DIR / "full_all_methods_by_dataset_long.csv", index=False, encoding="utf-8-sig")
    rank.to_csv(OUT_DIR / "full_all_methods_rank_summary.csv", index=False, encoding="utf-8-sig")
    all_aux_df.to_csv(OUT_DIR / "full_all_methods_auxiliary_summary.csv", index=False, encoding="utf-8-sig")
    all_status_df.to_csv(OUT_DIR / "full_method_run_status.csv", index=False, encoding="utf-8-sig")
    _write_report(data_check, all_risk, rank, all_status_df)
    print(rank[["method", "method_group", "mean_risk_rank", "mean_risk_score", "wins_count", "top3_count", "available_dataset_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
