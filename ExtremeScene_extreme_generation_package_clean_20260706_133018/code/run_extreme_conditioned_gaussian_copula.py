from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from evaluate_generation import EvalConfig, evaluate_generation
from risk_ranking_utils import write_risk_tables
from run_month_evt_copula_risk_selection import _add_month_season, _load_split, get_au_season
from traditional_statistical_extreme_baseline.traditional_copula_baseline import (
    CopulaConfig,
    GaussianCopulaGroupModel,
    physical_projection_np,
    temporal_correction,
)


BASE_DIR = Path(__file__).resolve().parent
METHOD_NAME = "evt_copula"

DATASET_PATHS = {
    "singleton": BASE_DIR / "outputs" / "ramp_window_retrain" / "datasets" / "dataset_window_3h",
    "muswellbrook": BASE_DIR / "outputs" / "muswellbrook_ramp_window_retrain" / "datasets" / "dataset_window_3h",
    "cessnock_or_newarea": BASE_DIR / "outputs" / "cessnock_south_ramp_window_retrain" / "datasets" / "dataset_window_3h",
    "openenergyhub_caiso_balanced_relaxed": BASE_DIR
    / "outputs"
    / "openenergyhub_caiso_threshold_sweep"
    / "balanced_relaxed"
    / "dataset",
}


@dataclass
class ExtremeConditionedCopulaConfig:
    """EVT-Copula 对照方法配置。

    该方法使用 EVT 得到的 extreme_prob 作为极端程度条件，只在训练集上
    按 EVT 概率分层拟合 Copula，用于检验“传统相关性建模 + EVT 极端
    程度调控”是否足以接近 RiskFirst TCN。
    """

    out_dir: Path = BASE_DIR / "results" / "evt_copula"
    dataset: str = "all"
    seed: int = 42
    min_samples: int = 8
    covariance_shrinkage: float = 0.08
    quantile_grid_size: int = 401
    temporal_smooth_strength: float = 0.15


def _ensure_nct(x: np.ndarray) -> np.ndarray:
    """保证风光荷样本为 [N, 3, T]，其中 3 对应 load/wind/solar。"""

    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"X must be 3D, got shape={arr.shape}")
    if arr.shape[1] == 3:
        return arr
    if arr.shape[2] == 3:
        return np.transpose(arr, (0, 2, 1))
    raise ValueError(f"Cannot infer channel dimension from shape={arr.shape}")


def _event_type_series(cond: pd.DataFrame) -> pd.Series:
    """提取事件类型条件；缺失时退化为全局事件类型。"""

    if "event_type" in cond.columns:
        return cond["event_type"].astype(str)
    if "event_type_code" in cond.columns:
        return cond["event_type_code"].astype(str)
    return pd.Series(["unknown"] * len(cond), index=cond.index, dtype="object")


def _extreme_prob_series(cond: pd.DataFrame) -> pd.Series:
    """提取 EVT 极端概率；缺失时使用 0.90 作为中高风险默认条件。"""

    if "extreme_prob" in cond.columns:
        values = pd.to_numeric(cond["extreme_prob"], errors="coerce")
    else:
        values = pd.Series([0.90] * len(cond), index=cond.index, dtype=float)
    return values.fillna(0.90).clip(0.50, 1.00)


def _risk_bin_from_prob(p: float) -> str:
    """根据 EVT extreme_prob 划分极端程度层。

    low_high_risk 表示较高风险但尚未进入尾部高段；
    high_risk 表示高风险；
    extreme_risk 表示极高风险尾部。
    """

    prob = float(np.clip(p, 0.50, 1.00))
    if prob < 0.75:
        return "low_high_risk"
    if prob < 0.90:
        return "high_risk"
    return "extreme_risk"


def _prepare_condition(cond: pd.DataFrame, meta: pd.DataFrame | None) -> pd.DataFrame:
    """补齐 month、澳洲季节、event_type_key 和 risk_bin 条件。"""

    out = _add_month_season(cond, meta).reset_index(drop=True)
    out["event_type_key"] = _event_type_series(out).to_numpy(dtype=object)
    out["extreme_prob_for_bin"] = _extreme_prob_series(out).to_numpy(dtype=float)
    out["risk_bin"] = out["extreme_prob_for_bin"].map(_risk_bin_from_prob)
    return out


def _copula_cfg(cfg: ExtremeConditionedCopulaConfig, seq_len: int, out_dir: Path) -> CopulaConfig:
    return CopulaConfig(
        data_dir="",
        output_dir=str(out_dir),
        seed=int(cfg.seed),
        group_cols="global",
        fallback_group_cols="global",
        min_group_size=1,
        covariance_shrinkage=float(cfg.covariance_shrinkage),
        quantile_grid_size=int(cfg.quantile_grid_size),
        n_per_condition=1,
        seq_len=int(seq_len),
        temporal_smooth_strength=float(cfg.temporal_smooth_strength),
        make_plots=False,
    )


def _mask_builder(cond_train: pd.DataFrame) -> dict[str, Callable[[pd.Series], np.ndarray]]:
    """构造测试条件到训练子集的回退查询函数。

    回退顺序严格为：
    month + event_type + risk_bin -> month + risk_bin -> season + risk_bin
    -> risk_bin -> global。
    """

    month = cond_train["month"].astype(int).to_numpy()
    season = cond_train["au_season"].astype(str).to_numpy()
    event_type = cond_train["event_type_key"].astype(str).to_numpy()
    risk_bin = cond_train["risk_bin"].astype(str).to_numpy()
    all_idx = np.arange(len(cond_train), dtype=np.int64)

    def month_event_risk(row: pd.Series) -> np.ndarray:
        return np.where(
            (month == int(row["month"]))
            & (event_type == str(row["event_type_key"]))
            & (risk_bin == str(row["risk_bin"]))
        )[0]

    def month_risk(row: pd.Series) -> np.ndarray:
        return np.where((month == int(row["month"])) & (risk_bin == str(row["risk_bin"])))[0]

    def season_risk(row: pd.Series) -> np.ndarray:
        return np.where((season == str(row["au_season"])) & (risk_bin == str(row["risk_bin"])))[0]

    def risk_only(row: pd.Series) -> np.ndarray:
        return np.where(risk_bin == str(row["risk_bin"]))[0]

    def global_all(row: pd.Series) -> np.ndarray:
        return all_idx

    return {
        "month_event_risk": month_event_risk,
        "month_risk": month_risk,
        "season_risk": season_risk,
        "risk_bin": risk_only,
        "global": global_all,
    }


def _group_name(group_type: str, row: pd.Series) -> str:
    if group_type == "month_event_risk":
        return f"month={int(row['month'])}|event_type={row['event_type_key']}|risk_bin={row['risk_bin']}"
    if group_type == "month_risk":
        return f"month={int(row['month'])}|risk_bin={row['risk_bin']}"
    if group_type == "season_risk":
        return f"season={row['au_season']}|risk_bin={row['risk_bin']}"
    if group_type == "risk_bin":
        return f"risk_bin={row['risk_bin']}"
    return "global"


def _select_train_group(
    row: pd.Series,
    mask_fns: dict[str, Callable[[pd.Series], np.ndarray]],
    min_samples: int,
) -> tuple[str, str, np.ndarray, str]:
    """按极端程度分层 Copula 的回退规则选择训练子集。"""

    fallback_notes: list[str] = []
    for group_type in ["month_event_risk", "month_risk", "season_risk", "risk_bin"]:
        idx = mask_fns[group_type](row)
        group_name = _group_name(group_type, row)
        if len(idx) >= int(min_samples):
            return group_type, group_name, idx.astype(np.int64), "; ".join(fallback_notes)
        fallback_notes.append(f"{group_name}: n={len(idx)} < min_samples={min_samples}")
    idx = mask_fns["global"](row)
    return "global", "global", idx.astype(np.int64), "; ".join(fallback_notes)


def _combined_condition_row(cond_row: pd.Series, meta_row: pd.Series | None) -> pd.DataFrame:
    """合并条件与元数据，供统一物理边界处理识别时间与夜间光伏。"""

    merged = cond_row.to_dict()
    if meta_row is not None:
        for key, value in meta_row.to_dict().items():
            if key not in merged:
                merged[key] = value
    return pd.DataFrame([merged])


def run_one_dataset(dataset: str, cfg: ExtremeConditionedCopulaConfig) -> pd.DataFrame:
    data_dir = DATASET_PATHS[dataset]
    out_dir = cfg.out_dir / dataset
    eval_dir = out_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train, cond_train_raw, meta_train, _mask_train = _load_split(data_dir, "train")
    x_test, cond_test_raw, meta_test, _mask_test = _load_split(data_dir, "test")
    x_train = _ensure_nct(x_train)
    x_test = _ensure_nct(x_test)
    cond_train = _prepare_condition(cond_train_raw, meta_train)
    cond_test = _prepare_condition(cond_test_raw, meta_test)

    cop_cfg = _copula_cfg(cfg, int(x_train.shape[2]), out_dir / "copula_models")
    rng = np.random.default_rng(int(cfg.seed))
    mask_fns = _mask_builder(cond_train)
    model_cache: dict[tuple[str, str], GaussianCopulaGroupModel] = {}
    fit_rows: list[dict] = []
    selection_rows: list[dict] = []
    generated: list[np.ndarray] = []

    for i, row in cond_test.iterrows():
        group_type, group_name, idx, fallback_reason = _select_train_group(row, mask_fns, int(cfg.min_samples))
        cache_key = (group_type, group_name)
        if cache_key not in model_cache:
            # 训练集子集拟合 Copula；测试真实曲线不参与任何参数估计。
            model_cache[cache_key] = GaussianCopulaGroupModel(group_name, x_train[idx], cop_cfg)
            fit_rows.append(
                {
                    "dataset": dataset,
                    "group_type": group_type,
                    "group_name": group_name,
                    "n_train_samples": int(len(idx)),
                    "risk_bin": row["risk_bin"] if group_type != "global" else "all",
                    "min_samples": int(cfg.min_samples),
                    "fallback_reason": fallback_reason,
                }
            )
        model = model_cache[cache_key]
        raw = model.sample_raw(1, rng)
        corrected = temporal_correction(raw, model, cop_cfg)
        meta_row = meta_test.iloc[i] if meta_test is not None and i < len(meta_test) else None
        cond_projection = _combined_condition_row(row, meta_row)
        projected = physical_projection_np(corrected, cond_projection, cop_cfg)
        generated.append(projected[0].astype(np.float32))
        selection_rows.append(
            {
                "dataset": dataset,
                "sample_index": int(i),
                "month": int(row["month"]),
                "season": row["au_season"],
                "event_type": row["event_type_key"],
                "extreme_prob": float(row["extreme_prob_for_bin"]),
                "risk_bin": row["risk_bin"],
                "group_type_used": group_type,
                "group_name_used": group_name,
                "n_train_samples_used": int(len(idx)),
                "fallback_reason": fallback_reason,
            }
        )

    gen = np.stack(generated, axis=0).astype(np.float32)
    gen_path = out_dir / f"generated_samples_{METHOD_NAME}.npy"
    np.save(gen_path, gen)
    pd.DataFrame(selection_rows).to_csv(out_dir / "group_selection_log.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fit_rows).to_csv(out_dir / "group_fit_summary.csv", index=False, encoding="utf-8-sig")
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    mask_path = data_dir / "event_mask_test.npy"
    summary = evaluate_generation(
        EvalConfig(
            real=str(data_dir / "X_test.npy"),
            generated=str(gen_path),
            cond=str(data_dir / "cond_test.csv"),
            meta=str(data_dir / "meta_test.csv"),
            out_dir=str(eval_dir),
            model_name=METHOD_NAME,
            event_mask=str(mask_path) if mask_path.exists() else None,
            ramp_metric_mode="window_3h",
            ramp_window_hours=3.0,
        )
    )
    metric_row = {"method": METHOD_NAME, **summary.get("metrics", {})}
    compare = pd.DataFrame([metric_row])
    compare.to_csv(out_dir / "compare_all_methods.csv", index=False, encoding="utf-8-sig")
    write_risk_tables(compare, out_dir, method_col="method")

    report = [
        f"# {METHOD_NAME} - {dataset}",
        "",
        "该方法在训练集上按 EVT extreme_prob 划分 low_high/high/extreme 三个风险层，",
        "再按 month + event_type + risk_bin 优先拟合 Gaussian Copula，样本不足时逐级回退。",
        "",
        f"- train samples: {len(x_train)}",
        f"- test samples: {len(x_test)}",
        f"- fitted groups: {len(model_cache)}",
        f"- min_samples: {cfg.min_samples}",
        f"- generated path: {gen_path}",
        "",
        "## Main Metrics",
        compare[[c for c in ["method", "q99_cum_deficit_error", "core_q99_cum_deficit_error", "netload_ramp_max_mae", "imbalance_duration_mae", "extreme_degree_match_rate"] if c in compare.columns]].to_markdown(index=False),
    ]
    (out_dir / "method_report.md").write_text("\n".join(report), encoding="utf-8")
    return compare


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EVT-Copula baseline with EVT extreme_prob stratification.")
    parser.add_argument("--dataset", default="all", choices=[*DATASET_PATHS.keys(), "all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--out-dir", default=str(BASE_DIR / "results" / "evt_copula"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExtremeConditionedCopulaConfig(
        out_dir=Path(args.out_dir),
        dataset=str(args.dataset),
        seed=int(args.seed),
        min_samples=int(args.min_samples),
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    datasets = list(DATASET_PATHS.keys()) if args.dataset == "all" else [str(args.dataset)]
    rows = []
    for dataset in datasets:
        print(f"[{METHOD_NAME}] running {dataset}")
        compare = run_one_dataset(dataset, cfg)
        tmp = compare.copy()
        tmp.insert(0, "dataset", dataset)
        rows.append(tmp)
    if rows:
        all_df = pd.concat(rows, ignore_index=True, sort=False)
        all_df.to_csv(cfg.out_dir / "all_datasets_compare_all_methods.csv", index=False, encoding="utf-8-sig")
    print(f"[{METHOD_NAME}] wrote outputs to {cfg.out_dir}")


if __name__ == "__main__":
    main()
