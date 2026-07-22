from __future__ import annotations

from pathlib import Path

import run_expanded_model_pool_experiments as expanded_pool
import run_model_pool_experiments as original_pool
from run_copula_guided_residual_diffusion import DatasetSpec
from run_targeted_expanded_pool_multiseed import _run_one_seed, _write_multiseed_summary


BASE_DIR = Path(__file__).resolve().parent


def _caiso_specs() -> list[DatasetSpec]:
    """中文注释：仅注册 OpenEnergyHub/CAISO 外部验证数据，不影响澳洲三组数据默认入口。"""
    return [
        DatasetSpec(
            name="OpenEnergyHub CAISO balanced relaxed",
            out_name="openenergyhub_caiso_balanced_relaxed",
            data_dir=BASE_DIR / "outputs" / "openenergyhub_caiso_threshold_sweep" / "balanced_relaxed" / "dataset",
            existing_results_dir=BASE_DIR / "results" / "openenergyhub_caiso_balanced_relaxed_main_compare_3y",
        )
    ]


def main() -> None:
    # 中文注释：model pool 脚本内部通过 _dataset_specs 获取数据集；这里临时替换为 CAISO，避免改动全局三地区实验。
    original_pool._dataset_specs = _caiso_specs
    expanded_pool._dataset_specs = _caiso_specs

    out_dir = BASE_DIR / "results" / "openenergyhub_caiso_targeted_model_pool"
    seeds = [45, 46, 47]
    completed: list[int] = []
    for seed in seeds:
        seed_dir = out_dir / f"seed{seed}"
        if (seed_dir / "all_datasets_rank_summary.csv").exists():
            completed.append(seed)
            continue
        _run_one_seed(
            seed=seed,
            root=out_dir,
            datasets=["openenergyhub_caiso_balanced_relaxed"],
            epochs=30,
            k_candidates=20,
            candidate_count=20,
        )
        completed.append(seed)
    _write_multiseed_summary(out_dir, completed)


if __name__ == "__main__":
    main()
