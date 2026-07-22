"""
第2章完整流程管线。

输入历史源荷数据，输出第3章所需：
- annual_background_near/mid/far.csv
- source_load_features.csv
- extreme_window_candidates.csv
- load_probability_boundary.csv
- renewable_probability_boundary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import json
import pandas as pd

from data_schema import Chapter2Config, DEFAULT_STAGE_CONFIGS, StageScenarioConfig
from load_uncertainty_model import build_load_probability_boundary, forecast_load_stage, build_classified_load_boundary
from renewable_uncertainty_model import build_renewable_probability_boundary, identify_low_renewable_risk
from source_load_coupling import compute_source_load_features, extract_top_extreme_windows
from annual_background_generator import generate_annual_background


def run_chapter2_pipeline(
    historical_df: pd.DataFrame,
    output_dir: str | Path,
    config: Chapter2Config | None = None,
    stage_configs: Dict[str, StageScenarioConfig] | None = None,
    year: int = 2026,
) -> Dict[str, str]:
    """
    运行第2章源荷动态增长预测流程。
    """
    config = config or Chapter2Config()
    stage_configs = stage_configs or DEFAULT_STAGE_CONFIGS
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_paths: Dict[str, str] = {}

    # 低新能源风险识别
    re_risk = identify_low_renewable_risk(historical_df, config)

    # 源荷耦合特征与极端窗口候选
    features = compute_source_load_features(re_risk, config)
    candidates = extract_top_extreme_windows(features, config)

    features_path = out_dir / "source_load_features.csv"
    candidates_path = out_dir / "extreme_window_candidates.csv"
    features.to_csv(features_path, index=False, encoding="utf-8-sig")
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    output_paths["source_load_features"] = str(features_path)
    output_paths["extreme_window_candidates"] = str(candidates_path)

    # 各阶段概率边界与年度背景
    load_boundaries = []
    renewable_boundaries = []
    summary = {"stages": {}}

    for name, stg in stage_configs.items():
        load_b = build_load_probability_boundary(historical_df, stg, config)
        re_b = build_renewable_probability_boundary(historical_df, stg, config)
        annual = generate_annual_background(historical_df, stg, config, year=year)

        load_boundaries.append(load_b)
        renewable_boundaries.append(re_b)

        annual_path = out_dir / f"annual_background_{name}.csv"
        annual.to_csv(annual_path, index=False, encoding="utf-8-sig")
        output_paths[f"annual_background_{name}"] = str(annual_path)

        # 分级负荷边界样例，按预测序列输出
        load_forecast = forecast_load_stage(historical_df, stg, config)
        classified = build_classified_load_boundary(load_forecast)
        classified_path = out_dir / f"classified_load_{name}.csv"
        classified.to_csv(classified_path, index=False, encoding="utf-8-sig")
        output_paths[f"classified_load_{name}"] = str(classified_path)

        summary["stages"][name] = {
            "load_growth_rate": stg.load_growth_rate,
            "wind_capacity_scale": stg.wind_capacity_scale,
            "pv_capacity_scale": stg.pv_capacity_scale,
            "annual_peak_load_kw": float(annual["load_kw"].max()),
            "annual_peak_net_load_kw": float(annual["net_load_kw"].max()),
            "annual_energy_load_kwh": float(annual["load_kw"].sum()),
        }

    load_boundary_all = pd.concat(load_boundaries, ignore_index=True)
    renewable_boundary_all = pd.concat(renewable_boundaries, ignore_index=True)

    load_boundary_path = out_dir / "load_probability_boundary.csv"
    renewable_boundary_path = out_dir / "renewable_probability_boundary.csv"
    load_boundary_all.to_csv(load_boundary_path, index=False, encoding="utf-8-sig")
    renewable_boundary_all.to_csv(renewable_boundary_path, index=False, encoding="utf-8-sig")
    output_paths["load_probability_boundary"] = str(load_boundary_path)
    output_paths["renewable_probability_boundary"] = str(renewable_boundary_path)

    summary["top_extreme_window"] = candidates.head(1).to_dict(orient="records")
    summary_path = out_dir / "chapter2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    output_paths["chapter2_summary"] = str(summary_path)

    return output_paths
