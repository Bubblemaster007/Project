"""Run the paper's exact event extraction/EVT labeling pipeline on project data.

This is the scenario-stage bridge; model training is intentionally optional and
the produced dataset can be consumed by the downstream fault/balance modules.
"""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "ExtremeScene_extreme_generation_package_clean_20260706_133018"
sys.path.insert(0, str(PKG / "code"))
from data_interface import ColumnMapping, DatasetBuildConfig, build_dataset_artifacts, build_event_samples
from evt_fit import EVTConfig
from risk_screening import RiskScreenConfig
from sample_metrics import MetricConfig

def main():
    src = ROOT / "data/demo/historical_source_load.csv"
    bridge = ROOT / "data/demo/paper_input.csv"
    df = pd.read_csv(src)
    df = df.rename(columns={"timestamp":"time", "load_kw":"load", "wind_kw":"wind_power", "pv_kw":"solar_power",
                            "temperature":"temp", "rainfall":"precipitation"})
    for col in ["time","load","wind_power","solar_power"]:
        if col not in df: raise ValueError(f"missing {col}")
    df.to_csv(bridge, index=False)
    out = ROOT / "outputs/paper_scenario_stage"; dataset = out / "dataset"; dataset.mkdir(parents=True, exist_ok=True)
    mapping = ColumnMapping()
    samples, evt_info = build_event_samples(df, mapping, intermediate_output_dir=dataset,
        evt_cfg=EVTConfig(metric_col="cum_deficit", threshold_quantile=.90),
        risk_screen_cfg=RiskScreenConfig(enabled=True, mode="medium", min_samples_after_screen=1),
        metric_cfg=MetricConfig())
    samples.to_csv(dataset / "samples_evt_labeled.csv", index=False, encoding="utf-8-sig")
    result = build_dataset_artifacts(df, samples, dataset, DatasetBuildConfig(seq_len=36, output_dir=str(dataset), split_group_col="sample_id"), mapping, evt_info)
    print(result["summary"])
    print(f"dataset={out / 'dataset'}")

if __name__ == "__main__": main()
