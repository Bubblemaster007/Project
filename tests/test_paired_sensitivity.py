from pathlib import Path

import pandas as pd

from scripts.run_paper_reference_case33 import run_conditional_event_samples, run_paired_sensitivity
from src.chapter4.case33_network_balance import Case33NetworkBalance, NetworkBalanceConfig


ROOT = Path(__file__).resolve().parents[1]


def test_paired_comparison_reuses_each_samples_failure_path(tmp_path):
    nodes = pd.read_csv(ROOT / "data/case33/nodes.csv")
    lines = pd.read_csv(ROOT / "data/case33/lines.csv")
    network = Case33NetworkBalance(nodes, lines, NetworkBalanceConfig())
    sequence = pd.DataFrame({"timestamp": pd.date_range("2021-01-01", periods=36, freq="h"),
                             "load_kw": [2000.0] * 36, "wind_kw": [0.0] * 36,
                             "pv_kw": [0.0] * 36})
    failed_sets = [[{1} if 12 <= hour < 24 else set() for hour in range(36)],
                   [set() for _ in range(36)]]
    event = {"event_id": 0, "wind_exposure_level": "high", "start_index": 0}
    run_conditional_event_samples(sequence, failed_sets, network, 0, 0, 825.0, 42, 2,
                                  tmp_path / "event_0_high")
    run_paired_sensitivity(sequence, network, failed_sets, [event], {0: 825.0}, tmp_path)
    detail = pd.read_csv(tmp_path / "paired_sensitivity_samples.csv")
    assert len(detail) == 8
    baseline = detail[(detail.sample_id == 0) & (detail.scenario == "baseline")].eens_kwh.iloc[0]
    no_fault = detail[(detail.sample_id == 0) & (detail.scenario == "no_line_failure")].eens_kwh.iloc[0]
    assert baseline > no_fault
    assert detail.groupby("scenario").sample_id.nunique().eq(2).all()
