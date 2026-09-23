"""Connect certified paper annual trajectories to the IEEE 33-bus balance model.

The calendar weather used here is a transparent placeholder for the missing
joint weather-generation step. It does not establish calibrated regional risk.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.chapter4.case33_network_balance import (
    Case33NetworkBalance,
    NetworkBalanceConfig,
    resource_limits_from_row,
    summarize_network_balance,
)
from src.chapter3.sequential_resource_states import simulate_resource_states
from src.common.io_utils import write_json


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "topic2_remaining_code"))

from resilience_reliability_planner import PlanningConfig, run_simple_planning
from strategy_trigger import StrategyThresholds, trigger_strategies
from scripts.run_paper_reference_case33 import (
    run_conditional_event_samples,
    sample_source_load_sequences,
    simulate_line_states,
)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _paper_preflight(paper_root: Path, dataset: str, regenerate: bool) -> None:
    required = [paper_root / name for name in (
        "annual_historical_calendar.py",
        "annual_transition_certified.py",
        "annual_calendar_assignment.py",
    )]
    required += [
        paper_root / "publication_protocol/data/main" / dataset / "fitted_parameters.json",
        paper_root / "publication_protocol/data/main" / dataset / "meta_train.csv",
        paper_root / "publication_protocol/data/main" / dataset / "event_mask_train.npy",
        paper_root / "publication_protocol/final_candidate/downstream" / dataset / "planning_conditions.csv",
    ]
    if regenerate:
        required += [paper_root / "publication_structure_calibrated.py",
                     paper_root / "publication_protocol/final_candidate/selected_main_seed123.csv"]
    else:
        required.append(paper_root / "publication_protocol/final_candidate/downstream" / dataset / "planning.npy")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "论文冻结方法文件缺失；请设置 CODE_PAPER_ROOT 指向完整论文仓库并补齐：\n"
            + "\n".join(missing)
        )


def run_paper_downstream(annual: pd.DataFrame, config: dict, out: Path,
                         provenance: dict | None = None) -> dict:
    """Dispatch the certified annual series without altering its source/load columns."""
    out.mkdir(parents=True, exist_ok=True)
    source = annual.copy()
    required = ["timestamp", "load_kw", "wind_kw", "pv_kw", "event_id", "wind_speed"]
    missing = [column for column in required if column not in source]
    if missing:
        raise ValueError(f"Paper annual trajectory is missing {missing}; joint weather input is required")
    source["timestamp"] = pd.to_datetime(source.timestamp)
    source = source.sort_values("timestamp").reset_index(drop=True)
    if source[required[1:]].isna().any().any() or source.timestamp.duplicated().any():
        raise ValueError("Paper annual trajectory has missing values or duplicate timestamps")
    if not source.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Paper annual trajectory must be hourly and continuous")
    if "random_sequence_id" not in source:
        source["random_sequence_id"] = "paper_sequence_001"
    if "sequence_weight" not in source:
        source["sequence_weight"] = 1.0
    input_hash = hashlib.sha256(source.to_csv(index=False).encode("utf-8")).hexdigest()
    base_source = source.copy()

    topology = config.get("topology", {})
    if topology.get("type", "case33") != "case33":
        raise ValueError("The current paper downstream requires case33 topology")
    case33_dir = _resolve(ROOT, topology.get("case33_dir", "data/case33"))
    nodes = pd.read_csv(case33_dir / "nodes.csv")
    lines = pd.read_csv(case33_dir / "lines.csv")
    cfg = NetworkBalanceConfig.from_project(config)
    network = Case33NetworkBalance(nodes, lines, cfg)
    seed = int(config.get("paper_method", {}).get("seed", config.get("random_seed", 42)))
    source, baseline_resource_failures = simulate_resource_states(base_source, cfg, seed, 0)
    source.to_csv(out / "annual_random_production_sequence.csv", index=False)
    event_ids = source.event_id.to_numpy(int)
    line_ids = lines.loc[lines.status.eq(1), "line"].astype(int).tolist()
    failed_sets, failures, rates = simulate_line_states(
        source.wind_speed.to_numpy(float), line_ids, seed, source.timestamp, event_ids
    )
    pd.DataFrame(failures, columns=["timestamp", "line", "repair_hours", "event_id",
                                    "failure_rate_per_hour"]).to_csv(out / "line_failures.csv", index=False)

    soc = cfg.soc_initial_ratio * cfg.storage_energy_kwh
    soc_before = {}
    event_starts = source.index[source.event_id.ge(0) & source.event_id.ne(source.event_id.shift(1))].tolist()
    for i in event_starts:
        soc_before[i] = None
    rows = []
    ac_rows = []
    for i, entry in enumerate(source.itertuples(index=False)):
        if i in soc_before:
            soc_before[i] = soc
        row, soc, node_state, ac = network.dispatch_hour_ac_balanced(
            entry.timestamp, entry.load_kw, entry.wind_kw, entry.pv_kw, failed_sets[i], soc,
            **resource_limits_from_row(entry)
        )
        row.update(event_id=int(entry.event_id), random_sequence_id=entry.random_sequence_id,
                   sequence_weight=float(entry.sequence_weight))
        rows.append(row)
        ac_rows.append({"timestamp": entry.timestamp, "event_id": int(entry.event_id), **ac,
                        "ac_grid_within_limit": ac["ac_root_required_kw"] <= row["grid_available_kw"] + 1e-5})
    hourly = pd.DataFrame(rows)
    if not np.allclose(hourly.load_kw, hourly.load_served_kw + hourly.power_deficit_kw, atol=1e-6):
        raise AssertionError("Original load is not conserved")
    physical_residual = (hourly.renewable_used_kw + hourly.ac_grid_dispatch_kw
                         + hourly.emergency_gen_kw + hourly.storage_discharge_kw
                         - hourly.load_served_kw - hourly.storage_charge_kw
                         - hourly.estimated_line_loss_kw)
    if float(physical_residual.abs().max()) > 1e-5:
        raise AssertionError("AC screened power ledger is not conserved")
    ac_frame = pd.DataFrame(ac_rows)
    balance_dir = out / "balance"
    balance_dir.mkdir(exist_ok=True)
    hourly.to_csv(balance_dir / "hourly_balance.csv", index=False)
    ac_frame.to_csv(balance_dir / "ac_screen.csv", index=False)
    if (~ac_frame.ac_voltage_within_bounds).any() or (~ac_frame.ac_grid_within_limit).any():
        raise RuntimeError("Paper annual AC screen failed; see balance/ac_screen.csv")
    metrics = summarize_network_balance(hourly)
    write_json(balance_dir / "metrics.json", metrics)

    stage_names = ["灾前准备"] * 6 + ["灾害冲击"] * 6 + ["灾害持续"] * 12 + ["灾后恢复"] * 12
    stage_rows = []
    event_status = []
    sample_count = int(config.get("paper_method", {}).get("event_samples", 10))
    if sample_count < 1:
        raise ValueError("event_samples must be at least one")
    sample_failure_sets = [failed_sets]
    sample_resource_sequences = [source]
    sample_source_sequences = sample_source_load_sequences(source, seed, sample_count)
    resource_failure_logs = [baseline_resource_failures]
    if event_starts and sample_count >= 2:
        for sample in range(1, sample_count):
            state, _, _ = simulate_line_states(source.wind_speed.to_numpy(float), line_ids,
                                               seed + 1000 * sample, source.timestamp, event_ids)
            sample_failure_sets.append(state)
            resource_sequence, resource_failures = simulate_resource_states(
                base_source, cfg, seed + 1000 * sample, sample)
            sample_resource_sequences.append(resource_sequence)
            resource_failure_logs.append(resource_failures)
    pd.concat(resource_failure_logs, ignore_index=True).to_csv(out / "resource_failures.csv", index=False)
    for event_id, group in source[source.event_id.ge(0)].groupby("event_id", sort=False):
        indices = group.index.to_numpy()
        if len(indices) != 36 or not np.array_equal(indices, np.arange(indices[0], indices[0] + 36)):
            event_status.append({"event_id": int(event_id), "status": "invalid_window",
                                 "hours": len(indices), "reason": "Expected one continuous 36-hour event"})
            continue
        start = int(indices[0])
        event_status.append({"event_id": int(event_id), "status": "analyzed", "hours": 36,
                             "reason": "6/6/12/12 test phase boundaries"})
        event_hourly = hourly.iloc[start:start + 36].copy()
        event_hourly["stage"] = stage_names
        for stage, part in event_hourly.groupby("stage", sort=False):
            stage_rows.append({"event_id": int(event_id), "stage": stage,
                               "phase_boundary_basis": "test_split_of_generated_36h_window",
                               "hours": len(part),
                               "lole_hours": int((part.power_deficit_kw > 1e-6).sum()),
                               "eens_kwh": float(part.power_deficit_kw.sum()),
                               "curtailment_kwh": float(part.curtailment_kw.sum()),
                               "critical_supply_ratio": float(part.critical_load_served_kw.sum() / part.critical_load_kw.sum()),
                               "important_supply_ratio": float(part.important_load_served_kw.sum() / part.important_load_kw.sum())})
        if sample_count >= 2:
            run_conditional_event_samples(source, sample_failure_sets, network, int(event_id),
                                          start, float(soc_before[start]), seed, sample_count,
                                          out / "events" / f"event_{int(event_id):04d}",
                                          resource_sequences=sample_resource_sequences,
                                          source_sequences=sample_source_sequences)
    pd.DataFrame(stage_rows, columns=["event_id", "stage", "phase_boundary_basis",
                                      "hours", "lole_hours", "eens_kwh",
                                      "curtailment_kwh", "critical_supply_ratio",
                                      "important_supply_ratio"]).to_csv(out / "event_phase_metrics.csv", index=False)
    pd.DataFrame(event_status, columns=["event_id", "status", "hours", "reason"]).to_csv(
        out / "event_stage_status.csv", index=False)
    triggers = trigger_strategies(metrics, StrategyThresholds(**config.get("strategy_thresholds", {})))
    triggers.to_csv(out / "strategies.csv", index=False)
    plan = run_simple_planning(metrics, triggers, PlanningConfig(**config.get("planning", {})))
    write_json(out / "planning.json", plan)
    provenance = provenance or {}
    summary = {
        "method": provenance.get("method", "paper annual downstream fixture"),
        "requested": provenance.get("requested"), "embedded": provenance.get("embedded"),
        "embedding_complete": provenance.get("complete"),
        "annual_status": "complete_dispatch", "valid_event_count": sum(x["status"] == "analyzed" for x in event_status),
        "event_stage_status": str(out / "event_stage_status.csv"),
        "weather_pairing": "reference calendar weather; generated extreme wind/solar may not be meteorologically consistent",
        "phase_boundary_basis": "test split of generated 36-hour window, not inferred physical disaster phases",
        "line_failure_model": "weather-conditional test rates, not calibrated fragility curves",
        "random_seed": seed, "conditional_event_samples": sample_count,
        "source_load_uncertainty": {
            "sample_count": sample_count,
            "model": "correlated multiplicative AR(1) residuals around the certified trajectory",
            "load_sigma": 0.03, "renewable_sigma": 0.06,
            "persistence": 0.85, "seed_offset": 700000,
            "reference_path_preserved": True,
        },
        "config_effective": {"topology_dir": str(case33_dir), "network": vars(cfg)},
        "input_hash_sha256": input_hash,
        "nodes_sha256": hashlib.sha256((case33_dir / "nodes.csv").read_bytes()).hexdigest(),
        "lines_sha256": hashlib.sha256((case33_dir / "lines.csv").read_bytes()).hexdigest(),
        "metrics": metrics,
        "chapter3": {"annual_random_production_sequence": str(out / "annual_random_production_sequence.csv"),
                     "resource_failures": str(out / "resource_failures.csv")},
        "chapter4": {"metrics": str(balance_dir / "metrics.json"),
                     "hourly_balance": str(balance_dir / "hourly_balance.csv"),
                     "event_phase_metrics": str(out / "event_phase_metrics.csv")},
        "chapter6": {"planning_result_json": str(out / "planning.json")},
    }
    write_json(out / "run_summary.json", summary)
    return summary


def run_current_pipeline(config_path, config):
    paper = config["paper_method"]
    dataset, year, seed = paper["dataset"], int(paper["year"]), int(paper["seed"])
    paper_root = Path(os.environ.get("CODE_PAPER_ROOT", paper["root"])).expanduser()
    _paper_preflight(paper_root, dataset, bool(paper.get("regenerate", True)))
    output_root = _resolve(ROOT, config.get("output_dir", "outputs")) / "current_paper"
    cmd = [sys.executable, str(ROOT / "scripts/run_current_paper_method.py"),
           "--paper-root", str(paper_root), "--dataset", dataset,
           "--year", str(year), "--seed", str(seed), "--output-root", str(output_root)]
    if paper.get("regenerate", True):
        cmd.append("--regenerate")
    source_override = os.environ.get("PAPER_SOURCE_CSV") or paper.get("source_csv")
    if source_override:
        cmd.extend(["--source-csv", str(_resolve(ROOT, source_override))])
    subprocess.run(cmd, check=True)
    out = output_root / dataset / str(year) / f"seed{seed}"
    annual = pd.read_csv(out / "annual_source_load.csv", parse_dates=["timestamp"])
    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    summary = run_paper_downstream(annual, config, out, provenance)
    summary["config"] = str(config_path)
    write_json(out / "run_summary.json", summary)
    return summary
