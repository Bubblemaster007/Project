"""First reproducible IEEE 33-bus integration using the paper's raw source/load data.

This is an integration baseline, not the unavailable frozen SC-RCRB candidate run.
Line failure and repair parameters are explicit test assumptions, not observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "topic2_remaining_code"))

from src.chapter4.case33_network_balance import Case33NetworkBalance, NetworkBalanceConfig, resource_limits_from_row, summarize_network_balance
from strategy_trigger import StrategyThresholds, trigger_strategies
from resilience_reliability_planner import PlanningConfig, run_simple_planning


def simulate_line_states(wind_speed: np.ndarray, line_ids: list[int], seed: int,
                         timestamps: pd.Series, event_ids: np.ndarray):
    """Sequential conditional line failures over the whole reference year."""
    q95, q99 = np.quantile(wind_speed, [0.95, 0.99])
    stress = np.clip((wind_speed - q95) / max(q99 - q95, 1e-6), 0.0, 1.0)
    # Explicit test rates per hour. Convert hazard rate to transition probability.
    rate = 0.00001 + 0.004 * stress
    rng = np.random.default_rng(seed)
    remaining_repair = {line: 0 for line in line_ids}
    failed_sets = []
    failures = []
    for hour, rate_now in enumerate(rate):
        failed = set()
        probability = -np.expm1(-rate_now)
        for line in line_ids:
            if remaining_repair[line] > 0:
                remaining_repair[line] -= 1
                failed.add(line)
            elif rng.random() < probability:
                repair_hours = int(rng.integers(5, 18))
                remaining_repair[line] = repair_hours - 1
                failed.add(line)
                failures.append({"timestamp": timestamps.iloc[hour], "line": line,
                                 "repair_hours": repair_hours,
                                 "event_id": int(event_ids[hour]),
                                 "failure_rate_per_hour": float(rate_now)})
        failed_sets.append(failed)
    return failed_sets, failures, rate


def select_wind_windows(wind_speed: pd.Series, timestamps: pd.Series) -> list[dict]:
    """Select nonoverlapping 36-hour windows using weather alone."""
    rolling = wind_speed.rolling(6, min_periods=6).mean().to_numpy(float)
    n = len(rolling)
    valid = np.arange(11, n - 25)
    selected = []
    for event_id, (level, quantile) in enumerate([("high", 1.0), ("medium", 0.99), ("low", 0.95)]):
        target = float(np.quantile(rolling[valid], quantile))
        candidate_starts = np.clip(valid - 11, 0, n - 36)
        permitted = np.array([all(abs(int(s) - old["start_index"]) >= 36 for old in selected)
                              for s in candidate_starts])
        if not permitted.any():
            raise ValueError("Cannot find nonoverlapping wind exposure windows")
        score = np.abs(rolling[valid] - target)
        score[~permitted] = np.inf
        chosen = int(np.argmin(score))
        start = int(candidate_starts[chosen])
        selected.append({"event_id": event_id, "wind_exposure_level": level,
                         "target_rolling_quantile": quantile, "start_index": start,
                         "start_time": str(timestamps.iloc[start]),
                         "six_hour_rolling_wind_speed": float(rolling[valid[chosen]]),
                         "window_mean_wind_speed": float(wind_speed.iloc[start:start + 36].mean())})
    return selected


def _reachable_sets(lines: pd.DataFrame, nodes: pd.DataFrame) -> dict[int, set[int]]:
    """Nodes still connected to bus 1 for each single open radial branch."""
    graph = {int(n): [] for n in nodes.node}
    for line in lines.itertuples():
        if int(line.status):
            graph[int(line.from_node)].append((int(line.to_node), int(line.line)))
            graph[int(line.to_node)].append((int(line.from_node), int(line.line)))

    def walk(open_line: int | None) -> set[int]:
        seen = {1}
        stack = [1]
        while stack:
            for other, line_id in graph[stack.pop()]:
                if line_id != open_line and other not in seen:
                    seen.add(other)
                    stack.append(other)
        return seen

    full = walk(None)
    if len(full) != 33 or len(lines[lines.status.eq(1)]) != 32:
        raise ValueError("Expected the classical 33-bus, 32-branch radial case")
    return {int(line.line): walk(int(line.line)) for line in lines.itertuples()}


def run_conditional_event_samples(sequence: pd.DataFrame, sample_failure_sets: list[list[set[int]]],
                                  network: Case33NetworkBalance, event_id: int, start: int,
                                  initial_soc_kwh: float, seed: int, sample_count: int,
                                  output_dir: Path,
                                  resource_sequences: list[pd.DataFrame] | None = None,
                                  source_sequences: list[pd.DataFrame] | None = None) -> dict:
    if sample_count < 2:
        raise ValueError("At least two conditional samples are needed for uncertainty intervals")
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = ["灾前准备"] * 6 + ["灾害冲击"] * 6 + ["灾害持续"] * 12 + ["灾后恢复"] * 12
    records = []
    for sample in range(sample_count):
        sample_seed = seed if sample == 0 else seed + 1000 * sample
        failures = sample_failure_sets[sample]
        soc = initial_soc_kwh
        for offset in range(36):
            i = start + offset
            source_entry = source_sequences[sample].iloc[i] if source_sequences is not None else sequence.iloc[i]
            resource_entry = resource_sequences[sample].iloc[i] if resource_sequences is not None else source_entry
            row, soc, state, ac = network.dispatch_hour_ac_balanced(
                source_entry.timestamp, source_entry.load_kw, source_entry.wind_kw, source_entry.pv_kw,
                failures[i], soc, **resource_limits_from_row(resource_entry))
            if not ac["ac_voltage_within_bounds"] or ac["ac_root_required_kw"] > row["grid_available_kw"] + 1e-5:
                raise RuntimeError(f"Conditional sample {sample} hour {offset} failed AC screen")
            records.append({"sample_id": sample, "seed": sample_seed,
                            "sample_weight": 1.0 / sample_count, "event_id": event_id,
                            "hour_offset": offset, "stage": stages[offset],
                            "timestamp": source_entry.timestamp, "ac_min_voltage_pu": ac["ac_min_voltage_pu"],
                            **row})
    event_hourly = pd.DataFrame(records)
    event_hourly.to_csv(output_dir / "event_conditional_hourly.csv", index=False)
    event_hourly.assign(shortage=(event_hourly.power_deficit_kw > 1e-6).astype(int)).groupby(
        "hour_offset", as_index=False).agg(timestamp=("timestamp", "first"),
                                           lolp=("shortage", "mean"),
                                           mean_deficit_kw=("power_deficit_kw", "mean")).to_csv(
        output_dir / "event_hourly_lolp.csv", index=False)
    per_sample = []
    for (sample_id, stage), group in event_hourly.groupby(["sample_id", "stage"], sort=False):
        per_sample.append({"sample_id": sample_id, "seed": int(group.seed.iloc[0]),
                           "event_id": event_id, "stage": stage, "hours": len(group),
                           "lole_hours": int((group.power_deficit_kw > 1e-6).sum()),
                           "event_shortage_indicator": int((group.power_deficit_kw > 1e-6).any()),
                           "eens_kwh": float(group.power_deficit_kw.sum()),
                           "curtailment_kwh": float(group.curtailment_kw.sum()),
                           "critical_demand_kwh": float(group.critical_load_kw.sum()),
                           "critical_served_kwh": float(group.critical_load_served_kw.sum()),
                           "important_demand_kwh": float(group.important_load_kw.sum()),
                           "important_served_kwh": float(group.important_load_served_kw.sum()),
                           "sample_max_deficit_kw": float(group.power_deficit_kw.max())})
    samples = pd.DataFrame(per_sample)
    samples.to_csv(output_dir / "event_phase_samples.csv", index=False)
    def interval(values: pd.Series) -> tuple[float, float]:
        mean = float(values.mean())
        half = float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
        return max(0.0, mean - half), mean + half
    metrics = []
    for stage in ["灾前准备", "灾害冲击", "灾害持续", "灾后恢复"]:
        group = samples[samples.stage.eq(stage)]
        hours = int(group.hours.iloc[0])
        lole_low, lole_high = interval(group.lole_hours)
        lole_high = min(float(hours), lole_high)
        eens_low, eens_high = interval(group.eens_kwh)
        metrics.append({"event_id": event_id, "stage": stage, "sample_count": sample_count,
                        "sample_weight": 1.0 / sample_count, "hours": hours,
                        "lole_hours": float(group.lole_hours.mean()),
                        "lole_ci95_low_hours": lole_low, "lole_ci95_high_hours": lole_high,
                        "time_average_lolp": float(group.lole_hours.mean() / hours),
                        "event_shortage_probability": float(group.event_shortage_indicator.mean()),
                        "eens_kwh": float(group.eens_kwh.mean()),
                        "eens_ci95_low_kwh": eens_low, "eens_ci95_high_kwh": eens_high,
                        "expected_curtailment_kwh": float(group.curtailment_kwh.mean()),
                        "critical_supply_ratio": float(group.critical_served_kwh.sum() / group.critical_demand_kwh.sum()),
                        "important_supply_ratio": float(group.important_served_kwh.sum() / group.important_demand_kwh.sum()),
                        "observed_max_deficit_kw": float(group.sample_max_deficit_kw.max()),
                        "mean_sample_max_deficit_kw": float(group.sample_max_deficit_kw.mean())})
    pd.DataFrame(metrics).to_csv(output_dir / "event_phase_conditional_metrics.csv", index=False)
    event_totals = samples.groupby("sample_id", sort=True).agg(eens_kwh=("eens_kwh", "sum"),
                                                                 lole_hours=("lole_hours", "sum"))
    convergence = pd.DataFrame({"sample_count": np.arange(1, sample_count + 1),
                                "mean_eens_kwh": event_totals.eens_kwh.cumsum().to_numpy() / np.arange(1, sample_count + 1),
                                "mean_lole_hours": event_totals.lole_hours.cumsum().to_numpy() / np.arange(1, sample_count + 1)})
    convergence.to_csv(output_dir / "event_convergence.csv", index=False)
    return {"sample_count": sample_count, "seed_rule": "sample 0 = seed; sample i = seed + 1000*i",
            "scope": "conditional on the selected reference weather/source-load trajectory and fixed pre-event SOC",
            "phase_metrics": str(output_dir / "event_phase_conditional_metrics.csv"),
            "hourly_lolp": str(output_dir / "event_hourly_lolp.csv"),
            "convergence": str(output_dir / "event_convergence.csv")}


def sample_source_load_sequences(sequence: pd.DataFrame, seed: int, sample_count: int,
                                 load_sigma: float = 0.03,
                                 renewable_sigma: float = 0.06,
                                 persistence: float = 0.85) -> list[pd.DataFrame]:
    """Generate paired source/load uncertainty paths around the certified path.

    The first path is the unmodified reference trajectory. Other paths use
    correlated multiplicative residuals, clipped to non-negative values. The
    perturbation is a downstream uncertainty sample; it never replaces the
    paper's certified annual trajectory or its event constraints.
    """
    if sample_count < 1 or not 0.0 <= persistence < 1.0:
        raise ValueError("sample_count must be positive and persistence must be in [0, 1)")
    base = sequence.copy()
    paths = [base]
    n = len(base)
    rng = np.random.default_rng(seed + 700_000)
    for sample_id in range(1, sample_count):
        out = base.copy()
        load_resid = np.zeros(n)
        renewable_resid = np.zeros(n)
        for i in range(1, n):
            load_resid[i] = persistence * load_resid[i - 1] + np.sqrt(1 - persistence ** 2) * rng.normal()
            renewable_resid[i] = persistence * renewable_resid[i - 1] + np.sqrt(1 - persistence ** 2) * rng.normal()
        load_factor = np.clip(1.0 + load_sigma * load_resid, 0.75, 1.25)
        renewable_factor = np.clip(1.0 + renewable_sigma * renewable_resid, 0.50, 1.50)
        out["load_kw"] = np.maximum(0.0, base["load_kw"].to_numpy(float) * load_factor)
        out["wind_kw"] = np.maximum(0.0, base["wind_kw"].to_numpy(float) * renewable_factor)
        out["pv_kw"] = np.maximum(0.0, base["pv_kw"].to_numpy(float) * renewable_factor)
        out["random_sequence_id"] = f"{base['random_sequence_id'].iloc[0]}_source_sample{sample_id}"
        out["sequence_weight"] = 1.0 / sample_count
        paths.append(out)
    return paths


def run_paired_sensitivity(sequence: pd.DataFrame, network: Case33NetworkBalance,
                           sample_failure_sets: list[list[set[int]]], events: list[dict],
                           soc_before: dict[int, float], output_dir: Path,
                           source_sequences: list[pd.DataFrame] | None = None) -> None:
    """Compare interventions with the same source/load trajectory and failure draws."""
    no_storage = Case33NetworkBalance(network.nodes, network.lines,
                                      replace(network.cfg, storage_power_kw=0.0, storage_energy_kwh=0.0))
    relaxed = Case33NetworkBalance(network.nodes, network.lines,
                                   replace(network.cfg, branch_limit_kw=1e9,
                                           voltage_min_pu=0.0, linear_voltage_guard_pu=0.0,
                                           voltage_max_pu=2.0))
    rows = []
    for event in events:
        eid, start = event["event_id"], event["start_index"]
        baseline = pd.read_csv(output_dir / f"event_{eid}_{event['wind_exposure_level']}" / "event_conditional_hourly.csv")
        for sample, failures in enumerate(sample_failure_sets):
            base = baseline[baseline.sample_id.eq(sample)]
            rows.append({"event_id": eid, "wind_exposure_level": event["wind_exposure_level"],
                         "sample_id": sample, "scenario": "baseline",
                         "eens_kwh": float(base.power_deficit_kw.sum()),
                         "lole_hours": int((base.power_deficit_kw > 1e-6).sum())})
            for label, model, ignore_failures, initial_soc in (
                ("no_line_failure", network, True, soc_before[eid]),
                ("no_storage", no_storage, False, 0.0),
                ("relaxed_network_ideal", relaxed, False, soc_before[eid]),
            ):
                soc = initial_soc
                deficits = []
                for offset in range(36):
                    i = start + offset
                    entry = source_sequences[sample].iloc[i] if source_sequences is not None else sequence.iloc[i]
                    row, soc, _, _ = model.dispatch_hour_ac_balanced(
                        entry.timestamp, entry.load_kw, entry.wind_kw, entry.pv_kw,
                        set() if ignore_failures else failures[i], soc,
                        **resource_limits_from_row(entry))
                    deficits.append(row["power_deficit_kw"])
                rows.append({"event_id": eid, "wind_exposure_level": event["wind_exposure_level"],
                             "sample_id": sample, "scenario": label,
                             "eens_kwh": float(sum(deficits)),
                             "lole_hours": int(sum(value > 1e-6 for value in deficits))})
    detail = pd.DataFrame(rows)
    detail.to_csv(output_dir / "paired_sensitivity_samples.csv", index=False)
    detail.groupby(["event_id", "wind_exposure_level", "scenario"], as_index=False).agg(
        sample_count=("sample_id", "nunique"), mean_eens_kwh=("eens_kwh", "mean"),
        mean_lole_hours=("lole_hours", "mean")).to_csv(
            output_dir / "paired_sensitivity_summary.csv", index=False)


def run(paper_csv: Path, output_dir: Path, seed: int = 42, event_samples: int = 30,
        project_config: dict | None = None) -> dict:
    project_config = project_config or {}
    raw = pd.read_csv(paper_csv, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    needed = ["time", "load", "wind_power", "solar_power", "wind_speed"]
    if raw[needed].isna().any().any() or raw.time.duplicated().any():
        raise ValueError("Paper source/load data contain missing values or duplicate timestamps")
    if len(raw) != 8760 or not raw.time.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Expected one continuous 8760-hour paper reference series")
    topology = project_config.get("topology", {})
    if topology.get("type", "case33") != "case33":
        raise ValueError("The paper reference entry currently requires case33 topology")
    case33_dir = Path(topology.get("case33_dir", "data/case33"))
    if not case33_dir.is_absolute():
        case33_dir = ROOT / case33_dir
    nodes = pd.read_csv(case33_dir / "nodes.csv")
    lines = pd.read_csv(case33_dir / "lines.csv")
    resources = NetworkBalanceConfig.from_project(project_config)
    connected_by_open = _reachable_sets(lines, nodes)
    all_nodes = set(nodes.node.astype(int))
    weights = nodes.set_index("node").pd_kw.astype(float)
    peak = float(weights.sum())
    scale = peak / float(raw.load.max())
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("Invalid source/load scale")

    # Exposure labels are selected solely from weather, never from deficit.
    events = select_wind_windows(raw.wind_speed, raw.time)
    start = events[0]["start_index"]
    event_ids = np.full(len(raw), -1, dtype=int)
    for event in events:
        s = event["start_index"]
        event_ids[s:s + 36] = event["event_id"]
    line_ids = lines.line.astype(int).tolist()
    failed_sets, failures, failure_rates = simulate_line_states(
        raw.wind_speed.to_numpy(float), line_ids, seed, raw.time, event_ids)
    reachable = np.empty(len(raw), dtype=float)
    wind_reachable = np.empty(len(raw), dtype=bool)
    pv_reachable = np.empty(len(raw), dtype=bool)
    failed_count = np.empty(len(raw), dtype=int)
    for hour in range(len(raw)):
        failed = failed_sets[hour]
        connected = all_nodes.copy()
        for line in failed:
            connected &= connected_by_open[line]
        reachable[hour] = float(weights.loc[list(connected)].sum()) / peak
        wind_reachable[hour] = resources.wind_node in connected
        pv_reachable[hour] = resources.pv_node in connected
        failed_count[hour] = len(failed)

    sequence = pd.DataFrame({
        "timestamp": raw.time,
        "load_kw": raw.load * scale,
        "wind_kw": raw.wind_power * scale,
        "pv_kw": raw.solar_power * scale,
        "event_id": event_ids,
        "is_extreme_condition": (event_ids >= 0).astype(int),
        "random_sequence_id": f"paper_reference_seed{seed}",
        "sequence_weight": 1.0,
        "wind_speed": raw.wind_speed,
        "line_failure_rate_per_hour": failure_rates,
        "failed_line_count": failed_count,
        "load_reachability_factor": reachable,
        "reachable_load_kw": raw.load * scale * reachable,
        "available_re_kw": raw.wind_power * scale * wind_reachable + raw.solar_power * scale * pv_reachable,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(events).to_csv(output_dir / "weather_selected_events.csv", index=False)
    sequence.to_csv(output_dir / "annual_sequence.csv", index=False)
    pd.DataFrame(failures, columns=["timestamp", "line", "repair_hours", "event_id", "failure_rate_per_hour"]).to_csv(output_dir / "line_failures.csv", index=False)
    network = Case33NetworkBalance(nodes, lines, resources)
    soc = resources.soc_initial_ratio * resources.storage_energy_kwh
    dispatch_rows = []
    ac_rows = []
    event_node_rows = []
    event_line_rows = []
    ac_hours = set(range(len(sequence)))
    soc_at_event_start = {}
    event_start_by_index = {event["start_index"]: event["event_id"] for event in events}
    for i, entry in enumerate(sequence.itertuples(index=False)):
        if i in event_start_by_index:
            soc_at_event_start[event_start_by_index[i]] = soc
        row, soc, node_state, ac = network.dispatch_hour_ac_balanced(
            entry.timestamp, entry.load_kw, entry.wind_kw, entry.pv_kw,
            failed_sets[i], soc, **resource_limits_from_row(entry))
        row["event_id"] = int(entry.event_id)
        row["random_sequence_id"] = entry.random_sequence_id
        row["sequence_weight"] = 1.0
        dispatch_rows.append(row)
        if int(entry.event_id) >= 0:
            connected_nodes = network.connected(failed_sets[i])
            for n, node_id in enumerate(network.node_ids):
                event_node_rows.append({"timestamp": entry.timestamp, "event_id": int(entry.event_id),
                                        "node": int(node_id),
                                        "demand_kw": float(entry.load_kw * network.load_weights[n] / network.peak_kw),
                                        "served_kw": float(node_state["served_kw"][n]),
                                        "unserved_kw": float(entry.load_kw * network.load_weights[n] / network.peak_kw
                                                             - node_state["served_kw"][n]),
                                        "critical_served_kw": float(node_state["critical_served_kw"][n]),
                                        "important_served_kw": float(node_state["important_served_kw"][n]),
                                        "ordinary_served_kw": float(node_state["ordinary_served_kw"][n]),
                                        "linear_voltage_pu": float(np.sqrt(node_state["voltage_squared"][n])),
                                        "connected_to_slack": int(n in connected_nodes)})
            for j, line_id in enumerate(network.line_ids):
                event_line_rows.append({"timestamp": entry.timestamp, "event_id": int(entry.event_id),
                                        "line": int(line_id), "from_node": int(network.node_ids[network.from_idx[j]]),
                                        "to_node": int(network.node_ids[network.to_idx[j]]),
                                        "flow_kw": float(node_state["branch_flow_kw"][j]),
                                        "available": int(int(line_id) not in failed_sets[i])})
        if i in ac_hours:
            physical_residual = (row["renewable_used_kw"] + row["ac_grid_dispatch_kw"]
                                 + row["emergency_gen_kw"] + row["storage_discharge_kw"]
                                 - row["load_served_kw"] - row["storage_charge_kw"]
                                 - row["estimated_line_loss_kw"])
            if abs(physical_residual) > 1e-5:
                raise AssertionError(f"AC screened energy balance failed at {entry.timestamp}")
            ac_rows.append({"timestamp": entry.timestamp, "event_id": int(entry.event_id), **ac,
                            "linear_grid_dispatch_kw": row["firm_grid_dispatch_kw"],
                            "ac_loss_reserve_error_kw": ac["ac_root_required_kw"] - row["firm_grid_dispatch_kw"],
                            "ac_line_loss_kw": row["estimated_line_loss_kw"],
                            "linear_loss_reserve_kw": row["loss_reserve_kw"],
                            "ac_grid_within_limit": bool(ac["ac_root_required_kw"] <= row["grid_available_kw"] + 1e-5)})
    hourly = pd.DataFrame(dispatch_rows)
    metrics = summarize_network_balance(hourly)
    if not np.allclose(hourly.load_kw, hourly.load_served_kw + hourly.power_deficit_kw, atol=1e-6):
        raise AssertionError("Hourly original load is not conserved")
    (output_dir / "balance").mkdir(exist_ok=True)
    hourly.to_csv(output_dir / "balance/hourly_balance.csv", index=False)
    ac_frame = pd.DataFrame(ac_rows)
    ac_frame.to_csv(output_dir / "balance/ac_screen.csv", index=False)
    pd.DataFrame(event_node_rows).to_csv(output_dir / "event_nodal_dispatch.csv", index=False)
    pd.DataFrame(event_line_rows).to_csv(output_dir / "event_branch_flows.csv", index=False)
    if (~ac_frame.ac_voltage_within_bounds).any() or (~ac_frame.ac_grid_within_limit).any():
        raise RuntimeError("AC screen found voltage or grid capacity violations; see balance/ac_screen.csv")
    (output_dir / "balance/metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # Boundaries are an explicit test split, not inferred disaster physics.
    phase_names = ["灾前准备"] * 6 + ["灾害冲击"] * 6 + ["灾害持续"] * 12 + ["灾后恢复"] * 12
    phase_parts = []
    for event in events:
        part = hourly.iloc[event["start_index"]:event["start_index"] + 36].copy()
        part["stage"] = phase_names
        part["event_id"] = event["event_id"]
        part["wind_exposure_level"] = event["wind_exposure_level"]
        phase_parts.append(part)
    phase = pd.concat(phase_parts, ignore_index=True)
    phase.to_csv(output_dir / "event_phase_hourly.csv", index=False)
    phase_rows = []
    for (event_id, name), group in phase.groupby(["event_id", "stage"], sort=False):
        phase_rows.append({"event_id": event_id, "wind_exposure_level": group.wind_exposure_level.iloc[0],
                           "stage": name, "hours": len(group),
                           "lole_hours": int((group.power_deficit_kw > 1e-6).sum()),
                           "time_average_lolp": float((group.power_deficit_kw > 1e-6).mean()),
                           "event_shortage_indicator": int((group.power_deficit_kw > 1e-6).any()),
                           "eens_kwh": float(group.power_deficit_kw.sum()),
                           "curtailment_kwh": float(group.curtailment_kw.sum()),
                           "critical_supply_ratio": float(group.critical_load_served_kw.sum() / group.critical_load_kw.sum()),
                           "important_supply_ratio": float(group.important_load_served_kw.sum() / group.important_load_kw.sum()),
                           "max_deficit_kw": float(group.power_deficit_kw.max())})
    pd.DataFrame(phase_rows).to_csv(output_dir / "event_phase_metrics.csv", index=False)
    sample_failure_sets = [failed_sets]
    source_sequences = sample_source_load_sequences(sequence, seed, event_samples)
    for sample in range(1, event_samples):
        other, _, _ = simulate_line_states(raw.wind_speed.to_numpy(float), line_ids,
                                           seed + 1000 * sample, raw.time, event_ids)
        sample_failure_sets.append(other)
    conditional = []
    conditional_frames = []
    for event in events:
        event_dir = output_dir / f"event_{event['event_id']}_{event['wind_exposure_level']}"
        result = run_conditional_event_samples(sequence, sample_failure_sets, network,
                                               event["event_id"], event["start_index"],
                                               float(soc_at_event_start[event["event_id"]]),
                                               seed, event_samples, event_dir,
                                               source_sequences=source_sequences)
        conditional.append({**event, **result})
        frame = pd.read_csv(event_dir / "event_phase_conditional_metrics.csv")
        frame["wind_exposure_level"] = event["wind_exposure_level"]
        conditional_frames.append(frame)
    pd.concat(conditional_frames, ignore_index=True).to_csv(
        output_dir / "wind_exposure_phase_comparison.csv", index=False)
    # Keep the primary high-wind event at the historical output paths.
    high_dir = output_dir / "event_0_high"
    for filename in ("event_conditional_hourly.csv", "event_hourly_lolp.csv",
                     "event_phase_samples.csv", "event_phase_conditional_metrics.csv",
                     "event_convergence.csv"):
        shutil.copyfile(high_dir / filename, output_dir / filename)
    run_paired_sensitivity(sequence, network, sample_failure_sets, events,
                           soc_at_event_start, output_dir,
                           source_sequences=source_sequences)
    triggers = trigger_strategies(metrics, StrategyThresholds(**project_config.get("strategy_thresholds", {})))
    triggers.to_csv(output_dir / "strategies.csv", index=False)
    plan = run_simple_planning(metrics, triggers, PlanningConfig(**project_config.get("planning", {})))
    (output_dir / "planning.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"status": "integration_baseline", "source": str(paper_csv),
               "source_sha256": hashlib.sha256(paper_csv.read_bytes()).hexdigest(),
               "nodes_sha256": hashlib.sha256((case33_dir / "nodes.csv").read_bytes()).hexdigest(),
               "lines_sha256": hashlib.sha256((case33_dir / "lines.csv").read_bytes()).hexdigest(),
               "effective_config_sha256": hashlib.sha256(json.dumps(project_config, sort_keys=True,
                                                                      ensure_ascii=False).encode("utf-8")).hexdigest(),
               "topology": "MATPOWER case33bw, 33 buses, 32 radial branches",
               "topology_source": str(case33_dir),
               "method": "raw paper source/load + illustrative sequential line failure/repair + hourly radial LinDistFlow dispatch",
               "scale_to_case33_peak_kw": peak, "scale_factor": scale, "seed": seed,
               "event_start": str(raw.time.iloc[start]),
               "event_definition": "nonoverlapping weather-selected 36-hour windows at six-hour rolling wind quantiles 1.00/0.99/0.95; test split 6/6/12/12",
               "weather_selected_events": events,
               "resources": vars(resources), "failures": len(failures), "metrics": metrics,
               "conditional_event": conditional,
               "source_load_uncertainty": {
                   "sample_count": event_samples,
                   "model": "correlated multiplicative AR(1) residuals around the certified trajectory",
                   "load_sigma": 0.03,
                   "renewable_sigma": 0.06,
                   "persistence": 0.85,
                   "seed_offset": 700000,
                   "reference_path_preserved": True,
               },
               "paired_sensitivity": str(output_dir / "paired_sensitivity_summary.csv"),
               "ac_screen": {"hours": len(ac_frame), "minimum_voltage_pu": float(ac_frame.ac_min_voltage_pu.min()),
                             "voltage_violations": int((~ac_frame.ac_voltage_within_bounds).sum()),
                             "grid_capacity_violations": int((~ac_frame.ac_grid_within_limit).sum()),
                             "maximum_ac_line_loss_kw": float(ac_frame.ac_line_loss_kw.max()),
                             "maximum_abs_loss_reserve_error_kw": float(ac_frame.ac_loss_reserve_error_kw.abs().max())},
               "limitations": ["Frozen SC-RCRB candidate bank and publication_protocol are unavailable locally",
                               "Weather-conditional line failure rates and repair durations are test assumptions",
                               "Branch thermal ratings are study assumptions; LinDistFlow is an approximation with AC screening, not an AC optimal dispatch",
                               "One reference sequence does not estimate a calibrated event probability or Alashankou risk"],
               "hourly_balance": str(output_dir / "balance/hourly_balance.csv")}
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/paper_reference_case33")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--event-samples", type=int, default=30)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    print(json.dumps(run(args.paper_csv, args.output_dir, args.seed, args.event_samples,
                         json.loads(args.config.read_text(encoding="utf-8-sig"))), ensure_ascii=False, indent=2))
