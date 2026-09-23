"""Hourly radial LinDistFlow dispatch for the classical 33-bus test feeder.

Active branch flow and squared-voltage equations use the lossless linear
approximation. Reactive demand follows each bus's case33bw Q/P ratio.
Branch ratings are explicit study assumptions because case33bw RATE_A is zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass(frozen=True)
class NetworkBalanceConfig:
    branch_limit_kw: float = 5000.0
    grid_limit_kw: float = 4000.0
    emergency_limit_kw: float = 500.0
    storage_power_kw: float = 500.0
    storage_energy_kwh: float = 1500.0
    soc_initial_ratio: float = 0.55
    soc_min_ratio: float = 0.10
    soc_max_ratio: float = 0.95
    eta_charge: float = 0.92
    eta_discharge: float = 0.90
    critical_ratio: float = 0.20
    important_ratio: float = 0.35
    wind_node: int = 18
    pv_node: int = 30
    voltage_min_pu: float = 0.95
    voltage_max_pu: float = 1.05
    linear_voltage_guard_pu: float = 0.955
    base_voltage_kv: float = 12.66

    @classmethod
    def from_project(cls, project: dict[str, Any]) -> "NetworkBalanceConfig":
        chapter4 = project.get("chapter4", {})
        network = project.get("case33_network", {})
        values = {
            "grid_limit_kw": float(chapter4.get("grid_channel_kw", 4000.0)) + float(chapter4.get("firm_supply_kw", 0.0)),
            "emergency_limit_kw": float(chapter4.get("emergency_gen_kw", 500.0)),
            "storage_power_kw": float(chapter4.get("storage_power_kw", 500.0)),
            "storage_energy_kwh": float(chapter4.get("storage_energy_kwh", 1500.0)),
            "soc_initial_ratio": float(chapter4.get("soc_initial_ratio", 0.55)),
            "soc_min_ratio": float(chapter4.get("soc_min_ratio", 0.10)),
            "soc_max_ratio": float(chapter4.get("soc_max_ratio", 0.95)),
            "eta_charge": float(chapter4.get("eta_charge", 0.92)),
            "eta_discharge": float(chapter4.get("eta_discharge", 0.90)),
            "critical_ratio": float(chapter4.get("critical_load_ratio", 0.20)),
            "important_ratio": float(chapter4.get("important_load_ratio", 0.35)),
        }
        for key in ("branch_limit_kw", "wind_node", "pv_node", "voltage_min_pu",
                    "voltage_max_pu", "linear_voltage_guard_pu", "base_voltage_kv"):
            if key in network:
                values[key] = network[key]
        cfg = cls(**values)
        if (cfg.branch_limit_kw <= 0 or cfg.grid_limit_kw < 0 or cfg.emergency_limit_kw < 0
                or cfg.storage_power_kw < 0 or cfg.storage_energy_kwh < 0
                or not 0 <= cfg.soc_min_ratio <= cfg.soc_initial_ratio <= cfg.soc_max_ratio <= 1
                or not 0 < cfg.eta_charge <= 1 or not 0 < cfg.eta_discharge <= 1
                or cfg.critical_ratio < 0 or cfg.important_ratio < 0
                or cfg.critical_ratio + cfg.important_ratio > 1):
            raise ValueError("Invalid case33 resource or load-tier configuration")
        return cfg


class Case33NetworkBalance:
    def __init__(self, nodes: pd.DataFrame, lines: pd.DataFrame, config: NetworkBalanceConfig):
        self.cfg = config
        self.nodes = nodes.sort_values("node").reset_index(drop=True)
        self.lines = lines.loc[lines.status.eq(1)].sort_values("line").reset_index(drop=True)
        self.node_ids = self.nodes.node.astype(int).to_numpy()
        self.line_ids = self.lines.line.astype(int).to_numpy()
        if len(self.nodes) != 33 or len(self.lines) != 32 or self.node_ids[0] != 1:
            raise ValueError("The network must be the classical radial 33-bus case")
        self.node_index = {int(n): i for i, n in enumerate(self.node_ids)}
        self.from_idx = np.array([self.node_index[int(n)] for n in self.lines.from_node])
        self.to_idx = np.array([self.node_index[int(n)] for n in self.lines.to_node])
        self.load_weights = self.nodes.pd_kw.to_numpy(float)
        self.peak_kw = float(self.load_weights.sum())
        self.q_over_p = np.divide(self.nodes.qd_kvar.to_numpy(float), self.load_weights,
                                  out=np.zeros(33), where=self.load_weights > 0)
        self.subtree = np.zeros((32, 33))
        for j, node in enumerate(self.to_idx):
            descendants = {int(node)}
            while True:
                more = {int(self.to_idx[k]) for k, parent in enumerate(self.from_idx) if int(parent) in descendants}
                if more <= descendants:
                    break
                descendants |= more
            self.subtree[j, list(descendants)] = 1.0
        # Variables: served critical/important/ordinary by bus, wind, PV,
        # root grid/emergency/discharge/charge, branch P, squared bus voltage.
        self.s_c = slice(0, 33)
        self.s_i = slice(33, 66)
        self.s_o = slice(66, 99)
        self.i_wind, self.i_pv = 99, 100
        self.i_grid, self.i_emergency, self.i_discharge, self.i_charge = 101, 102, 103, 104
        self.s_flow = slice(105, 137)
        self.s_voltage = slice(137, 170)
        self.nvar = 170
        self.c = np.zeros(self.nvar)
        self.c[self.s_c] = -1000.0
        self.c[self.s_i] = -100.0
        self.c[self.s_o] = -10.0
        self.c[[self.i_grid, self.i_emergency, self.i_discharge, self.i_charge]] = [0.1, 0.5, 0.3, -0.02]
        # Equal-price renewables can switch arbitrarily between LP vertices,
        # making the AC loss feedback discontinuous. Break the tie using the
        # smaller resistance distance from the slack as a transparent heuristic.
        parent = {int(b): (int(a), float(self.lines.iloc[j].r_ohm))
                  for j, (a, b) in enumerate(zip(self.from_idx, self.to_idx))}
        def path_resistance(node: int) -> float:
            total = 0.0
            while node != 0:
                node, resistance = parent[node]
                total += resistance
            return total
        wind_path = path_resistance(self.node_index[config.wind_node])
        pv_path = path_resistance(self.node_index[config.pv_node])
        self.c[self.i_wind] = -0.011 if wind_path < pv_path else -0.010
        self.c[self.i_pv] = -0.011 if pv_path < wind_path else -0.010
        self.aeq = np.zeros((65, self.nvar))
        # Nodal active-power balance: injection + incoming - outgoing - served = 0.
        for n in range(33):
            for served in (self.s_c, self.s_i, self.s_o):
                self.aeq[n, served.start + n] = -1.0
        for j, (a, b) in enumerate(zip(self.from_idx, self.to_idx)):
            self.aeq[a, self.s_flow.start + j] = -1.0
            self.aeq[b, self.s_flow.start + j] = 1.0
        self.aeq[self.node_index[config.wind_node], self.i_wind] = 1.0
        self.aeq[self.node_index[config.pv_node], self.i_pv] = 1.0
        self.aeq[0, [self.i_grid, self.i_emergency, self.i_discharge, self.i_charge]] = [1, 1, 1, -1]
        # u_to - u_from + 2(rP+xQ)/V_LL^2 = 0, with Q from served load.
        voltage_factor = 2.0 / (config.base_voltage_kv * 1000.0) ** 2 * 1000.0
        for j, line in enumerate(self.lines.itertuples()):
            row = 33 + j
            self.aeq[row, self.s_voltage.start + self.to_idx[j]] = 1.0
            self.aeq[row, self.s_voltage.start + self.from_idx[j]] = -1.0
            self.aeq[row, self.s_flow.start + j] = voltage_factor * float(line.r_ohm)
            qcoeff = voltage_factor * float(line.x_ohm) * self.subtree[j] * self.q_over_p
            for served in (self.s_c, self.s_i, self.s_o):
                self.aeq[row, served] = qcoeff
        self.beq = np.zeros(65)
        self.all_nodes = set(range(33))

    def connected(self, failed_lines: set[int]) -> set[int]:
        adjacency = [[] for _ in range(33)]
        for j, line in enumerate(self.line_ids):
            if int(line) not in failed_lines:
                a, b = int(self.from_idx[j]), int(self.to_idx[j])
                adjacency[a].append(b)
                adjacency[b].append(a)
        seen = {0}
        stack = [0]
        while stack:
            for other in adjacency[stack.pop()]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        return seen

    def dispatch_hour(self, timestamp, load_kw: float, wind_kw: float, pv_kw: float,
                      failed_lines: set[int], soc_kwh: float,
                      grid_available_kw: float | None = None,
                      emergency_available_kw: float | None = None,
                      storage_available_kw: float | None = None,
                      renewable_available_kw: float | None = None,
                      loss_reserve_kw: float = 0.0) -> tuple[dict, float, dict]:
        cfg = self.cfg
        if not np.isfinite(loss_reserve_kw) or loss_reserve_kw < 0:
            raise ValueError(f"Invalid loss reserve at {timestamp}: {loss_reserve_kw}")
        def available_limit(name: str, configured: float, reported: float | None) -> float:
            if reported is None:
                return configured
            if not np.isfinite(reported) or reported < 0:
                raise ValueError(f"Invalid {name} capacity at {timestamp}: {reported}")
            return min(configured, float(reported))
        grid_cap = available_limit("grid", cfg.grid_limit_kw, grid_available_kw)
        emergency_cap = available_limit("emergency", cfg.emergency_limit_kw, emergency_available_kw)
        storage_cap = available_limit("storage", cfg.storage_power_kw, storage_available_kw)
        re_cap = available_limit("renewable", max(0.0, wind_kw + pv_kw), renewable_available_kw)
        re_factor = min(1.0, re_cap / max(wind_kw + pv_kw, 1e-9))
        connected = self.connected(failed_lines)
        demand = self.load_weights / self.peak_kw * float(load_kw)
        bounds: list[tuple[float | None, float | None]] = []
        for ratio in (cfg.critical_ratio, cfg.important_ratio, 1 - cfg.critical_ratio - cfg.important_ratio):
            if ratio < 0:
                raise ValueError("Load priority ratios exceed one")
            bounds.extend((0.0, max(0.0, demand[n] * ratio) if n in connected else 0.0) for n in range(33))
        bounds += [(0.0, max(0.0, wind_kw) * re_factor if self.node_index[cfg.wind_node] in connected else 0.0),
                   (0.0, max(0.0, pv_kw) * re_factor if self.node_index[cfg.pv_node] in connected else 0.0)]
        discharge_max = min(storage_cap, max(0.0, soc_kwh - cfg.soc_min_ratio * cfg.storage_energy_kwh) * cfg.eta_discharge)
        charge_max = min(storage_cap, max(0.0, cfg.soc_max_ratio * cfg.storage_energy_kwh - soc_kwh) / cfg.eta_charge)
        bounds += [(0.0, grid_cap), (0.0, emergency_cap),
                   (0.0, discharge_max), (0.0, charge_max)]
        bounds += [(-cfg.branch_limit_kw, cfg.branch_limit_kw) if int(line) not in failed_lines else (0.0, 0.0)
                   for line in self.line_ids]
        bounds += [(1.0, 1.0)] + [(max(cfg.voltage_min_pu, cfg.linear_voltage_guard_pu) ** 2,
                                   cfg.voltage_max_pu ** 2)] * 32
        beq = self.beq.copy()
        beq[0] = loss_reserve_kw
        solved = linprog(self.c, A_eq=self.aeq, b_eq=beq, bounds=bounds, method="highs")
        if not solved.success:
            raise RuntimeError(f"LinDistFlow dispatch failed at {timestamp}: {solved.message}")
        x = solved.x
        served_by_tier = [float(x[s].sum()) for s in (self.s_c, self.s_i, self.s_o)]
        served = sum(served_by_tier)
        deficit = max(float(load_kw) - served, 0.0)
        renew_used = float(x[self.i_wind] + x[self.i_pv])
        curtailment = max(re_cap - renew_used, 0.0)
        unavailable_re = max(float(wind_kw + pv_kw) - re_cap, 0.0)
        if deficit < 1e-6:
            deficit = 0.0
        if curtailment < 1e-6:
            curtailment = 0.0
        soc_next = soc_kwh + cfg.eta_charge * x[self.i_charge] - x[self.i_discharge] / cfg.eta_discharge
        row = {"timestamp": timestamp, "load_kw": float(load_kw), "load_served_kw": served,
               "power_deficit_kw": deficit, "potential_re_kw": float(wind_kw + pv_kw),
               "available_re_kw": re_cap,
               "renewable_used_kw": renew_used, "curtailment_kw": curtailment,
               "renewable_unavailable_kw": unavailable_re,
               "loss_reserve_kw": float(loss_reserve_kw),
               "firm_grid_dispatch_kw": float(x[self.i_grid]), "emergency_gen_kw": float(x[self.i_emergency]),
               "storage_discharge_kw": float(x[self.i_discharge]), "storage_charge_kw": float(x[self.i_charge]),
               "soc": soc_next / cfg.storage_energy_kwh if cfg.storage_energy_kwh else 0.0,
               "critical_load_kw": float(load_kw * cfg.critical_ratio),
               "critical_load_served_kw": served_by_tier[0],
               "important_load_kw": float(load_kw * cfg.important_ratio),
               "important_load_served_kw": served_by_tier[1],
               "ordinary_load_served_kw": served_by_tier[2],
               "min_linear_voltage_pu": float(np.sqrt(x[self.s_voltage].min())),
               "max_linear_voltage_pu": float(np.sqrt(x[self.s_voltage].max())),
               "max_abs_branch_flow_kw": float(np.abs(x[self.s_flow]).max()),
               "failed_line_count": len(failed_lines),
               "grid_available_kw": grid_cap,
               "emergency_available_kw": emergency_cap,
               "storage_available_kw": storage_cap,
               "renewable_available_kw": re_cap,
               "reachable_load_kw": float(demand[list(connected)].sum())}
        node_state = {"served_kw": x[self.s_c] + x[self.s_i] + x[self.s_o],
                      "critical_served_kw": x[self.s_c].copy(),
                      "important_served_kw": x[self.s_i].copy(),
                      "ordinary_served_kw": x[self.s_o].copy(),
                      "wind_used_kw": float(x[self.i_wind]), "pv_used_kw": float(x[self.i_pv]),
                      "storage_discharge_kw": float(x[self.i_discharge]),
                      "storage_charge_kw": float(x[self.i_charge]),
                      "emergency_kw": float(x[self.i_emergency]),
                      "branch_flow_kw": x[self.s_flow].copy(), "voltage_squared": x[self.s_voltage].copy()}
        return row, float(soc_next), node_state

    def dispatch_hour_ac_balanced(self, timestamp, load_kw: float, wind_kw: float, pv_kw: float,
                                  failed_lines: set[int], soc_kwh: float,
                                  **resource_limits) -> tuple[dict, float, dict, dict]:
        """Iterate a root loss reserve until the AC energy ledger closes."""
        reserve = 0.0
        trail = []
        for iteration in range(40):
            row, soc_next, state = self.dispatch_hour(
                timestamp, load_kw, wind_kw, pv_kw, failed_lines, soc_kwh,
                loss_reserve_kw=reserve, **resource_limits)
            ac = self.ac_screen(state, failed_lines)
            actual_loss = max(0.0, ac["ac_root_required_kw"] - row["firm_grid_dispatch_kw"] + reserve)
            trail.append((reserve, actual_loss, row["firm_grid_dispatch_kw"], row["load_served_kw"],
                          row["renewable_used_kw"], row["emergency_gen_kw"],
                          row["storage_discharge_kw"], row["storage_charge_kw"]))
            if abs(actual_loss - reserve) <= 1e-5:
                row["ac_grid_dispatch_kw"] = ac["ac_root_required_kw"]
                row["estimated_line_loss_kw"] = actual_loss
                row["ac_min_voltage_pu"] = ac["ac_min_voltage_pu"]
                row["ac_loss_iterations"] = iteration + 1
                return row, soc_next, state, ac
            reserve = actual_loss
        raise RuntimeError(f"AC loss reserve did not converge at {timestamp}: {trail[-5:]}")


    def ac_screen(self, node_state: dict, failed_lines: set[int]) -> dict[str, float | bool]:
        """Backward/forward sweep of the dispatched constant-PQ injections.

        dispatch_hour_ac_balanced uses this result to iterate a loss reserve.
        """
        connected = self.connected(failed_lines)
        children = [[] for _ in range(33)]
        parent_line = np.full(33, -1, dtype=int)
        for j, (a, b) in enumerate(zip(self.from_idx, self.to_idx)):
            if int(self.line_ids[j]) not in failed_lines and int(a) in connected and int(b) in connected:
                children[a].append(int(b))
                parent_line[b] = j
        order = [0]
        for node in order:
            order.extend(children[node])
        p = np.asarray(node_state["served_kw"], dtype=float).copy()
        p[self.node_index[self.cfg.wind_node]] -= node_state["wind_used_kw"]
        p[self.node_index[self.cfg.pv_node]] -= node_state["pv_used_kw"]
        p[0] += node_state["storage_charge_kw"] - node_state["storage_discharge_kw"] - node_state["emergency_kw"]
        q = np.asarray(node_state["served_kw"], dtype=float) * self.q_over_p
        # case33bw specifies 10 MVA, 12.66 kV base.
        s = (p + 1j * q) / 10000.0
        impedance_base = self.cfg.base_voltage_kv ** 2 / 10.0
        z = (self.lines.r_ohm.to_numpy(float) + 1j * self.lines.x_ohm.to_numpy(float)) / impedance_base
        voltage = np.ones(33, dtype=complex)
        for iteration in range(60):
            injection = np.zeros(33, dtype=complex)
            for node in connected:
                injection[node] = np.conj(s[node] / voltage[node])
            branch_current = np.zeros(32, dtype=complex)
            accumulated = injection.copy()
            for node in reversed(order[1:]):
                j = parent_line[node]
                branch_current[j] = accumulated[node]
                accumulated[int(self.from_idx[j])] += accumulated[node]
            updated = np.ones(33, dtype=complex)
            for node in order[1:]:
                j = parent_line[node]
                updated[node] = updated[int(self.from_idx[j])] - z[j] * branch_current[j]
            error = float(np.max(np.abs(updated[list(connected)] - voltage[list(connected)])))
            voltage = updated
            if error < 1e-10:
                break
        if error >= 1e-10:
            raise RuntimeError("AC backward/forward sweep did not converge")
        root_injection_kw = float((voltage[0] * np.conj(accumulated[0])).real * 10000.0)
        return {"ac_min_voltage_pu": float(np.min(np.abs(voltage[list(connected)]))),
                "ac_max_voltage_pu": float(np.max(np.abs(voltage[list(connected)]))),
                "ac_root_required_kw": root_injection_kw,
                "ac_converged": True,
                "ac_voltage_within_bounds": bool(np.all(np.abs(voltage[list(connected)]) >= self.cfg.voltage_min_pu - 1e-8)
                                                  and np.all(np.abs(voltage[list(connected)]) <= self.cfg.voltage_max_pu + 1e-8))}


def resource_limits_from_row(row) -> dict[str, float]:
    """Pass only explicitly supplied hourly capacity fields to dispatch_hour."""
    mapping = {
        "available_kw_grid_channel": "grid_available_kw",
        "available_kw_emergency_gen": "emergency_available_kw",
        "available_kw_storage": "storage_available_kw",
        "available_kw_renewable": "renewable_available_kw",
    }
    values = {}
    def has_column(column: str) -> bool:
        return (isinstance(row, pd.Series) and column in row) or hasattr(row, column)
    def field(column: str) -> float:
        return float(row[column] if isinstance(row, pd.Series) else getattr(row, column))
    for column, argument in mapping.items():
        if has_column(column):
            values[argument] = field(column)
    if has_column("available_kw_transformer"):
        transformer = field("available_kw_transformer")
        if not np.isfinite(transformer) or transformer < 0:
            raise ValueError(f"Invalid transformer capacity: {transformer}")
        values["grid_available_kw"] = min(values.get("grid_available_kw", float("inf")), transformer)
    if has_column("available_kw_line"):
        raise ValueError("Aggregate available_kw_line cannot identify the failed case33 branch; provide line-level failures")
    return values


def summarize_network_balance(hourly: pd.DataFrame) -> dict[str, float]:
    if hourly.empty:
        raise ValueError("No dispatch hours")
    def ratio(num: str, den: str) -> float:
        denominator = float(hourly[den].sum())
        return float(hourly[num].sum() / denominator) if denominator else 1.0
    shortage = hourly.power_deficit_kw > 1e-6
    return {"loss_of_load_probability": float(shortage.mean()),
            "loss_of_load_expectation_hours": float(shortage.sum()),
            "curtailment_probability": float((hourly.curtailment_kw > 1e-6).mean()),
            "max_power_deficit_kw": float(hourly.power_deficit_kw.max()),
            "max_curtailment_kw": float(hourly.curtailment_kw.max()),
            "expected_energy_deficit_kwh": float(hourly.power_deficit_kw.sum()),
            "expected_curtailment_energy_kwh": float(hourly.curtailment_kw.sum()),
            "critical_load_supply_ratio": ratio("critical_load_served_kw", "critical_load_kw"),
            "important_load_restoration_ratio": ratio("important_load_served_kw", "important_load_kw"),
            "mobile_reachability": float((hourly.reachable_load_kw / hourly.load_kw.replace(0, np.nan)).fillna(1).mean()),
            "island_required_hours": float((hourly.reachable_load_kw < hourly.load_kw - 1e-6).sum())}
