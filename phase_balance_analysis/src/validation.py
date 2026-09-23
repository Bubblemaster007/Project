"""六类自动化正确性测试与文本报告生成。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable
import math

import pandas as pd

from .hourly_balance import calculate_hourly_balance
from .phase_metrics import compute_phase_metrics, total_expected_energy


STAGES = [
    ("灾前准备", 1, 6),
    ("灾害冲击", 2, 6),
    ("灾害持续", 3, 12),
    ("灾后恢复", 4, 12),
]


def _base_frame() -> pd.DataFrame:
    rows = []
    timestamp = pd.Timestamp("2026-01-01 00:00:00")
    hour = 0
    for stage, order, count in STAGES:
        for _ in range(count):
            rows.append(
                {
                    "timestamp": timestamp + pd.Timedelta(hours=hour),
                    "simulation_id": "test_sim_001",
                    "event_id": "test_event_001",
                    "event_hour": hour,
                    "stage": stage,
                    "stage_order": order,
                    "load_mw": 20.0,
                    "wind_available_mw": 0.0,
                    "pv_available_mw": 0.0,
                    "grid_dispatch_mw": 0.0,
                    "storage_charge_mw": 0.0,
                    "storage_discharge_mw": 0.0,
                    "emergency_mw": 0.0,
                    "other_generation_mw": 20.0,
                    "export_mw": 0.0,
                }
            )
            hour += 1
    return pd.DataFrame(rows)


def _metrics(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly, _ = calculate_hourly_balance(data)
    main, _ = compute_phase_metrics(hourly, 1.0, 1e-6, "典型仿真测试数据")
    return main, hourly


def _row(main: pd.DataFrame, stage: str) -> pd.Series:
    return main.loc[main["stage"] == stage].iloc[0]


def _assert_close(actual: float, expected: float, message: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError(f"{message}: actual={actual}, expected={expected}")


def _test_complete_balance() -> str:
    main, _ = _metrics(_base_frame())
    numeric = [
        "electricity_shortage_probability_pct",
        "curtailment_probability_pct",
        "maximum_electricity_shortage_mw",
        "maximum_curtailment_power_mw",
        "expected_energy_not_served_mwh",
        "expected_curtailed_energy_mwh",
    ]
    if not (main[numeric].abs().to_numpy() <= 1e-9).all():
        raise AssertionError("完全平衡场景的六项指标应全部为 0")
    return "供电能力始终等于负荷，六项指标均为 0。"


def _test_fixed_shortage() -> str:
    data = _base_frame()
    mask = data["stage"] == "灾害冲击"
    data.loc[mask, "other_generation_mw"] = 10.0
    main, _ = _metrics(data)
    row = _row(main, "灾害冲击")
    _assert_close(row["maximum_electricity_shortage_mw"], 10.0, "缺额极大值")
    _assert_close(row["expected_energy_not_served_mwh"], 60.0, "缺电量")
    return "灾害冲击阶段固定缺口 10 MW、持续 6 h，极大值 10 MW、EENS 60 MWh。"


def _test_fixed_curtailment() -> str:
    data = _base_frame()
    mask = data["stage"] == "灾害持续"
    data.loc[mask, "other_generation_mw"] = 28.0
    main, _ = _metrics(data)
    row = _row(main, "灾害持续")
    _assert_close(row["maximum_curtailment_power_mw"], 8.0, "弃电极大值")
    _assert_close(row["expected_curtailed_energy_mwh"], 96.0, "弃电量")
    return "灾害持续阶段固定富余 8 MW、持续 12 h，极大值 8 MW、EEC 96 MWh。"


def _test_failure_worsening() -> str:
    normal = _base_frame()
    degraded = _base_frame()
    normal["equipment_availability"] = 1.0
    degraded["equipment_availability"] = 0.60
    normal["other_generation_mw"] = 22.0 * normal["equipment_availability"]
    degraded["other_generation_mw"] = 22.0 * degraded["equipment_availability"]
    m1, _ = _metrics(normal)
    m2, _ = _metrics(degraded)
    cols = [
        "electricity_shortage_probability_pct",
        "maximum_electricity_shortage_mw",
        "expected_energy_not_served_mwh",
    ]
    if not (m2[cols].to_numpy() >= m1[cols].to_numpy() - 1e-9).all():
        raise AssertionError("设备可用系数下降后缺电指标不应下降")
    return "设备可用系数由 1.00 降至 0.60，三项缺电指标均未下降。"


def _test_more_support() -> str:
    base = _base_frame()
    base["other_generation_mw"] = 15.0
    supported = base.copy()
    supported["emergency_mw"] = 4.0
    m1, _ = _metrics(base)
    m2, _ = _metrics(supported)
    cols = [
        "electricity_shortage_probability_pct",
        "maximum_electricity_shortage_mw",
        "expected_energy_not_served_mwh",
    ]
    if not (m2[cols].to_numpy() <= m1[cols].to_numpy() + 1e-9).all():
        raise AssertionError("增加应急电源后缺电指标不应上升")
    return "应急支撑增加 4 MW 后，三项缺电指标均未上升。"


def _test_phase_consistency() -> str:
    data = _base_frame()
    data.loc[data["stage"] == "灾害冲击", "other_generation_mw"] = 12.0
    data.loc[data["stage"] == "灾害持续", "other_generation_mw"] = 16.0
    main, hourly = _metrics(data)
    if len(hourly) != sum(count for _, _, count in STAGES):
        raise AssertionError("四阶段时间步之和不等于完整事件窗口")
    total_loss, total_curt = total_expected_energy(hourly, 1.0)
    _assert_close(
        main["expected_energy_not_served_mwh"].sum(),
        total_loss,
        "四阶段 EENS 之和",
    )
    _assert_close(
        main["expected_curtailed_energy_mwh"].sum(),
        total_curt,
        "四阶段 EEC 之和",
    )
    return "四阶段共 36 h，分阶段电量之和与完整事件窗口电量一致。"


def run_validation_suite(report_path: Path | None = None) -> list[dict[str, object]]:
    """执行六项测试，并按 PASS/FAIL 输出报告。"""
    tests: list[tuple[str, Callable[[], str]]] = [
        ("测试1：完全平衡场景", _test_complete_balance),
        ("测试2：固定功率缺口场景", _test_fixed_shortage),
        ("测试3：固定弃电场景", _test_fixed_curtailment),
        ("测试4：设备故障加重", _test_failure_worsening),
        ("测试5：储能或应急电源增加", _test_more_support),
        ("测试6：阶段聚合一致性", _test_phase_consistency),
    ]
    results: list[dict[str, object]] = []
    for name, test in tests:
        try:
            detail = test()
            results.append({"name": name, "passed": True, "detail": detail})
        except Exception as exc:  # noqa: BLE001 - 报告必须保留具体失败原因
            results.append({
                "name": name,
                "passed": False,
                "detail": f"{type(exc).__name__}: {exc}",
            })

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "极端事件全过程分阶段电力电量平衡指标自动化测试报告",
            "=" * 64,
        ]
        for item in results:
            status = "PASS" if item["passed"] else "FAIL"
            lines.append(f"[{status}] {item['name']}")
            lines.append(f"  {item['detail']}")
        passed = sum(bool(item["passed"]) for item in results)
        lines.extend(["", f"汇总：{passed}/{len(results)} 项测试通过。"])
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def validate_phase_evolution(
    hourly: pd.DataFrame,
    metrics: pd.DataFrame,
    logic_config: dict[str, object],
    report_path: Path | None = None,
) -> list[dict[str, object]]:
    """检查四阶段结果是否符合输入设定的灾害演化物理逻辑。"""
    metric_rows = metrics.set_index("stage")
    hourly_means = hourly.groupby("stage").agg(
        equipment_availability=("equipment_availability", "mean"),
        hazard_intensity=("hazard_intensity", "mean"),
    )
    checks: list[dict[str, object]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    preparation_lolp = float(
        metric_rows.loc["灾前准备", "electricity_shortage_probability_pct"]
    )
    maximum_preparation = float(
        logic_config.get("maximum_preparation_shortage_probability_pct", 5.0)
    )
    record(
        "灾前供电基本平衡",
        preparation_lolp <= maximum_preparation,
        f"灾前电力不足概率 {preparation_lolp:.2f}% ≤ {maximum_preparation:.2f}%。",
    )

    prep_avail = float(hourly_means.loc["灾前准备", "equipment_availability"])
    impact_avail = float(hourly_means.loc["灾害冲击", "equipment_availability"])
    sustained_avail = float(hourly_means.loc["灾害持续", "equipment_availability"])
    recovery_avail = float(hourly_means.loc["灾后恢复", "equipment_availability"])
    record(
        "设备可用率先降后升",
        prep_avail > impact_avail > sustained_avail
        and recovery_avail > sustained_avail,
        "阶段平均设备可用率："
        f"灾前 {prep_avail:.3f}、冲击 {impact_avail:.3f}、"
        f"持续 {sustained_avail:.3f}、恢复 {recovery_avail:.3f}。",
    )

    prep_hazard = float(hourly_means.loc["灾前准备", "hazard_intensity"])
    impact_hazard = float(hourly_means.loc["灾害冲击", "hazard_intensity"])
    sustained_hazard = float(hourly_means.loc["灾害持续", "hazard_intensity"])
    recovery_hazard = float(hourly_means.loc["灾后恢复", "hazard_intensity"])
    record(
        "灾害强度先升后降",
        prep_hazard < impact_hazard < sustained_hazard
        and recovery_hazard < sustained_hazard,
        "阶段平均灾害强度："
        f"灾前 {prep_hazard:.3f}、冲击 {impact_hazard:.3f}、"
        f"持续 {sustained_hazard:.3f}、恢复 {recovery_hazard:.3f}。",
    )

    impact_eens = float(
        metric_rows.loc["灾害冲击", "expected_energy_not_served_mwh"]
    )
    sustained_eens = float(
        metric_rows.loc["灾害持续", "expected_energy_not_served_mwh"]
    )
    recovery_eens = float(
        metric_rows.loc["灾后恢复", "expected_energy_not_served_mwh"]
    )
    if bool(logic_config.get("require_sustained_eens_above_impact", True)):
        record(
            "持续阶段累计缺电量不低于冲击阶段",
            sustained_eens >= impact_eens,
            f"冲击 EENS={impact_eens:.3f} MWh，持续 EENS={sustained_eens:.3f} MWh。",
        )
    if bool(logic_config.get("require_recovery_eens_below_sustained", True)):
        record(
            "恢复阶段累计缺电量低于持续阶段",
            recovery_eens < sustained_eens,
            f"持续 EENS={sustained_eens:.3f} MWh，恢复 EENS={recovery_eens:.3f} MWh。",
        )

    if bool(logic_config.get("require_limited_recovery_curtailment", False)):
        recovery_pcr = float(
            metric_rows.loc["灾后恢复", "curtailment_probability_pct"]
        )
        minimum_recovery_pcr = float(
            logic_config.get(
                "minimum_recovery_curtailment_probability_pct", 0.0
            )
        )
        maximum_recovery_pcr = float(
            logic_config.get(
                "maximum_recovery_curtailment_probability_pct", 100.0
            )
        )
        record(
            "恢复阶段存在有限弃电",
            minimum_recovery_pcr <= recovery_pcr <= maximum_recovery_pcr,
            "灾后恢复弃电概率 "
            f"{recovery_pcr:.2f}% 位于设定范围 "
            f"[{minimum_recovery_pcr:.2f}%, {maximum_recovery_pcr:.2f}%]。",
        )

    simultaneous = (
        (hourly["power_loss_mw"] > 1e-6)
        & (hourly["curtailment_mw"] > 1e-6)
    )
    record(
        "缺额与弃电不同时发生",
        int(simultaneous.sum()) == 0,
        f"同时发生记录数={int(simultaneous.sum())}。",
    )

    if report_path is not None:
        lines = [
            "四阶段灾害演化逻辑校验报告",
            "=" * 64,
        ]
        for item in checks:
            status = "PASS" if item["passed"] else "FAIL"
            lines.append(f"[{status}] {item['name']}")
            lines.append(f"  {item['detail']}")
        passed = sum(bool(item["passed"]) for item in checks)
        lines.extend(["", f"汇总：{passed}/{len(checks)} 项逻辑校验通过。"])
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checks
