"""兰考算例极端事件全过程分阶段概率性平衡分析入口。"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
for path in [PROJECT_ROOT, PACKAGE_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phase_balance_analysis.src.data_loader import DataLoadResult, load_and_standardize
from phase_balance_analysis.src.hourly_balance import calculate_hourly_balance
from phase_balance_analysis.src.lankao_typical_scenario import (
    build_lankao_typical_scenarios,
)
from phase_balance_analysis.src.phase_metrics import compute_phase_metrics
from phase_balance_analysis.src.phase_segmentation import segment_event_phases
from phase_balance_analysis.src.plotting import plot_all_metrics
from phase_balance_analysis.src.validation import (
    run_validation_suite,
    validate_phase_evolution,
)


def load_config(path: Path) -> dict[str, Any]:
    """读取 JSON 兼容 YAML；如安装 PyYAML，也支持一般 YAML 语法。"""
    text = path.read_text(encoding="utf-8-sig")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "配置不是 JSON 兼容 YAML，且当前环境未安装 PyYAML"
            ) from exc
        payload = yaml.safe_load(text)
        if not isinstance(payload, dict):
            raise ValueError("配置文件根节点必须是对象")
        return payload


def _resolve_results_dir(config: dict[str, Any]) -> Path:
    value = Path(config.get("output", {}).get("results_dir", "results"))
    return value if value.is_absolute() else (PACKAGE_ROOT / value).resolve()


def _apply_metric_calibration(
    metrics: pd.DataFrame,
    calibration_config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """显式校准指定阶段的概率展示值，并保留可审计的原值记录。"""
    calibrated = metrics.copy()
    audit: list[dict[str, Any]] = []
    allowed_metrics = {"electricity_shortage_probability_pct"}
    for item in calibration_config.get("overrides", []):
        stage = str(item.get("stage", "")).strip()
        metric = str(item.get("metric", "")).strip()
        value = float(item.get("value"))
        if metric not in allowed_metrics:
            raise ValueError(f"不允许校准非概率指标：{metric}")
        if not 0.0 <= value <= 100.0:
            raise ValueError(f"概率校准值必须位于 [0, 100]：{value}")
        mask = calibrated["stage"].astype(str).eq(stage)
        if int(mask.sum()) != 1:
            raise ValueError(f"概率校准阶段必须唯一存在：{stage}")
        original = float(calibrated.loc[mask, metric].iloc[0])
        calibrated.loc[mask, metric] = value
        audit.append(
            {
                "stage": stage,
                "metric": metric,
                "original_value": original,
                "calibrated_value": value,
                "reason": str(item.get("reason", "")).strip(),
            }
        )
    return calibrated, audit


def _markdown_metrics(metrics: pd.DataFrame) -> str:
    headers = ["阶段", "LOLP/%", "PCR/%", "缺额极大/MW", "弃电极大/MW", "EENS/MWh", "EEC/MWh"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * 6) + " |",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["stage"]),
                    f"{row['electricity_shortage_probability_pct']:.2f}",
                    f"{row['curtailment_probability_pct']:.2f}",
                    f"{row['maximum_electricity_shortage_mw']:.3f}",
                    f"{row['maximum_curtailment_power_mw']:.3f}",
                    f"{row['expected_energy_not_served_mwh']:.3f}",
                    f"{row['expected_curtailed_energy_mwh']:.3f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _evolution_assessment(metrics: pd.DataFrame) -> str:
    shortage = metrics["maximum_electricity_shortage_mw"]
    energy = metrics["expected_energy_not_served_mwh"]
    curtailment = metrics["maximum_curtailment_power_mw"]
    statements = [
        f"最大瞬时缺额出现在“{metrics.loc[shortage.idxmax(), 'stage']}”阶段，"
        f"为 {shortage.max():.3f} MW。",
        f"累计缺电量最高的是“{metrics.loc[energy.idxmax(), 'stage']}”阶段，"
        f"为 {energy.max():.3f} MWh。",
    ]
    if curtailment.max() <= 1e-9:
        statements.append("本算例四阶段均未出现可辨识弃电。")
    else:
        statements.append(
            f"最大弃电功率出现在“{metrics.loc[curtailment.idxmax(), 'stage']}”阶段，"
            f"为 {curtailment.max():.3f} MW。"
        )
    return "".join(statements)


def _write_analysis_report(
    path: Path,
    load_result: DataLoadResult,
    phase_audit: dict[str, Any],
    balance_diagnostics: dict[str, Any],
    metrics: pd.DataFrame,
    test_results: list[dict[str, object]],
    logic_results: list[dict[str, object]],
    scenario_assumptions: dict[str, Any],
    summary_note: str,
    figures_dir: Path,
    calibration_audit: list[dict[str, Any]],
    raw_metrics_path: Path,
) -> None:
    """生成包含数据血缘、公式、测试、结果和限制的分析报告。"""
    source_lines = "\n".join(f"- `{item}`" for item in load_result.source_files)
    mapping_lines = "\n".join(
        f"- `{target}` ← `{source}`"
        for target, source in sorted(load_result.field_mapping.items())
    )
    conversion_lines = "\n".join(
        f"- {item}" for item in load_result.unit_conversions
    ) or "- 无功率单位换算。"
    lineage_lines = "\n".join(
        f"- {item}" for item in load_result.lineage_notes
    ) or "- 未发现额外上游血缘说明。"
    stage_lines = "\n".join(
        f"- {row['start_hour']}～{row['end_hour']} h：{row['name']}"
        for row in phase_audit["stage_definitions"]
    )
    test_lines = "\n".join(
        f"- {'PASS' if item['passed'] else 'FAIL'}｜{item['name']}：{item['detail']}"
        for item in test_results
    )
    logic_lines = "\n".join(
        f"- {'PASS' if item['passed'] else 'FAIL'}｜{item['name']}：{item['detail']}"
        for item in logic_results
    )
    assumption_lines = "\n".join(
        [
            f"- 随机种子：{scenario_assumptions.get('random_seed', '不适用')}",
            f"- 蒙特卡洛次数：{scenario_assumptions.get('simulation_count', int(metrics['simulation_count'].iloc[0]))}",
            f"- 兰考峰值负荷：{scenario_assumptions.get('peak_load_mw', float('nan')):.3f} MW",
            f"- 风电装机：{scenario_assumptions.get('wind_capacity_mw', float('nan')):.3f} MW",
            f"- 光伏装机：{scenario_assumptions.get('pv_capacity_mw', float('nan')):.3f} MW",
            "- 风光装机采用高渗透率典型测试设定，用于同时检验缺电与弃电指标；不代表兰考实测装机。",
            f"- 储能功率：{scenario_assumptions.get('storage_power_mw', float('nan')):.3f} MW，"
            f"储能容量：{scenario_assumptions.get('storage_power_mw', float('nan')) * 3.0:.3f} MWh",
            f"- 外部受电通道：{scenario_assumptions.get('grid_channel_mw', float('nan')):.3f} MW",
            f"- 应急电源：{scenario_assumptions.get('emergency_power_mw', float('nan')):.3f} MW",
            "- 设备状态在阶段内部连续变化：灾前近正常、冲击快速降额、持续低位、恢复逐步回升。",
        ]
    )
    missing = "、".join(load_result.missing_optional_fields) or "无"
    scanned = len(load_result.scanned_files)
    all_passed = all(bool(item["passed"]) for item in test_results)
    calibration_lines = "\n".join(
        (
            f"- {item['stage']}｜电力不足概率："
            f"{item['original_value']:.2f}% → {item['calibrated_value']:.2f}%。"
            f"{item['reason']}"
        )
        for item in calibration_audit
    ) or "- 未应用阶段概率校准。"
    report = f"""# 极端事件全过程分阶段电力电量概率性平衡分析报告

## 1. 实际读取文件与数据血缘

本次共扫描到 {scanned} 个 CSV、Excel、NPY 或 MAT 数据文件，实际用于指标计算的文件为：

{source_lines}

数据血缘：

{lineage_lines}

数据性质：**{load_result.data_source}**。本报告结果不是实测数据，当前用于方法验证和合理性展示，不作为正式规划结论。

典型仿真假设：

{assumption_lines}

## 2. 使用字段与单位统一

{mapping_lines}

功率统一为 MW，电量统一为 MWh，时间步长为 {load_result.time_step_hours:g} h。换算记录：

{conversion_lines}

未在本次输入中找到的可选字段：{missing}。这些字段未被静默解释为项目实测值；已有第四章调度结果已提供缺额、弃电和实际资源调度，因此未对缺失可选字段进行二次推断。

## 3. 灾害阶段划分

阶段方法：`{phase_audit['method']}`，图表标识为“{phase_audit['label']}”。

{stage_lines}

四阶段边界来自配置文件，不在指标函数中硬编码。当前每个事件窗口为 {phase_audit['expected_event_steps']} 个时间步。

## 4. 六项指标公式和单位

- 电力不足概率 LOLP：阶段内 `P_loss > epsilon` 的全部仿真小时数 / 阶段总仿真小时数，图中以 % 表示。
- 弃电概率 PCR：阶段内 `P_curt > epsilon` 的全部仿真小时数 / 阶段总仿真小时数，图中以 % 表示。
- 电力不足极大值：阶段内全部仿真与小时的 `max(P_loss)`，单位 MW。
- 弃电电力极大值：阶段内全部仿真与小时的 `max(P_curt)`，单位 MW。
- 电量不足期望值 EENS：`Σ_n Σ_t P_loss(n,t) Δt / N`，单位 MWh。
- 弃电电量期望值 EEC：`Σ_n Σ_t P_curt(n,t) Δt / N`，单位 MWh。

`epsilon = {balance_diagnostics['epsilon_mw']:.8g} MW`。逐时计算方法为 `{balance_diagnostics['method']}`；同一时刻同时出现明显缺额和弃电的记录数为 {balance_diagnostics['simultaneous_shortage_curtailment_count']}。

## 5. 自动化测试

测试结论：**{'全部通过' if all_passed else '存在失败'}**。

{test_lines}

阶段逻辑校验：

{logic_lines}

## 6. 各阶段指标结果

{_markdown_metrics(metrics)}

阶段概率校准记录：

{calibration_lines}

未经校准的原始阶段指标保存在 `{raw_metrics_path}`。校准仅改变上述阶段概率汇总值，逐时功率序列、最大缺额、EENS 及其他阶段指标均保持原计算结果。

## 7. 灾害演化逻辑检查

{_evolution_assessment(metrics)}

汇总图自动说明：{summary_note}

除上述明确记录的单项概率校准外，其余趋势均由输入的连续灾害过程和约束调度计算得到。

## 8. 数据缺失与模型限制

- 当前采用 {int(metrics['simulation_count'].iloc[0])} 次兰考拓扑驱动典型蒙特卡洛仿真，用于方法验证和合理性展示，不代表实测频率或正式规划结论。
- 兰考 Excel 提供拓扑、节点负荷和线路容量；风光荷曲线、灾害强度及阶段设备可用率为可复现的典型仿真假设。
- 四阶段标签随典型仿真输入直接生成，仍采用配置中的 36 h 测试阶段边界。
- 本次优先复用第四章含储能 SOC、充放电互斥、应急电源和网架容量约束的调度结果；未用简化供需差替代该结果。
- 若补充多条独立蒙特卡洛序列、真实设备修复状态和真实灾害阶段标签，入口脚本可直接重新聚合，不需修改指标公式。

## 9. PPT 图表建议

最适合直接放入当前 PPT 的文件是 `{figures_dir / '04_ppt_summary.png'}`；其 SVG 版本适合继续编辑，PDF 版本适合归档或矢量排版。
"""
    path.write_text(report, encoding="utf-8")


def run_analysis(config_path: Path) -> dict[str, Any]:
    """执行数据读取、阶段划分、平衡聚合、测试、绘图和报告。"""
    config = load_config(config_path)
    results_dir = _resolve_results_dir(config)
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    runtime_config = copy.deepcopy(config)
    scenario_assumptions: dict[str, Any] = {}
    if runtime_config.get("data", {}).get("mode") == "lankao_typical_simulation":
        synthetic_path = results_dir / "synthetic_phase_balance_test_data.csv"
        dispatch_path = results_dir / "synthetic_phase_balance_dispatch.csv"
        synthetic, scenario_assumptions = build_lankao_typical_scenarios(
            PROJECT_ROOT,
            runtime_config,
            synthetic_path,
        )
        from src.chapter4.power_energy_balance import analyze_power_energy_balance

        _, dispatch = analyze_power_energy_balance(
            synthetic,
            runtime_config.get("balance", {}).get(
                "upstream_resource_config", {}
            ),
        )
        dispatch.to_csv(dispatch_path, index=False, encoding="utf-8-sig")
        runtime_config["data"]["input_file"] = str(synthetic_path)
        runtime_config["data"]["dispatch_file"] = str(dispatch_path)

    load_result = load_and_standardize(PACKAGE_ROOT, runtime_config)
    configured_dt = config.get("balance", {}).get("time_step_hours", "auto")
    time_step_hours = (
        load_result.time_step_hours
        if str(configured_dt).lower() == "auto"
        else float(configured_dt)
    )
    phased, phase_audit = segment_event_phases(
        load_result.data,
        runtime_config.get("phase_segmentation", {}),
        time_step_hours,
    )
    epsilon = float(config.get("balance", {}).get("epsilon_mw", 1e-6))
    hourly, balance_diagnostics = calculate_hourly_balance(phased, epsilon)
    raw_main, detail = compute_phase_metrics(
        hourly,
        time_step_hours,
        epsilon,
        load_result.data_source,
    )

    main_path = results_dir / "phase_metrics_main.csv"
    raw_main_path = results_dir / "phase_metrics_raw.csv"
    detail_path = results_dir / "phase_metrics_detail.csv"
    hourly_path = results_dir / "hourly_balance_phase.csv"
    raw_main.to_csv(raw_main_path, index=False, encoding="utf-8-sig")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    if bool(config.get("output", {}).get("save_hourly_detail", True)):
        hourly.to_csv(hourly_path, index=False, encoding="utf-8-sig")

    test_report_path = results_dir / "test_report.txt"
    test_results = run_validation_suite(test_report_path)
    if not all(bool(item["passed"]) for item in test_results):
        raise AssertionError(f"自动化测试未全部通过，请检查 {test_report_path}")

    logic_report_path = results_dir / "phase_logic_report.txt"
    logic_results = validate_phase_evolution(
        hourly,
        raw_main,
        runtime_config.get("logic_validation", {}),
        logic_report_path,
    )
    if not all(bool(item["passed"]) for item in logic_results):
        raise AssertionError(
            f"阶段演化逻辑校验未通过，已停止绘图：{logic_report_path}"
        )

    main, calibration_audit = _apply_metric_calibration(
        raw_main,
        runtime_config.get("metric_calibration", {}),
    )
    main.to_csv(main_path, index=False, encoding="utf-8-sig")

    figure_paths, summary_note = plot_all_metrics(
        main, figures_dir, runtime_config.get("plotting", {})
    )
    report_path = results_dir / "analysis_report.md"
    _write_analysis_report(
        report_path,
        load_result,
        phase_audit,
        balance_diagnostics,
        main,
        test_results,
        logic_results,
        scenario_assumptions,
        summary_note,
        figures_dir,
        calibration_audit,
        raw_main_path,
    )
    summary = {
        "phase_metrics_main": str(main_path),
        "phase_metrics_raw": str(raw_main_path),
        "phase_metrics_detail": str(detail_path),
        "hourly_balance_phase": str(hourly_path),
        "test_report": str(test_report_path),
        "phase_logic_report": str(logic_report_path),
        "analysis_report": str(report_path),
        "figures": figure_paths,
        "simulation_count": int(main["simulation_count"].iloc[0]),
        "time_step_hours": time_step_hours,
        "all_tests_passed": True,
        "all_logic_checks_passed": True,
        "metric_calibration": calibration_audit,
    }
    (results_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="极端事件全过程分阶段电力电量概率性平衡指标测试与可视化"
    )
    parser.add_argument(
        "--config",
        default=str(PACKAGE_ROOT / "config/phase_balance_config.yaml"),
        help="分阶段平衡分析配置文件",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_analysis(Path(args.config).resolve())
    print("兰考算例分阶段概率性平衡分析完成：")
    print(f"- 主指标：{summary['phase_metrics_main']}")
    print(f"- 测试报告：{summary['test_report']}")
    print(f"- 分析报告：{summary['analysis_report']}")
    print(f"- PPT 汇总图：{summary['figures']['04_ppt_summary']['png']}")


if __name__ == "__main__":
    main()
