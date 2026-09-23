from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass
TOPIC2_REMAINING = ROOT / "topic2_remaining_code"
if str(TOPIC2_REMAINING) not in sys.path:
    sys.path.insert(0, str(TOPIC2_REMAINING))

from annual_sequence_embedder import embed_extreme_condition
from coupled_condition_builder import build_coupled_condition
from fault_probability_model import FaultModelConfig, build_device_state_sequence
from resilience_reliability_planner import PlanningConfig, planning_result_to_dataframe, run_simple_planning
from strategy_trigger import StrategyThresholds, trigger_strategies

from chapter2_pipeline import run_chapter2_pipeline
from data_schema import Chapter2Config, StageScenarioConfig
from src.chapter3.extreme_generator_adapter import build_extreme_scene
from src.chapter3.lankao_topology_adapter import (
    build_device_params_from_lankao,
    build_device_params_from_case33,
    build_load_reachability,
)
from src.chapter4.balance_analyzer_adapter import run_balance_analysis
from src.common.demo_data import save_demo_history
from src.common.io_utils import ensure_dir, load_config, read_csv, write_csv, write_json


def build_stage_configs(config: dict[str, Any]) -> dict[str, StageScenarioConfig]:
    stages = {}
    for name, values in config["chapter2"]["stages"].items():
        stages[name] = StageScenarioConfig(name=name, **values)
    return stages


def build_chapter2_config(config: dict[str, Any]) -> Chapter2Config:
    chapter2 = config.get("chapter2", {})
    return Chapter2Config(
        random_seed=int(config.get("random_seed", 42)),
        extreme_top_k=int(chapter2.get("extreme_top_k", 20)),
        extreme_window_hours=int(chapter2.get("extreme_window_hours", 36)),
    )


def run_pipeline(config_path: str | Path, demo: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    if config.get("scenario_method") == "current_paper":
        from src.chapter3.current_paper_adapter import run_current_pipeline
        return run_current_pipeline(config_path, config)
    output_root = ensure_dir(config.get("output_dir", "outputs"))
    data_root = ensure_dir(config.get("data_dir", "data"))
    demo_dir = ensure_dir(data_root / "demo")
    random_seed = int(config.get("random_seed", 42))

    topology_cfg = config.get("topology", {})
    lankao_excel = Path(topology_cfg.get("lankao_excel", "兰考算例数据.xlsx"))
    topology_type = topology_cfg.get("type", "lankao")
    case33_dir = ROOT / topology_cfg.get("case33_dir", "data/case33")

    historical_csv = Path(config["chapter2"].get("historical_csv", config["demo"]["historical_csv"]))
    if demo or not historical_csv.exists():
        historical_csv = save_demo_history(
            output_dir=demo_dir,
            excel_path=(case33_dir / "nodes.csv") if topology_type == "case33" else lankao_excel,
            year=int(config.get("demo", {}).get("year", 2025)),
            random_seed=random_seed,
        )
    historical_df = read_csv(historical_csv)

    chapter2_out = ensure_dir(output_root / "chapter2")
    chapter2_paths = run_chapter2_pipeline(
        historical_df=historical_df,
        output_dir=chapter2_out,
        config=build_chapter2_config(config),
        stage_configs=build_stage_configs(config),
        year=int(config["chapter2"].get("year", 2026)),
    )

    chapter3_out = ensure_dir(output_root / "chapter3")
    source_load_features = read_csv(chapter2_paths["source_load_features"])
    extreme_candidates = read_csv(chapter2_paths["extreme_window_candidates"])
    extreme_scene, hazard_36h, extreme_meta = build_extreme_scene(
        source_load_features,
        extreme_candidates,
        {**config.get("extreme_generator", {}), "random_seed": random_seed},
    )
    extreme_path = write_csv(extreme_scene, chapter3_out / "extreme_36h.csv")
    hazard_path = write_csv(hazard_36h, chapter3_out / "hazard_36h.csv")

    if topology_type == "case33":
        device_params, lines, nodes, topology_meta = build_device_params_from_case33(case33_dir, topology_cfg)
    else:
        device_params, lines, nodes, topology_meta = build_device_params_from_lankao(lankao_excel, topology_cfg)
    device_params_path = write_csv(device_params, chapter3_out / "device_params.csv")
    lines_path = write_csv(lines, chapter3_out / "lankao_lines.csv")
    nodes_path = write_csv(nodes, chapter3_out / "lankao_nodes.csv")

    states, capacity = build_device_state_sequence(
        hazard_36h,
        device_params,
        config=FaultModelConfig(random_seed=random_seed),
    )
    load_reachability = build_load_reachability(hazard_36h)
    coupled = build_coupled_condition(
        extreme_scene,
        capacity,
        load_reachability_df=load_reachability,
        external_grid_available_kw=float(config.get("chapter3", {}).get("external_grid_available_kw", 0.0)),
    )

    embed_stage = config.get("chapter3", {}).get("embed_stage", "near")
    annual_background_path = chapter2_paths[f"annual_background_{embed_stage}"]
    annual_background = read_csv(annual_background_path)
    annual_sequence = embed_extreme_condition(
        annual_background,
        coupled,
        embed_start_time=config.get("chapter3", {}).get("embed_start_time", "2026-01-15 00:00:00"),
        sequence_id=config.get("chapter3", {}).get("sequence_id", "seq_lankao_demo_001"),
        sequence_weight=float(config.get("chapter3", {}).get("sequence_weight", 1.0)),
    )

    states_path = write_csv(states, chapter3_out / "device_state_sequence_36h.csv")
    capacity_path = write_csv(capacity, chapter3_out / "available_capacity_36h.csv")
    reachability_path = write_csv(load_reachability, chapter3_out / "load_reachability_36h.csv")
    coupled_path = write_csv(coupled, chapter3_out / "coupled_condition_36h.csv")
    annual_sequence_path = write_csv(annual_sequence, chapter3_out / "annual_random_production_sequence.csv")
    write_json(
        chapter3_out / "topology_summary.json",
        {
            "topology_type": topology_type,
            "topology_source": str(case33_dir if topology_type == "case33" else lankao_excel),
            "metadata": topology_meta,
            "device_type_counts": device_params["device_type"].value_counts().to_dict(),
            "extreme_generator": extreme_meta,
        },
    )

    chapter4_out = ensure_dir(output_root / "chapter4")
    metrics, hourly_balance, chapter4_paths = run_balance_analysis(
        annual_sequence,
        chapter4_out,
        config.get("chapter4", {}),
    )

    chapter5_out = ensure_dir(output_root / "chapter5")
    thresholds = StrategyThresholds(**config.get("strategy_thresholds", {}))
    trigger_df = trigger_strategies(metrics, thresholds)
    strategy_path = write_csv(trigger_df, chapter5_out / "strategy_trigger_result.csv")

    chapter6_out = ensure_dir(output_root / "chapter6")
    planning_config = PlanningConfig(**config.get("planning", {}))
    planning_result = run_simple_planning(metrics, trigger_df, planning_config)
    planning_df = planning_result_to_dataframe(planning_result)
    planning_csv_path = write_csv(planning_df, chapter6_out / "planning_result.csv")
    planning_json_path = write_json(chapter6_out / "planning_result.json", planning_result)

    summary = {
        "config": str(config_path),
        "demo": demo,
        "historical_csv": str(historical_csv),
        "chapter2": chapter2_paths,
        "chapter3": {
            "extreme_36h": str(extreme_path),
            "hazard_36h": str(hazard_path),
            "device_params": str(device_params_path),
            "lankao_lines": str(lines_path),
            "lankao_nodes": str(nodes_path),
            "device_state_sequence_36h": str(states_path),
            "available_capacity_36h": str(capacity_path),
            "load_reachability_36h": str(reachability_path),
            "coupled_condition_36h": str(coupled_path),
            "annual_random_production_sequence": str(annual_sequence_path),
        },
        "chapter4": chapter4_paths,
        "chapter5": {"strategy_trigger_result": str(strategy_path)},
        "chapter6": {
            "planning_result_csv": str(planning_csv_path),
            "planning_result_json": str(planning_json_path),
        },
        "metrics": metrics,
        "planning_recommended_plan": planning_result["recommended_plan"],
    }
    write_json(output_root / "run_summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the topic 2 full source-load-to-planning pipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--demo", action="store_true", help="Generate demo source-load data before running")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    summary = run_pipeline(args.config, demo=args.demo)
    print("课题2全流程运行完成。关键输出：")
    print(f"- 第3章8760 h随机生产模拟序列：{summary['chapter3']['annual_random_production_sequence']}")
    print(f"- 第4章平衡指标：{summary['chapter4']['metrics']}")
    print(f"- 第6章规划建议：{summary['chapter6']['planning_result_json']}")
    return summary


if __name__ == "__main__":
    main()
