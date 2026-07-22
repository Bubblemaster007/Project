from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
TOPIC2_REMAINING = ROOT / "topic2_remaining_code"
if str(TOPIC2_REMAINING) not in sys.path:
    sys.path.insert(0, str(TOPIC2_REMAINING))

from resilience_reliability_planner import PlanningConfig, planning_result_to_dataframe, run_simple_planning
from strategy_trigger import StrategyThresholds, trigger_strategies

from src.common.io_utils import ensure_dir, load_config, write_csv, write_json


def main() -> None:
    config = load_config("config.yaml")
    metrics = json.loads((Path(config["output_dir"]) / "chapter4" / "metrics.json").read_text(encoding="utf-8"))
    trigger_df = trigger_strategies(metrics, StrategyThresholds(**config.get("strategy_thresholds", {})))
    ch5 = ensure_dir(Path(config["output_dir"]) / "chapter5")
    ch6 = ensure_dir(Path(config["output_dir"]) / "chapter6")
    print(write_csv(trigger_df, ch5 / "strategy_trigger_result.csv"))
    result = run_simple_planning(metrics, trigger_df, PlanningConfig(**config.get("planning", {})))
    print(write_csv(planning_result_to_dataframe(result), ch6 / "planning_result.csv"))
    print(write_json(ch6 / "planning_result.json", result))


if __name__ == "__main__":
    main()
