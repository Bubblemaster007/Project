from __future__ import annotations

from pathlib import Path

import pandas as pd

from chapter2_pipeline import run_chapter2_pipeline
from data_schema import Chapter2Config, StageScenarioConfig
from src.common.io_utils import ensure_dir, load_config


def main() -> None:
    config = load_config("config.yaml")
    historical_csv = Path(config["chapter2"]["historical_csv"])
    if not historical_csv.exists():
        raise FileNotFoundError(f"请先运行 scripts/make_demo_data.py，缺少：{historical_csv}")
    stages = {
        name: StageScenarioConfig(name=name, **values)
        for name, values in config["chapter2"]["stages"].items()
    }
    paths = run_chapter2_pipeline(
        pd.read_csv(historical_csv, encoding="utf-8-sig"),
        ensure_dir(Path(config["output_dir"]) / "chapter2"),
        config=Chapter2Config(
            random_seed=int(config.get("random_seed", 42)),
            extreme_top_k=int(config["chapter2"].get("extreme_top_k", 20)),
            extreme_window_hours=int(config["chapter2"].get("extreme_window_hours", 36)),
        ),
        stage_configs=stages,
        year=int(config["chapter2"].get("year", 2026)),
    )
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
