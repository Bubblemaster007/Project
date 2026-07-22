from __future__ import annotations

from pathlib import Path

from src.chapter4.balance_analyzer_adapter import run_balance_analysis
from src.common.io_utils import load_config


def main() -> None:
    config = load_config("config.yaml")
    sequence = Path(config["output_dir"]) / "chapter3" / "annual_random_production_sequence.csv"
    _, _, paths = run_balance_analysis(sequence, Path(config["output_dir"]) / "chapter4", config.get("chapter4", {}))
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
