from __future__ import annotations

import argparse
from pathlib import Path

from src.common.demo_data import save_demo_history
from src.common.io_utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo historical source-load data.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="data/demo")
    args = parser.parse_args()
    config = load_config(args.config)
    path = save_demo_history(
        output_dir=Path(args.output),
        excel_path=config.get("topology", {}).get("lankao_excel", "兰考算例数据.xlsx"),
        year=int(config.get("demo", {}).get("year", 2025)),
        random_seed=int(config.get("random_seed", 42)),
    )
    print(f"demo数据已生成：{path}")


if __name__ == "__main__":
    main()
