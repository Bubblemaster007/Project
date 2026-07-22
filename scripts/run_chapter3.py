from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TOPIC2_REMAINING = ROOT / "topic2_remaining_code"
if str(TOPIC2_REMAINING) not in sys.path:
    sys.path.insert(0, str(TOPIC2_REMAINING))

from annual_sequence_embedder import embed_extreme_condition
from coupled_condition_builder import build_coupled_condition
from fault_probability_model import FaultModelConfig, build_device_state_sequence

from src.chapter3.extreme_generator_adapter import build_extreme_scene
from src.chapter3.lankao_topology_adapter import build_device_params_from_lankao, build_load_reachability
from src.common.io_utils import ensure_dir, load_config, read_csv, write_csv


def main() -> None:
    config = load_config("config.yaml")
    out = ensure_dir(Path(config["output_dir"]) / "chapter3")
    ch2 = Path(config["output_dir"]) / "chapter2"
    features = read_csv(ch2 / "source_load_features.csv")
    candidates = read_csv(ch2 / "extreme_window_candidates.csv")
    scene, hazard, _ = build_extreme_scene(features, candidates, config.get("extreme_generator", {}))
    device_params, _, _, _ = build_device_params_from_lankao(config["topology"]["lankao_excel"], config.get("topology", {}))
    states, capacity = build_device_state_sequence(hazard, device_params, FaultModelConfig(random_seed=int(config.get("random_seed", 42))))
    reachability = build_load_reachability(hazard)
    coupled = build_coupled_condition(scene, capacity, reachability)
    annual = read_csv(ch2 / f"annual_background_{config['chapter3'].get('embed_stage', 'near')}.csv")
    sequence = embed_extreme_condition(annual, coupled, config["chapter3"]["embed_start_time"])
    for name, df in {
        "extreme_36h.csv": scene,
        "hazard_36h.csv": hazard,
        "device_params.csv": device_params,
        "device_state_sequence_36h.csv": states,
        "available_capacity_36h.csv": capacity,
        "coupled_condition_36h.csv": coupled,
        "annual_random_production_sequence.csv": sequence,
    }.items():
        print(write_csv(df, out / name))


if __name__ == "__main__":
    main()
