from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from annual_background_generator import generate_annual_timestamps
from chapter2_pipeline import run_chapter2_pipeline
from data_schema import Chapter2Config, StageScenarioConfig
from src.common.demo_data import make_demo_history


class Chapter2PipelineTests(unittest.TestCase):
    def test_chapter2_pipeline_outputs_required_files(self) -> None:
        history = make_demo_history(year=2025, peak_load_kw=5000, random_seed=7).head(24 * 60)
        with tempfile.TemporaryDirectory() as tmp:
            paths = run_chapter2_pipeline(
                history,
                Path(tmp),
                config=Chapter2Config(extreme_top_k=3, extreme_window_hours=36, random_seed=7),
                stage_configs={
                    "near": StageScenarioConfig(
                        name="near",
                        load_growth_rate=0.05,
                        wind_capacity_scale=1.05,
                        pv_capacity_scale=1.05,
                        noise_scale=0.02,
                    )
                },
                year=2026,
            )
            for key in [
                "source_load_features",
                "extreme_window_candidates",
                "annual_background_near",
                "load_probability_boundary",
                "renewable_probability_boundary",
                "chapter2_summary",
            ]:
                self.assertTrue(Path(paths[key]).exists(), key)

        self.assertEqual(len(generate_annual_timestamps(2026)), 8760)
