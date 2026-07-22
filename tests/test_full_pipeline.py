from __future__ import annotations

from pathlib import Path
import unittest

from run_full_pipeline import run_pipeline


class FullPipelineTests(unittest.TestCase):
    def test_full_pipeline_demo_smoke(self) -> None:
        summary = run_pipeline("config.yaml", demo=True)
        self.assertTrue(Path(summary["chapter3"]["annual_random_production_sequence"]).exists())
        self.assertTrue(Path(summary["chapter4"]["metrics"]).exists())
        self.assertTrue(Path(summary["chapter6"]["planning_result_json"]).exists())
        self.assertGreaterEqual(summary["planning_recommended_plan"]["storage_power_kw"], 0)
