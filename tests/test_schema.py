from __future__ import annotations

import unittest

import pandas as pd

from src.common.io_utils import load_config
from src.common.validators import require_columns


class SchemaTests(unittest.TestCase):
    def test_config_loads_and_has_required_sections(self) -> None:
        config = load_config("config.yaml")
        for section in ["chapter2", "chapter3", "chapter4", "strategy_thresholds", "planning"]:
            self.assertIn(section, config)

    def test_require_columns_reports_missing_fields(self) -> None:
        df = pd.DataFrame({"timestamp": ["2026-01-01"], "load_kw": [1.0]})
        require_columns(df, ["timestamp", "load_kw"], "demo")
        with self.assertRaisesRegex(ValueError, "wind_kw"):
            require_columns(df, ["timestamp", "wind_kw"], "demo")
