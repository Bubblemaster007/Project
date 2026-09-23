"""六类分阶段电力电量平衡指标自动化测试。"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase_balance_analysis.src.validation import run_validation_suite


class PhaseBalanceMetricTests(unittest.TestCase):
    """逐项检查 validation.py 中定义的六类场景。"""

    def test_all_required_scenarios_pass(self) -> None:
        results = run_validation_suite()
        failures = [item for item in results if not item["passed"]]
        self.assertEqual(failures, [], failures)


if __name__ == "__main__":
    unittest.main()
