from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "dev" / "run_regression_suite.py"

spec = importlib.util.spec_from_file_location("run_regression_suite", SCRIPT_PATH)
run_regression_suite = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = run_regression_suite
spec.loader.exec_module(run_regression_suite)


class RegressionRunnerSelectionTests(unittest.TestCase):
    def test_formatter_change_runs_output_contract_and_functional_tests(self) -> None:
        selection = run_regression_suite.select_targets(["src/utils/formatter.py"])

        self.assertEqual(
            [
                "tests/test_formatter.py",
                "tests/test_aibom_output_contract.py",
                "tests/test_functional_regression.py",
            ],
            selection.pytest_targets,
        )
        self.assertFalse(selection.docs_only)
        self.assertFalse(selection.full_recommended)

    def test_docs_only_change_runs_clean_state_without_product_tests(self) -> None:
        selection = run_regression_suite.select_targets(
            ["README.md", "docs/dev/regression-testing.md"]
        )

        self.assertEqual([], selection.pytest_targets)
        self.assertTrue(selection.run_clean_state)
        self.assertTrue(selection.docs_only)

    def test_unmapped_source_change_uses_default_fallback(self) -> None:
        selection = run_regression_suite.select_targets(["src/models/new_surface.py"])

        self.assertEqual(
            [
                "tests/test_functional_regression.py",
                "tests/test_aibom_output_contract.py",
            ],
            selection.pytest_targets,
        )
        self.assertTrue(selection.full_recommended)

    def test_clean_state_change_runs_its_unit_test_and_guardrail(self) -> None:
        selection = run_regression_suite.select_targets(["scripts/dev/check_clean_state.py"])

        self.assertEqual(["tests/test_clean_state.py"], selection.pytest_targets)
        self.assertTrue(selection.run_clean_state)


if __name__ == "__main__":
    unittest.main()
