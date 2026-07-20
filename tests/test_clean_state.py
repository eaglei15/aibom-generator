from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "dev" / "check_clean_state.py"

spec = importlib.util.spec_from_file_location("check_clean_state", SCRIPT_PATH)
check_clean_state = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = check_clean_state
spec.loader.exec_module(check_clean_state)


class CleanStateGuardrailTests(unittest.TestCase):
    def test_private_analysis_path_staged_fails(self) -> None:
        findings = check_clean_state.collect_findings(
            ["docs/analysis/local-report.md"],
            allow_dependency_files=False,
            allow_source_files=False,
            allow_future_mode_files=False,
        )

        self.assertIn(
            "staged local analysis/design/reference/requirements material",
            {finding.reason for finding in findings},
        )

    def test_private_analysis_path_unstaged_warns_only(self) -> None:
        warnings = check_clean_state.collect_warnings(["docs/analysis/local-report.md"])
        findings = check_clean_state.collect_findings(
            [],
            allow_dependency_files=False,
            allow_source_files=False,
            allow_future_mode_files=False,
        )

        self.assertEqual([], findings)
        self.assertIn(
            "local analysis/design/reference/requirements material needs human review before staging",
            {warning.reason for warning in warnings},
        )

    def test_dependency_file_staged_fails(self) -> None:
        findings = check_clean_state.collect_findings(
            ["requirements.txt"],
            allow_dependency_files=False,
            allow_source_files=False,
            allow_future_mode_files=False,
        )

        self.assertEqual(
            ["staged dependency file requires explicit review"],
            [finding.reason for finding in findings],
        )

    def test_dependency_file_modified_only_warns(self) -> None:
        warnings = check_clean_state.collect_warnings(["uv.lock"])

        self.assertEqual(
            ["dependency file needs human review before staging"],
            [warning.reason for warning in warnings],
        )

    def test_no_staged_private_files_passes(self) -> None:
        findings = check_clean_state.collect_findings(
            [
                "docs/dev/cleanup-guardrails.md",
                "scripts/dev/check_clean_state.py",
                "tests/test_clean_state.py",
            ],
            allow_dependency_files=False,
            allow_source_files=False,
            allow_future_mode_files=False,
        )

        self.assertEqual([], findings)

    def test_regression_tooling_staged_set_passes(self) -> None:
        findings = check_clean_state.collect_findings(
            [
                ".codex/skills/regression-suite/SKILL.md",
                "docs/dev/cleanup-guardrails.md",
                "docs/dev/regression-testing.md",
                "scripts/dev/check_clean_state.py",
                "scripts/dev/run_regression_suite.py",
                "tests/test_clean_state.py",
                "tests/test_regression_runner.py",
            ],
            allow_dependency_files=False,
            allow_source_files=False,
            allow_future_mode_files=False,
        )

        self.assertEqual([], findings)

    def test_source_file_modified_only_warns(self) -> None:
        warnings = check_clean_state.collect_warnings(["src/models/service.py"])

        self.assertEqual(
            ["source file change needs human review before staging"],
            [warning.reason for warning in warnings],
        )


if __name__ == "__main__":
    unittest.main()
