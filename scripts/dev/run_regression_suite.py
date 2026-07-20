from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE_TESTS = [
    "tests/test_model_file_extraction.py",
    "tests/test_gguf_metadata.py",
    "tests/test_safetensors_metadata.py",
    "tests/test_hyperparameter_wiring.py",
    "tests/test_aibom_hyperparameters.py",
]

TARGETED_TEST_MAP = {
    "src/controllers/cli_controller.py": [
        "tests/test_cli_controller.py",
        "tests/test_aibom_output_contract.py",
    ],
    "src/controllers/web_controller.py": [
        "tests/test_web_controller.py",
        "tests/test_aibom_output_contract.py",
    ],
    "src/utils/formatter.py": [
        "tests/test_formatter.py",
        "tests/test_aibom_output_contract.py",
        "tests/test_functional_regression.py",
    ],
    "src/utils/validation.py": [
        "tests/test_validation.py",
        "tests/test_aibom_output_contract.py",
    ],
    "src/models/service.py": [
        "tests/test_service.py",
        "tests/test_functional_regression.py",
        "tests/test_aibom_output_contract.py",
    ],
    "src/models/model_file_extractors.py": MODEL_FILE_TESTS,
    "src/models/gguf_metadata.py": MODEL_FILE_TESTS,
    "src/models/safetensors_metadata.py": MODEL_FILE_TESTS,
    "src/models/config_parsing.py": MODEL_FILE_TESTS,
    "src/utils/license_utils.py": [
        "tests/test_license_utils.py",
        "tests/test_service.py",
    ],
    "src/models/aibom_domain.py": [
        "tests/test_aibom_domain.py",
    ],
    "src/models/aibom_normalizer.py": [
        "tests/test_aibom_normalizer.py",
    ],
    "scripts/dev/check_clean_state.py": [
        "tests/test_clean_state.py",
    ],
    "scripts/dev/run_regression_suite.py": [
        "tests/test_regression_runner.py",
    ],
}

DEFAULT_TARGETED_TESTS = [
    "tests/test_functional_regression.py",
    "tests/test_aibom_output_contract.py",
]

DOC_ONLY_PREFIXES = ("docs/",)
DOC_ONLY_FILES = {"README.md", "CONTRIBUTING.md"}


@dataclass(frozen=True)
class Selection:
    changed_files: list[str]
    groups: list[str]
    pytest_targets: list[str]
    run_clean_state: bool
    docs_only: bool
    full_recommended: bool


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int


def normalize_path(path: str) -> str:
    return path.strip().replace("\\", "/")


def unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def is_docs_only_file(path: str) -> bool:
    normalized = normalize_path(path)
    return normalized in DOC_ONLY_FILES or normalized.startswith(DOC_ONLY_PREFIXES)


def command_text(command: list[str]) -> str:
    return " ".join(command)


def clean_state_command() -> list[str]:
    script = "scripts/dev/check_clean_state.py"
    if sys.platform.startswith("win") and shutil.which("py"):
        return ["py", script]
    return [sys.executable, script]


def pytest_command(targets: list[str]) -> list[str]:
    return [sys.executable, "-m", "pytest", *targets]


def collect_git_files(args: list[str]) -> list[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        raise SystemExit(completed.returncode)
    return [normalize_path(line) for line in completed.stdout.splitlines() if line.strip()]


def changed_files() -> list[str]:
    files: list[str] = []
    files.extend(collect_git_files(["diff", "--name-only"]))
    files.extend(collect_git_files(["diff", "--cached", "--name-only"]))
    files.extend(collect_git_files(["ls-files", "--others", "--exclude-standard"]))
    return unique(files)


def select_targets(files: list[str]) -> Selection:
    normalized_files = unique([normalize_path(path) for path in files])
    if normalized_files and all(is_docs_only_file(path) for path in normalized_files):
        return Selection(
            changed_files=normalized_files,
            groups=["docs-only guardrail"],
            pytest_targets=[],
            run_clean_state=True,
            docs_only=True,
            full_recommended=False,
        )

    groups: list[str] = []
    pytest_targets: list[str] = []
    run_clean_state = False

    for path in normalized_files:
        mapped_tests = TARGETED_TEST_MAP.get(path)
        if mapped_tests:
            groups.append(path)
            pytest_targets.extend(mapped_tests)
            if path == "scripts/dev/check_clean_state.py":
                run_clean_state = True
            continue

        if path.startswith("tests/test_") and path.endswith(".py"):
            groups.append("changed test file")
            pytest_targets.append(path)

    pytest_targets = unique(pytest_targets)

    if not pytest_targets and not run_clean_state:
        groups.append("default functional/output-contract regression")
        pytest_targets = DEFAULT_TARGETED_TESTS.copy()

    return Selection(
        changed_files=normalized_files,
        groups=unique(groups),
        pytest_targets=pytest_targets,
        run_clean_state=run_clean_state,
        docs_only=False,
        full_recommended="default functional/output-contract regression" in groups,
    )


def print_file_list(title: str, files: list[str]) -> None:
    print(f"\n{title}")
    if not files:
        print("- none")
        return
    for path in files:
        print(f"- {path}")


def print_selection(selection: Selection) -> None:
    print_file_list("Changed files detected:", selection.changed_files)
    print_file_list("Selected test groups:", selection.groups)
    print_file_list("Selected pytest targets:", selection.pytest_targets)
    if selection.docs_only:
        print("\nDocs-only changes detected; product tests are not run by targeted mode by default.")
    if selection.full_recommended:
        print("\nNo targeted mapping matched; full suite is recommended before staging.")


def run_command(command: list[str]) -> CommandResult:
    print(f"\nRunning: {command_text(command)}")
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    status = "PASS" if completed.returncode == 0 else "FAIL"
    print(f"Result: {status} ({completed.returncode})")
    return CommandResult(command=command, returncode=completed.returncode)


def summarize(results: list[CommandResult], next_action: str) -> int:
    failed = [result for result in results if result.returncode != 0]
    print("\nCommands run:")
    if not results:
        print("- none")
    for result in results:
        status = "PASS" if result.returncode == 0 else "FAIL"
        print(f"- {command_text(result.command)} -> {status} ({result.returncode})")

    overall = 1 if failed else 0
    print(f"\nOverall status: {'FAIL' if failed else 'PASS'}")
    print(f"Next recommended action: {next_action}")
    return overall


def run_targeted() -> int:
    selection = select_targets(changed_files())
    print_selection(selection)

    results: list[CommandResult] = []
    if selection.pytest_targets:
        results.append(run_command(pytest_command(selection.pytest_targets)))
    if selection.run_clean_state:
        results.append(run_command(clean_state_command()))

    if selection.docs_only:
        next_action = "Review docs changes, then run full or pre-stage mode before staging if product behavior changed."
    elif selection.full_recommended:
        next_action = "Run --mode full before staging because targeted mode used the default fallback."
    else:
        next_action = "Run --mode full or --mode pre-stage before staging or committing."
    return summarize(results, next_action)


def run_full() -> int:
    results = [
        run_command(pytest_command(["tests"])),
        run_command(clean_state_command()),
        run_command(["git", "diff", "--cached", "--name-only"]),
    ]
    return summarize(results, "Address any failures, then run --mode pre-stage before manual staging.")


def run_pre_stage() -> int:
    print("\nStaging remains manual. This runner does not stage or commit files.")
    results = [
        run_command(pytest_command(["tests"])),
        run_command(clean_state_command()),
        run_command(["git", "diff", "--cached", "--name-only"]),
        run_command(["git", "status", "--short", "-uall"]),
    ]
    return summarize(results, "If all checks pass, manually review git status before staging.")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repo regression checks.")
    parser.add_argument(
        "--mode",
        choices=("targeted", "full", "pre-stage"),
        required=True,
        help="Regression mode to run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.mode == "targeted":
        return run_targeted()
    if args.mode == "full":
        return run_full()
    if args.mode == "pre-stage":
        return run_pre_stage()
    raise AssertionError(f"Unexpected mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
