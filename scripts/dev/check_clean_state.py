"""Phase A cleanup guardrail for staged files.

This script is read-only. It does not modify files, stage files, commit files,
validate CycloneDX output, or prove AIBOM correctness.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from dataclasses import dataclass


FORBIDDEN_DOC_PREFIXES = (
    "docs/analysis/",
    "docs/design/",
    "docs/reference/",
    "docs/requirements/",
    "docs/analysis/current-output-samples/",
)

PHASE_1_LOCAL_FILES = {
    "src/models/aibom_domain.py",
    "src/models/aibom_normalizer.py",
    "tests/test_aibom_domain.py",
    "tests/test_aibom_normalizer.py",
    "scripts/dev/check_cyclonedx_capabilities.py",
}

DEPENDENCY_FILES = {
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "uv.lock",
    "poetry.lock",
    "Pipfile",
    "Pipfile.lock",
    "setup.py",
    "setup.cfg",
}

SOURCE_PREFIXES = (
    "src/",
)

TEST_PREFIXES = (
    "tests/",
)

ALLOWED_GUARDRAIL_TEST_FILES = {
    "tests/test_clean_state.py",
    "tests/test_regression_runner.py",
}

WORKSET_PREFIXES = (
    ".ai-workset/",
)

PYTEST_CACHE_PREFIXES = (
    "pytest-cache-files-",
)

GENERATED_OR_SYNTHETIC_MARKERS = (
    "current-output-samples/",
    "synthetic",
    "generated",
)

GENERATED_OR_SYNTHETIC_PREFIXES = (
    "dist/",
    "build/",
    "sboms/",
)


@dataclass(frozen=True)
class Finding:
    path: str
    reason: str


def run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise RuntimeError(message)
    return result.stdout


def normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def staged_paths() -> list[str]:
    output = run_git(["diff", "--cached", "--name-only", "--diff-filter=ACMRTD"])
    return [normalize_path(line) for line in output.splitlines() if line.strip()]


def unstaged_or_untracked_paths() -> list[str]:
    output = run_git(["status", "--porcelain=v1", "-uall"])
    paths: list[str] = []

    for line in output.splitlines():
        if len(line) < 4:
            continue
        index_status = line[0]
        worktree_status = line[1]
        raw_path = line[3:]

        if " -> " in raw_path:
            raw_path = raw_path.rsplit(" -> ", maxsplit=1)[1]

        if index_status == "?" or worktree_status not in {" ", "!"}:
            paths.append(normalize_path(raw_path.strip('"')))

    paths.extend(root_pytest_cache_dirs())
    return sorted(set(paths))


def root_pytest_cache_dirs() -> list[str]:
    return [
        path.name
        for path in Path.cwd().iterdir()
        if path.is_dir() and path.name.startswith(PYTEST_CACHE_PREFIXES)
    ]


def is_generated_or_synthetic(path: str) -> bool:
    normalized = normalize_path(path).lower()
    if any(normalized.startswith(prefix) for prefix in GENERATED_OR_SYNTHETIC_PREFIXES):
        return True
    return any(marker in normalized for marker in GENERATED_OR_SYNTHETIC_MARKERS)


def is_source(path: str) -> bool:
    normalized = normalize_path(path)
    return normalized.startswith(SOURCE_PREFIXES)


def is_test(path: str) -> bool:
    normalized = normalize_path(path)
    return normalized.startswith(TEST_PREFIXES)


def is_allowed_guardrail_test(path: str) -> bool:
    normalized = normalize_path(path)
    return normalized in ALLOWED_GUARDRAIL_TEST_FILES


def is_readme(path: str) -> bool:
    normalized = normalize_path(path).lower()
    return normalized in {"readme", "readme.md"} or normalized.startswith("readme.")


def is_workset(path: str) -> bool:
    normalized = normalize_path(path)
    return normalized.startswith(WORKSET_PREFIXES)


def is_pytest_cache_files(path: str) -> bool:
    normalized = normalize_path(path)
    return any(normalized.startswith(prefix) for prefix in PYTEST_CACHE_PREFIXES)


def collect_findings(
    paths: list[str],
    allow_dependency_files: bool,
    allow_source_files: bool,
    allow_future_mode_files: bool,
) -> list[Finding]:
    findings: list[Finding] = []

    for path in paths:
        normalized = normalize_path(path)

        if any(normalized.startswith(prefix) for prefix in FORBIDDEN_DOC_PREFIXES):
            findings.append(Finding(normalized, "staged local analysis/design/reference/requirements material"))

        if is_generated_or_synthetic(normalized):
            findings.append(Finding(normalized, "staged generated or synthetic sample material"))

        if normalized in DEPENDENCY_FILES and not allow_dependency_files:
            findings.append(Finding(normalized, "staged dependency file requires explicit review"))

        if is_source(normalized) and not allow_source_files:
            findings.append(Finding(normalized, "staged source file is outside Phase A guardrail scope"))

        if normalized in PHASE_1_LOCAL_FILES and not allow_future_mode_files:
            findings.append(Finding(normalized, "staged local Phase 1 implementation-shaped file requires explicit review"))

        if is_test(normalized) and not is_allowed_guardrail_test(normalized) and not allow_future_mode_files:
            findings.append(Finding(normalized, "staged test file is outside Phase A guardrail scope"))

        if is_readme(normalized) and not allow_future_mode_files:
            findings.append(Finding(normalized, "staged README or end-user doc is outside Phase A guardrail scope"))

        if is_workset(normalized) and not allow_future_mode_files:
            findings.append(Finding(normalized, "staged workset artifact is outside Phase A guardrail scope"))

        if is_pytest_cache_files(normalized) and not allow_future_mode_files:
            findings.append(Finding(normalized, "staged pytest cache file is generated local material"))

    return findings


def collect_warnings(paths: list[str]) -> list[Finding]:
    warnings: list[Finding] = []

    for path in paths:
        normalized = normalize_path(path)

        if any(normalized.startswith(prefix) for prefix in FORBIDDEN_DOC_PREFIXES):
            warnings.append(Finding(normalized, "local analysis/design/reference/requirements material needs human review before staging"))

        if is_generated_or_synthetic(normalized):
            warnings.append(Finding(normalized, "generated or synthetic sample material needs human review before staging"))

        if normalized in DEPENDENCY_FILES:
            warnings.append(Finding(normalized, "dependency file needs human review before staging"))

        if normalized in PHASE_1_LOCAL_FILES:
            warnings.append(Finding(normalized, "local Phase 1 implementation-shaped file needs human review before staging"))

        if is_source(normalized):
            warnings.append(Finding(normalized, "source file change needs human review before staging"))

        if is_workset(normalized):
            warnings.append(Finding(normalized, "workset artifact needs human review before staging"))

        if is_pytest_cache_files(normalized):
            warnings.append(Finding(normalized, "pytest cache temp directory should be cleaned or ignored before cleanup fixes"))

    return warnings


def print_status_note() -> None:
    status = run_git(["status", "--short", "-uall"])
    if status.strip():
        print("Working tree has unstaged or untracked changes:")
        print(status.rstrip())
    else:
        print("Working tree is clean.")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Phase A cleanup guardrails for staged files.")
    parser.add_argument(
        "--allow-dependency-files",
        action="store_true",
        help="Allow staged dependency files after explicit human review.",
    )
    parser.add_argument(
        "--allow-source-files",
        action="store_true",
        help="Allow staged source files after leaving Phase A guardrail mode.",
    )
    parser.add_argument(
        "--allow-future-mode-files",
        action="store_true",
        help="Allow staged tests, README, workset artifacts, and local investigation files after explicit future-mode review.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    print("Phase A cleanup guardrail check")
    print("Note: this script is read-only and does not validate CycloneDX output or prove AIBOM correctness.")
    print()

    try:
        paths = staged_paths()
        findings = collect_findings(
            paths,
            args.allow_dependency_files,
            args.allow_source_files,
            args.allow_future_mode_files,
        )
        warnings = collect_warnings(unstaged_or_untracked_paths())
        print_status_note()
    except RuntimeError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2

    print()
    if not paths:
        print("No staged files found.")

    if findings:
        print("FAIL: staged files violate Phase A cleanup guardrails.")
        print("Affected files:")
        for finding in findings:
            print(f"- {finding.path}: {finding.reason}")
        return 1

    if warnings:
        print("WARN: risky unstaged or untracked files exist and need human review before staging.")
        print("Affected files:")
        for warning in warnings:
            print(f"- {warning.path}: {warning.reason}")
        print()

    print("PASS: no staged files violate Phase A cleanup guardrails.")
    print("PASS does not mean the repository is clean; review any WARN lines and git status output above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
