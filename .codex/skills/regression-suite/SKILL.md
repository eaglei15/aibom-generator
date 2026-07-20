# Regression Suite

Use this skill after implementation changes in this repository.

## Required Checks

- After every implementation change, run targeted regression at minimum:
  `.\.venv\Scripts\python.exe scripts\dev\run_regression_suite.py --mode targeted`
- Before staging or committing, run full or pre-stage regression:
  `.\.venv\Scripts\python.exe scripts\dev\run_regression_suite.py --mode full`
  or
  `.\.venv\Scripts\python.exe scripts\dev\run_regression_suite.py --mode pre-stage`
- Use pytest through the runner. Do not rely only on `unittest discover`.
- Never hide test failures. Report the exact command, exit status, and failing check.
- Never stage or commit unless the user explicitly asks.

## Decision Guidance

- Small localized fix: run targeted mode.
- Controller, output, formatter, schema, or validation fix: run targeted mode and make sure output-contract tests are included.
- Broad model, service, schema, or cross-path change: run full mode.
- Before commit or staging: run pre-stage mode.

## Manual Smoke Reminder

Real Hugging Face generation remains optional and manual because the automated regression suite is offline and deterministic. Mention this residual integration risk in final responses when relevant.
