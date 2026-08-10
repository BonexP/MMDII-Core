# Real-Data Training Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real Dataset v0.2 environment checks and a one-step ModernTCN-MIL smoke command without changing formal training behavior.

**Architecture:** Put reusable report and smoke logic in `mmdii.training.readiness`. Keep `scripts/check_training_environment.py` and `scripts/smoke_train.py` as thin argument-parsing wrappers. Reuse `ExperimentConfig`, `DatasetIndex`, `WeldWindowDataset`, existing model construction, and the existing weighted BCE path.

**Tech Stack:** Python 3.11, NumPy, SciPy, optional PyTorch/scikit-learn, TOML, unittest.

---

### Task 1: Readiness report tests

**Files:**
- Create: `tests/test_training_readiness.py`
- Create: `src/mmdii/training/readiness.py`

- [ ] Write tests for a report containing package, dataset, output and error fields, with dependency checks injected so tests do not require PyTorch.
- [ ] Write tests that invalid fold and non-positive batch size are rejected before data loading.
- [ ] Run the focused tests and confirm they fail because the readiness module does not exist.
- [ ] Implement the smallest report builder with lazy optional imports, DatasetIndex validation, nearest-existing-parent output writability check, and deterministic JSON-compatible values.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Environment CLI

**Files:**
- Create: `scripts/check_training_environment.py`
- Modify: `tests/test_training_readiness.py`

- [ ] Write a CLI test for `--json` output and non-zero status on an unsuccessful report.
- [ ] Implement `--config`, `--release-dir`, `--output-dir`, and `--json`; print either compact human-readable lines or sorted JSON.
- [ ] Run the focused CLI tests and confirm they pass.

### Task 3: Real-data smoke implementation

**Files:**
- Modify: `src/mmdii/training/readiness.py`
- Modify: `tests/test_training_readiness.py`

- [ ] Write a train-extra-dependent test for a finite one-step summary, skipped only when PyTorch is absent, plus a missing-train-extra error test.
- [ ] Implement real fold selection, fold-local normalization, `WeldWindowDataset`, one collated batch, the configured ModernTCN-MIL model, weighted BCE, one AdamW update, and finite-value/parameter-delta checks.
- [ ] Ensure the smoke path does not call `run_cross_validation` or write formal output files.
- [ ] Run focused tests and confirm they pass or skip only for the documented optional dependency.

### Task 4: Smoke CLI and documentation

**Files:**
- Create: `scripts/smoke_train.py`
- Modify: `README.md`
- Create: `docs/research-progress.md`
- Modify: `tests/test_training_readiness.py`

- [ ] Write CLI tests for required config, fold/batch/device arguments, and machine-readable completion output.
- [ ] Implement the thin smoke CLI and document the exact environment-check, smoke, and formal-training commands.
- [ ] Add the consolidated research progress ledger with data counts, target limitations, model sequence, implementation boundary, and pending training-host acceptance.
- [ ] Run Core tests and compileall.

### Task 5: Integration verification

- [ ] Run the real release environment check from the model worktree.
- [ ] Run the real-data smoke command; if the train extra is absent, record the clear dependency failure rather than substituting synthetic data.
- [ ] Run top-level tests, `git diff --check`, and inspect parent/Core status.
- [ ] Commit Core changes, update the parent submodule pointer, and report the training-host commands and remaining acceptance boundary.
