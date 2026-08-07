# ModernTCN-MIL Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a self-contained, leakage-safe ModernTCN-style windowed MIL baseline for Dataset v0.2, with OOF weld-level evaluation and training-host scripts.

**Architecture:** Dataset v0.2 is indexed once, folds are trusted only after independent validation, and windows are generated dynamically inside each fold. A small pure-PyTorch ModernTCN-style encoder produces window embeddings; one of four fixed aggregators produces weld logits for `flash`, `blur`, and `tunnel`. The Core package owns all code and the top-level data release remains immutable.

**Tech Stack:** Python 3.11, NumPy, SciPy, PyTorch (train extra), scikit-learn (train extra), TOML, unittest.

---

### Task 1: Branch and package setup

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/mmdii/models/__init__.py`
- Modify: `src/mmdii/training/__init__.py`
- Modify: `src/mmdii/evaluation/__init__.py`

- [ ] Confirm the Core branch starts at the Dataset v0.2 commit and record the package version `0.2.0`.
- [ ] Add optional `train` dependencies for PyTorch and scikit-learn without making data-only imports require either package.
- [ ] Keep package `__init__` files free of eager PyTorch imports.
- [ ] Run the existing Core test suite before implementation; expected result is 29 passing tests.
- [ ] Commit `chore: prepare core for model training`.

### Task 2: Dataset index and target parsing

**Files:**
- Create: `src/mmdii/data/training_dataset.py`
- Create: `tests/test_training_dataset.py`

- [ ] Write failing tests for exact sample/label/fold ID equality, target-code filtering, rejection of duplicate IDs and image-group leakage.
- [ ] Implement `DatasetIndex.from_release` using stdlib CSV/JSON and existing `validate_dataset_release`.
- [ ] Represent one weld as an immutable record containing sample ID, weld ID, image group, fold, target vector, signal path, and metadata.
- [ ] Keep `pore` in parsed metadata but exclude it from the target vector when `target_codes=("flash", "blur", "tunnel")`.
- [ ] Run the focused tests and commit `feat: index dataset v0.2 for training`.

### Task 3: Resampling, windowing and fold-local normalization

**Files:**
- Modify: `src/mmdii/data/training_dataset.py`
- Modify: `tests/test_training_dataset.py`

- [ ] Write failing tests for a seconds-based window count, deterministic starts, rational resampling, short-signal padding, and a mask that excludes padded windows.
- [ ] Implement `WindowSpec`, `window_signal`, and a SciPy `resample_poly` path that bypasses resampling when the source rate already equals the target.
- [ ] Implement `FoldNormalizer.fit` from training-fold signal values only, with zero standard deviations clamped to one.
- [ ] Implement `WeldWindowDataset` and a collator returning padded `[B,N,C,L]`, `[B,N]` masks, targets, IDs and window starts.
- [ ] Verify focused tests pass and commit `feat: add leakage-safe signal windows`.

### Task 4: Statistical lower bound

**Files:**
- Create: `src/mmdii/models/statistical.py`
- Create: `tests/test_statistical.py`

- [ ] Write failing tests for deterministic per-channel mean, standard deviation, RMS, minimum, maximum and peak-to-peak features.
- [ ] Implement `extract_statistical_features` with a fixed documented column order.
- [ ] Implement fold-local one-vs-rest logistic regression for the configured target codes and return weld-level probabilities in the common OOF schema.
- [ ] Raise a clear error if a training fold contains only one class for a requested target rather than emitting a misleading model.
- [ ] Run focused tests and commit `feat: add statistical weld baseline`.

### Task 5: Full-signal mode and ModernTCN-style encoder

**Files:**
- Modify: `src/mmdii/data/training_dataset.py`
- Modify: `tests/test_training_dataset.py`
- Create: `src/mmdii/models/modern_tcn.py`
- Create: `tests/test_modern_tcn.py`

- [ ] Write failing data tests for `FullSignalSpec`, proving the entire weld is deterministically rescaled to a fixed point count without being cut into windows.
- [ ] Write failing shape and gradient tests, skipping only when PyTorch is unavailable.
- [ ] Implement the full-signal transform with SciPy resampling and return one valid instance per weld.
- [ ] Implement `ModernTCNSmall` with an input projection, two residual large-kernel depthwise blocks, pointwise cross-channel mixing, normalization, dropout and adaptive temporal pooling.
- [ ] Validate `[B,C,L] -> [B,D]`, finite output, configurable channel count and deterministic eval mode.
- [ ] Run model tests on the training host or any environment with the train extra and commit `feat: add modern tcn small encoder`.

### Task 6: Weld MIL heads

**Files:**
- Create: `src/mmdii/models/mil.py`
- Create: `tests/test_mil.py`

- [ ] Write failing tests for `mean`, `max`, `topk_mean` and `gated_attention` with padded windows, checking padded values cannot affect logits.
- [ ] Implement a common `WeldMIL` head with a linear window classifier and per-code gated attention for the attention mode.
- [ ] Return logits `[B,K]` and optional attention `[B,K,N]`; normalize attention only over valid windows.
- [ ] Run focused tests and commit `feat: add masked weld mil aggregators`.

### Task 7: Multilabel loss and metrics

**Files:**
- Create: `src/mmdii/evaluation/multilabel.py`
- Create: `tests/test_multilabel.py`

- [ ] Write failing tests for fold-local class weights, ignored `pore`, PR-AUC/average precision, recall, macro F1 and undefined-positive reporting.
- [ ] Implement weighted BCE loss construction and metric functions with explicit `valid_positive_count` fields; never silently score an absent positive class as zero quality.
- [ ] Use scikit-learn only inside evaluation functions and raise a clear optional-dependency error if unavailable.
- [ ] Run focused tests and commit `feat: add weld-level multilabel evaluation`.

### Task 8: Cross-validation runner and config

**Files:**
- Create: `src/mmdii/training/cross_validation.py`
- Create: `configs/moderntcn_mil_v0_1.toml`
- Create: `tests/test_cross_validation.py`

- [ ] Write failing tests for one-fold synthetic training, deterministic seeding, train-only normalization, one OOF row per weld, and output artifact fields.
- [ ] Implement `run_cross_validation` with `statistical`, `full_signal`, and `window_mil` experiment modes plus explicit device, seed, fold, optimizer, epoch and aggregator settings; compute class weights separately per training fold.
- [ ] Emit `oof_predictions.csv`, `fold_metrics.json`, `run_config.json` and a compact `training_summary.json` under the configured output directory.
- [ ] Run the CPU smoke test when PyTorch is installed; otherwise run import/config tests and leave the command ready for the training host.
- [ ] Commit `feat: add cross validation training runner`.

### Task 9: Training-host entry point and documentation

**Files:**
- Create: `scripts/train_baseline.py`
- Modify: `README.md`
- Modify: `tests/test_package.py`

- [ ] Write failing CLI tests for required release path, config loading and machine-readable completion output.
- [ ] Implement a thin CLI that loads Dataset v0.2, runs the configured aggregator, and prints the output directory plus OOF summary.
- [ ] Document installation, command examples, experiment order, target-code boundary and the fact that attention is candidate evidence rather than localization truth.
- [ ] Run Core tests, CLI tests and compileall; commit `feat: expose baseline training command`.

### Task 10: Final verification and handoff

- [ ] Run all data, model and training tests in an environment with the train extra.
- [ ] Run a tiny synthetic smoke experiment and verify one OOF prediction per weld, no cross-fold image group, finite logits and reproducible seed.
- [ ] Run `git diff --check` and inspect both parent and Core status.
- [ ] Report the training-host command and the exact boundary between the immutable Dataset v0.2 release and Core model code.
