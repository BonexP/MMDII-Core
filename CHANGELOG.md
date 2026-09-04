# Changelog

All notable changes to MMDII-Core are recorded here. Versions follow Semantic
Versioning.

## [Unreleased]

### Added

- Resumable `nohup` launcher for unattended B0/E0/E1 comparison runs on Linux
  and JupyterLab training hosts.
- Dataset v0.2 overnight baseline analysis report with raw fold/OOF result
  tables and model-selection conclusions.
- Robustness experiment launcher for multi-seed B0/E1c runs.
- Tracked-output validator for OOF, fold, completion-marker, and probability
  consistency checks.
- Nonlinear Random Forest reference using the existing fold-local statistical
  features.
- Cross-fitted per-class OOF threshold calibration utility.

### Changed

- Training CLI now supports seed, epoch, optimizer, learning-rate,
  regularization, gradient-clipping, and conservative training-loss early-stop
  overrides while preserving the v0.3.0 defaults.

## [0.3.0] - 2026-08-13

### Added

- Self-contained Dataset v0.2 artifacts, bandwidth audit, deterministic grouped
  folds, and training dataset indexing.
- Statistical, full-signal, and windowed ModernTCN-MIL baselines with weld-level
  multilabel evaluation and out-of-fold training.
- Real-dataset environment checks, CPU smoke training, and formal training
  command-line workflows.
- Reusable CPU-only training-host setup based on Python 3.11 and uv.

### Changed

- Bounded window encoder memory and strengthened metric validation.

## [0.2.0] - 2026-07-30

### Added

- Dataset v0.1 MAT parsing, mapping validation, force audit, staged preparation,
  and immutable release publication.

## [0.1.0] - 2026-07-15

### Added

- Initial MMDII-Core package and MAT header inspection APIs.
