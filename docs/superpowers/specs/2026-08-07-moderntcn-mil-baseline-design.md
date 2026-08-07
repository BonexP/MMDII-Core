# ModernTCN-MIL Baseline Design

## Goal

Build a self-contained, leakage-safe first training loop for Dataset v0.2.
The first implemented deep route is a small ModernTCN-style encoder over
time-based windows followed by a configurable weld-level aggregator. The
training unit remains the weld, not the window.

## Scope and boundaries

This phase implements B0 statistical features, E0 full-signal support,
E1 window plus aggregation training, and the infrastructure needed for later
PatchTST and Mamba replacements. It does not claim that attention weights are
defect boundaries, does not train on `pore`, and does not add image inputs.

The three formal targets are `flash`, `blur`, and `tunnel`. `pore` remains in
the release metadata and is reported as excluded because it has one positive
image group. A completed empty defect list is represented by an all-zero target
vector; no separate normal class is created.

## Repository boundary

All reusable code, model code, experiment configuration, and training scripts
live in `MMDII-Core`. The top-level repository only supplies and publishes the
Dataset v0.2 release. A training machine can clone Core and point it at a
published release directory without importing the annotation application or
the original MAT directory.

## Data flow

```text
Dataset v0.2 release
    -> index samples, labels and preassigned folds
    -> select train/validation by weld fold
    -> fit train-fold channel statistics
    -> resample and cut each weld into seconds-based windows
    -> pad variable window counts with a boolean mask
    -> ModernTCN-small window embeddings
    -> mean / max / top-k mean / gated attention aggregation
    -> weld-level 3-label logits
    -> weighted BCE loss and weld-level metrics
    -> one out-of-fold prediction per weld
```

Folds are read from `folds.csv`; the training code never randomly assigns
windows. Normalization statistics are calculated from training-fold windows
only. The loader retains `sample_id`, `weld_id`, `image_group`, fold and window
start time for traceability.

## Stable interfaces

### Dataset API

`mmdii.data.training_dataset` provides:

- `DatasetIndex.from_release(path, target_codes)` to validate the release and
  build an in-memory index;
- `WindowSpec(target_fs, window_seconds, stride_seconds)`;
- `FullSignalSpec(target_fs, output_samples)` for a complete-weld baseline that
  rescales the entire accepted signal to one fixed-length observation;
- `FoldNormalizer.fit(welds, window_spec)` and `.transform(signal)`;
- `window_signal(signal, time, spec)` returning padded windows and a mask;
- `WeldWindowDataset` yielding one weld bag at a time.

The dataset returns arrays as `[windows, channels, samples]` and a mask as
`[windows]`. The collator pads only the number of windows, never the signal
channels, and carries a weld identifier with every batch item.

### Model API

`mmdii.models.modern_tcn.ModernTCNSmall` accepts `[batch, channels, samples]`
and returns `[batch, embedding]` for one window. The implementation is a
small, documented ModernTCN-style block: pointwise channel projection,
depthwise large-kernel temporal convolution, normalization, residual path, and
pointwise cross-channel mixing. It is intentionally not a vendored copy of the
official benchmark runner.

`mmdii.models.mil.WeldMIL` accepts padded window embeddings and a boolean mask,
then returns weld-level logits and optional attention weights. The aggregation
mode is a configuration value: `mean`, `max`, `topk_mean`, or `gated_attention`.
All modes use the same encoder and output dimension so comparisons change one
experimental axis at a time. Gated attention is per target code, because two
defect types may depend on different windows.

### Training and evaluation API

`mmdii.training.cross_validation.run_cross_validation` consumes a validated
`DatasetIndex` and an experiment config, trains one model per held-out fold,
and returns weld-level OOF logits. `mmdii.evaluation.multilabel` calculates
per-code average precision/PR-AUC, recall at a configured threshold, macro F1,
and the count of valid positive folds. Class weights are fit from the current
training fold only.

`mmdii.models.statistical` supplies the B0 lower bound: per-channel mean,
standard deviation, RMS, minimum, maximum and peak-to-peak features followed
by one fold-local logistic regression per target. It uses the same OOF folds
and output schema as the deep models.

## Engineering choices

- Use native PyTorch modules and the existing NumPy/SciPy stack. Add PyTorch
  and scikit-learn as a `train` optional dependency; importing data-only APIs
  must not require PyTorch.
- Use TOML configuration and explicit seeds. Do not add a training framework,
  experiment tracker, database, or registry in this phase.
- Keep the release immutable. Dynamic windows are cheaper and less error-prone
  than materializing overlapping arrays in Dataset v0.2.
- Reject missing IDs, duplicate IDs, invalid folds, and image-group leakage at
  the loader boundary.
- Treat short bags with zero padding and masks. Masked pooling must have the
  same result as unpadded pooling.

## Experiment sequence

1. B0: deterministic hand-crafted summary features plus logistic regression.
2. E0: the complete signal rescaled to a fixed length, then ModernTCN-small
   with one weld-level head.
3. E1a: window encoder with mean aggregation.
4. E1b: max and top-k mean aggregation.
5. E1c: gated attention MIL.
6. E2: replace only the encoder with a small PatchTST-style patch encoder.

Every experiment uses the same target codes, folds, preprocessing policy,
threshold selection rule and OOF output schema. The first success criterion is
not a particular score: it is a reproducible, leakage-free OOF run whose
artifacts can be inspected and rerun on the training host.

## Testing

Data tests cover seconds-based resampling, deterministic windows, masks and
fold-local normalization. Model tests cover tensor shapes, masked pooling and
finite logits. A CPU smoke test runs only when PyTorch is installed and trains
on a tiny synthetic release for one fold. The real 101-weld run is a training
host acceptance test, not a required local unit test.
