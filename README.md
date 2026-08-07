# MMDII-Core

MMDII-Core is the standalone Python code repository for MMDII. It owns reusable
MAT readers, dataset-preparation logic, multimodal models, training workflows,
and evaluation code.

It is used by the parent MMDII integration repository as a Git submodule. The
parent owns raw-data locations, labels, project documentation, and workflow
orchestration. Core code receives those locations through explicit arguments or
configuration paths and must not hard-code parent-repository paths.

## Layout

- `src/mmdii/data/`: Dataset v0.2 indexing, signal quality and preparation code.
- `src/mmdii/models/`: statistical lower bound, ModernTCN and weld MIL heads.
- `src/mmdii/training/`: deterministic five-fold OOF training workflows.
- `src/mmdii/evaluation/`: weld-level multi-label evaluation workflows.
- `tests/`: standalone package tests.

Run the package checks from this directory:

```powershell
python -m unittest discover -s tests -v
```

## MAT Header Inspection

Install the Core package into the active Python environment:

```powershell
python -m pip install -e .
```

`mmdii.data.inspect_mat_directory` reads MAT variable names, MATLAB classes,
and shapes through `scipy.io.whosmat`. It does not load signal arrays and does
not require every file to contain the same variables. Unreadable files are
reported individually while the remaining files are inspected.

## Dataset v0.2 training

Install the standalone training dependencies on the training host:

```powershell
python -m pip install -e ".[train]"
```

Run the configured windowed ModernTCN-MIL experiment:

```powershell
python scripts/train_baseline.py --config configs/moderntcn_mil_v0_1.toml
```

The release path may be overridden when the dataset is stored outside this
checkout:

```powershell
python scripts/train_baseline.py `
  --config configs/moderntcn_mil_v0_1.toml `
  --release-dir D:\datasets\mmdii-v0-2\releases\<release-id> `
  --output-dir outputs\mmdii-v0-2-gated
```

The runner writes one weld-level OOF prediction per accepted sample, fold
metrics, the resolved configuration and a summary JSON. `pore` remains in the
release metadata but is excluded from the first three formal targets. Attention
weights are candidate windows for review, not validated defect locations.

Experiment order is B0 statistical features, E0 full-signal ModernTCN, then
windowed mean/max/top-k/gated MIL. The same preassigned image-group folds are
used for every comparison.
