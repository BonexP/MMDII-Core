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
- `src/mmdii/models/`: statistical lower bound, random forest, ModernTCN and weld MIL heads.
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

For a reproducible Linux/JupyterLab setup with an uv-managed Python 3.11 and
CPU-only PyTorch, see [`docs/training-host-cpu-setup.md`](docs/training-host-cpu-setup.md).

Install the standalone training dependencies on the training host:

```powershell
python -m pip install -e ".[train]"
```

The primary protocol is Dataset v0.2.1 with `weld_independent` folds. The
published release is data, not Git content: place
`weld-independent-v0-2-1-r2` on the training host before running. Check the
environment and immutable release before allocating a full run:

```powershell
python scripts/check_training_environment.py `
  --config configs/moderntcn_mil_weld_independent_v0_2_1.toml `
  --release-dir D:\datasets\mmdii-v0-2\releases\weld-independent-v0-2-1-r2
```

Perform one optimizer update using real Dataset v0.2 windows. This command does
not write formal OOF artifacts or use synthetic data:

```powershell
python scripts/smoke_train.py `
  --config configs/moderntcn_mil_weld_independent_v0_2_1.toml `
  --release-dir D:\datasets\mmdii-v0-2\releases\weld-independent-v0-2-1-r2 `
  --fold 0 --batch-size 1
```

After both checks pass, run the configured windowed ModernTCN-MIL experiment:

```powershell
python scripts/train_baseline.py --config configs/moderntcn_mil_weld_independent_v0_2_1.toml
```

The same entry point also supports the nonlinear statistical reference:

```powershell
python scripts/train_baseline.py `
  --config configs/moderntcn_mil_v0_1.toml `
  --release-dir D:\datasets\mmdii-v0-2\releases\<release-id> `
  --output-dir outputs\random-forest-v0-2 `
  --mode random_forest
```

Calibrate thresholds without using the held-out fold labels:

```powershell
python scripts/calibrate_oof_thresholds.py `
  --oof outputs\mmdii-v0-2-gated\oof_predictions.csv `
  --output outputs\mmdii-v0-2-gated\threshold-calibration.json
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
windowed mean/max/top-k/gated MIL. The v0.2.1 main protocol uses deterministic
weld-independent folds; the same release also carries `folds_image_group.csv`
for strict unseen-image-source comparison. The older v0.2.0 release remains
available for reproducing the original image-group results.

### Primary GPU host command

After checking out the release branch and transferring the published dataset
directory to the host, run the complete v0.2.1 suite with this Linux command:

```bash
RELEASE=/data/mmdii/releases/weld-independent-v0-2-1-r2
PYTHON=.venv/bin/python \
  CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_overnight_suite.sh "$RELEASE" outputs/weld-independent-v0-2-1
```

The launcher uses the v0.2.1 weld-independent configuration by default. It
starts B0, E0, E1a, E1b-max, E1b-top-k and E1c sequentially under `nohup`, so
it survives SSH/JupyterLab disconnection. Follow it with
`tail -f outputs/weld-independent-v0-2-1/suite.log`; rerun the same command to
resume only incomplete experiments. Use the same release and set
`--fold-scheme image_group` only for the separate strict comparison run.

### Unattended Linux/JupyterLab run

Run the complete comparison suite in a process that survives terminal or
JupyterLab disconnection:

```bash
RELEASE=/path/to/mmdii-v0-2/releases/<release-id>
bash scripts/run_overnight_suite.sh "$RELEASE"
```

The launcher returns immediately and prints the PID, output directory, and
commands for following the log and current status. It runs B0, E0, E1a mean,
E1b max, E1b top-k mean, and E1c gated attention sequentially. Every experiment
gets a separate directory. Passing the same output directory again skips
completed experiments and resumes at the first incomplete experiment:

```bash
bash scripts/run_overnight_suite.sh "$RELEASE" outputs/overnight-20260902-210000
```

For protocol alignment across seeds, use the multi-seed launcher. It runs the
same six overnight models plus the Random Forest reference under the selected
release and fold scheme:

```bash
RELEASE=/path/to/mmdii-v0-2/releases/weld-independent-v0-2-1-r2
SEEDS="7 17 27" \
  PYTHON=.venv/bin/python CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_robustness_suite.sh \
    "$RELEASE" outputs/alignment-weld-independent-v0-2-1 \
    configs/moderntcn_mil_weld_independent_v0_2_1.toml
```

To test a deeper ModernTCN without changing the training budget, use the
depth-only configuration below. It changes `block_count` from 2 to 4 and
keeps the data protocol, width, kernel, optimizer, and 20 epochs fixed:

```bash
RELEASE=/path/to/mmdii-v0-2/releases/weld-independent-v0-2-1-r2
SEEDS="7 17 27" \
  PYTHON=.venv/bin/python CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_robustness_suite.sh \
    "$RELEASE" outputs/alignment-weld-independent-deep-v0-2-1 \
    configs/moderntcn_mil_weld_independent_deep_v0_2_1.toml
```

Run this before increasing epochs so the depth effect remains identifiable.

### Time-frequency matrix

The time-frequency experiment stack provides STFT (`stft_256`, `stft_512`),
Morlet CWT (`cwt_morl`), and five-level `db4` SWT (`dwt_swt_db4`) window
representations. Each can use `cnn2d`, `separable_cnn2d`, `resnet2d_small`, or
`convnext2d_lite`, with all four MIL aggregators. The matrix also includes raw
ModernTCN plus STFT/CWT fusion. All runs use the weld-independent five-fold
protocol and seeds 7, 17, and 27.

Run a one-fold smoke matrix before the full detached suite:

```bash
RELEASE=/data/mmdii/releases/weld-independent-v0-2-1-r2
SMOKE_ONLY=1 SMOKE_FOLD=0 PYTHON=.venv/bin/python \
  bash scripts/run_time_frequency_suite.sh "$RELEASE" outputs/time-frequency-smoke
```

After smoke validation succeeds, launch the full 103-configuration matrix:

```bash
SEEDS="7 17 27" PYTHON=.venv/bin/python CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_time_frequency_suite.sh \
  "$RELEASE" outputs/time-frequency-v0-2-1
```

Summarize completed runs across seeds and compare them with raw gated
attention and the statistical baselines:

```bash
.venv/bin/python scripts/summarize_time_frequency_matrix.py \
  --root outputs/time-frequency-v0-2-1
```

The worker uses `.venv/bin/python` directly, so shell activation is not needed
after launch. `nohup` protects against terminal disconnection; it cannot keep a
process alive if the hosting platform suspends or destroys the entire instance.
