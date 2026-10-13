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

### Training progress logs

Training entry points emit line-based progress to **stderr**; stdout stays
machine-readable JSON. The library is silent by default, and progress is enabled
only by the CLI and suite entry points. Set the level with `MMDII_LOG_LEVEL`
(default `info`):

```bash
MMDII_LOG_LEVEL=info python scripts/train_baseline.py --config configs/moderntcn_mil_weld_independent_v0_2_1.toml
```

Each fold logs its split and metric summary. Neural-network folds also log
per-epoch loss/learning-rate/gradient statistics with elapsed time and ETA,
throttled batch progress, and an early-stop notice when applicable. Statistical
folds log feature extraction progress; time-frequency runs log transform
progress and cache hits. The final results table is written after cross-validation.
Because detached suites capture stderr into each run's `train.log`,
`tail -f <run>/train.log` follows live progress.

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

The matrix requires the Core training environment, including PyWavelets for
CWT/DWT. After pulling a Core update, refresh an existing virtual environment
before launching:

```bash
uv pip install --python .venv/bin/python -e ".[train]"
```

The suite performs this dependency check before starting any run and prints the
same command if a package is missing.

After smoke validation succeeds, launch the full 103-configuration matrix:

```bash
RELEASE=/data/mmdii/releases/weld-independent-v0-2-1-r2
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

### v0.3 deep-embedding tree suite

The v0.3 second-stage suite tests whether a tree classifier can use a
fold-safe deep representation more effectively than the neural classification
head. It is a fixed 4-candidate matrix:

| Candidate | Representation | Encoder | Aggregator |
| --- | --- | --- | --- |
| raw + STFT | `stft_256` | `resnet2d_small` | mean |
| raw + CWT | `cwt_morl` | `convnext2d_lite` | mean |
| raw + DWT | `dwt_swt_db4` | `resnet2d_small` | mean |
| raw + STFT | `stft_256` | `cnn2d` | gated attention |

Each candidate is evaluated with `embedding` and `both` features and with
`random_forest` and `extra_trees`, using the same weld-independent five-fold
protocol. The tree is trained only on out-of-sample inner-fold embeddings;
the outer fold remains untouched. The default plan is 16 runs per seed and
48 runs across seeds 7, 17, and 27:

```bash
RELEASE=/path/to/weld-independent-v0-2-1-r2
SEEDS="7 17 27" \
  PYTHON=.venv/bin/python CUDA_VISIBLE_DEVICES=0 \
  PIPELINE_BACKEND=cuda \
  bash scripts/run_tree_on_deep_v0_3_suite.sh \
  "$RELEASE" outputs/tree-on-deep-v0-3
```

The suite can resume into the same output directory. It writes
`run-metadata.txt`, `configuration-manifest.txt`, one validated output
directory per run, and a suite-level `status.txt`. `TREE_LEARNERS` and
`FEATURE_SOURCES` may be narrowed for a staged run; the fixed candidate list
itself is intentionally not configurable.

Set `PIPELINE_BACKEND=cpu` for a CPU-only check. On a GPU host,
`PREPROCESS_CACHE_DIRECTORY=/path/to/cache` enables the explicit preprocessing
cache path and `GPU_CACHE_GIB` controls the in-memory dataset cache budget.

If a GPU host raises `cuFFT error: CUFFT_INTERNAL_ERROR` while preparing an
STFT, keep model training on CUDA and disable only the GPU STFT preprocessing:

```bash
cd ~/PROJECT/MMDII/MMDII-Core
export RELEASE=/data/mmdii/releases/weld-independent-v0-2-1-r2
export PYTHON="$PWD/.venv/bin/python"
MMDII_GPU_STFT=0 CUDA_VISIBLE_DEVICES=0 \
  "$PYTHON" scripts/train_baseline.py \
  --config configs/time_frequency_v0_3_0_matrix.toml \
  --release-dir "$RELEASE" \
  --output-dir outputs/v0-3-smoke/seed-7-raw-plus-stft-resnet2d-small-mean-cpu-stft \
  --mode window_mil \
  --representation stft_256 \
  --encoder resnet2d_small \
  --fusion raw_plus_stft \
  --aggregator mean \
  --seed 7 \
  --fold 0 \
  --epochs 20 \
  --batch-size 4 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --optimizer adamw \
  --early-stopping-patience 0 \
  --device auto \
  --pipeline-backend cuda \
  --fold-scheme weld_independent
```

`MMDII_GPU_STFT=auto` is the default: it attempts GPU STFT and falls back to
CPU STFT on a cuFFT/CUDA runtime error, while the neural model remains on the
GPU. The selected value is recorded in suite `run-metadata.txt`. Set
`MMDII_GPU_STFT=1` only when diagnosing the GPU STFT path.
