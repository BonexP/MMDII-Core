#!/usr/bin/env bash
set -Eeuo pipefail

# v0.3.1 pure-deep suite:
#   raw + DWT/SWT, mean aggregation, five fixed encoder configurations,
#   three seeds, and the weld-independent five-fold protocol.
#
# Usage:
#   bash scripts/run_time_frequency_v0_3_1_pure_deep_suite.sh RELEASE_DIR [OUTPUT_DIR] [CONFIG]

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SEEDS="${SEEDS:-7 17 27}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
DEVICE="${DEVICE:-auto}"
PIPELINE_BACKEND="${PIPELINE_BACKEND:-cuda}"
MMDII_TRANSFORM_WORKERS="${MMDII_TRANSFORM_WORKERS:-2}"
MMDII_GPU_STFT="${MMDII_GPU_STFT:-auto}"
PREPROCESS_CACHE_DIRECTORY="${PREPROCESS_CACHE_DIRECTORY:-}"
GPU_CACHE_GIB="${GPU_CACHE_GIB:-4.0}"
FOLDS="${FOLDS:-}"
ENCODERS=(resnet2d_small resnet2d_wide resnet2d_separable convnext2d_lite convnext2d_deep)

usage() {
  echo "Usage: bash scripts/run_time_frequency_v0_3_1_pure_deep_suite.sh RELEASE_DIR [OUTPUT_DIR] [CONFIG]"
}

check_dependencies() {
  "$PYTHON" -c 'import scipy, pywt, sklearn, torch' >/dev/null 2>&1 || {
    echo "Missing training dependencies in $PYTHON." >&2
    echo "Install them with: uv sync --extra train" >&2
    return 1
  }
}

validate_run() {
  local destination="$1"
  "$PYTHON" scripts/validate_tracked_outputs.py \
    --root "$destination" \
    --allow-unmarked \
    --expected-weld-count 101 \
    --expected-fold-count 5 \
    --expected-fold-scheme weld_independent
}

report_metrics() {
  local destination="$1"
  "$PYTHON" - "$destination" <<'PY' || true
import json
import statistics
import sys

run = sys.argv[1]
try:
    summary = json.loads(open(run + "/training_summary.json", encoding="utf-8").read())
    folds = json.loads(open(run + "/fold_metrics.json", encoding="utf-8").read())
except (OSError, ValueError):
    raise SystemExit(0)

def mean(key):
    values = [row[key] for row in folds if isinstance(row, dict) and row.get(key) is not None]
    return f"{statistics.mean(values):.4f}" if values else "n/a"

print(
    f"  metrics: samples={summary.get('sample_count', 'n/a')} "
    f"folds={summary.get('completed_fold_count', 'n/a')} "
    f"macro_f1={mean('macro_f1')} macro_recall={mean('macro_recall')} "
    f"macro_pr_auc={mean('macro_pr_auc')}"
)
PY
}

run_one() {
  local name="$1" encoder="$2" seed="$3"
  local destination="$OUTPUT_ROOT/$name"
  printf '%s\n' "$name" >> "$CONFIG_MANIFEST"
  PLANNED=$((PLANNED + 1))
  if [[ -f "$destination/.complete" ]]; then
    if validate_run "$destination" >/dev/null 2>&1; then
      echo "[$(date --iso-8601=seconds)] skip validated run: $name"
      return 0
    fi
    echo "[$(date --iso-8601=seconds)] remove stale completion marker: $name"
    rm -f "$destination/.complete"
  fi

  mkdir -p "$destination"
  echo "[$(date --iso-8601=seconds)] start experiment: $name"
  echo "  live: tail -f '$destination/train.log'"
  local args=(
    --config "$BASE_CONFIG"
    --release-dir "$RELEASE_DIR"
    --output-dir "$destination"
    --mode window_mil
    --representation dwt_swt_db4
    --encoder "$encoder"
    --fusion raw_plus_dwt
    --aggregator mean
    --seed "$seed"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --learning-rate "$LEARNING_RATE"
    --weight-decay "$WEIGHT_DECAY"
    --optimizer adamw
    --early-stopping-patience 0
    --device "$DEVICE"
    --pipeline-backend "$PIPELINE_BACKEND"
    --fold-scheme weld_independent
  )
  if [[ -n "$PREPROCESS_CACHE_DIRECTORY" ]]; then
    args+=(--preprocess-cache-directory "$PREPROCESS_CACHE_DIRECTORY")
  fi
  if [[ -n "$GPU_CACHE_GIB" ]]; then
    args+=(--gpu-cache-gib "$GPU_CACHE_GIB")
  fi
  if [[ -n "$FOLDS" ]]; then
    read -r -a fold_array <<< "$FOLDS"
    for fold in "${fold_array[@]}"; do args+=(--fold "$fold"); done
    echo "FOLDS is intended for smoke/debug runs; .complete is withheld for partial runs." >&2
  fi

  "$PYTHON" scripts/train_baseline.py "${args[@]}" 2>&1 | tee "$destination/train.log"
  if [[ -n "$FOLDS" ]]; then
    echo "[$(date --iso-8601=seconds)] partial run left without .complete: $name"
    return 0
  fi
  validate_run "$destination"
  touch "$destination/.complete"
  report_metrics "$destination"
  echo "[$(date --iso-8601=seconds)] complete experiment: $name"
}

worker() {
  cd "$ROOT"
  export MMDII_TRANSFORM_WORKERS MMDII_GPU_STFT
  check_dependencies
  mkdir -p "$OUTPUT_ROOT"
  CONFIG_MANIFEST="$OUTPUT_ROOT/configuration-manifest.txt"
  : > "$CONFIG_MANIFEST"
  PLANNED=0
  trap 'code=$?; printf "status=failed\nfinished_at=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$code" > "$OUTPUT_ROOT/status.txt"; exit "$code"' ERR

  cp "$BASE_CONFIG" "$OUTPUT_ROOT/base-config.toml"
  printf 'started_at=%s\ncommit=%s\nrelease=%s\nbase_config=%s\nseeds=%s\nencoders=%s\nepochs=%s\nbatch_size=%s\nlearning_rate=%s\nweight_decay=%s\noptimizer=adamw\nfold_scheme=weld_independent\nfolds=%s\ndevice=%s\npipeline_backend=%s\ngpu_stft=%s\ntransform_workers=%s\n' \
    "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" "$RELEASE_DIR" "$BASE_CONFIG" \
    "$SEEDS" "${ENCODERS[*]}" "$EPOCHS" "$BATCH_SIZE" "$LEARNING_RATE" "$WEIGHT_DECAY" \
    "$FOLDS" "$DEVICE" "$PIPELINE_BACKEND" "$MMDII_GPU_STFT" "$MMDII_TRANSFORM_WORKERS" \
    > "$OUTPUT_ROOT/run-metadata.txt"

  read -r -a seed_array <<< "$SEEDS"
  local seed encoder
  for seed in "${seed_array[@]}"; do
    for encoder in "${ENCODERS[@]}"; do
      run_one "seed-${seed}-raw-plus-dwt-${encoder}-mean" "$encoder" "$seed"
    done
  done

  if [[ -z "$FOLDS" ]]; then
    "$PYTHON" scripts/validate_tracked_outputs.py \
      --root "$OUTPUT_ROOT" \
      --expected-run-count "$PLANNED" \
      --expected-configurations-per-seed "${#ENCODERS[@]}" \
      --expected-weld-count 101 \
      --expected-fold-count 5 \
      --expected-fold-scheme weld_independent \
      --expected-seeds "${seed_array[@]}"
    printf 'status=complete\nfinished_at=%s\nrun_count=%s\n' \
      "$(date --iso-8601=seconds)" "$PLANNED" > "$OUTPUT_ROOT/status.txt"
  else
    printf 'status=partial\nfinished_at=%s\nrun_count=%s\nfolds=%s\n' \
      "$(date --iso-8601=seconds)" "$PLANNED" "$FOLDS" > "$OUTPUT_ROOT/status.txt"
  fi
  trap - ERR
  echo "planned per seed: ${#ENCODERS[@]}; total planned runs: $PLANNED"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  [[ $# -eq 3 ]] || { usage >&2; exit 2; }
  RELEASE_DIR="$1"
  OUTPUT_ROOT="$2"
  BASE_CONFIG="$3"
  worker
  exit
fi

[[ $# -ge 1 && $# -le 3 ]] || { usage >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "Python executable not found: $PYTHON" >&2; exit 2; }
RELEASE_DIR="$(cd "$1" && pwd)"
OUTPUT_ROOT="${2:-$ROOT/outputs/time-frequency-v0-3-1-pure-deep-$(date +%Y%m%d-%H%M%S)}"
BASE_CONFIG="${3:-$ROOT/configs/time_frequency_v0_3_1_pure_deep.toml}"
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1
nohup bash "$ROOT/scripts/run_time_frequency_v0_3_1_pure_deep_suite.sh" --worker \
  "$RELEASE_DIR" "$OUTPUT_ROOT" "$BASE_CONFIG" \
  > "$OUTPUT_ROOT/suite.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$OUTPUT_ROOT/suite.pid"
echo "Started v0.3.1 pure-deep suite: $OUTPUT_ROOT"
