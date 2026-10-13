#!/usr/bin/env bash
set -Eeuo pipefail

# v0.3.0 confirmation matrix:
#   3 dual branches x 3 encoders x 2 aggregators = 18 deep runs/seed
#   3 fold-local controls (statistical, Random Forest, Extra Trees) = 3 runs/seed
#
# Tree second-stage and offline probability fusion are separate experiments.
# Keeping them out of this matrix makes each output directory describe one
# training protocol and preserves the standard OOF schema.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SEEDS="${SEEDS:-7 17 27}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
DEVICE="${DEVICE:-auto}"
PIPELINE_BACKEND="${PIPELINE_BACKEND:-}"
PREPROCESS_CACHE_DIRECTORY="${PREPROCESS_CACHE_DIRECTORY:-}"
GPU_CACHE_GIB="${GPU_CACHE_GIB:-}"
MMDII_TRANSFORM_WORKERS="${MMDII_TRANSFORM_WORKERS:-2}"
BRANCHES=(raw_plus_stft raw_plus_cwt raw_plus_dwt)
ENCODERS=(cnn2d resnet2d_small convnext2d_lite)
AGGREGATORS=(mean gated_attention)

usage() {
  echo "Usage: bash scripts/run_time_frequency_v0_3_0_suite.sh RELEASE_DIR [OUTPUT_DIR] [CONFIG]"
}

check_dependencies() {
  if ! "$PYTHON" -c 'import scipy, pywt, sklearn, torch' >/dev/null 2>&1; then
    echo "Missing training dependencies in $PYTHON." >&2
    echo "Install them with: uv pip install --python $PYTHON -e \".[train]\"" >&2
    return 1
  fi
}

report_metrics() {
  local destination="$1"
  "$PYTHON" - "$destination" <<'PY' || true
import json
import statistics
import sys

run = sys.argv[1]
try:
    summary = json.load(open(run + "/training_summary.json", encoding="utf-8"))
    folds = json.load(open(run + "/fold_metrics.json", encoding="utf-8"))
except (OSError, ValueError):
    sys.exit(0)

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

common_args() {
  COMMON_ARGS=(
    --config "$BASE_CONFIG"
    --release-dir "$RELEASE_DIR"
    --output-dir "$DESTINATION"
    --seed "$SEED"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --learning-rate "$LEARNING_RATE"
    --weight-decay "$WEIGHT_DECAY"
    --optimizer adamw
    --early-stopping-patience 0
    --device "$DEVICE"
    --fold-scheme weld_independent
  )
  if [[ -n "$PIPELINE_BACKEND" ]]; then
    COMMON_ARGS+=(--pipeline-backend "$PIPELINE_BACKEND")
  fi
  if [[ -n "$PREPROCESS_CACHE_DIRECTORY" ]]; then
    COMMON_ARGS+=(--preprocess-cache-directory "$PREPROCESS_CACHE_DIRECTORY")
  fi
  if [[ -n "$GPU_CACHE_GIB" ]]; then
    COMMON_ARGS+=(--gpu-cache-gib "$GPU_CACHE_GIB")
  fi
}

validate_run() {
  local destination="$1"
  "$PYTHON" scripts/validate_tracked_outputs.py --root "$destination" \
    --expected-weld-count 101 --expected-fold-count 5 \
    --expected-fold-scheme weld_independent
}

run_one() {
  local name="$1" representation="$2" encoder="$3" fusion="$4" aggregator="$5" seed="$6"
  DESTINATION="$OUTPUT_ROOT/$name"
  printf '%s\n' "$name" >> "$CONFIG_MANIFEST"
  PLANNED=$((PLANNED + 1))
  [[ -f "$DESTINATION/.complete" ]] && return
  mkdir -p "$DESTINATION"
  echo "[$(date --iso-8601=seconds)] start experiment: $name"
  echo "  live: tail -f '$DESTINATION/train.log'"
  SEED="$seed"
  common_args
  "$PYTHON" scripts/train_baseline.py "${COMMON_ARGS[@]}" \
    --mode window_mil --aggregator "$aggregator" \
    --representation "$representation" --encoder "$encoder" --fusion "$fusion" \
    2>&1 | tee "$DESTINATION/train.log"
  touch "$DESTINATION/.complete"
  if ! validate_run "$DESTINATION"; then
    rm -f "$DESTINATION/.complete"
    return 1
  fi
  report_metrics "$DESTINATION"
  echo "[$(date --iso-8601=seconds)] complete experiment: $name"
}

run_control() {
  local name="$1" mode="$2" learner="$3" seed="$4"
  DESTINATION="$OUTPUT_ROOT/$name"
  printf '%s\n' "$name" >> "$CONFIG_MANIFEST"
  PLANNED=$((PLANNED + 1))
  [[ -f "$DESTINATION/.complete" ]] && return
  mkdir -p "$DESTINATION"
  echo "[$(date --iso-8601=seconds)] start experiment: $name"
  echo "  live: tail -f '$DESTINATION/train.log'"
  SEED="$seed"
  common_args
  CONTROL_ARGS=(--mode "$mode" --aggregator mean)
  if [[ -n "$learner" ]]; then
    CONTROL_ARGS+=(--tree-learner "$learner")
  fi
  "$PYTHON" scripts/train_baseline.py "${COMMON_ARGS[@]}" "${CONTROL_ARGS[@]}" \
    2>&1 | tee "$DESTINATION/train.log"
  touch "$DESTINATION/.complete"
  if ! validate_run "$DESTINATION"; then
    rm -f "$DESTINATION/.complete"
    return 1
  fi
  report_metrics "$DESTINATION"
  echo "[$(date --iso-8601=seconds)] complete experiment: $name"
}

worker() {
  cd "$ROOT"
  export MMDII_TRANSFORM_WORKERS
  check_dependencies
  mkdir -p "$OUTPUT_ROOT"
  CONFIG_MANIFEST="$OUTPUT_ROOT/configuration-manifest.txt"
  : > "$CONFIG_MANIFEST"
  PLANNED=0
  trap 'code=$?; printf "status=failed\nfinished_at=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$code" > "$OUTPUT_ROOT/status.txt"; exit "$code"' ERR

  cp "$ROOT/configs/time_frequency_v0_3_0_matrix.toml" "$OUTPUT_ROOT/matrix-config.toml"
  printf 'started_at=%s\ncommit=%s\nrelease=%s\nbase_config=%s\nseeds=%s\nepochs=%s\nbatch_size=%s\nlearning_rate=%s\nweight_decay=%s\noptimizer=adamw\nearly_stopping_patience=0\nfold_scheme=weld_independent\ndevice=%s\npipeline_backend=%s\npreprocess_cache_directory=%s\ngpu_cache_gib=%s\ntransform_workers=%s\n' \
    "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" "$RELEASE_DIR" "$BASE_CONFIG" \
    "$SEEDS" "$EPOCHS" "$BATCH_SIZE" "$LEARNING_RATE" "$WEIGHT_DECAY" "$DEVICE" \
    "$PIPELINE_BACKEND" "$PREPROCESS_CACHE_DIRECTORY" "$GPU_CACHE_GIB" "$MMDII_TRANSFORM_WORKERS" \
    > "$OUTPUT_ROOT/run-metadata.txt"

  for seed in $SEEDS; do
    run_control "seed-${seed}-b0-statistical" statistical "" "$seed"
    run_control "seed-${seed}-random-forest" random_forest random_forest "$seed"
    run_control "seed-${seed}-extra-trees" random_forest extra_trees "$seed"

    for branch in "${BRANCHES[@]}"; do
      if [[ "$branch" == "raw_plus_stft" ]]; then
        representation=stft_256
      elif [[ "$branch" == "raw_plus_cwt" ]]; then
        representation=cwt_morl
      else
        representation=dwt_swt_db4
      fi
      for encoder in "${ENCODERS[@]}"; do
        for aggregator in "${AGGREGATORS[@]}"; do
          run_one "seed-${seed}-${branch}-${encoder}-${aggregator}" \
            "$representation" "$encoder" "$branch" "$aggregator" "$seed"
        done
      done
    done
  done

  read -r -a seed_array <<< "$SEEDS"
  per_seed=$((3 + ${#BRANCHES[@]} * ${#ENCODERS[@]} * ${#AGGREGATORS[@]}))
  "$PYTHON" scripts/validate_tracked_outputs.py --root "$OUTPUT_ROOT" \
    --expected-run-count "$PLANNED" \
    --expected-configurations-per-seed "$per_seed" \
    --expected-weld-count 101 --expected-fold-count 5 \
    --expected-fold-scheme weld_independent --expected-seeds "${seed_array[@]}"
  trap - ERR
  printf 'status=complete\nfinished_at=%s\nrun_count=%s\n' \
    "$(date --iso-8601=seconds)" "$PLANNED" > "$OUTPUT_ROOT/status.txt"
  echo "planned per seed: $per_seed; total planned runs: $PLANNED"
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
OUTPUT_ROOT="${2:-$ROOT/outputs/time-frequency-v0-3-0-$(date +%Y%m%d-%H%M%S)}"
BASE_CONFIG="${3:-$ROOT/configs/time_frequency_v0_3_0_matrix.toml}"
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1
nohup bash "$ROOT/scripts/run_time_frequency_v0_3_0_suite.sh" --worker \
  "$RELEASE_DIR" "$OUTPUT_ROOT" "$BASE_CONFIG" \
  > "$OUTPUT_ROOT/suite.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$OUTPUT_ROOT/suite.pid"
echo "Started v0.3.0 time-frequency suite: $OUTPUT_ROOT"
