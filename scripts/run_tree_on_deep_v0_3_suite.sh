#!/usr/bin/env bash
# v0.3 second-stage tree suite: fit a tree on leak-safe deep embeddings.
#
# Each deep candidate is trained inside the outer fold.  The embedding arm
# uses only the harvested representation; the both arm concatenates the same
# representation with the 18 fold-local signal statistics.  Inner-fold
# cross-fitting is handled by train_baseline.py/second_stage.py.
#
# Usage:
#   bash scripts/run_tree_on_deep_v0_3_suite.sh RELEASE_DIR [OUTPUT_DIR] [CONFIG]
#
# Knobs (environment):
#   SEEDS             default "7 17 27"
#   TREE_LEARNERS     default "random_forest extra_trees"
#   FEATURE_SOURCES   default "embedding both"
#   INNER_FOLDS       default 4
#   ENCODER_EPOCHS    default 0 (0 means inherit EPOCHS)
#   EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, DEVICE, FOLDS
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SEEDS="${SEEDS:-7 17 27}"
TREE_LEARNERS="${TREE_LEARNERS:-random_forest extra_trees}"
FEATURE_SOURCES="${FEATURE_SOURCES:-embedding both}"
INNER_FOLDS="${INNER_FOLDS:-4}"
ENCODER_EPOCHS="${ENCODER_EPOCHS:-0}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
DEVICE="${DEVICE:-auto}"
PIPELINE_BACKEND="${PIPELINE_BACKEND:-cuda}"
PREPROCESS_CACHE_DIRECTORY="${PREPROCESS_CACHE_DIRECTORY:-}"
GPU_CACHE_GIB="${GPU_CACHE_GIB:-4.0}"
FOLD_SCHEME="${FOLD_SCHEME:-weld_independent}"
FOLDS="${FOLDS:-}"
MMDII_TRANSFORM_WORKERS="${MMDII_TRANSFORM_WORKERS:-2}"
MMDII_GPU_STFT="${MMDII_GPU_STFT:-auto}"

usage() {
  echo "Usage: bash scripts/run_tree_on_deep_v0_3_suite.sh RELEASE_DIR [OUTPUT_DIR] [CONFIG]"
}

check_dependencies() {
  if ! "$PYTHON" -c 'import scipy, sklearn, torch, pywt' >/dev/null 2>&1; then
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
    values = [
        row[key]
        for row in folds
        if isinstance(row, dict) and row.get(key) is not None
    ]
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
  local name="$1"
  local fusion="$2"
  local representation="$3"
  local encoder="$4"
  local aggregator="$5"
  local source="$6"
  local learner="$7"
  local seed="$8"
  local destination="$OUTPUT_ROOT/$name"

  printf '%s\n' "$name" >> "$CONFIG_MANIFEST"
  PLANNED=$((PLANNED + 1))
  [[ -f "$destination/.complete" ]] && return

  mkdir -p "$destination"
  echo "[$(date --iso-8601=seconds)] start experiment: $name"
  echo "  live: tail -f '$destination/train.log'"

  local args=(
    --config "$BASE_CONFIG"
    --release-dir "$RELEASE_DIR"
    --output-dir "$destination"
    --mode second_stage
    --second-stage-source "$source"
    --second-stage-mode window_mil
    --second-stage-inner-folds "$INNER_FOLDS"
    --second-stage-encoder-epochs "$ENCODER_EPOCHS"
    --aggregator "$aggregator"
    --representation "$representation"
    --encoder "$encoder"
    --fusion "$fusion"
    --tree-learner "$learner"
    --seed "$seed"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --learning-rate "$LEARNING_RATE"
    --weight-decay "$WEIGHT_DECAY"
    --optimizer adamw
    --early-stopping-patience 0
    --device "$DEVICE"
    --pipeline-backend "$PIPELINE_BACKEND"
    --gpu-cache-gib "$GPU_CACHE_GIB"
    --fold-scheme "$FOLD_SCHEME"
  )
  if [[ -n "$PREPROCESS_CACHE_DIRECTORY" ]]; then
    args+=(--preprocess-cache-directory "$PREPROCESS_CACHE_DIRECTORY")
  fi
  if [[ -n "$FOLDS" ]]; then
    read -r -a fold_array <<< "$FOLDS"
    for fold in "${fold_array[@]}"; do
      args+=(--fold "$fold")
    done
  fi

  "$PYTHON" scripts/train_baseline.py "${args[@]}" 2>&1 | tee "$destination/train.log"
  touch "$destination/.complete"
  if [[ -z "$FOLDS" ]]; then
    if ! "$PYTHON" scripts/validate_tracked_outputs.py \
      --root "$destination" \
      --expected-weld-count 101 \
      --expected-fold-count 5 \
      --expected-fold-scheme "$FOLD_SCHEME"; then
      rm -f "$destination/.complete"
      return 1
    fi
  else
    if ! "$PYTHON" scripts/validate_tracked_outputs.py \
      --root "$destination" \
      --allow-partial \
      --expected-fold-count "${#fold_array[@]}" \
      --expected-fold-scheme "$FOLD_SCHEME"; then
      rm -f "$destination/.complete"
      return 1
    fi
  fi
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
  local fold_count_for_validation=5
  local -a fold_array=()
  if [[ -n "$FOLDS" ]]; then
    read -r -a fold_array <<< "$FOLDS"
    fold_count_for_validation="${#fold_array[@]}"
  fi
  trap 'code=$?; printf "status=failed\nfinished_at=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$code" > "$OUTPUT_ROOT/status.txt"; exit "$code"' ERR

  cp "$BASE_CONFIG" "$OUTPUT_ROOT/base-config.toml"
  printf 'started_at=%s\ncommit=%s\nrelease=%s\nbase_config=%s\nseeds=%s\ntree_learners=%s\nfeature_sources=%s\ninner_folds=%s\nencoder_epochs=%s\nepochs=%s\nbatch_size=%s\nlearning_rate=%s\nweight_decay=%s\noptimizer=adamw\ndevice=%s\npipeline_backend=%s\ngpu_stft=%s\npreprocess_cache_directory=%s\ngpu_cache_gib=%s\nfold_scheme=%s\nfolds=%s\n' \
    "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" "$RELEASE_DIR" "$BASE_CONFIG" \
    "$SEEDS" "$TREE_LEARNERS" "$FEATURE_SOURCES" "$INNER_FOLDS" "$ENCODER_EPOCHS" \
    "$EPOCHS" "$BATCH_SIZE" "$LEARNING_RATE" "$WEIGHT_DECAY" "$DEVICE" "$PIPELINE_BACKEND" "$MMDII_GPU_STFT" \
    "$PREPROCESS_CACHE_DIRECTORY" "$GPU_CACHE_GIB" "$FOLD_SCHEME" "$FOLDS" \
    > "$OUTPUT_ROOT/run-metadata.txt"

  # The v0.3 candidate list is fixed so that tree comparisons do not silently
  # change when the broader v0.2.1 matrix is extended.
  local candidates=(
    "raw_plus_stft|stft_256|resnet2d_small|mean"
    "raw_plus_cwt|cwt_morl|convnext2d_lite|mean"
    "raw_plus_dwt|dwt_swt_db4|resnet2d_small|mean"
    "raw_plus_stft|stft_256|cnn2d|gated_attention"
  )
  printf 'candidate=%s\n' "${candidates[@]}" >> "$OUTPUT_ROOT/run-metadata.txt"

  read -r -a seed_array <<< "$SEEDS"
  read -r -a learner_array <<< "$TREE_LEARNERS"
  read -r -a source_array <<< "$FEATURE_SOURCES"
  local per_seed=$((${#candidates[@]} * ${#learner_array[@]} * ${#source_array[@]}))
  local seed learner source candidate fusion representation encoder aggregator name
  for seed in "${seed_array[@]}"; do
    for candidate in "${candidates[@]}"; do
      IFS='|' read -r fusion representation encoder aggregator <<< "$candidate"
      for source in "${source_array[@]}"; do
        for learner in "${learner_array[@]}"; do
          name="seed-${seed}-tree-${learner}-${source}-${fusion}-${encoder}-${aggregator}"
          run_one "$name" "$fusion" "$representation" "$encoder" "$aggregator" \
            "$source" "$learner" "$seed"
        done
      done
    done
  done

  if [[ -z "$FOLDS" ]]; then
    "$PYTHON" scripts/validate_tracked_outputs.py \
      --root "$OUTPUT_ROOT" \
      --expected-run-count "$PLANNED" \
      --expected-configurations-per-seed "$per_seed" \
      --expected-weld-count 101 \
      --expected-fold-count 5 \
      --expected-fold-scheme "$FOLD_SCHEME" \
      --expected-seeds "${seed_array[@]}"
  else
    "$PYTHON" scripts/validate_tracked_outputs.py \
      --root "$OUTPUT_ROOT" \
      --allow-partial \
      --expected-run-count "$PLANNED" \
      --expected-configurations-per-seed "$per_seed" \
      --expected-fold-count "$fold_count_for_validation" \
      --expected-fold-scheme "$FOLD_SCHEME" \
      --expected-seeds "${seed_array[@]}"
  fi
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
OUTPUT_ROOT="${2:-$ROOT/outputs/tree-on-deep-v0-3-$(date +%Y%m%d-%H%M%S)}"
BASE_CONFIG="${3:-$ROOT/configs/time_frequency_v0_3_0_matrix.toml}"
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1
nohup bash "$ROOT/scripts/run_tree_on_deep_v0_3_suite.sh" --worker \
  "$RELEASE_DIR" "$OUTPUT_ROOT" "$BASE_CONFIG" \
  > "$OUTPUT_ROOT/suite.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$OUTPUT_ROOT/suite.pid"
echo "Started tree-on-deep v0.3 suite: $OUTPUT_ROOT"
