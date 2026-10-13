#!/usr/bin/env bash
# Tree-fusion experiment suite: tabular tree controls plus the three
# second-stage feature arms.
#
# This script is deliberately separate from run_time_frequency_suite.sh. The
# canonical suite hard-codes 103 configurations and its archives are frozen;
# duplicating it here keeps those numbers intact. The planned run count is
# accumulated in PLANNED instead of being hard-coded.
#
# Offline probability fusion is intentionally NOT part of this suite: it is a
# zero-GPU step over archived OOF files, run separately with scripts/fuse_oof.py.
#
# Usage:
#   bash scripts/run_tree_fusion_suite.sh RELEASE_DIR [OUTPUT_DIR] [CONFIG]
#
# Knobs (environment):
#   SEEDS             default "7 17 27"
#   CONTROL_LEARNERS  default "extra_trees random_forest"
#   ARM_LEARNERS      default "extra_trees"
#   ARMS              default "stats embedding both"
#   INNER_FOLDS       default 4
#   ENCODER_EPOCHS    default 0 (0 means inherit EPOCHS)
#   EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, DEVICE, FOLDS
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SEEDS="${SEEDS:-7 17 27}"
CONTROL_LEARNERS="${CONTROL_LEARNERS:-extra_trees random_forest}"
ARM_LEARNERS="${ARM_LEARNERS:-extra_trees}"
ARMS="${ARMS:-stats embedding both}"
INNER_FOLDS="${INNER_FOLDS:-4}"
ENCODER_EPOCHS="${ENCODER_EPOCHS:-0}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
DEVICE="${DEVICE:-auto}"
FOLD_SCHEME="${FOLD_SCHEME:-weld_independent}"
FOLDS="${FOLDS:-}"
MMDII_TRANSFORM_WORKERS="${MMDII_TRANSFORM_WORKERS:-2}"

usage() { echo "Usage: bash scripts/run_tree_fusion_suite.sh RELEASE_DIR [OUTPUT_DIR] [CONFIG]"; }

check_dependencies() {
  if ! "$PYTHON" -c 'import scipy, sklearn' >/dev/null 2>&1; then
    echo "Missing training dependencies in $PYTHON." >&2
    return 1
  fi
  if ! "$PYTHON" -c 'import torch' >/dev/null 2>&1; then
    echo "The deep arms require torch; the stats-only arms do not." >&2
  fi
}

report_metrics() {
  local destination="$1"
  "$PYTHON" - "$destination" <<'PY' || true
import json, statistics, sys
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

run_one() {
  local name="$1"; shift
  local destination="$OUTPUT_ROOT/$name"
  printf '%s\n' "$name" >> "$CONFIG_MANIFEST"
  [[ -f "$destination/.complete" ]] && return
  mkdir -p "$destination"
  echo "[$(date --iso-8601=seconds)] start experiment: $name"
  echo "  live: tail -f '$destination/train.log'"
  local args=(--config "$BASE_CONFIG" --release-dir "$RELEASE_DIR" --output-dir "$destination"
    --seed "$SEED" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE"
    --learning-rate "$LEARNING_RATE" --weight-decay "$WEIGHT_DECAY"
    --optimizer adamw --early-stopping-patience 0 --device "$DEVICE"
    --fold-scheme "$FOLD_SCHEME")
  if [[ -n "$FOLDS" ]]; then
    read -r -a fold_array <<< "$FOLDS"
    for fold in "${fold_array[@]}"; do args+=(--fold "$fold"); done
  fi
  "$PYTHON" scripts/train_baseline.py "${args[@]}" "$@" 2>&1 | tee "$destination/train.log"
  touch "$destination/.complete"
  if [[ -z "$FOLDS" ]]; then
    if ! "$PYTHON" scripts/validate_tracked_outputs.py --root "$destination" \
      --expected-weld-count 101 --expected-fold-count 5 --expected-fold-scheme "$FOLD_SCHEME"; then
      rm -f "$destination/.complete"
      return 1
    fi
  else
    if ! "$PYTHON" scripts/validate_tracked_outputs.py --root "$destination" --allow-partial \
      --expected-fold-count "${#FOLDS}" --expected-fold-scheme "$FOLD_SCHEME"; then
      rm -f "$destination/.complete"
      return 1
    fi
  fi
  report_metrics "$destination"
  echo "[$(date --iso-8601=seconds)] complete experiment: $name"
}

worker() {
  cd "$ROOT"
  export MMDII_TRANSFORM_WORKERS
  check_dependencies
  mkdir -p "$OUTPUT_ROOT"
  CONFIG_MANIFEST="$OUTPUT_ROOT/configuration-manifest.txt"
  : > "$CONFIG_MANIFEST"
  trap 'code=$?; printf "status=failed\nfinished_at=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$code" > "$OUTPUT_ROOT/status.txt"; exit "$code"' ERR
  cp "$BASE_CONFIG" "$OUTPUT_ROOT/base-config.toml"
  printf 'started_at=%s\ncommit=%s\nrelease=%s\nbase_config=%s\nseeds=%s\ncontrol_learners=%s\narm_learners=%s\narms=%s\ninner_folds=%s\nencoder_epochs=%s\nepochs=%s\nbatch_size=%s\nlearning_rate=%s\nweight_decay=%s\noptimizer=adamw\nfold_scheme=%s\nfolds=%s\n' \
    "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" "$RELEASE_DIR" "$BASE_CONFIG" \
    "$SEEDS" "$CONTROL_LEARNERS" "$ARM_LEARNERS" "$ARMS" "$INNER_FOLDS" "$ENCODER_EPOCHS" \
    "$EPOCHS" "$BATCH_SIZE" "$LEARNING_RATE" "$WEIGHT_DECAY" "$FOLD_SCHEME" "$FOLDS" \
    > "$OUTPUT_ROOT/run-metadata.txt"

  read -r -a seed_array <<< "$SEEDS"
  read -r -a control_array <<< "$CONTROL_LEARNERS"
  read -r -a arm_learner_array <<< "$ARM_LEARNERS"
  read -r -a arm_array <<< "$ARMS"
  local per_seed=$(( ${#control_array[@]} + ${#arm_array[@]} * ${#arm_learner_array[@]} ))
  PLANNED=0

  local seed learner arm label
  for seed in "${seed_array[@]}"; do
    SEED="$seed"
    for learner in "${control_array[@]}"; do
      label="${learner//_/-}"
      run_one "seed-${seed}-${label}" --mode random_forest --tree-learner "$learner"
      PLANNED=$((PLANNED + 1))
    done
    for arm in "${arm_array[@]}"; do
      for learner in "${arm_learner_array[@]}"; do
        label="${learner//_/-}"
        if [[ "$arm" == "stats" ]]; then
          run_one "seed-${seed}-second-stage-stats-${label}" \
            --mode second_stage --second-stage-source stats --tree-learner "$learner"
        else
          run_one "seed-${seed}-second-stage-${arm}-${label}" \
            --mode second_stage --second-stage-source "$arm" \
            --second-stage-inner-folds "$INNER_FOLDS" \
            --second-stage-encoder-epochs "$ENCODER_EPOCHS" \
            --tree-learner "$learner"
        fi
        PLANNED=$((PLANNED + 1))
      done
    done
  done

  if [[ -z "$FOLDS" ]]; then
    "$PYTHON" scripts/validate_tracked_outputs.py --root "$OUTPUT_ROOT" \
      --expected-run-count "$PLANNED" \
      --expected-configurations-per-seed "$per_seed" \
      --expected-weld-count 101 \
      --expected-fold-count 5 --expected-fold-scheme "$FOLD_SCHEME" \
      --expected-seeds "${seed_array[@]}"
  else
    "$PYTHON" scripts/validate_tracked_outputs.py --root "$OUTPUT_ROOT" --allow-partial \
      --expected-run-count "$PLANNED" \
      --expected-configurations-per-seed "$per_seed" \
      --expected-fold-count "${#FOLDS}" --expected-fold-scheme "$FOLD_SCHEME" \
      --expected-seeds "${seed_array[@]}"
  fi
  trap - ERR
  printf 'status=complete\nfinished_at=%s\nrun_count=%s\n' "$(date --iso-8601=seconds)" "$PLANNED" \
    > "$OUTPUT_ROOT/status.txt"
  echo "planned per seed: $per_seed; total planned runs: $PLANNED"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift; [[ $# -eq 3 ]] || { usage >&2; exit 2; }
  RELEASE_DIR="$1"; OUTPUT_ROOT="$2"; BASE_CONFIG="$3"; worker; exit
fi

[[ $# -ge 1 && $# -le 3 ]] || { usage >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "Python executable not found: $PYTHON" >&2; exit 2; }
RELEASE_DIR="$(cd "$1" && pwd)"
OUTPUT_ROOT="${2:-$ROOT/outputs/tree-fusion-$(date +%Y%m%d-%H%M%S)}"
BASE_CONFIG="${3:-$ROOT/configs/moderntcn_mil_weld_independent_v0_2_1.toml}"
mkdir -p "$OUTPUT_ROOT"; OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1
nohup bash "$ROOT/scripts/run_tree_fusion_suite.sh" --worker "$RELEASE_DIR" "$OUTPUT_ROOT" "$BASE_CONFIG" \
  > "$OUTPUT_ROOT/suite.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$OUTPUT_ROOT/suite.pid"
echo "Started tree-fusion suite: $OUTPUT_ROOT"
