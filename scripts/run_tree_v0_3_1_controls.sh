#!/usr/bin/env bash
set -Eeuo pipefail

# v0.3.1 tree controls:
#   statistical logistic, legacy/tuned Random Forest, and legacy/tuned ExtraTrees.
#   Tuning is delegated to the strict inner-fold implementation behind
#   --tree-tune; this script never labels a fixed-parameter run as tuned.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SEEDS="${SEEDS:-7 17 27}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
DEVICE="${DEVICE:-auto}"
PIPELINE_BACKEND="${PIPELINE_BACKEND:-cuda}"
FOLD_SCHEME="${FOLD_SCHEME:-weld_independent}"
FOLDS="${FOLDS:-}"
MMDII_TRANSFORM_WORKERS="${MMDII_TRANSFORM_WORKERS:-2}"

usage() {
  echo "Usage: bash scripts/run_tree_v0_3_1_controls.sh RELEASE_DIR [OUTPUT_DIR] [CONFIG]"
}

check_dependencies() {
  "$PYTHON" -c 'import scipy, sklearn' >/dev/null 2>&1 || {
    echo "Missing statistical dependencies in $PYTHON." >&2
    echo "Install them with: uv sync --extra train" >&2
    return 1
  }
}

validate_run() {
  local destination="$1"
  "$PYTHON" scripts/validate_tracked_outputs.py \
    --root "$destination" --allow-unmarked --expected-weld-count 101 \
    --expected-fold-count 5 --expected-fold-scheme "$FOLD_SCHEME"
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
  local name="$1"; shift
  local destination="$OUTPUT_ROOT/$name"
  printf '%s\n' "$name" >> "$CONFIG_MANIFEST"
  PLANNED=$((PLANNED + 1))
  if [[ -f "$destination/.complete" ]]; then
    if validate_run "$destination" >/dev/null 2>&1; then
      echo "[$(date --iso-8601=seconds)] skip validated run: $name"
      return 0
    fi
    rm -f "$destination/.complete"
  fi

  mkdir -p "$destination"
  echo "[$(date --iso-8601=seconds)] start experiment: $name"
  echo "  live: tail -f '$destination/train.log'"
  local args=(
    --config "$BASE_CONFIG" --release-dir "$RELEASE_DIR" --output-dir "$destination"
    --seed "$SEED" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE"
    --learning-rate "$LEARNING_RATE" --weight-decay "$WEIGHT_DECAY"
    --optimizer adamw --early-stopping-patience 0 --device "$DEVICE"
    --pipeline-backend "$PIPELINE_BACKEND" --fold-scheme "$FOLD_SCHEME"
  )
  if [[ -n "$FOLDS" ]]; then
    read -r -a fold_array <<< "$FOLDS"
    for fold in "${fold_array[@]}"; do args+=(--fold "$fold"); done
  fi
  "$PYTHON" scripts/train_baseline.py "${args[@]}" "$@" 2>&1 | tee "$destination/train.log"
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
  export MMDII_TRANSFORM_WORKERS
  check_dependencies
  mkdir -p "$OUTPUT_ROOT"
  CONFIG_MANIFEST="$OUTPUT_ROOT/configuration-manifest.txt"
  : > "$CONFIG_MANIFEST"
  PLANNED=0
  trap 'code=$?; printf "status=failed\nfinished_at=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$code" > "$OUTPUT_ROOT/status.txt"; exit "$code"' ERR
  cp "$BASE_CONFIG" "$OUTPUT_ROOT/base-config.toml"
  printf 'started_at=%s\ncommit=%s\nrelease=%s\nbase_config=%s\nseeds=%s\nfold_scheme=%s\nfolds=%s\n' \
    "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" "$RELEASE_DIR" "$BASE_CONFIG" \
    "$SEEDS" "$FOLD_SCHEME" "$FOLDS" > "$OUTPUT_ROOT/run-metadata.txt"

  read -r -a seed_array <<< "$SEEDS"
  local seed
  for seed in "${seed_array[@]}"; do
    SEED="$seed"
    run_one "seed-${seed}-statistical-logistic" \
      --mode statistical --aggregator mean
    run_one "seed-${seed}-random-forest-legacy" \
      --mode random_forest --aggregator mean --tree-learner random_forest \
      --tree-n-estimators 300 --tree-min-samples-leaf 1
    run_one "seed-${seed}-random-forest-tuned" \
      --mode random_forest --aggregator mean --tree-learner random_forest \
      --tree-tune
    run_one "seed-${seed}-extra-trees-legacy" \
      --mode random_forest --aggregator mean --tree-learner extra_trees \
      --tree-n-estimators 300 --tree-min-samples-leaf 1
    run_one "seed-${seed}-extra-trees-tuned" \
      --mode random_forest --aggregator mean --tree-learner extra_trees \
      --tree-tune
  done

  if [[ -z "$FOLDS" ]]; then
    "$PYTHON" scripts/validate_tracked_outputs.py \
      --root "$OUTPUT_ROOT" --expected-run-count "$PLANNED" \
      --expected-configurations-per-seed 5 --expected-weld-count 101 \
      --expected-fold-count 5 --expected-fold-scheme "$FOLD_SCHEME" \
      --expected-seeds "${seed_array[@]}"
    printf 'status=complete\nfinished_at=%s\nrun_count=%s\n' \
      "$(date --iso-8601=seconds)" "$PLANNED" > "$OUTPUT_ROOT/status.txt"
  else
    printf 'status=partial\nfinished_at=%s\nrun_count=%s\nfolds=%s\n' \
      "$(date --iso-8601=seconds)" "$PLANNED" "$FOLDS" > "$OUTPUT_ROOT/status.txt"
  fi
  trap - ERR
  echo "planned per seed: 5; total planned runs: $PLANNED"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  [[ $# -eq 3 ]] || { usage >&2; exit 2; }
  RELEASE_DIR="$1"; OUTPUT_ROOT="$2"; BASE_CONFIG="$3"; worker; exit
fi

[[ $# -ge 1 && $# -le 3 ]] || { usage >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "Python executable not found: $PYTHON" >&2; exit 2; }
RELEASE_DIR="$(cd "$1" && pwd)"
OUTPUT_ROOT="${2:-$ROOT/outputs/tree-v0-3-1-controls-$(date +%Y%m%d-%H%M%S)}"
BASE_CONFIG="${3:-$ROOT/configs/time_frequency_v0_3_1_tree.toml}"
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1
nohup bash "$ROOT/scripts/run_tree_v0_3_1_controls.sh" --worker \
  "$RELEASE_DIR" "$OUTPUT_ROOT" "$BASE_CONFIG" \
  > "$OUTPUT_ROOT/suite.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$OUTPUT_ROOT/suite.pid"
echo "Started v0.3.1 tree controls suite: $OUTPUT_ROOT"
