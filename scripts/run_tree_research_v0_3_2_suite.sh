#!/usr/bin/env bash
set -Eeuo pipefail

# v0.3.2: 12 bootstrap comparisons, 6 model-family comparisons, and 6
# ExtraTrees feature ablations. Runs are serial so the host can resume safely.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SEEDS="${SEEDS:-7 17 27}"
FOLD_SCHEME="weld_independent"

LEARNERS=(random_forest extra_trees)
BOOTSTRAP_VALUES=(true false)
FEATURE_SETS=(stats15_no_ptp stats12_no_ptp_rms)

usage() {
  echo "Usage: bash scripts/run_tree_research_v0_3_2_suite.sh RELEASE_DIR [OUTPUT_DIR] [CONFIG]"
}

check_dependencies() {
  "$PYTHON" -c 'import scipy, sklearn' >/dev/null 2>&1 || {
    echo "Missing statistical dependencies in $PYTHON." >&2
    echo "Install them with: uv sync --extra train" >&2
    return 1
  }
}

validate_run() {
  local destination="$1" seed="$2" mode="$3" learner="$4"
  local feature_set="$5" bootstrap="$6" allow_unmarked="${7:-false}"
  local validator_args=(
    --root "$destination" --expected-weld-count 101
    --expected-fold-count 5 --expected-fold-scheme "$FOLD_SCHEME"
  )
  if [[ "$allow_unmarked" == "true" ]]; then
    validator_args+=(--allow-unmarked)
  fi
  "$PYTHON" scripts/validate_tracked_outputs.py "${validator_args[@]}"
  "$PYTHON" - "$destination/run_config.json" "$RELEASE_DIR" "$seed" \
    "$mode" "$learner" "$feature_set" "$bootstrap" <<'PY'
import json
import sys
from pathlib import Path

config_path, release, seed, mode, learner, feature_set, bootstrap = sys.argv[1:]
config = json.loads(Path(config_path).read_text(encoding="utf-8"))
expected = {
    "seed": int(seed),
    "mode": mode,
    "fold_scheme": "weld_independent",
    "release_directory": str(Path(release).resolve()),
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(f"run config mismatch for {key}: expected {value!r}, found {config.get(key)!r}")
if learner:
    tree = config.get("tree", {})
    for key, value in (
        ("learner", learner),
        ("feature_set", feature_set),
        ("bootstrap", bootstrap == "true"),
    ):
        if tree.get(key) != value:
            raise SystemExit(f"run config mismatch for tree.{key}: expected {value!r}, found {tree.get(key)!r}")
PY
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
    f"macro_pr_auc={mean('macro_pr_auc')} macro_f1={mean('macro_f1')} "
    f"macro_recall={mean('macro_recall')} macro_brier={mean('macro_brier')}"
)
PY
}

run_one() {
  local name="$1" seed="$2" mode="$3" learner="$4"
  local feature_set="$5" bootstrap="$6" tune="$7"
  local destination="$OUTPUT_ROOT/$name"
  printf '%s\n' "$name" >> "$CONFIG_MANIFEST"
  PLANNED=$((PLANNED + 1))

  if [[ -f "$destination/.complete" ]]; then
    if validate_run "$destination" "$seed" "$mode" "$learner" "$feature_set" "$bootstrap" \
      >/dev/null 2>&1; then
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
    --seed "$seed" --fold-scheme "$FOLD_SCHEME" --mode "$mode" --aggregator mean
  )
  if [[ -n "$learner" ]]; then
    args+=(--tree-learner "$learner" --tree-bootstrap "$bootstrap" --tree-feature-set "$feature_set")
  fi
  if [[ "$tune" == "true" ]]; then
    args+=(--tree-tune)
  fi
  "$PYTHON" scripts/train_baseline.py "${args[@]}" 2>&1 | tee "$destination/train.log"
  validate_run "$destination" "$seed" "$mode" "$learner" "$feature_set" "$bootstrap" true
  touch "$destination/.complete"
  report_metrics "$destination"
  echo "[$(date --iso-8601=seconds)] complete experiment: $name"
}

worker() {
  cd "$ROOT"
  check_dependencies
  mkdir -p "$OUTPUT_ROOT"
  CONFIG_MANIFEST="$OUTPUT_ROOT/configuration-manifest.txt"
  : > "$CONFIG_MANIFEST"
  PLANNED=0
  trap 'code=$?; printf "status=failed\nfinished_at=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$code" > "$OUTPUT_ROOT/status.txt"; exit "$code"' ERR
  cp "$BASE_CONFIG" "$OUTPUT_ROOT/base-config.toml"
  printf 'started_at=%s\ncommit=%s\nrelease=%s\nbase_config=%s\nseeds=%s\nfold_scheme=%s\nruns_per_seed=8\n' \
    "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" "$RELEASE_DIR" "$BASE_CONFIG" \
    "$SEEDS" "$FOLD_SCHEME" > "$OUTPUT_ROOT/run-metadata.txt"

  for seed in $SEEDS; do
    for learner in "${LEARNERS[@]}"; do
      for bootstrap in "${BOOTSTRAP_VALUES[@]}"; do
        run_one "seed-${seed}-${learner}-bootstrap-${bootstrap}-stats18" \
          "$seed" random_forest "$learner" stats18 "$bootstrap" false
      done
    done

    run_one "seed-${seed}-gradient-boosting-stats18" \
      "$seed" gradient_boosting "" stats18 "" true
    run_one "seed-${seed}-rbf-svm-stats18" \
      "$seed" rbf_svm "" stats18 "" true

    for feature_set in "${FEATURE_SETS[@]}"; do
      run_one "seed-${seed}-extra-trees-bootstrap-false-${feature_set}" \
        "$seed" random_forest extra_trees "$feature_set" false false
    done
  done

  read -r -a seed_array <<< "$SEEDS"
  expected_runs=$((8 * ${#seed_array[@]}))
  "$PYTHON" scripts/validate_tracked_outputs.py \
    --root "$OUTPUT_ROOT" --expected-run-count "$PLANNED" \
    --expected-configurations-per-seed 8 --expected-weld-count 101 \
    --expected-fold-count 5 --expected-fold-scheme "$FOLD_SCHEME" \
    --expected-seeds "${seed_array[@]}"
  trap - ERR
  printf 'status=complete\nfinished_at=%s\nrun_count=%s\nexpected_runs=%s\n' \
    "$(date --iso-8601=seconds)" "$PLANNED" "$expected_runs" > "$OUTPUT_ROOT/status.txt"
  echo "planned per seed: 8; total planned runs: $PLANNED"
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
[[ -d "$1" ]] || { echo "Release directory not found: $1" >&2; exit 2; }
RELEASE_DIR="$(cd "$1" && pwd)"
OUTPUT_ROOT="${2:-$ROOT/outputs/tree-research-v0-3-2-$(date +%Y%m%d-%H%M%S)}"
BASE_CONFIG="${3:-$ROOT/configs/tree_research_v0_3_2.toml}"
[[ -f "$BASE_CONFIG" ]] || { echo "Config file not found: $BASE_CONFIG" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
export PYTHONUNBUFFERED=1
nohup bash "$ROOT/scripts/run_tree_research_v0_3_2_suite.sh" --worker \
  "$RELEASE_DIR" "$OUTPUT_ROOT" "$BASE_CONFIG" \
  > "$OUTPUT_ROOT/suite.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$OUTPUT_ROOT/suite.pid"
echo "Started v0.3.2 tree research suite: $OUTPUT_ROOT"
