#!/usr/bin/env bash
set -Eeuo pipefail

# Full 2x2x2 feature-block design; runs serially and resumes validated arms.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SEEDS="${SEEDS:-7 17 27}"
FOLD_SCHEME="weld_independent"

usage() {
  echo "Usage: bash scripts/run_tree_interval_v0_3_3_suite.sh RELEASE_DIR [OUTPUT_DIR] [CONFIG]"
}

validate_run() {
  local destination="$1" seed="$2" interval="$3" process="$4" duration="$5"
  local allow_unmarked="${6:-false}"
  local validator_args=(
    --root "$destination" --expected-weld-count 101
    --expected-fold-count 5 --expected-fold-scheme "$FOLD_SCHEME"
  )
  [[ "$allow_unmarked" == "true" ]] && validator_args+=(--allow-unmarked)
  "$PYTHON" scripts/validate_tracked_outputs.py "${validator_args[@]}"
  "$PYTHON" - "$destination/run_config.json" "$RELEASE_DIR" "$seed" \
    "$interval" "$process" "$duration" <<'PY'
import json
import sys
from pathlib import Path

path, release, seed, interval, process, duration = sys.argv[1:]
config = json.loads(Path(path).read_text(encoding="utf-8"))
for key, value in {
    "seed": int(seed),
    "mode": "random_forest",
    "fold_scheme": "weld_independent",
    "release_directory": str(Path(release).resolve()),
}.items():
    if config.get(key) != value:
        raise SystemExit(f"run config mismatch for {key}: expected {value!r}, found {config.get(key)!r}")
tree = config.get("tree", {})
expected = {
    "learner": "extra_trees",
    "n_estimators": 300,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "max_depth": None,
    "criterion": "gini",
    "class_weight": "balanced",
    "bootstrap": False,
    "feature_set": "stats18",
    "interval_features": interval == "true",
    "process_features": process == "true",
    "duration_feature": duration == "true",
}
for key, value in expected.items():
    if tree.get(key) != value:
        raise SystemExit(f"run config mismatch for tree.{key}: expected {value!r}, found {tree.get(key)!r}")
statistical = tree.get("statistical_features", {})
expected_count = 18 + 18 * (interval == "true") + 4 * (process == "true") + (duration == "true")
if statistical.get("feature_count") != expected_count:
    raise SystemExit(f"expected {expected_count} statistical features, found {statistical.get('feature_count')!r}")
PY
}

run_one() {
  local seed="$1" interval="$2" process="$3" duration="$4"
  local suffix="stats18"
  [[ "$interval" == "true" ]] && suffix+="+interval3"
  [[ "$process" == "true" ]] && suffix+="+process"
  [[ "$duration" == "true" ]] && suffix+="+duration"
  local name="seed-${seed}-${suffix}"
  local destination="$OUTPUT_ROOT/$name"
  printf '%s\n' "$name" >> "$CONFIG_MANIFEST"
  PLANNED=$((PLANNED + 1))

  if [[ -f "$destination/.complete" ]]; then
    if validate_run "$destination" "$seed" "$interval" "$process" "$duration" \
      >/dev/null 2>&1; then
      echo "[$(date --iso-8601=seconds)] skip validated run: $name"
      return
    fi
    rm -f "$destination/.complete"
  fi
  mkdir -p "$destination"
  echo "[$(date --iso-8601=seconds)] start experiment: $name"
  local args=(
    --config "$BASE_CONFIG" --release-dir "$RELEASE_DIR" --output-dir "$destination"
    --seed "$seed" --fold-scheme "$FOLD_SCHEME" --mode random_forest
    --aggregator mean --tree-learner extra_trees --tree-bootstrap false
    --tree-feature-set stats18
  )
  [[ "$interval" == "true" ]] && args+=(--tree-interval-features)
  [[ "$process" == "true" ]] && args+=(--tree-process-features)
  [[ "$duration" == "true" ]] && args+=(--tree-duration-feature)
  "$PYTHON" scripts/train_baseline.py "${args[@]}" 2>&1 | tee "$destination/train.log"
  validate_run "$destination" "$seed" "$interval" "$process" "$duration" true
  touch "$destination/.complete"
  echo "[$(date --iso-8601=seconds)] complete experiment: $name"
}

worker() {
  cd "$ROOT"
  "$PYTHON" -c 'import scipy, sklearn' >/dev/null
  mkdir -p "$OUTPUT_ROOT"
  CONFIG_MANIFEST="$OUTPUT_ROOT/configuration-manifest.txt"
  : > "$CONFIG_MANIFEST"
  PLANNED=0
  trap 'code=$?; printf "status=failed\nexit_code=%s\n" "$code" > "$OUTPUT_ROOT/status.txt"; exit "$code"' ERR
  cp "$BASE_CONFIG" "$OUTPUT_ROOT/base-config.toml"
  printf 'started_at=%s\ncommit=%s\nrelease=%s\nbase_config=%s\nseeds=%s\nruns_per_seed=8\nplanned_runs=24\n' \
    "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" "$RELEASE_DIR" "$BASE_CONFIG" "$SEEDS" \
    > "$OUTPUT_ROOT/run-metadata.txt"
  for seed in $SEEDS; do
    for interval in false true; do
      for process in false true; do
        for duration in false true; do
          run_one "$seed" "$interval" "$process" "$duration"
        done
      done
    done
  done
  read -r -a seed_array <<< "$SEEDS"
  "$PYTHON" scripts/validate_tracked_outputs.py \
    --root "$OUTPUT_ROOT" --expected-run-count "$PLANNED" \
    --expected-configurations-per-seed 8 --expected-weld-count 101 \
    --expected-fold-count 5 --expected-fold-scheme "$FOLD_SCHEME" \
    --expected-seeds "${seed_array[@]}"
  trap - ERR
  printf 'status=complete\nfinished_at=%s\nrun_count=%s\n' \
    "$(date --iso-8601=seconds)" "$PLANNED" > "$OUTPUT_ROOT/status.txt"
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
OUTPUT_ROOT="${2:-$ROOT/outputs/tree-interval-v0-3-3-$(date +%Y%m%d-%H%M%S)}"
BASE_CONFIG="${3:-$ROOT/configs/tree_interval_v0_3_3.toml}"
[[ -f "$BASE_CONFIG" ]] || { echo "Config file not found: $BASE_CONFIG" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
export PYTHONUNBUFFERED=1
nohup bash "$ROOT/scripts/run_tree_interval_v0_3_3_suite.sh" --worker \
  "$RELEASE_DIR" "$OUTPUT_ROOT" "$BASE_CONFIG" \
  > "$OUTPUT_ROOT/suite.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$OUTPUT_ROOT/suite.pid"
echo "Started v0.3.3 tree interval suite: $OUTPUT_ROOT"
