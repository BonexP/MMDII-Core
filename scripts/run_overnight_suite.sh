#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/scripts/run_overnight_suite.sh"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

usage() {
    cat <<'EOF'
Usage: bash scripts/run_overnight_suite.sh RELEASE_DIR [OUTPUT_DIR] [CONFIG]

Starts the complete baseline suite with nohup and returns immediately. Reuse
the same OUTPUT_DIR to skip experiments that already finished successfully.

Environment variables:
  CUDA_VISIBLE_DEVICES  GPU selection passed to training (default: 0)
  PYTHON                Python executable (default: .venv/bin/python)
EOF
}

status() {
    printf 'status=%s\nexperiment=%s\nupdated_at=%s\n' \
        "$1" "$2" "$(date --iso-8601=seconds)" > "$OUTPUT_ROOT/status.txt"
}

run_experiment() {
    local name="$1"
    shift
    local destination="$OUTPUT_ROOT/$name"

    if [[ -f "$destination/.complete" ]]; then
        echo "[$(date --iso-8601=seconds)] skip completed experiment: $name"
        return
    fi

    CURRENT_EXPERIMENT="$name"
    status running "$name"
    mkdir -p "$destination"
    echo "[$(date --iso-8601=seconds)] start experiment: $name"
    "$PYTHON" scripts/train_baseline.py \
        --config "$CONFIG" \
        --release-dir "$RELEASE_DIR" \
        --output-dir "$destination" \
        "$@" 2>&1 | tee "$destination/train.log"

    [[ -s "$destination/training_summary.json" ]]
    [[ -s "$destination/fold_metrics.json" ]]
    [[ -s "$destination/oof_predictions.csv" ]]
    [[ "$(wc -l < "$destination/oof_predictions.csv")" -eq "$EXPECTED_CSV_ROWS" ]]
    touch "$destination/.complete"
    echo "[$(date --iso-8601=seconds)] complete experiment: $name"
}

worker() {
    RELEASE_DIR="$1"
    OUTPUT_ROOT="$2"
    CONFIG="$3"
    CURRENT_EXPERIMENT="startup"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    export PYTHONUNBUFFERED=1

    cd "$ROOT"
    mkdir -p "$OUTPUT_ROOT"
    trap 'code=$?; if [[ $code -eq 0 ]]; then status complete all; else status failed "$CURRENT_EXPERIMENT"; fi' EXIT
    trap 'exit 130' INT TERM

    {
        echo "started_at=$(date --iso-8601=seconds)"
        echo "repository=$ROOT"
        echo "git_commit=$(git rev-parse HEAD)"
        echo "python=$($PYTHON --version 2>&1)"
        echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
    } > "$OUTPUT_ROOT/run-metadata.txt"

    status checking_environment environment
    "$PYTHON" scripts/check_training_environment.py \
        --config "$CONFIG" \
        --release-dir "$RELEASE_DIR" \
        --output-dir "$OUTPUT_ROOT" \
        --json > "$OUTPUT_ROOT/environment-check.json"
    EXPECTED_CSV_ROWS="$($PYTHON -c 'import json, sys; print(json.load(open(sys.argv[1]))["dataset"]["sample_count"] + 1)' "$OUTPUT_ROOT/environment-check.json")"

    run_experiment b0-statistical --mode statistical --aggregator mean
    run_experiment e0-full-signal --mode full_signal --aggregator mean
    run_experiment e1a-mean --mode window_mil --aggregator mean
    run_experiment e1b-max --mode window_mil --aggregator max
    run_experiment e1b-topk-mean --mode window_mil --aggregator topk_mean
    run_experiment e1c-gated-attention --mode window_mil --aggregator gated_attention
}

if [[ "${1:-}" == "--worker" ]]; then
    shift
    [[ $# -eq 3 ]] || { usage >&2; exit 2; }
    worker "$@"
    exit
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit
fi

[[ $# -ge 1 && $# -le 3 ]] || { usage >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "Python executable not found: $PYTHON" >&2; exit 2; }
[[ -d "$1" ]] || { echo "Release directory not found: $1" >&2; exit 2; }

RELEASE_DIR="$(cd "$1" && pwd)"
OUTPUT_ROOT="${2:-$ROOT/outputs/overnight-$(date +%Y%m%d-%H%M%S)}"
CONFIG="${3:-$ROOT/configs/moderntcn_mil_v0_1.toml}"
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }

if [[ -f "$OUTPUT_ROOT/suite.pid" ]] && kill -0 "$(cat "$OUTPUT_ROOT/suite.pid")" 2>/dev/null; then
    echo "Suite is already running with PID $(cat "$OUTPUT_ROOT/suite.pid")" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
nohup bash "$SCRIPT" --worker "$RELEASE_DIR" "$OUTPUT_ROOT" "$CONFIG" \
    > "$OUTPUT_ROOT/suite.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUTPUT_ROOT/suite.pid"

echo "Started overnight suite."
echo "PID: $PID"
echo "Output: $OUTPUT_ROOT"
echo "Follow: tail -f '$OUTPUT_ROOT/suite.log'"
echo "Status: cat '$OUTPUT_ROOT/status.txt'"
