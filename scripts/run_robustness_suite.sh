#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SEEDS="${SEEDS:-7 17 27}"
EPOCHS="${EPOCHS:-20}"
OPTIMIZER="${OPTIMIZER:-adamw}"

usage() {
    cat <<'EOF'
Usage: bash scripts/run_robustness_suite.sh RELEASE_DIR [OUTPUT_DIR] [CONFIG]

Runs the predeclared B0 and E1c candidate baselines for multiple seeds. The
process is detached with nohup; override SEEDS, EPOCHS, OPTIMIZER, or
CUDA_VISIBLE_DEVICES before invoking it when needed.
EOF
}

run_one() {
    local name="$1" mode="$2" aggregator="$3" seed="$4"
    local destination="$OUTPUT_ROOT/$name"
    [[ -f "$destination/.complete" ]] && return
    mkdir -p "$destination"
    "$PYTHON" scripts/train_baseline.py \
        --config "$CONFIG" --release-dir "$RELEASE_DIR" --output-dir "$destination" \
        --mode "$mode" --aggregator "$aggregator" --seed "$seed" \
        --epochs "$EPOCHS" --optimizer "$OPTIMIZER" 2>&1 | tee "$destination/train.log"
    "$PYTHON" scripts/validate_tracked_outputs.py --root "$destination"
    touch "$destination/.complete"
}

worker() {
    cd "$ROOT"
    mkdir -p "$OUTPUT_ROOT"
    printf 'started_at=%s\ncommit=%s\nseeds=%s\nepochs=%s\noptimizer=%s\n' \
        "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" "$SEEDS" "$EPOCHS" "$OPTIMIZER" \
        > "$OUTPUT_ROOT/run-metadata.txt"
    for seed in $SEEDS; do
        run_one "seed-${seed}-b0-statistical" statistical mean "$seed"
        run_one "seed-${seed}-e1c-gated-attention" window_mil gated_attention "$seed"
    done
    printf 'status=complete\nfinished_at=%s\n' "$(date --iso-8601=seconds)" > "$OUTPUT_ROOT/status.txt"
}

if [[ "${1:-}" == "--worker" ]]; then
    shift
    [[ $# -eq 3 ]] || { usage >&2; exit 2; }
    RELEASE_DIR="$1"
    OUTPUT_ROOT="$2"
    CONFIG="$3"
    worker
    exit
fi

[[ "${1:-}" == "--help" || "${1:-}" == "-h" ]] && { usage; exit 0; }
[[ $# -ge 1 && $# -le 3 ]] || { usage >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "Python executable not found: $PYTHON" >&2; exit 2; }
[[ -d "$1" ]] || { echo "Release directory not found: $1" >&2; exit 2; }
RELEASE_DIR="$(cd "$1" && pwd)"
OUTPUT_ROOT="${2:-$ROOT/outputs/robustness-$(date +%Y%m%d-%H%M%S)}"
CONFIG="${3:-$ROOT/configs/moderntcn_mil_v0_1.toml}"
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1
nohup bash "$ROOT/scripts/run_robustness_suite.sh" --worker "$RELEASE_DIR" "$OUTPUT_ROOT" "$CONFIG" \
    > "$OUTPUT_ROOT/suite.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$OUTPUT_ROOT/suite.pid"
echo "Started robustness suite: $OUTPUT_ROOT"
