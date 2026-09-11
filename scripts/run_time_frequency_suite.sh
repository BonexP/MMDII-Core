#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SEEDS="${SEEDS:-7 17 27}"
EPOCHS="${EPOCHS:-20}"
DEVICE="${DEVICE:-auto}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"

usage() { echo "Usage: bash scripts/run_time_frequency_suite.sh RELEASE_DIR [OUTPUT_DIR] [CONFIG]"; }
run_one() {
  local name="$1" representation="$2" encoder="$3" fusion="$4" aggregator="$5" seed="$6"
  local destination="$OUTPUT_ROOT/$name"
  [[ -f "$destination/.complete" ]] && return
  mkdir -p "$destination"
  local args=(--config "$BASE_CONFIG" --release-dir "$RELEASE_DIR" --output-dir "$destination"
    --mode window_mil --aggregator "$aggregator" --seed "$seed" --epochs "$EPOCHS"
    --device "$DEVICE" --representation "$representation" --encoder "$encoder" --fusion "$fusion")
  "$PYTHON" scripts/train_baseline.py "${args[@]}" 2>&1 | tee "$destination/train.log"
  touch "$destination/.complete"
  if ! "$PYTHON" scripts/validate_tracked_outputs.py --root "$destination"; then
    rm -f "$destination/.complete"
    return 1
  fi
}
run_control() {
  local name="$1" mode="$2" aggregator="$3" seed="$4"
  local destination="$OUTPUT_ROOT/$name"
  [[ -f "$destination/.complete" ]] && return
  mkdir -p "$destination"
  "$PYTHON" scripts/train_baseline.py --config "$BASE_CONFIG" --release-dir "$RELEASE_DIR" --output-dir "$destination" \
    --mode "$mode" --aggregator "$aggregator" --seed "$seed" --epochs "$EPOCHS" --device "$DEVICE" 2>&1 | tee "$destination/train.log"
  touch "$destination/.complete"
  if ! "$PYTHON" scripts/validate_tracked_outputs.py --root "$destination"; then rm -f "$destination/.complete"; return 1; fi
}
worker() {
  cd "$ROOT"
  mkdir -p "$OUTPUT_ROOT"
  printf 'started_at=%s\ncommit=%s\nseeds=%s\nepochs=%s\n' "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" "$SEEDS" "$EPOCHS" > "$OUTPUT_ROOT/run-metadata.txt"
  local count=0
  for seed in $SEEDS; do
    run_control "seed-${seed}-b0-statistical" statistical mean "$seed"; count=$((count + 1))
    run_control "seed-${seed}-random-forest" random_forest mean "$seed"; count=$((count + 1))
    run_control "seed-${seed}-e0-full-signal" full_signal mean "$seed"; count=$((count + 1))
    run_control "seed-${seed}-e1a-mean" window_mil mean "$seed"; count=$((count + 1))
    run_control "seed-${seed}-e1b-max" window_mil max "$seed"; count=$((count + 1))
    run_control "seed-${seed}-e1b-topk-mean" window_mil topk_mean "$seed"; count=$((count + 1))
    run_control "seed-${seed}-e1c-gated-attention" window_mil gated_attention "$seed"; count=$((count + 1))
    for representation in stft_256 stft_512 cwt_morl dwt_swt_db4; do
      for encoder in cnn2d separable_cnn2d resnet2d_small convnext2d_lite; do
        for aggregator in mean max topk_mean gated_attention; do
          run_one "seed-${seed}-${representation}-${encoder}-${aggregator}" "$representation" "$encoder" none "$aggregator" "$seed"
          count=$((count + 1))
          if [[ "$SMOKE_ONLY" == 1 && "$count" -ge 8 ]]; then break 4; fi
        done
      done
    done
    for fusion in raw_plus_stft raw_plus_cwt; do
      if [[ "$fusion" == raw_plus_stft ]]; then representation=stft_256; else representation=cwt_morl; fi
      for encoder in cnn2d separable_cnn2d resnet2d_small convnext2d_lite; do
        for aggregator in mean max topk_mean gated_attention; do
          run_one "seed-${seed}-${fusion}-${encoder}-${aggregator}" "$representation" "$encoder" "$fusion" "$aggregator" "$seed"
          count=$((count + 1))
          if [[ "$SMOKE_ONLY" == 1 && "$count" -ge 10 ]]; then break 4; fi
        done
      done
    done
  done
  printf 'status=complete\nfinished_at=%s\nrun_count=%s\n' "$(date --iso-8601=seconds)" "$count" > "$OUTPUT_ROOT/status.txt"
}
if [[ "${1:-}" == "--worker" ]]; then
  shift; [[ $# -eq 3 ]] || { usage >&2; exit 2; }
  RELEASE_DIR="$1"; OUTPUT_ROOT="$2"; BASE_CONFIG="$3"; worker; exit
fi
[[ $# -ge 1 && $# -le 3 ]] || { usage >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "Python executable not found: $PYTHON" >&2; exit 2; }
RELEASE_DIR="$(cd "$1" && pwd)"
OUTPUT_ROOT="${2:-$ROOT/outputs/time-frequency-$(date +%Y%m%d-%H%M%S)}"
BASE_CONFIG="${3:-$ROOT/configs/moderntcn_mil_weld_independent_v0_2_1.toml}"
mkdir -p "$OUTPUT_ROOT"; OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1
nohup bash "$ROOT/scripts/run_time_frequency_suite.sh" --worker "$RELEASE_DIR" "$OUTPUT_ROOT" "$BASE_CONFIG" > "$OUTPUT_ROOT/suite.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$OUTPUT_ROOT/suite.pid"
echo "Started time-frequency suite: $OUTPUT_ROOT"
