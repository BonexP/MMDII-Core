#!/usr/bin/env bash
set -Eeuo pipefail

# Reduced follow-up suite.  It deliberately leaves the canonical 103-run
# suite unchanged and keeps a small max/top-k sentinel set for reversal checks.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
SEEDS="${SEEDS:-17 27}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
DEVICE="${DEVICE:-auto}"
MMDII_TRANSFORM_WORKERS="${MMDII_TRANSFORM_WORKERS:-2}"
INCLUDE_SENTINELS="${INCLUDE_SENTINELS:-1}"

usage() { echo "Usage: bash scripts/run_time_frequency_focused_suite.sh RELEASE_DIR [OUTPUT_DIR] [CONFIG]"; }

check_dependencies() {
  if ! "$PYTHON" -c 'import scipy, pywt, sklearn, torch' >/dev/null 2>&1; then
    echo "Missing training dependencies in $PYTHON." >&2
    echo "Install them with: uv pip install --python $PYTHON -e \".[train]\"" >&2
    return 1
  fi
}

run_one() {
  local name="$1" representation="$2" encoder="$3" fusion="$4" aggregator="$5" seed="$6"
  local destination="$OUTPUT_ROOT/$name"
  printf '%s\n' "$name" >> "$CONFIG_MANIFEST"
  PLANNED=$((PLANNED + 1))
  [[ -f "$destination/.complete" ]] && return
  mkdir -p "$destination"
  "$PYTHON" scripts/train_baseline.py \
    --config "$BASE_CONFIG" --release-dir "$RELEASE_DIR" --output-dir "$destination" \
    --mode window_mil --aggregator "$aggregator" --seed "$seed" --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" --learning-rate "$LEARNING_RATE" --weight-decay "$WEIGHT_DECAY" \
    --optimizer adamw --early-stopping-patience 0 --fold-scheme weld_independent \
    --device "$DEVICE" --representation "$representation" --encoder "$encoder" --fusion "$fusion" \
    2>&1 | tee "$destination/train.log"
  touch "$destination/.complete"
  if ! "$PYTHON" scripts/validate_tracked_outputs.py --root "$destination" \
    --expected-weld-count 101 --expected-fold-count 5 --expected-fold-scheme weld_independent; then
    rm -f "$destination/.complete"
    return 1
  fi
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
  cp "$ROOT/configs/time_frequency_focused_matrix.toml" "$OUTPUT_ROOT/matrix-config.toml"
  printf 'started_at=%s\ncommit=%s\nrelease=%s\nbase_config=%s\nseeds=%s\nepochs=%s\nbatch_size=%s\nlearning_rate=%s\nweight_decay=%s\noptimizer=adamw\nearly_stopping_patience=0\nfold_scheme=weld_independent\ntransform_workers=%s\ninclude_sentinels=%s\n' \
    "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)" "$RELEASE_DIR" "$BASE_CONFIG" "$SEEDS" "$EPOCHS" \
    "$BATCH_SIZE" "$LEARNING_RATE" "$WEIGHT_DECAY" "$MMDII_TRANSFORM_WORKERS" "$INCLUDE_SENTINELS" > "$OUTPUT_ROOT/run-metadata.txt"

  for seed in $SEEDS; do
    for representation in stft_256 stft_512 cwt_morl dwt_swt_db4; do
      for encoder in cnn2d separable_cnn2d resnet2d_small convnext2d_lite; do
        for aggregator in mean gated_attention; do
          run_one "seed-${seed}-${representation}-${encoder}-${aggregator}" "$representation" "$encoder" none "$aggregator" "$seed"
        done
      done
    done
    for fusion in raw_plus_stft raw_plus_cwt; do
      if [[ "$fusion" == raw_plus_stft ]]; then representation=stft_256; else representation=cwt_morl; fi
      for encoder in cnn2d separable_cnn2d resnet2d_small convnext2d_lite; do
        for aggregator in mean gated_attention; do
          run_one "seed-${seed}-${fusion}-${encoder}-${aggregator}" "$representation" "$encoder" "$fusion" "$aggregator" "$seed"
        done
      done
    done
    if [[ "$INCLUDE_SENTINELS" == 1 ]]; then
      run_one "seed-${seed}-stft_256-cnn2d-max" stft_256 cnn2d none max "$seed"
      run_one "seed-${seed}-stft_256-cnn2d-topk_mean" stft_256 cnn2d none topk_mean "$seed"
      run_one "seed-${seed}-cwt_morl-resnet2d_small-max" cwt_morl resnet2d_small none max "$seed"
      run_one "seed-${seed}-cwt_morl-resnet2d_small-topk_mean" cwt_morl resnet2d_small none topk_mean "$seed"
      run_one "seed-${seed}-dwt_swt_db4-resnet2d_small-max" dwt_swt_db4 resnet2d_small none max "$seed"
      run_one "seed-${seed}-dwt_swt_db4-resnet2d_small-topk_mean" dwt_swt_db4 resnet2d_small none topk_mean "$seed"
      run_one "seed-${seed}-raw_plus_cwt-resnet2d_small-max" cwt_morl resnet2d_small raw_plus_cwt max "$seed"
      run_one "seed-${seed}-raw_plus_cwt-resnet2d_small-topk_mean" cwt_morl resnet2d_small raw_plus_cwt topk_mean "$seed"
    fi
  done
  read -r -a seed_array <<< "$SEEDS"
  "$PYTHON" scripts/validate_tracked_outputs.py --root "$OUTPUT_ROOT" \
    --expected-run-count "$PLANNED" --expected-configurations-per-seed "$((PLANNED / ${#seed_array[@]}))" \
    --expected-weld-count 101 --expected-fold-count 5 --expected-fold-scheme weld_independent \
    --expected-seeds "${seed_array[@]}"
  trap - ERR
  printf 'status=complete\nfinished_at=%s\nrun_count=%s\n' "$(date --iso-8601=seconds)" "$PLANNED" > "$OUTPUT_ROOT/status.txt"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift; [[ $# -eq 3 ]] || { usage >&2; exit 2; }
  RELEASE_DIR="$1"; OUTPUT_ROOT="$2"; BASE_CONFIG="$3"; worker; exit
fi
[[ $# -ge 1 && $# -le 3 ]] || { usage >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "Python executable not found: $PYTHON" >&2; exit 2; }
RELEASE_DIR="$(cd "$1" && pwd)"
OUTPUT_ROOT="${2:-$ROOT/outputs/time-frequency-focused-$(date +%Y%m%d-%H%M%S)}"
BASE_CONFIG="${3:-$ROOT/configs/moderntcn_mil_weld_independent_v0_2_1.toml}"
mkdir -p "$OUTPUT_ROOT"; OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1
nohup bash "$ROOT/scripts/run_time_frequency_focused_suite.sh" --worker "$RELEASE_DIR" "$OUTPUT_ROOT" "$BASE_CONFIG" > "$OUTPUT_ROOT/suite.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$OUTPUT_ROOT/suite.pid"
echo "Started focused time-frequency suite: $OUTPUT_ROOT"
