#!/bin/bash

## Wrapper to run the 7 experiments. Optionally runs calibration first.
## Usage:
##   ./Script/run_experiments_after_calibrate.sh [--calibrate] [extra args...]
## If --calibrate is provided, this will call ../calibrater/calibrate_model.sh first.
## Extra args (e.g. -m 1b) are forwarded to all experiment runs.

CALIBRATE=0
if [ "$1" = "--calibrate" ]; then
  CALIBRATE=1
  shift
fi

# Save remaining args before calibration parsing can consume them
EXTRA_ARGS=("$@")

if [ $CALIBRATE -eq 1 ]; then
  # Extract -m MODEL and optional -g GPU_ID for calibration
  MODEL=""
  GPU_ID=""
  for ((i=0; i<${#EXTRA_ARGS[@]}; i++)); do
    case "${EXTRA_ARGS[$i]}" in
      -m) MODEL="${EXTRA_ARGS[$((i+1))]}" ;;
      -g) GPU_ID="${EXTRA_ARGS[$((i+1))]}" ;;
    esac
  done

  if [ -z "${MODEL}" ]; then
    echo "Error: -m MODEL is required for calibration."
    exit 1
  fi

  CAL_CMD=("../calibrater/calibrate_model.sh" -m "${MODEL}")
  if [ -n "${GPU_ID}" ]; then
    CAL_CMD+=( -g "${GPU_ID}" )
  fi

  echo "=== Running calibration: ${CAL_CMD[*]} ==="
  "${CAL_CMD[@]}"
  echo ""
fi

# --- Shared quantization args ---
QUANT_ARGS=(-w 4 -a 8 -k 8 -G 128 --sym --kv_ex 8 --proj_ex 15)

# --- Define experiments: (name, mode, gpu, extra_flags...) ---
EXPERIMENTS=(
  "full|full|1|"
  "baseline|baseline|2|"
  "quarot|quarot|3|"
  "dart|dart|4|"
  "baseline_static|baseline|5|--static-act"
  "quarot_static|quarot|6|--static-act"
  "dart_static|dart|7|--static-act"
)

SCRIPT_DIR_BASE="$(cd "$(dirname "$0")/../.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR_BASE}/data/cached_results"

rm -f ${RESULTS_DIR}/*_results.out ${RESULTS_DIR}/*_results.err 2>/dev/null

echo "=== Launching 7 experiments ==="
echo ""

PIDS=()
for exp in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r name mode gpu extra <<< "$exp"

  if [ "$mode" = "full" ]; then
    CMD=(./Script/dart_gptq_wxaykvz.sh "$mode" -g "$gpu" "${EXTRA_ARGS[@]}")
  else
    CMD=(./Script/dart_gptq_wxaykvz.sh "$mode" -g "$gpu" "${QUANT_ARGS[@]}")
    [ -n "$extra" ] && CMD+=($extra)
    CMD+=("${EXTRA_ARGS[@]}")
  fi

  echo "  [GPU $gpu] $name: ${CMD[*]}"
  "${CMD[@]}" > "${RESULTS_DIR}/${name}_results.out" 2> "${RESULTS_DIR}/${name}_results.err" &
  PIDS+=("$!:$name")
done

echo ""
echo "=== Waiting for all experiments ==="

FAILED=0
for entry in "${PIDS[@]}"; do
  IFS=':' read -r pid name <<< "$entry"
  if wait "$pid"; then
    echo "  [done] $name"
  else
    echo "  [FAIL] $name (see ${RESULTS_DIR}/${name}_results.err)"
    FAILED=$((FAILED + 1))
  fi
done

echo ""
if [ $FAILED -eq 0 ]; then
  echo "=== All 7 experiments completed successfully ==="
else
  echo "=== $FAILED experiment(s) failed ==="
  exit 1
fi
