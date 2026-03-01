#!/bin/bash

## Wrapper to run the 7 experiments. Optionally runs calibration first.
## Usage:
##   ./Script/run_experiments_after_calibrate.sh [--calibrate] [args...]
## If --calibrate is provided, this will call ../calibrater/calibrate_model.sh with the same args.

set -e

CALIBRATE=0
if [ "$1" = "--calibrate" ]; then
  CALIBRATE=1
  shift
fi

if [ $CALIBRATE -eq 1 ]; then
  # Extract only -m MODEL and optional -g GPU_ID from the passed args
  MODEL=""
  GPU_ID=""
  i=1
  # iterate over args
  while [ "$#" -gt 0 ]; do
    case "$1" in
      -m)
        MODEL="$2"
        shift 2
        ;;
      -g)
        GPU_ID="$2"
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done

  if [ -z "${MODEL}" ]; then
    echo "Error: -m MODEL is required for calibration. Provide it when invoking this script."
    exit 1
  fi

  CAL_CMD=("../calibrater/calibrate_model.sh" -m "${MODEL}")
  if [ -n "${GPU_ID}" ]; then
    CAL_CMD+=( -g "${GPU_ID}" )
  fi

  echo "Running calibrate: ${CAL_CMD[*]}"
  "${CAL_CMD[@]}"
fi

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

rm /tmp/*_results.out || true
rm /tmp/*_results.err || true

./Script/dart_gptq_wxaykvz.sh full -g 1 "$@" > /tmp/full_results.out 2> /tmp/full_results.err &
./Script/dart_gptq_wxaykvz.sh baseline -g 2 -w 4 -a 8 -k 8 -G 128 --sym --kv_ex 8 --proj_ex 15 "$@" > /tmp/baseline_results.out 2> /tmp/baseline_results.err &
./Script/dart_gptq_wxaykvz.sh quarot -g 3 -w 4 -a 8 -k 8 -G 128 --sym --kv_ex 8 --proj_ex 15 "$@" > /tmp/quarot_results.out 2> /tmp/quarot_results.err &
./Script/dart_gptq_wxaykvz.sh dart -g 4 -w 4 -a 8 -k 8 -G 128 --sym --kv_ex 8 --proj_ex 15 "$@" > /tmp/dart_results.out 2> /tmp/dart_results.err &

# --- Static activation quantization runs (reuse GPTQ checkpoints from above) ---
./Script/dart_gptq_wxaykvz.sh baseline -g 5 -w 4 -a 8 -k 8 -G 128 --sym --static-act --kv_ex 8 --proj_ex 15 "$@" > /tmp/baseline_static_results.out 2> /tmp/baseline_static_results.err &
./Script/dart_gptq_wxaykvz.sh quarot -g 6 -w 4 -a 8 -k 8 -G 128 --sym --static-act --kv_ex 8 --proj_ex 15 "$@" > /tmp/quarot_static_results.out 2> /tmp/quarot_static_results.err &
./Script/dart_gptq_wxaykvz.sh dart -g 7 -w 4 -a 8 -k 8 -G 128 --sym --static-act --kv_ex 8 --proj_ex 15 "$@" > /tmp/dart_static_results.out 2> /tmp/dart_static_results.err &

wait

echo "All experiments completed"
