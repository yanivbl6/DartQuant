#!/bin/bash

## all arguments passed through to each script

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/data/cached_results"

rm ${RESULTS_DIR}/*_results.out
rm ${RESULTS_DIR}/*_results.err

./Script/dart_gptq_wxaykvz.sh full -g 1 "$@" > ${RESULTS_DIR}/full_results.out 2> ${RESULTS_DIR}/full_results.err &
./Script/dart_gptq_wxaykvz.sh baseline -g 2 -w 4 -a 8 -k 8 -G 128 --sym --kv_ex 8 --proj_ex 15 "$@" > ${RESULTS_DIR}/baseline_results.out 2> ${RESULTS_DIR}/baseline_results.err &
./Script/dart_gptq_wxaykvz.sh quarot -g 3 -w 4 -a 8 -k 8 -G 128 --sym --kv_ex 8 --proj_ex 15 "$@" > ${RESULTS_DIR}/quarot_results.out 2> ${RESULTS_DIR}/quarot_results.err &
./Script/dart_gptq_wxaykvz.sh dart -g 4 -w 4 -a 8 -k 8 -G 128 --sym --kv_ex 8 --proj_ex 15 "$@" > ${RESULTS_DIR}/dart_results.out 2> ${RESULTS_DIR}/dart_results.err &

# --- Static activation quantization runs (reuse GPTQ checkpoints from above) ---
./Script/dart_gptq_wxaykvz.sh baseline -g 5 -w 4 -a 8 -k 8 -G 128 --sym --static-act --kv_ex 8 --proj_ex 15 "$@" > ${RESULTS_DIR}/baseline_static_results.out 2> ${RESULTS_DIR}/baseline_static_results.err &
./Script/dart_gptq_wxaykvz.sh quarot -g 6 -w 4 -a 8 -k 8 -G 128 --sym --static-act --kv_ex 8 --proj_ex 15 "$@" > ${RESULTS_DIR}/quarot_static_results.out 2> ${RESULTS_DIR}/quarot_static_results.err &
./Script/dart_gptq_wxaykvz.sh dart -g 7 -w 4 -a 8 -k 8 -G 128 --sym --static-act --kv_ex 8 --proj_ex 15 "$@" > ${RESULTS_DIR}/dart_static_results.out 2> ${RESULTS_DIR}/dart_static_results.err &

# --- R3/R4 exclusion experiments (replace online rotation with higher-bit quant) ---
#./Script/dart_gptq_wxaykvz.sh dart -g 5 -w 4 -a 8 -k 4 -G 128 --sym --kv_ex 8 "$@" > ${RESULTS_DIR}/dart_kvex8_results.out 2> ${RESULTS_DIR}/dart_kvex8_results.err &
#./Script/dart_gptq_wxaykvz.sh dart -g 6 -w 4 -a 8 -k 4 -G 128 --sym --proj_ex 8 "$@" > ${RESULTS_DIR}/dart_projex8_results.out 2> ${RESULTS_DIR}/dart_projex8_results.err &
#./Script/dart_gptq_wxaykvz.sh dart -g 7 -w 4 -a 8 -k 4 -G 128 --sym --kv_ex 8 --proj_ex 8 "$@" > ${RESULTS_DIR}/dart_kvex8_projex8_results.out 2> ${RESULTS_DIR}/dart_kvex8_projex8_results.err &

wait

echo "All experiments completed"
