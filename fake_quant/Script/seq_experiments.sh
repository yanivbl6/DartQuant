#!/bin/bash

## all arguments passed through to each script

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7



SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/data/cached_results"

rm ${RESULTS_DIR}/*_results.out
rm ${RESULTS_DIR}/*_results.err

./Script/dart_gptq_wxaykvz.sh full -g 1 "$@"
./Script/dart_gptq_wxaykvz.sh baseline -g 2 -w 4 -a 8 -k 4 -G 128 --sym "$@" 
./Script/dart_gptq_wxaykvz.sh quarot -g 3 -w 4 -a 8 -k 4 -G 128 --sym "$@"
./Script/dart_gptq_wxaykvz.sh dart -g 4 -w 4 -a 8 -k 4 -G 128 --sym "$@"

# --- Static activation quantization runs (reuse GPTQ checkpoints from above) ---
./Script/dart_gptq_wxaykvz.sh baseline -g 5 -w 4 -a 8 -k 4 -G 128 --sym --static-act "$@"
./Script/dart_gptq_wxaykvz.sh quarot -g 6 -w 4 -a 8 -k 4 -G 128 --sym --static-act "$@"
./Script/dart_gptq_wxaykvz.sh dart -g 7 -w 4 -a 8 -k 4 -G 128 --sym --static-act  "$@"

# --- R3/R4 exclusion experiments (replace online rotation with higher-bit quant) ---
#./Script/dart_gptq_wxaykvz.sh dart -g 4 -w 4 -a 8 -k 4 -G 128 --sym --kv_ex 8 "$@"
#./Script/dart_gptq_wxaykvz.sh dart -g 4 -w 4 -a 8 -k 4 -G 128 --sym --proj_ex 8 "$@"
#./Script/dart_gptq_wxaykvz.sh dart -g 4 -w 4 -a 8 -k 4 -G 128 --sym --kv_ex 8 --proj_ex 8 "$@"

wait

echo "All experiments completed"
