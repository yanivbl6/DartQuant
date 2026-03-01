#!/bin/bash

## all arguments passed through to each script

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

rm /tmp/*_results.out
rm /tmp/*_results.err

./Script/dart_gptq_wxaykvz.sh full -g 1 "$@" > /tmp/full_results.out 2> /tmp/full_results.err &
./Script/dart_gptq_wxaykvz.sh baseline -g 2 -w 4 -a 8 -k 4 -G 128 --sym "$@" > /tmp/baseline_results.out 2> /tmp/baseline_results.err &
./Script/dart_gptq_wxaykvz.sh quarot -g 3 -w 4 -a 8 -k 4 -G 128 --sym "$@" > /tmp/quarot_results.out 2> /tmp/quarot_results.err &
./Script/dart_gptq_wxaykvz.sh dart -g 4 -w 4 -a 8 -k 4 -G 128 --sym "$@" > /tmp/dart_results.out 2> /tmp/dart_results.err &

# --- Static activation quantization runs (reuse GPTQ checkpoints from above) ---
./Script/dart_gptq_wxaykvz.sh baseline -g 5 -w 4 -a 8 -k 4 -G 128 --sym --static-act "$@" > /tmp/baseline_static_results.out 2> /tmp/baseline_static_results.err &
./Script/dart_gptq_wxaykvz.sh quarot -g 6 -w 4 -a 8 -k 4 -G 128 --sym --static-act "$@" > /tmp/quarot_static_results.out 2> /tmp/quarot_static_results.err &
./Script/dart_gptq_wxaykvz.sh dart -g 7 -w 4 -a 8 -k 4 -G 128 --sym --static-act "$@" > /tmp/dart_static_results.out 2> /tmp/dart_static_results.err &

# --- R3/R4 exclusion experiments (replace online rotation with higher-bit quant) ---
#./Script/dart_gptq_wxaykvz.sh dart -g 5 -w 4 -a 8 -k 4 -G 128 --sym --kv_ex 8 "$@" > /tmp/dart_kvex8_results.out 2> /tmp/dart_kvex8_results.err &
#./Script/dart_gptq_wxaykvz.sh dart -g 6 -w 4 -a 8 -k 4 -G 128 --sym --proj_ex 8 "$@" > /tmp/dart_projex8_results.out 2> /tmp/dart_projex8_results.err &
#./Script/dart_gptq_wxaykvz.sh dart -g 7 -w 4 -a 8 -k 4 -G 128 --sym --kv_ex 8 --proj_ex 8 "$@" > /tmp/dart_kvex8_projex8_results.out 2> /tmp/dart_kvex8_projex8_results.err &
#  --kv_ex 8 --proj_ex 15 -k 8
wait

echo "All experiments completed"
