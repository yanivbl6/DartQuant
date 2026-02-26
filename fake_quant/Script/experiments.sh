#!/bin/bash

## all arguments passed through to each script

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7





./Script/dart_gptq_wxaykvz.sh full -g 4 "$@" > /tmp/full_results.out 2> /tmp/full_results.err &
./Script/dart_gptq_wxaykvz.sh baseline -g 5 -w 4 -a 8 -k 4 -G 128 --sym "$@" > /tmp/baseline_results.out 2> /tmp/baseline_results.err &
./Script/dart_gptq_wxaykvz.sh quarot -g 6 -w 4 -a 8 -k 4 -G 128 --sym "$@" > /tmp/quarot_results.out 2> /tmp/quarot_results.err &
./Script/dart_gptq_wxaykvz.sh dart -g 7 -w 4 -a 8 -k 4 -G 128 --sym "$@" > /tmp/dart_results.out 2> /tmp/dart_results.err &

./Script/dart_gptq_wxaykvz.sh baseline -g 5 -w 4 -a 8 -k 4 -G 128 --sym --static-act "$@" > /tmp/baseline_results.out 2> /tmp/baseline_results.err &
./Script/dart_gptq_wxaykvz.sh quarot -g 6 -w 4 -a 8 -k 4 -G 128 --sym --static-act "$@" > /tmp/quarot_results.out 2> /tmp/quarot_results.err &
./Script/dart_gptq_wxaykvz.sh dart -g 7 -w 4 -a 8 -k 4 -G 128 --sym --static-act "$@" > /tmp/dart_results.out 2> /tmp/dart_results.err &



wait

echo "All experiments completed"