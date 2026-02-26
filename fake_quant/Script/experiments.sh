#!/bin/bash


model="1b"

if [[ $# -gt 0 ]]; then
    model=$1
fi

./Script/dart_gptq_wxaykvz.sh full -m $model    

./Script/dart_gptq_wxaykvz.sh baseline -w 4 -a 8 -k 4 -G 128 --sym -m $model
./Script/dart_gptq_wxaykvz.sh quarot -w 4 -a 8 -k 4 -G 128 --sym -m $model
./Script/dart_gptq_wxaykvz.sh dart -w 4 -a 8 -k 4 -G 128 --sym -m $model
