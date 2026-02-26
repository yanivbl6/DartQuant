#!/bin/bash


./Script/dart_gptq_wxaykvz.sh full

./Script/dart_gptq_wxaykvz.sh baseline -w 4 -a 8 -k 4 -G 128 --sym
./Script/dart_gptq_wxaykvz.sh quarot -w 4 -a 8 -k 4 -G 128 --sym
./Script/dart_gptq_wxaykvz.sh dart -w 4 -a 8 -k 4 -G 128 --sym
