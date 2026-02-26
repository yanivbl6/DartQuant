#!/bin/bash

# =============================================================================
# Calibrate R1/R2 rotation matrices for a given model
# =============================================================================
#
# Runs the full calibration pipeline:
#   1. get_train_data.py  — capture layer activations
#   2. r1_base_qr.py      — train R1 rotation matrix
#   3. r2_base_qr.py      — train per-layer R2 rotation matrices
#
# Usage:
#   ./calibrate_model.sh -m /path/to/model [-g GPU_ID]
#
# Examples:
#   ./calibrate_model.sh -m /data/users/sashas/LLMC/Models/meta-llama/Llama-3.2-1B-Instruct
#   ./calibrate_model.sh -m /data/users/sashas/LLMC/Models/meta-llama/Llama-3.2-3B-Instruct -g 1
# =============================================================================

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 -m MODEL [-g GPU_ID]

Options:
  -m MODEL    Model path or HF model name (required)
  -g GPU_ID   GPU device ID (default: 0)
  -h          Show this help message
EOF
    exit 0
}

GPU_ID=0
MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m) MODEL="$2";  shift 2 ;;
        -g) GPU_ID="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "Error: -m MODEL is required."
    usage
fi

MODEL_NAME=$(basename "${MODEL%/}")
echo "=== Calibrating ${MODEL_NAME} on GPU ${GPU_ID} ==="

# Hyperparameters matching the existing 7b calibration
R1_LR=0.0015
R1_EP=10
R1_BSZ=64
R1_SUBSET=0.1
R1_ACCUM=1

R2_LR=0.001
R2_EP=10
R2_BSZ=64
R2_ACCUM=2

# --- Step 1: Generate training data ---
echo ""
echo "=== Step 1/3: Generating training data ==="
CUDA_VISIBLE_DEVICES=${GPU_ID} python get_train_data.py \
    --model ${MODEL} \
    --calib_dataset wikitext2 \
    --nsamples 128 \
    --seqlen 2048

# --- Step 2: Train R1 ---
echo ""
echo "=== Step 2/3: Training R1 ==="
CUDA_VISIBLE_DEVICES=${GPU_ID} python r1_base_qr.py \
    --model ${MODEL} \
    --calib_dataset wikitext2 \
    --nsamples 128 \
    --calib_sample 128 \
    --optim sgd \
    --lr ${R1_LR} \
    --mom 0.9 \
    --ep ${R1_EP} \
    --bsz ${R1_BSZ} \
    --train_subset_size ${R1_SUBSET} \
    --accumulation_steps ${R1_ACCUM} \
    --init_mode hadamard \
    --save_model

# --- Step 3: Train R2 ---
echo ""
echo "=== Step 3/3: Training R2 ==="
CUDA_VISIBLE_DEVICES=${GPU_ID} python r2_base_qr.py \
    --model ${MODEL} \
    --calib_dataset wikitext2 \
    --nsamples 128 \
    --calib_sample 128 \
    --optim sgd \
    --lr ${R2_LR} \
    --mom 0.9 \
    --ep ${R2_EP} \
    --bsz ${R2_BSZ} \
    --accumulation_steps ${R2_ACCUM} \
    --save_model

echo ""
echo "=== Calibration complete for ${MODEL_NAME} ==="
echo "Trained rotations saved under: ../data/trained_rotation/wikitext2_128samples/${MODEL_NAME}/"
