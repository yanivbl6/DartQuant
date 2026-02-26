#!/bin/bash

# =============================================================================
# DartQuant / QuaRot / Baseline GPTQ Evaluation Script
# =============================================================================
#
# Supports four modes:
#   full     : Full-precision (FP16) evaluation, no quantization, no rotations.
#   baseline : Plain GPTQ quantization, no rotations, no norm fusion.
#   quarot   : QuaRot-style all-Hadamard rotations (R1-R4), no learned matrices.
#   dart     : DartQuant with learned R1/R2 rotations + online R3/R4 Hadamard.
#
# Run with -h for full usage information.
# =============================================================================

usage() {
    cat <<EOF
Usage: $0 <MODE> [OPTIONS]

MODE (required, first argument):
  full       Full-precision (FP16), no quantization, no rotations.
  baseline   GPTQ quantization, no rotations.
  quarot     QuaRot: all-Hadamard rotations (R1-R4).
  dart       DartQuant: learned R1/R2 + online R3/R4 Hadamard.

Options:
  -g GPU_ID        GPU device ID                       (default: 0)
  -m MODEL         Model path or HF model name         (default: meta-llama/Llama-2-7b-hf)
  -w W_BITS        Weight bit-width                    (default: 4)
  -a A_BITS        Activation bit-width                (default: 8)
  -k KV_BITS       KV-cache bit-width                  (default: 4)
  -G GROUPSIZE     Group size for W, K, V              (default: 128)
  --sym            Use symmetric quantization for W/K/V (default: asymmetric)
  --overwrite      Ignore cached results, re-run all
  -h               Show this help message

Notes:
  - "full" mode forces W/A/KV to 16-bit (options -w/-a/-k are ignored).
  - Activations are always asymmetric (--sym only affects W, K, V).

Examples:
  $0 full -g 0 -m /path/to/model
  $0 quarot -w 8 -a 8 -k 8
  $0 dart -w 4 -a 8 -k 4 --sym
  $0 baseline -w 4 -a 8 -k 4 -G 64
EOF
    exit 0
}

# --- Parse MODE (must be first arg) ---
if [ $# -eq 0 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    usage
fi

MODE=$1
shift

case "$MODE" in
    full|baseline|quarot|dart) ;;
    *)
        echo "Error: MODE must be one of: full, baseline, quarot, dart (got '$MODE')"
        echo "Run '$0 -h' for help."
        exit 1
        ;;
esac

# --- Defaults ---
GPU_ID=0
MODEL="/data/users/sashas/LLMC/Models/meta-llama/Llama-2-7b-hf/"
W_BITS=4
A_BITS=8
KV_BITS=4
GROUPSIZE=128
SYM=0
OVERWRITE=0

# --- Parse options ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        -g)       GPU_ID="$2";    shift 2 ;;
        -m)       MODEL="$2";     shift 2 ;;
        -w)       W_BITS="$2";    shift 2 ;;
        -a)       A_BITS="$2";    shift 2 ;;
        -k)       KV_BITS="$2";   shift 2 ;;
        -G)       GROUPSIZE="$2"; shift 2 ;;
        --sym)    SYM=1;          shift   ;;
        --overwrite) OVERWRITE=1; shift   ;;
        -h|--help) usage ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 -h' for help."
            exit 1
            ;;
    esac
done

# --- Full mode overrides ---
if [ "$MODE" == "full" ]; then
    W_BITS=16
    A_BITS=16
    KV_BITS=16
fi

# --- Symmetry flags ---
if [ "$SYM" == "1" ]; then
    W_ASYM_FLAG=""
    K_ASYM_FLAG=""
    V_ASYM_FLAG=""
    SYM_TAG="wSym_kSym_vSym"
else
    W_ASYM_FLAG=""       # w is symmetric by default in the python code
    K_ASYM_FLAG="--k_asym"
    V_ASYM_FLAG="--v_asym"
    SYM_TAG="kAsym_vAsym"
fi

MODEL_NAME=${MODEL##*/}

# --- Build rotation flags based on mode ---
if [ "$MODE" == "full" ] || [ "$MODE" == "baseline" ]; then
    ROTATION_FLAGS="\
    --no-fuse_norm \
    --no-use_r1 \
    --use_r2 none \
    --no-use_r3 \
    --no-use_r4"
    SAVE_PREFIX="$MODE"

elif [ "$MODE" == "quarot" ]; then
    ROTATION_FLAGS="\
    --fuse_norm \
    --use_r1 \
    --use_r2 offline \
    --use_r3 \
    --use_r4 \
    --o_per_head"
    SAVE_PREFIX="quarot"

elif [ "$MODE" == "dart" ]; then
    R2_PATH="../data/trained_rotation/wikitext2_128samples/r2/sgd.0.001.0.9.10.64.2"
    R1_PATH="../data/trained_rotation/wikitext2_128samples/r1/sgd.0.0015.0.9.10.64.0.1.1"

    ROTATION_FLAGS="\
    --fuse_norm \
    --use_r1 \
    --r1_path ${R1_PATH} \
    --use_r2 offline \
    --r2_path ${R2_PATH} \
    --use_r3 \
    --use_r4 \
    --o_per_head"
    SAVE_PREFIX="dart"
fi

# --- Descriptive tag encoding the quantization config ---
QUANT_TAG="w${W_BITS}a${A_BITS}k${KV_BITS}v${KV_BITS}_g${GROUPSIZE}_aAsym_${SYM_TAG}"

# Result cache: /tmp/<mode>_<quant_tag>_results.pb  (JSON format)
CACHE_PATH="/tmp/${SAVE_PREFIX}_${QUANT_TAG}_results.pb"
OVERWRITE_FLAG=""
if [ "$OVERWRITE" == "1" ]; then
    OVERWRITE_FLAG="--overwrite"
fi

CUDA_VISIBLE_DEVICES=${GPU_ID} python main_for_test.py \
    --model ${MODEL} \
    ${ROTATION_FLAGS} \
    --gptq_checkpoint_path /tmp/${SAVE_PREFIX}_${QUANT_TAG} \
    --cache_path ${CACHE_PATH} \
    ${OVERWRITE_FLAG} \
    --w_groupsize ${GROUPSIZE} \
    --w_clip \
    --a_asym \
    --a_clip_ratio 0.9 \
    --w_bits ${W_BITS} \
    --a_bits ${A_BITS} \
    --k_bits ${KV_BITS} \
    --v_bits ${KV_BITS} \
    --k_groupsize ${GROUPSIZE} \
    --v_groupsize ${GROUPSIZE} \
    ${K_ASYM_FLAG} \
    ${V_ASYM_FLAG} \
    --percdamp 0.1 \
    --no-w_ft \
    --ft_percdamp 0.0 \
    --save_name ${SAVE_PREFIX}_${QUANT_TAG} \
    --distribute \
    --ppl_eval \
    --ppl_eval_batch_size 1 \
    --ppl_eval_dataset wikitext2 ptb c4 \
    --lm_eval \
    --lm_eval_batch_size 2 \
    --tasks piqa hellaswag arc_easy arc_challenge winogrande lambada_openai social_iqa openbookqa mmlu
