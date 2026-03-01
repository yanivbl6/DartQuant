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
  -m MODEL         Model path, HF name, or shorthand:   (default: 7b)
                     1b  -> Llama-3.2-1B-Instruct
                     3b  -> Llama-3.2-3B-Instruct
                     7b  -> Llama-2-7b-hf
  -w W_BITS        Weight bit-width                    (default: 4)
  -a A_BITS        Activation bit-width                (default: 8)
  -k KV_BITS       KV-cache bit-width                  (default: 4)
  -G GROUPSIZE     Group size for W, K, V              (default: 128)
  --sym            Use symmetric quantization for W/K/V (default: asymmetric)
  --overwrite      Ignore cached results, re-run all
  --gptq           Delete cached GPTQ checkpoint and re-quantize
  --kv_ex N        K-cache quant without R3 rotation            (default: 0=off)
  --proj_ex N      Down-proj input quant without R4 rotation   (default: 0=off)
  --static-act     Use pre-calibrated static activation scales
  -F FAST          Enable fast model
  -h               Show this help message

Notes:
  - "full" mode forces W/A/KV to 16-bit (options -w/-a/-k are ignored).
  - Activations are always asymmetric (--sym only affects W, K, V).

Examples:
  $0 full -m 1b
  $0 quarot -m 3b -w 8 -a 8 -k 8
  $0 dart -m 1b -w 4 -a 8 -k 4 --sym
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
MODEL="7b"
W_BITS=4
A_BITS=8
KV_BITS=4
GROUPSIZE=128
SYM=0
OVERWRITE=0
FAST=0
STATIC_ACT=0
REDO_GPTQ=0
KV_EX=0
PROJ_EX=0

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
        --static-act) STATIC_ACT=1; shift ;;
        --gptq)   REDO_GPTQ=1;   shift   ;;
        --kv_ex)  KV_EX="$2";   shift 2 ;;
        --proj_ex) PROJ_EX="$2"; shift 2 ;;
        -F|--fast) FAST=1;        shift   ;;
        -h|--help) usage ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 -h' for help."
            exit 1
            ;;
    esac
done

# --- Resolve model shorthands ---
MODEL_BASE="/data/users/sashas/LLMC/Models/meta-llama"
case "$MODEL" in
    1b) MODEL="${MODEL_BASE}/Llama-3.2-1B-Instruct" ;;
    3b) MODEL="${MODEL_BASE}/Llama-3.2-3B-Instruct" ;;
    7b) MODEL="${MODEL_BASE}/Llama-2-7b-hf" ;;
esac

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

MODEL_NAME=$(basename "${MODEL%/}")

# Cap K_GROUPSIZE at head_dim for models with small heads (1B: head_dim=64)
case "$MODEL_NAME" in
    *1B*) K_GROUPSIZE=64 ;;
    *)    K_GROUPSIZE=$GROUPSIZE ;;
esac

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
    case "$MODEL_NAME" in
        Llama-2-7b-hf)
            R2_PATH="../data/trained_rotation/wikitext2_128samples/r2/sgd.0.001.0.9.10.64.2"
            R1_PATH="../data/trained_rotation/wikitext2_128samples/r1/sgd.0.0015.0.9.10.64.0.1.1"
            ;;
        Llama-3.2-1B-Instruct)
            R2_PATH="../data/trained_rotation/wikitext2_128samples/Llama-3.2-1B-Instruct/r2/sgd.0.001.0.9.10.64.2"
            R1_PATH="../data/trained_rotation/wikitext2_128samples/Llama-3.2-1B-Instruct/r1/sgd.0.0015.0.9.10.64.0.1.1"
            ;;
        Llama-3.2-3B-Instruct)
            R2_PATH="../data/trained_rotation/wikitext2_128samples/Llama-3.2-3B-Instruct/r2/sgd.0.001.0.9.10.64.2"
            R1_PATH="../data/trained_rotation/wikitext2_128samples/Llama-3.2-3B-Instruct/r1/sgd.0.0015.0.9.10.64.0.1.1"
            ;;
        *)
            echo "Error: dart mode requires trained R1/R2 matrices for ${MODEL_NAME}."
            echo "Train them first: cd calibrater && ./calibrate_model.sh -m <MODEL>"
            exit 1
            ;;
    esac

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
if [ "$KV_EX" != "0" ]; then
    QUANT_TAG="${QUANT_TAG}_kvex${KV_EX}"
fi
if [ "$PROJ_EX" != "0" ]; then
    QUANT_TAG="${QUANT_TAG}_projex${PROJ_EX}"
fi

OVERWRITE_FLAG=""
if [ "$OVERWRITE" == "1" ]; then
    OVERWRITE_FLAG="--overwrite"
fi

# --- Static activation scales ---
STATIC_ACT_FLAG=""
STATIC_TAG=""
if [ "$STATIC_ACT" == "1" ]; then
    ACT_SCALES_DIR="../data/act_scales/${MODEL_NAME}"
    ACT_SCALES_FILE="${ACT_SCALES_DIR}/${SAVE_PREFIX}_${QUANT_TAG}.pt"
    if [ ! -f "$ACT_SCALES_FILE" ]; then
        echo "Static act scales not found at ${ACT_SCALES_FILE}"
        echo "Run calibration first:"
        echo "  cd calibrater && python calibrate_act_scales.py --model ${MODEL} --mode ${MODE} --save_path ${ACT_SCALES_FILE} ..."
        exit 1
    fi
    STATIC_ACT_FLAG="--act_scales_path ${ACT_SCALES_FILE}"
    STATIC_TAG="static_"
fi

if [ "$FAST" == "0" ]; then
    tasks="--tasks piqa hellaswag arc_easy arc_challenge winogrande lambada_openai social_iqa openbookqa mmlu"

    ppl_t="wikitext2 ptb c4"
else
    tasks=""
    ppl_t="wikitext2"
fi

if [ "$REDO_GPTQ" == "1" ]; then
    GPTQ_DIR="/tmp/${SAVE_PREFIX}_${MODEL_NAME}_${QUANT_TAG}"
    if [ -z "$SAVE_PREFIX" ] || [ -z "$MODEL_NAME" ] || [ -z "$QUANT_TAG" ]; then
        echo "Error: refusing to rm -rf with empty path components"
        exit 1
    fi
    if [ -d "$GPTQ_DIR" ]; then
        echo "Removing cached GPTQ checkpoint: $GPTQ_DIR"
        rm -rf "$GPTQ_DIR"
    fi
fi

CUDA_VISIBLE_DEVICES=${GPU_ID} \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python main_for_test.py \
    --model ${MODEL} \
    ${ROTATION_FLAGS} \
    --gptq_checkpoint_path /tmp/${SAVE_PREFIX}_${MODEL_NAME}_${QUANT_TAG} \
    --cache_path /tmp/${STATIC_TAG}${SAVE_PREFIX}_${MODEL_NAME}_${QUANT_TAG}_results.pb \
    ${OVERWRITE_FLAG} \
    --w_groupsize ${GROUPSIZE} \
    --w_clip \
    --a_asym \
    --a_clip_ratio 0.9 \
    --w_bits ${W_BITS} \
    --a_bits ${A_BITS} \
    --k_bits ${KV_BITS} \
    --v_bits ${KV_BITS} \
    --k_groupsize ${K_GROUPSIZE} \
    --v_groupsize ${GROUPSIZE} \
    ${K_ASYM_FLAG} \
    ${V_ASYM_FLAG} \
    ${STATIC_ACT_FLAG} \
    --kv_ex ${KV_EX} \
    --proj_ex ${PROJ_EX} \
    --percdamp 0.1 \
    --no-w_ft \
    --ft_percdamp 0.0 \
    --save_name ${STATIC_TAG}${SAVE_PREFIX}_${QUANT_TAG} \
    --distribute \
    --ppl_eval \
    --ppl_eval_batch_size 1 \
    --ppl_eval_dataset ${ppl_t} \
    --lm_eval \
    --lm_eval_batch_size 2 \
    ${tasks}
