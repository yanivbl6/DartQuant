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
  -k KV_BITS       K-cache bit-width                    (default: 4)
  -v V_BITS        V-cache bit-width                    (default: same as -k)
  -G GROUPSIZE     Group size for W, K, V              (default: 128)
  --sym            Use symmetric quantization for W/K/V (default: asymmetric)
  --overwrite      Ignore cached results, re-run all
  --gptq           Delete cached GPTQ checkpoint and re-quantize
  --kv_ex N        K-cache quant without R3 rotation            (default: 0=off)
  --proj_ex N      Down-proj input quant without R4 rotation   (default: 0=off)
  --static-act     Use pre-calibrated static activation scales
  --pwl_act            Replace activations with PWL approximation
  --pwl_n_segments N   Number of PWL segments                     (default: 9)
  --pwl_input_bits N   PWL input quantizer bit-width              (default: 16)
  --pwl_output_bits N  PWL output quantizer bit-width             (default: 16)
  --pwl_no_hw_sim      Disable HW precision simulation (pure float PWL)
  --int_gemm           Use integer GEMM with capped accumulator
  --acc_bits N         Accumulator bit-width                         (default: 32)
  --acc_block_k N      K-block size for accumulator capping          (default: 32)
  --acc_wrap           Use wrap-around instead of saturation on overflow
  --smq N              Softmax output quantization bits (0=disabled, default: 0)
  --sd_check T         Compare static vs dynamic quantization per-layer (threshold T, 0=off)
  --sd_check_norm N    Norm for sd_check: 1, 2, or inf                   (default: inf)
  --selective-dyn P    Comma-separated layer patterns to force dynamic    (e.g., "v_proj,o_proj")
  --weights_stats F  Write weight sparsity stats to file F
  --gguf TYPE      Use GGUF pre-quantized weights (e.g. Q4_K_M, Q4_K_S, or path)
  --quant_warnings Warn on GGUF vs config quantization mismatches
  --wait           Wait for a clear GPU (polls every 20s, overrides -g)
  --max_used_mb N  Max used memory (MiB) for a GPU to be "clear"   (default: 200)
  -F               Fast mode (skip lm_eval tasks)
  --very-fast      Very fast mode (skip lm_eval, PPL on wikitext2 only)
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
V_BITS=""
GROUPSIZE=128
SYM=0
OVERWRITE=0
FAST=0
STATIC_ACT=0
REDO_GPTQ=0
KV_EX=0
PROJ_EX=0
PWL_ACT=0
PWL_N_SEGMENTS=9
PWL_INPUT_BITS=16
PWL_OUTPUT_BITS=16
PWL_NO_HW_SIM=0
INT_GEMM=0
ACC_BITS=32
ACC_BLOCK_K=32
ACC_WRAP=0
SMQ=0
SD_CHECK=0
SD_CHECK_NORM="inf"
SELECTIVE_DYN=""
WEIGHTS_STATS=""
GGUF=""
QUANT_WARNINGS=0
WAIT_GPU=0
MAX_USED_MB=200

# --- Parse options ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        -g)       GPU_ID="$2";    shift 2 ;;
        -m)       MODEL="$2";     shift 2 ;;
        -w)       W_BITS="$2";    shift 2 ;;
        -a)       A_BITS="$2";    shift 2 ;;
        -k)       KV_BITS="$2";   shift 2 ;;
        -v)       V_BITS="$2";    shift 2 ;;
        -G)       GROUPSIZE="$2"; shift 2 ;;
        --sym)    SYM=1;          shift   ;;
        --overwrite) OVERWRITE=1; shift   ;;
        --static-act) STATIC_ACT=1; shift ;;
        --gptq)   REDO_GPTQ=1;   shift   ;;
        --kv_ex)  KV_EX="$2";   shift 2 ;;
        --proj_ex) PROJ_EX="$2"; shift 2 ;;
        --pwl_act) PWL_ACT=1;    shift   ;;
        --pwl_n_segments) PWL_N_SEGMENTS="$2"; shift 2 ;;
        --pwl_input_bits) PWL_INPUT_BITS="$2"; shift 2 ;;
        --pwl_output_bits) PWL_OUTPUT_BITS="$2"; shift 2 ;;
        --pwl_no_hw_sim) PWL_NO_HW_SIM=1; shift ;;
        --int_gemm)    INT_GEMM=1;      shift   ;;
        --acc_bits)    ACC_BITS="$2";    shift 2 ;;
        --acc_block_k) ACC_BLOCK_K="$2"; shift 2 ;;
        --acc_wrap)    ACC_WRAP=1;        shift   ;;
        --smq)         SMQ="$2";         shift 2 ;;
        --sd_check)    SD_CHECK="$2";    shift 2 ;;
        --sd_check_norm) SD_CHECK_NORM="$2"; shift 2 ;;
        --selective-dyn) SELECTIVE_DYN="$2"; shift 2 ;;
        --weights_stats) WEIGHTS_STATS="$2"; shift 2 ;;
        --gguf)        GGUF="$2";           shift 2 ;;
        --quant_warnings) QUANT_WARNINGS=1; shift   ;;
        --wait)        WAIT_GPU=1;         shift   ;;
        --max_used_mb) MAX_USED_MB="$2";   shift 2 ;;
        -F|--fast) FAST=1;        shift   ;;
        --very-fast) FAST=2;     shift   ;;
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

# --- Default V_BITS to KV_BITS if not explicitly set ---
V_BITS=${V_BITS:-$KV_BITS}

# --- Full mode overrides ---
if [ "$MODE" == "full" ]; then
    W_BITS=16
    A_BITS=16
    KV_BITS=16
    V_BITS=16
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
SCRIPT_BASE="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_DIR="${SCRIPT_BASE}/data"

# --- Resolve GGUF path ---
# GGUF can be a quant-type shorthand (Q4_K_M, Q4_K_S) or an explicit path.
if [ -n "$GGUF" ] && [[ "$GGUF" != */* ]] && [[ "$GGUF" != *.gguf ]]; then
    # Shorthand like Q4_K_M -> look up in quantized_models/
    GGUF_DIR="${DATA_DIR}/quantized_models"
    GGUF_FILE="${GGUF_DIR}/${MODEL_NAME}-${GGUF}.gguf"
    if [ ! -f "$GGUF_FILE" ]; then
        echo "Error: GGUF file not found: ${GGUF_FILE}"
        echo "Available: $(ls ${GGUF_DIR}/${MODEL_NAME}*.gguf 2>/dev/null)"
        exit 1
    fi
    GGUF="$GGUF_FILE"
    echo "Resolved GGUF: $GGUF"
fi

# K_GROUPSIZE follows GROUPSIZE; Python clamps to valid divisor of head_dim if needed
K_GROUPSIZE=$GROUPSIZE

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
QUANT_TAG="w${W_BITS}a${A_BITS}k${KV_BITS}v${V_BITS}_g${GROUPSIZE}_aAsym_${SYM_TAG}"
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

# --- PWL activation flags & tag (mirrors pwl_utils.pwl_tag()) ---
PWL_ACT_FLAG=""
if [ "$PWL_ACT" == "1" ]; then
    PWL_ACT_FLAG="--pwl_act --pwl_n_segments ${PWL_N_SEGMENTS} --pwl_input_bits ${PWL_INPUT_BITS} --pwl_output_bits ${PWL_OUTPUT_BITS}"
    PWL_TAG="_pwl"
    if [ "$PWL_N_SEGMENTS" != "9" ]; then
        PWL_TAG="${PWL_TAG}_${PWL_N_SEGMENTS}p"
    fi
    if [ "$PWL_INPUT_BITS" != "16" ]; then
        PWL_TAG="${PWL_TAG}_in${PWL_INPUT_BITS}"
    fi
    if [ "$PWL_OUTPUT_BITS" != "16" ]; then
        PWL_TAG="${PWL_TAG}_out${PWL_OUTPUT_BITS}"
    fi
    if [ "$PWL_NO_HW_SIM" == "1" ]; then
        PWL_ACT_FLAG="${PWL_ACT_FLAG} --pwl_no_hw_sim"
        PWL_TAG="${PWL_TAG}_nohw"
    fi
    QUANT_TAG="${QUANT_TAG}${PWL_TAG}"
fi

# --- Integer GEMM flags & tag (mirrors int_acc_gemm.int_gemm_tag()) ---
INT_GEMM_FLAG=""
if [ "$INT_GEMM" == "1" ]; then
    INT_GEMM_FLAG="--int_gemm --acc_bits ${ACC_BITS} --acc_block_k ${ACC_BLOCK_K}"
    INTGEMM_TAG="_intgemm"
    if [ "$ACC_BITS" != "32" ]; then
        INTGEMM_TAG="${INTGEMM_TAG}_acc${ACC_BITS}"
    fi
    if [ "$ACC_BLOCK_K" != "32" ]; then
        INTGEMM_TAG="${INTGEMM_TAG}_bk${ACC_BLOCK_K}"
    fi
    if [ "$ACC_WRAP" == "1" ]; then
        INT_GEMM_FLAG="${INT_GEMM_FLAG} --acc_wrap"
        INTGEMM_TAG="${INTGEMM_TAG}_wrap"
    fi
    QUANT_TAG="${QUANT_TAG}${INTGEMM_TAG}"
fi

# --- SMQ tag ---
if [ "$SMQ" != "0" ]; then
    QUANT_TAG="${QUANT_TAG}_smq${SMQ}"
fi

# --- GGUF tag ---
GGUF_FLAG=""
QUANT_WARN_FLAG=""
if [ -n "$GGUF" ]; then
    GGUF_BASENAME=$(basename "$GGUF" .gguf)
    GGUF_QTYPE=$(echo "$GGUF_BASENAME" | grep -oP '(?<=-)(Q[^.]+)$' || echo "gguf")
    # Use dashes in tag: gguf-Q4-K-M
    GGUF_TAG=$(echo "$GGUF_QTYPE" | tr '_' '-')
    QUANT_TAG="${QUANT_TAG}_gguf-${GGUF_TAG}"
    GGUF_FLAG="--gguf_path ${GGUF}"
fi
if [ "$QUANT_WARNINGS" == "1" ]; then
    QUANT_WARN_FLAG="--quant_warnings"
fi

# --- SMQ flags ---
SMQ_FLAG=""
if [ "$SMQ" != "0" ]; then
    SMQ_FLAG="--smq ${SMQ}"
fi

# --- Static vs Dynamic check flags ---
SD_CHECK_FLAG=""
if [ "$SD_CHECK" != "0" ]; then
    SD_CHECK_FLAG="--sd_check ${SD_CHECK} --sd_check_norm ${SD_CHECK_NORM}"
fi

# --- Selective dynamic flags ---
SELECTIVE_DYN_FLAG=""
if [ -n "$SELECTIVE_DYN" ]; then
    SELECTIVE_DYN_FLAG="--selective-dyn ${SELECTIVE_DYN}"
    STATIC_TAG="seldyn_"
fi

# --- Weight stats flag (append _<MODE> before extension) ---
WEIGHTS_STATS_FLAG=""
if [ -n "$WEIGHTS_STATS" ]; then
    WS_EXT="${WEIGHTS_STATS##*.}"
    WS_BASE="${WEIGHTS_STATS%.*}"
    if [ "$WS_EXT" != "$WEIGHTS_STATS" ]; then
        WS_FILE="${WS_BASE}_${MODE}.${WS_EXT}"
    else
        WS_FILE="${WEIGHTS_STATS}_${MODE}"
    fi
    WEIGHTS_STATS_FLAG="--weights_stats ${WS_FILE}"
fi

# --- Static activation scales (after all tag components are finalized) ---
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
    LM_EVAL_FLAG="--lm_eval"
    ppl_t="wikitext2 ptb c4"
elif [ "$FAST" == "2" ]; then
    tasks=""
    LM_EVAL_FLAG=""
    ppl_t="wikitext2"
else
    tasks=""
    LM_EVAL_FLAG=""
    ppl_t="wikitext2 ptb c4"
fi

if [ "$REDO_GPTQ" == "1" ]; then
    GPTQ_DIR="${DATA_DIR}/gptq_checkpoints/${SAVE_PREFIX}_${MODEL_NAME}_${QUANT_TAG}"
    if [ -z "$SAVE_PREFIX" ] || [ -z "$MODEL_NAME" ] || [ -z "$QUANT_TAG" ]; then
        echo "Error: refusing to rm -rf with empty path components"
        exit 1
    fi
    if [ -d "$GPTQ_DIR" ]; then
        echo "Removing cached GPTQ checkpoint: $GPTQ_DIR"
        rm -rf "$GPTQ_DIR"
    fi
fi

# --- Wait for a clear GPU if requested ---
if [ "$WAIT_GPU" == "1" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    GPU_WAIT_SCRIPT="${SCRIPT_DIR}/../../utils/gpu_wait.py"
    echo "Waiting for a clear GPU (max_used_mb=${MAX_USED_MB}) ..."
    GPU_ID=$(python "$GPU_WAIT_SCRIPT" --max_used_mb "$MAX_USED_MB")
    echo "Selected GPU ${GPU_ID}"
fi

CUDA_VISIBLE_DEVICES=${GPU_ID} \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python main_for_test.py \
    --model ${MODEL} \
    ${ROTATION_FLAGS} \
    --gptq_checkpoint_path ${DATA_DIR}/gptq_checkpoints/${SAVE_PREFIX}_${MODEL_NAME}_${QUANT_TAG} \
    --cache_path ${DATA_DIR}/cached_results/${STATIC_TAG}${SAVE_PREFIX}_${MODEL_NAME}_${QUANT_TAG}_results.pb \
    ${OVERWRITE_FLAG} \
    --w_groupsize ${GROUPSIZE} \
    --w_clip \
    --a_asym \
    --a_clip_ratio 0.9 \
    --w_bits ${W_BITS} \
    --a_bits ${A_BITS} \
    --k_bits ${KV_BITS} \
    --v_bits ${V_BITS} \
    --k_groupsize ${K_GROUPSIZE} \
    --v_groupsize ${GROUPSIZE} \
    ${K_ASYM_FLAG} \
    ${V_ASYM_FLAG} \
    ${STATIC_ACT_FLAG} \
    ${PWL_ACT_FLAG} \
    ${INT_GEMM_FLAG} \
    ${SMQ_FLAG} \
    ${SD_CHECK_FLAG} \
    ${SELECTIVE_DYN_FLAG} \
    ${WEIGHTS_STATS_FLAG} \
    ${GGUF_FLAG} \
    ${QUANT_WARN_FLAG} \
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
    ${LM_EVAL_FLAG} \
    --lm_eval_batch_size 2 \
    ${tasks}
