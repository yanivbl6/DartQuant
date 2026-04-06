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
  --sym            Use symmetric quantization for W/K/V (default: asymmetric K/V)
  --w_asym         Use asymmetric weight quantization  (default: symmetric W)
  --overwrite      Ignore cached results, re-run all
  --gptq           Delete cached GPTQ checkpoint and re-quantize
  --kv_ex N        K-cache quant without R3 rotation            (default: 0=off)
  --proj_ex N      Down-proj input quant without R4 rotation   (default: 0=off)
  --no_r4          Disable R4 rotation on down_proj            (default: off)
  --down_bits N    Override down_proj input activation bits     (default: a_bits)
  --eq             Enable per-channel equalization on down_proj (default: off)
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
  --acc_dtype S        Tier-2 accumulator dtype (e.g. fp16, int24)     (default: float)
  --quant_out MODE Output quantization: none, up, mlp, spec, speco, all, r4, res, mm, ex
  --smq N              Softmax output quantization bits (0=disabled, default: 0)
  --sd_check T         Compare static vs dynamic quantization per-layer (threshold T, 0=off)
  --sd_check_norm N    Norm for sd_check: 1, 2, or inf                   (default: inf)
  --selective-dyn P    Comma-separated layer patterns to force dynamic    (e.g., "v_proj,o_proj")
  --weights_stats F  Write weight sparsity stats to file F
  --gptq_strength F  GPTQ error-propagation strength (0.0–1.0, default: 1.0)
  --gguf TYPE      Use GGUF pre-quantized weights (scheme name like Q4_K_S, Q4_K_M, Q4_K_L, or path)
  --quant_warnings Warn on GGUF vs config quantization mismatches
  --wait           Wait for a clear GPU (polls every 20s, overrides -g)
  --max_used_mb N  Max used memory (MiB) for a GPU to be "clear"   (default: 200)
  --sim_version N  Simulation version tag for A/B comparisons       (default: 0=omitted)
  --fp32           Run model in float32 instead of float16 (isolate precision effects)
  --realint        Force real integer quantize/dequantize even at 16 bits
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
W_ASYM=0
OVERWRITE=0
FAST=0
STATIC_ACT=0
REDO_GPTQ=0
KV_EX=0
PROJ_EX=0
NO_R4=0
DOWN_BITS=""
EQ=0
PWL_ACT=0
PWL_N_SEGMENTS=9
PWL_INPUT_BITS=16
PWL_OUTPUT_BITS=16
PWL_NO_HW_SIM=0
INT_GEMM=0
ACC_BITS=32
ACC_BLOCK_K=32
ACC_WRAP=0
ACC_DTYPE="float"
SMQ=0
SD_CHECK=0
SD_CHECK_NORM="inf"
IG_COMPARE=0
SELECTIVE_DYN=""
WEIGHTS_STATS=""
GGUF=""
IMITATE_GGUF=""
GSCALER=""
GPTQ_STRENGTH=""
QUANT_WARNINGS=0
WAIT_GPU=0
MAX_USED_MB=200
SIM_VERSION=0
FP32=0
REALINT=0
QUANT_OUT="none"
LATE_ROT4=0
R4_STATS=""
R4_STATS_BATCHES=0
STOCHASTIC_QUANT=0

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
        --w_asym) W_ASYM=1;       shift   ;;
        --overwrite) OVERWRITE=1; shift   ;;
        --static-act) STATIC_ACT=1; shift ;;
        --gptq)   REDO_GPTQ=1;   shift   ;;
        --kv_ex)  KV_EX="$2";   shift 2 ;;
        --proj_ex) PROJ_EX="$2"; shift 2 ;;
        --no_r4)  NO_R4=1;     shift   ;;
        --down_bits) DOWN_BITS="$2"; shift 2 ;;
        --eq)     EQ=1;        shift   ;;
        --pwl_act) PWL_ACT=1;    shift   ;;
        --pwl_n_segments) PWL_N_SEGMENTS="$2"; shift 2 ;;
        --pwl_input_bits) PWL_INPUT_BITS="$2"; shift 2 ;;
        --pwl_output_bits) PWL_OUTPUT_BITS="$2"; shift 2 ;;
        --pwl_no_hw_sim) PWL_NO_HW_SIM=1; shift ;;
        --int_gemm)    INT_GEMM=1;      shift   ;;
        --acc_bits)    ACC_BITS="$2";    shift 2 ;;
        --acc_block_k) ACC_BLOCK_K="$2"; shift 2 ;;
        --acc_wrap)    ACC_WRAP=1;        shift   ;;
        --acc_dtype)   ACC_DTYPE="$2";   shift 2 ;;
        --smq)         SMQ="$2";         shift 2 ;;
        --sd_check)    SD_CHECK="$2";    shift 2 ;;
        --sd_check_norm) SD_CHECK_NORM="$2"; shift 2 ;;
        --ig_compare)  IG_COMPARE=1;  shift   ;;
        --selective-dyn) SELECTIVE_DYN="$2"; shift 2 ;;
        --weights_stats) WEIGHTS_STATS="$2"; shift 2 ;;
        --gguf)        GGUF="$2";           shift 2 ;;
        --imitate_gguf) IMITATE_GGUF="$2"; shift 2 ;;
        --gscaler)     GSCALER="$2";        shift 2 ;;
        --gptq_strength) GPTQ_STRENGTH="$2"; shift 2 ;;
        --quant_warnings) QUANT_WARNINGS=1; shift   ;;
        --sim_version) SIM_VERSION="$2";   shift 2 ;;
        --fp32)        FP32=1;             shift   ;;
        --realint)     REALINT=1;          shift   ;;
        --quant_out)   QUANT_OUT="$2";     shift 2 ;;
        --late_rot4)   LATE_ROT4=1;        shift   ;;
        --r4_stats)    R4_STATS="$2";      shift 2 ;;
        --r4_stats_batches) R4_STATS_BATCHES="$2"; shift 2 ;;
        --stochastic_quant) STOCHASTIC_QUANT=1; shift ;;
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
else
    W_ASYM_FLAG=""
    K_ASYM_FLAG="--k_asym"
    V_ASYM_FLAG="--v_asym"
fi

# --w_asym can be set independently (or via --sym=0 doesn't imply it)
if [ "$W_ASYM" == "1" ]; then
    W_ASYM_FLAG="--w_asym"
fi

MODEL_NAME=$(basename "${MODEL%/}")
SCRIPT_BASE="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_DIR="${SCRIPT_BASE}/data"

# --- Resolve GGUF path ---
# GGUF can be a quant scheme name (e.g. Q4_K_S, Q4_K_M, Q4_K_L) or an explicit path.
if [ -n "$GGUF" ] && [[ "$GGUF" != */* ]] && [[ "$GGUF" != *.gguf ]]; then
    # Scheme name -> look up in quantized_models/<ModelName>-<scheme>.gguf
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

OVERWRITE_FLAG=""
if [ "$OVERWRITE" == "1" ]; then
    OVERWRITE_FLAG="--overwrite"
fi

# --- Build flags for forwarding to Python ---
PWL_ACT_FLAG=""
if [ "$PWL_ACT" == "1" ]; then
    PWL_ACT_FLAG="--pwl_act --pwl_n_segments ${PWL_N_SEGMENTS} --pwl_input_bits ${PWL_INPUT_BITS} --pwl_output_bits ${PWL_OUTPUT_BITS}"
    if [ "$PWL_NO_HW_SIM" == "1" ]; then
        PWL_ACT_FLAG="${PWL_ACT_FLAG} --pwl_no_hw_sim"
    fi
fi

INT_GEMM_FLAG=""
if [ "$INT_GEMM" == "1" ]; then
    INT_GEMM_FLAG="--int_gemm --acc_bits ${ACC_BITS} --acc_block_k ${ACC_BLOCK_K}"
    if [ "$ACC_WRAP" == "1" ]; then
        INT_GEMM_FLAG="${INT_GEMM_FLAG} --acc_wrap"
    fi
    if [ "$ACC_DTYPE" != "float" ]; then
        INT_GEMM_FLAG="${INT_GEMM_FLAG} --acc_dtype ${ACC_DTYPE}"
    fi
fi

SMQ_FLAG=""
if [ "$SMQ" != "0" ]; then
    SMQ_FLAG="--smq ${SMQ}"
fi

GSCALER_FLAG=""
if [ -n "$GSCALER" ]; then
    GSCALER_FLAG="--gscaler ${GSCALER}"
fi

GPTQ_STRENGTH_FLAG=""
if [ -n "$GPTQ_STRENGTH" ]; then
    GPTQ_STRENGTH_FLAG="--gptq_strength ${GPTQ_STRENGTH}"
fi

FP32_FLAG=""
if [ "$FP32" == "1" ]; then
    FP32_FLAG="--fp32"
fi

REALINT_FLAG=""
if [ "$REALINT" == "1" ]; then
    REALINT_FLAG="--realint"
fi

QUANT_OUT_FLAG=""
if [ "$QUANT_OUT" != "none" ]; then
    QUANT_OUT_FLAG="--quant_out ${QUANT_OUT}"
fi

LATE_ROT4_FLAG=""
if [ "$LATE_ROT4" == "1" ]; then
    LATE_ROT4_FLAG="--late_rot4"
fi

STOCHASTIC_QUANT_FLAG=""
if [ "$STOCHASTIC_QUANT" == "1" ]; then
    STOCHASTIC_QUANT_FLAG="--stochastic_quant"
fi

R4_STATS_FLAG=""
if [ -n "$R4_STATS" ]; then
    mkdir -p "$(dirname "$R4_STATS")"
    R4_STATS_FLAG="--r4_stats ${R4_STATS}"
    if [ "$R4_STATS_BATCHES" != "0" ]; then
        R4_STATS_FLAG="${R4_STATS_FLAG} --r4_stats_batches ${R4_STATS_BATCHES}"
    fi
fi

GGUF_FLAG=""
GGUF_TAG_FLAG=""
QUANT_WARN_FLAG=""
if [ -n "$GGUF" ]; then
    GGUF_FLAG="--gguf_path ${GGUF}"
    GGUF_TAG_FLAG="--gguf ${GGUF}"
fi
IMITATE_GGUF_FLAG=""
IMITATE_GGUF_TAG_FLAG=""
if [ -n "$IMITATE_GGUF" ]; then
    # Resolve shorthand to path (same logic as GGUF)
    if [[ "$IMITATE_GGUF" != */* ]] && [[ "$IMITATE_GGUF" != *.gguf ]]; then
        IMITATE_GGUF_DIR="${DATA_DIR}/quantized_models"
        IMITATE_GGUF_FILE="${IMITATE_GGUF_DIR}/${MODEL_NAME}-${IMITATE_GGUF}.gguf"
        if [ ! -f "$IMITATE_GGUF_FILE" ]; then
            echo "Error: imitate_gguf file not found: ${IMITATE_GGUF_FILE}"
            exit 1
        fi
        IMITATE_GGUF="$IMITATE_GGUF_FILE"
        echo "Resolved imitate_gguf: $IMITATE_GGUF"
    fi
    IMITATE_GGUF_FLAG="--imitate_gguf ${IMITATE_GGUF}"
    IMITATE_GGUF_TAG_FLAG="--imitate_gguf ${IMITATE_GGUF}"
fi
if [ "$QUANT_WARNINGS" == "1" ]; then
    QUANT_WARN_FLAG="--quant_warnings"
fi

# --- Build quant tag via centralized Python function ---
TAG_ARGS="-w ${W_BITS} -a ${A_BITS} -k ${KV_BITS} -v ${V_BITS} -G ${GROUPSIZE} -m ${MODEL}"
[ "$SYM" == "1" ] && TAG_ARGS="${TAG_ARGS} --sym"
[ "$W_ASYM" == "1" ] && TAG_ARGS="${TAG_ARGS} --w_asym"
[ "$KV_EX" != "0" ] && TAG_ARGS="${TAG_ARGS} --kv_ex ${KV_EX}"
[ "$PROJ_EX" != "0" ] && TAG_ARGS="${TAG_ARGS} --proj_ex ${PROJ_EX}"
[ "$PROJ_EX" == "0" ] && [ "$NO_R4" == "1" ] && TAG_ARGS="${TAG_ARGS} --no_r4"
[ "$PROJ_EX" == "0" ] && [ -n "$DOWN_BITS" ] && TAG_ARGS="${TAG_ARGS} --down_bits ${DOWN_BITS}"
[ "$EQ" == "1" ] && TAG_ARGS="${TAG_ARGS} --eq"
[ -n "$PWL_ACT_FLAG" ] && TAG_ARGS="${TAG_ARGS} ${PWL_ACT_FLAG}"
[ -n "$INT_GEMM_FLAG" ] && TAG_ARGS="${TAG_ARGS} ${INT_GEMM_FLAG}"
[ -n "$SMQ_FLAG" ] && TAG_ARGS="${TAG_ARGS} ${SMQ_FLAG}"
[ -n "$GGUF_TAG_FLAG" ] && TAG_ARGS="${TAG_ARGS} ${GGUF_TAG_FLAG}"
[ -n "$IMITATE_GGUF_TAG_FLAG" ] && TAG_ARGS="${TAG_ARGS} ${IMITATE_GGUF_TAG_FLAG}"
[ -n "$GPTQ_STRENGTH_FLAG" ] && TAG_ARGS="${TAG_ARGS} ${GPTQ_STRENGTH_FLAG}"
[ -n "$GSCALER_FLAG" ] && TAG_ARGS="${TAG_ARGS} ${GSCALER_FLAG}"
[ "$SIM_VERSION" != "0" ] && TAG_ARGS="${TAG_ARGS} --sim_version ${SIM_VERSION}"
[ "$FP32" == "1" ] && TAG_ARGS="${TAG_ARGS} --fp32"
[ "$REALINT" == "1" ] && TAG_ARGS="${TAG_ARGS} --realint"
[ "$QUANT_OUT" != "none" ] && TAG_ARGS="${TAG_ARGS} --quant_out ${QUANT_OUT}"
[ "$LATE_ROT4" == "1" ] && TAG_ARGS="${TAG_ARGS} --late_rot4"
[ "$STOCHASTIC_QUANT" == "1" ] && TAG_ARGS="${TAG_ARGS} --stochastic_quant"
[ "$IG_COMPARE" == "1" ] && TAG_ARGS="${TAG_ARGS} --ig_compare"

SCRIPT_DIR_BASE="$(cd "$(dirname "$0")/../.." && pwd)"
QUANT_TAG=$(python "${SCRIPT_DIR_BASE}/experiment_config.py" ${TAG_ARGS})
if [ $? -ne 0 ] || [ -z "$QUANT_TAG" ]; then
    echo "Error: failed to build quant tag"
    exit 1
fi

# Build a separate cache tag for GPTQ checkpoints (excludes gptq_strength
# so all strength values share a single cached GPTQ run).
GPTQ_CACHE_TAG=$(python "${SCRIPT_DIR_BASE}/experiment_config.py" ${TAG_ARGS} --for_gptq_cache)
if [ $? -ne 0 ] || [ -z "$GPTQ_CACHE_TAG" ]; then
    GPTQ_CACHE_TAG="$QUANT_TAG"  # fallback
fi

# Build calibration cache tag (late_rot4 reuses normal R4 calibration scales)
CAL_TAG=$(python "${SCRIPT_DIR_BASE}/experiment_config.py" ${TAG_ARGS} --for_cal_cache)
if [ $? -ne 0 ] || [ -z "$CAL_TAG" ]; then
    CAL_TAG="$QUANT_TAG"  # fallback
fi

# --- Static vs Dynamic check flags ---
SD_CHECK_FLAG=""
if [ "$SD_CHECK" != "0" ]; then
    SD_CHECK_FLAG="--sd_check ${SD_CHECK} --sd_check_norm ${SD_CHECK_NORM}"
fi
IG_COMPARE_FLAG=""
if [ "$IG_COMPARE" = "1" ]; then
    IG_COMPARE_FLAG="--ig_compare"
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
    ACT_SCALES_FILE="${ACT_SCALES_DIR}/${SAVE_PREFIX}_${CAL_TAG}.pt"
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
    GPTQ_DIR="${DATA_DIR}/gptq_checkpoints/${SAVE_PREFIX}_${MODEL_NAME}_${GPTQ_CACHE_TAG}"
    if [ -z "$SAVE_PREFIX" ] || [ -z "$MODEL_NAME" ] || [ -z "$GPTQ_CACHE_TAG" ]; then
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
    --gptq_checkpoint_path ${DATA_DIR}/gptq_checkpoints/${SAVE_PREFIX}_${MODEL_NAME}_${GPTQ_CACHE_TAG} \
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
    ${W_ASYM_FLAG} \
    ${K_ASYM_FLAG} \
    ${V_ASYM_FLAG} \
    ${STATIC_ACT_FLAG} \
    ${PWL_ACT_FLAG} \
    ${INT_GEMM_FLAG} \
    ${SMQ_FLAG} \
    ${SD_CHECK_FLAG} \
    ${IG_COMPARE_FLAG} \
    ${SELECTIVE_DYN_FLAG} \
    ${WEIGHTS_STATS_FLAG} \
    ${GGUF_FLAG} \
    ${IMITATE_GGUF_FLAG} \
    ${QUANT_WARN_FLAG} \
    ${GSCALER_FLAG} \
    ${GPTQ_STRENGTH_FLAG} \
    ${FP32_FLAG} \
    ${REALINT_FLAG} \
    ${QUANT_OUT_FLAG} \
    ${LATE_ROT4_FLAG} \
    ${R4_STATS_FLAG} \
    ${STOCHASTIC_QUANT_FLAG} \
    --kv_ex ${KV_EX} \
    --proj_ex ${PROJ_EX} \
    $([ "$NO_R4" == "1" ] && echo "--no_r4") \
    $([ -n "$DOWN_BITS" ] && echo "--down_bits ${DOWN_BITS}") \
    $([ "$EQ" == "1" ] && echo "--eq") \
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
