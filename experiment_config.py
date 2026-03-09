"""Shared configuration and argument definitions for DartQuant scripts.

Imported by:
  - calibrater/multi_calibration.py
  - calibrater/calibrate_act_scales.py
  - fake_quant/run_experiments.py
"""

import argparse
import glob
import os

# ── Model paths ──────────────────────────────────────────────────────────────

MODEL_BASE = "/data/users/sashas/LLMC/Models/meta-llama"
MODEL_MAP = {
    '1b': f'{MODEL_BASE}/Llama-3.2-1B-Instruct',
    '3b': f'{MODEL_BASE}/Llama-3.2-3B-Instruct',
    '7b': f'{MODEL_BASE}/Llama-2-7b-hf',
}

# ── Trained rotation lookup ──────────────────────────────────────────────────

R1_PATHS = {
    'Llama-2-7b-hf':         '../data/trained_rotation/wikitext2_128samples/r1/sgd.0.0015.0.9.10.64.0.1.1',
    'Llama-3.2-1B-Instruct': '../data/trained_rotation/wikitext2_128samples/Llama-3.2-1B-Instruct/r1/sgd.0.0015.0.9.10.64.0.1.1',
    'Llama-3.2-3B-Instruct': '../data/trained_rotation/wikitext2_128samples/Llama-3.2-3B-Instruct/r1/sgd.0.0015.0.9.10.64.0.1.1',
}
R2_PATHS = {
    'Llama-2-7b-hf':         '../data/trained_rotation/wikitext2_128samples/r2/sgd.0.001.0.9.10.64.2',
    'Llama-3.2-1B-Instruct': '../data/trained_rotation/wikitext2_128samples/Llama-3.2-1B-Instruct/r2/sgd.0.001.0.9.10.64.2',
    'Llama-3.2-3B-Instruct': '../data/trained_rotation/wikitext2_128samples/Llama-3.2-3B-Instruct/r2/sgd.0.001.0.9.10.64.2',
}


def resolve_r_path(base_path):
    """Append the .pt file inside the directory."""
    if base_path and '.pt' not in base_path and '.bin' not in base_path:
        return base_path + '/' + base_path.split('/')[-1] + '.pt'
    return base_path


def r1r2_exist(model_name):
    """Check whether pre-trained R1 and R2 .pt files exist.

    NOTE: paths are relative -- call from calibrater/ directory.
    """
    r1_base = R1_PATHS.get(model_name)
    r2_base = R2_PATHS.get(model_name)
    if not r1_base or not r2_base:
        return False
    return os.path.isfile(resolve_r_path(r1_base)) and os.path.isfile(resolve_r_path(r2_base))


def resolve_model(model_short):
    """Resolve model shorthand (1b, 3b, 7b) to full path."""
    return MODEL_MAP.get(model_short, model_short)


def model_name_from_path(model_path):
    """Extract model name from a full path."""
    return os.path.basename(model_path.rstrip('/'))


# ── GGUF resolution ──────────────────────────────────────────────────────────

GGUF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'quantized_models')


def resolve_gguf_path(gguf_arg, model_path):
    """Resolve --gguf value to an actual .gguf file path.

    gguf_arg can be:
      - None      -> return None
      - 'Q4_K_M'  -> lookup ../quantized_models/<ModelName>-Q4_K_M.gguf
      - 'Q4_K_S'  -> lookup ../quantized_models/<ModelName>-Q4_K_S.gguf
      - '/path/to/file.gguf' -> return as-is
    """
    if gguf_arg is None:
        return None
    # If it looks like a path (contains / or ends with .gguf), use directly
    if '/' in gguf_arg or gguf_arg.endswith('.gguf'):
        return gguf_arg
    # Otherwise treat as a quant-type shorthand (e.g. Q4_K_M, Q4_K_S)
    model_name = model_name_from_path(model_path)
    candidate = os.path.join(GGUF_DIR, f'{model_name}-{gguf_arg}.gguf')
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(
        f'No GGUF file found: {candidate}\n'
        f'Available: {glob.glob(os.path.join(GGUF_DIR, f"{model_name}*.gguf"))}')


# ── Default experiment definitions ───────────────────────────────────────────
# (name, mode, extra_flags_for_shell_script)

EXPERIMENTS = [
    ("full",             "full",     []),
    ("baseline",         "baseline", []),
    ("quarot",           "quarot",   []),
    ("dart",             "dart",     []),
    ("baseline_static",  "baseline", ["--static-act"]),
    ("quarot_static",    "quarot",   ["--static-act"]),
    ("dart_static",      "dart",     ["--static-act"]),
]


# ── Shared argument helpers ──────────────────────────────────────────────────

def add_model_arg(parser):
    """Add -m/--model argument."""
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model path, HF name, or shorthand: 1b, 3b, 7b')


def add_quant_args(parser):
    """Add quantization arguments shared across experiment / calibration scripts."""
    parser.add_argument('-w', '--w_bits', type=int, default=4,
                        help='Weight bit-width (default: 4)')
    parser.add_argument('-a', '--a_bits', type=int, default=8,
                        help='Activation bit-width (default: 8)')
    parser.add_argument('-k', '--k_bits', type=int, default=4,
                        help='K-cache bit-width (default: 4)')
    parser.add_argument('-v', '--v_bits', type=int, default=None,
                        help='V-cache bit-width (default: same as -k)')
    parser.add_argument('-G', '--groupsize', type=int, default=128,
                        help='Group size for W, K, V (default: 128)')
    parser.add_argument('--sym', action='store_true',
                        help='Symmetric quantization for W/K/V')
    parser.add_argument('--kv_ex', type=int, default=0,
                        help='K-cache quant without R3 rotation (0=off)')
    parser.add_argument('--proj_ex', type=int, default=0,
                        help='Down-proj input quant without R4 rotation (0=off)')

    # PWL activation
    parser.add_argument('--pwl_act', action='store_true',
                        help='Use PWL activation approximation')
    parser.add_argument('--pwl_n_segments', type=int, default=9)
    parser.add_argument('--pwl_input_bits', type=int, default=16)
    parser.add_argument('--pwl_output_bits', type=int, default=16)
    parser.add_argument('--pwl_no_hw_sim', action='store_true',
                        help='Disable HW precision simulation')

    # Integer GEMM
    parser.add_argument('--int_gemm', action='store_true',
                        help='Integer GEMM with capped accumulator')
    parser.add_argument('--acc_bits', type=int, default=32)
    parser.add_argument('--acc_block_k', type=int, default=32)
    parser.add_argument('--acc_wrap', action='store_true',
                        help='Wrap-around instead of saturation')

    # Softmax Output Quantization
    parser.add_argument('--smq', type=int, default=0,
                        help='Softmax output quantization bits (0=disabled)')

    # GGUF pre-quantized weights
    parser.add_argument('--gguf', type=str, default=None,
                        help='Use GGUF pre-quantized weights. Pass a quant type '
                             '(e.g. Q4_K_M, Q4_K_S) to lookup from quantized_models/, '
                             'or an explicit path to a .gguf file.')
    parser.add_argument('--quant_warnings', action='store_true',
                        help='Warn when quantization params mismatch GGUF tensor specs')

    # Weight stats
    parser.add_argument('--weights_stats', type=str, default=None,
                        help='Base path for weight sparsity stats (mode suffix added automatically)')


def resolve_v_bits(args):
    """Default v_bits to k_bits when not explicitly given."""
    if args.v_bits is None:
        args.v_bits = args.k_bits


def build_quant_args(args):
    """Serialize parsed quant args back to a CLI arg list for forwarding."""
    cmd = ['-w', str(args.w_bits),
           '-a', str(args.a_bits),
           '-k', str(args.k_bits),
           '-v', str(args.v_bits),
           '-G', str(args.groupsize)]
    if args.sym:
        cmd.append('--sym')
    if args.kv_ex:
        cmd += ['--kv_ex', str(args.kv_ex)]
    if args.proj_ex:
        cmd += ['--proj_ex', str(args.proj_ex)]
    if args.pwl_act:
        cmd += ['--pwl_act',
                '--pwl_n_segments', str(args.pwl_n_segments),
                '--pwl_input_bits', str(args.pwl_input_bits),
                '--pwl_output_bits', str(args.pwl_output_bits)]
        if args.pwl_no_hw_sim:
            cmd.append('--pwl_no_hw_sim')
    if args.int_gemm:
        cmd += ['--int_gemm',
                '--acc_bits', str(args.acc_bits),
                '--acc_block_k', str(args.acc_block_k)]
        if args.acc_wrap:
            cmd.append('--acc_wrap')
    if getattr(args, 'smq', 0) > 0:
        cmd += ['--smq', str(args.smq)]
    if getattr(args, 'gguf', None):
        cmd += ['--gguf', args.gguf]
    if getattr(args, 'quant_warnings', False):
        cmd.append('--quant_warnings')
    if getattr(args, 'weights_stats', None):
        cmd += ['--weights_stats', args.weights_stats]
    return cmd
