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

GGUF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'quantized_models')


def resolve_gguf_path(gguf_arg, model_path):
    """Resolve --gguf value to an actual .gguf file path.

    gguf_arg can be:
      - None      -> return None
      - 'Q4_K_M'  -> lookup data/quantized_models/<ModelName>-Q4_K_M.gguf
      - 'Q4_K_S'  -> lookup data/quantized_models/<ModelName>-Q4_K_S.gguf
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
    parser.add_argument('--w_asym', action='store_true',
                        help='Asymmetric weight quantization (default: symmetric)')
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

    # GGUF imitation (per-layer bit-width matching)
    parser.add_argument('--imitate_gguf', type=str, default=None,
                        help='Match per-layer weight bit-widths from a GGUF file. '
                             'Pass a quant type (e.g. Q4_K_M) or explicit .gguf path. '
                             'Does NOT load actual GGUF weights (use --gguf for that).')

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

    # GPTQ strength
    parser.add_argument('--gptq_strength', type=float, default=1.0,
                        help='GPTQ error-propagation strength (0.0=no compensation, 1.0=full). '
                             'Values < 1.0 weaken GPTQ for bring-your-own-weights flows.')

    # Group scale quantization
    parser.add_argument('--gscaler', type=str, default=None,
                        help='Group scale format: M5S3, M6E4b2, M6S4l2, etc. '
                             '(default: None = FP32 scales)')


def resolve_v_bits(args):
    """Default v_bits to k_bits when not explicitly given."""
    if args.v_bits is None:
        args.v_bits = args.k_bits


# ── Quant-tag and path helpers ───────────────────────────────────────────────

def build_quant_tag(args, for_gptq_cache=False):
    """Build the canonical quant-tag string from parsed args.

    This is the SINGLE SOURCE OF TRUTH for tag construction.
    Called by: calibrate_act_scales.py, analyze_scales.py, run_experiments.py,
    and dart_gptq_wxaykvz.sh (via ``python experiment_config.py``).

    If *for_gptq_cache* is True, the gptq_strength suffix is omitted so that
    all strength values share the same GPTQ checkpoint cache.
    """
    w_asym = getattr(args, 'w_asym', False) and not args.sym
    if args.sym:
        sym_tag = "wSym_kSym_vSym"
    elif w_asym:
        sym_tag = "wAsym_kAsym_vAsym"
    else:
        sym_tag = "kAsym_vAsym"
    w_tag = "w0" if getattr(args, 'imitate_gguf', None) else f"w{args.w_bits}"
    tag = f"{w_tag}a{args.a_bits}k{args.k_bits}v{args.v_bits}_g{args.groupsize}_aAsym_{sym_tag}"
    if args.kv_ex != 0:
        tag += f"_kvex{args.kv_ex}"
    if args.proj_ex != 0:
        tag += f"_projex{args.proj_ex}"
    # PWL activation tag
    if getattr(args, 'pwl_act', False):
        parts = ["_pwl"]
        if getattr(args, 'pwl_n_segments', 9) != 9:
            parts.append(f"{args.pwl_n_segments}p")
        if getattr(args, 'pwl_input_bits', 16) != 16:
            parts.append(f"in{args.pwl_input_bits}")
        if getattr(args, 'pwl_output_bits', 16) != 16:
            parts.append(f"out{args.pwl_output_bits}")
        if getattr(args, 'pwl_no_hw_sim', False):
            parts.append("nohw")
        tag += "_".join(parts)
    # Integer GEMM tag
    if getattr(args, 'int_gemm', False):
        parts = ["_intgemm"]
        if getattr(args, 'acc_bits', 32) != 32:
            parts.append(f"acc{args.acc_bits}")
        if getattr(args, 'acc_block_k', 32) != 32:
            parts.append(f"bk{args.acc_block_k}")
        if getattr(args, 'acc_wrap', False):
            parts.append("wrap")
        tag += "_".join(parts)
    # SMQ tag
    if getattr(args, 'smq', 0) > 0:
        tag += f"_smq{args.smq}"
    # GGUF tag
    if getattr(args, 'gguf', None):
        gguf_path = resolve_gguf_path(args.gguf, args.model)
        basename = os.path.basename(gguf_path).replace('.gguf', '')
        parts = basename.split('-')
        qtype = '-'.join(p for p in parts if p.startswith('Q')) or 'gguf'
        tag += f"_gguf-{qtype.replace('_', '-')}"
    # Imitate-GGUF tag
    if getattr(args, 'imitate_gguf', None):
        gguf_path = resolve_gguf_path(args.imitate_gguf, args.model)
        basename = os.path.basename(gguf_path).replace('.gguf', '')
        parts = basename.split('-')
        first_q = next((i for i, p in enumerate(parts) if p.startswith('Q')), None)
        imit_label = '-'.join(parts[first_q:]) if first_q is not None else 'gguf'
        tag += f"_imitate-{imit_label.replace('_', '-')}"
    # GPTQ strength tag (omitted for GPTQ cache so all strengths share one checkpoint)
    if not for_gptq_cache:
        _gs = getattr(args, 'gptq_strength', 1.0)
        if _gs == 0.0:
            tag += "_no-gptq"
        elif _gs != 1.0:
            tag += f"_gptqs{round(_gs * 100)}"
    # Group scaler tag
    if getattr(args, 'gscaler', None):
        tag += f"_G-scaler-{args.gscaler}"
    return tag


_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def resolve_act_scales_path(model_path, mode, quant_tag):
    """Resolve calibration .pt file path."""
    model_name = model_name_from_path(model_path)
    return os.path.join(_DATA_DIR, 'act_scales', model_name, f'{mode}_{quant_tag}.pt')


def resolve_gptq_checkpoint_dir(model_path, mode, quant_tag, w_bits, imitate_gguf=False):
    """Resolve GPTQ checkpoint directory (containing .pth files)."""
    model_name = model_name_from_path(model_path)
    w_suffix = 'w0' if imitate_gguf else f'w{w_bits}'
    return os.path.join(
        _DATA_DIR, 'gptq_checkpoints',
        f'{mode}_{model_name}_{quant_tag}',
        f'{model_name}_{w_suffix}',
    )


def build_quant_args(args):
    """Serialize parsed quant args back to a CLI arg list for forwarding."""
    cmd = ['-w', str(args.w_bits),
           '-a', str(args.a_bits),
           '-k', str(args.k_bits),
           '-v', str(args.v_bits),
           '-G', str(args.groupsize)]
    if args.sym:
        cmd.append('--sym')
    if getattr(args, 'w_asym', False):
        cmd.append('--w_asym')
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
    if getattr(args, 'imitate_gguf', None):
        cmd += ['--imitate_gguf', args.imitate_gguf]
    if getattr(args, 'gguf', None):
        cmd += ['--gguf', args.gguf]
    if getattr(args, 'quant_warnings', False):
        cmd.append('--quant_warnings')
    if getattr(args, 'weights_stats', None):
        cmd += ['--weights_stats', args.weights_stats]
    if getattr(args, 'gptq_strength', 1.0) != 1.0:
        cmd += ['--gptq_strength', str(args.gptq_strength)]
    if getattr(args, 'gscaler', None):
        cmd += ['--gscaler', args.gscaler]
    return cmd


# ── CLI entry point — print quant tag from args ─────────────────────────────
# Used by dart_gptq_wxaykvz.sh:  QUANT_TAG=$(python experiment_config.py ...)

if __name__ == '__main__':
    import argparse as _ap
    _parser = _ap.ArgumentParser(description='Print the quant tag for the given args')
    add_model_arg(_parser)
    add_quant_args(_parser)
    _parser.add_argument('--for_gptq_cache', action='store_true',
                         help='Omit gptq_strength from tag (for shared GPTQ cache)')
    _args = _parser.parse_args()
    resolve_v_bits(_args)
    _args.model = resolve_model(_args.model)
    print(build_quant_tag(_args, for_gptq_cache=_args.for_gptq_cache))
