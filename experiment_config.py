"""Shared configuration and argument definitions for DartQuant scripts.

Imported by:
  - calibrater/multi_calibration.py
  - calibrater/calibrate_act_scales.py
  - fake_quant/run_experiments.py
"""

import argparse
import glob
import os
import re

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
      - '<scheme>' -> lookup data/quantized_models/<ModelName>-<scheme>.gguf
                      (e.g. Q4_K_M, Q4_K_S, Q4_K_L, ...)
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
    parser.add_argument('--weight_group_mode', type=str, default='all',
                        choices=['all', 'down', 'down_o'],
                        help='Per-Linear weight groupsize policy. '
                             '"all" (default): every Linear uses --groupsize. '
                             '"down": only down_proj uses --groupsize; all '
                             'other Linears go per-channel (-1). '
                             '"down_o": down_proj AND o_proj use --groupsize; '
                             'all other Linears go per-channel (-1).')
    parser.add_argument('--sym', action='store_true',
                        help='Symmetric quantization for W/K/V')
    parser.add_argument('--w_asym', action='store_true',
                        help='Asymmetric weight quantization (default: symmetric)')
    parser.add_argument('--fp4', type=str, default='none',
                        choices=['all', 'down', 'none'],
                        help='FP4 weight quantization: all / down (down_proj only) / none (default)')
    parser.add_argument('--kv_ex', type=int, default=0,
                        help='K-cache quant without R3 rotation (0=off)')
    parser.add_argument('--proj_ex', type=int, default=0,
                        help='Down-proj input quant without R4 rotation (0=off). Shorthand for --no_r4 --down_bits X')
    parser.add_argument('--no_r4', action='store_true',
                        help='Disable R4 rotation on down_proj (without changing bits)')
    parser.add_argument('--late_rot4', action='store_true', default=False,
                        help='Quantize weights first, then apply R4 rotation: Q(W)@H instead of Q(W@H). '
                             'Incompatible with --int_gemm on down_proj.')
    parser.add_argument('--down_bits', type=int, default=None,
                        help='Override down_proj input activation bits (without disabling R4)')
    parser.add_argument('--oproj_bits', type=int, default=None,
                        help='Override o_proj input activation bits (e.g. 16 for int16 decomposition)')
    parser.add_argument('--eq', action='store_true',
                        help='Enable per-channel equalization on down_proj inputs '
                             '(legacy: online division at down_proj input)')
    parser.add_argument('--ud_eq', action='store_true',
                        help='Branch equalization (up + down): per-channel scale on '
                             'up_proj output, folded into W_up rows and W_down cols. '
                             'Mutually exclusive with --eq and --ugd_eq.')
    parser.add_argument('--ugd_eq', action='store_true',
                        help='Branch equalization (up + gate + down): independent '
                             'per-channel scales on silu(gate) and up_proj output. '
                             'PWL path uses per-channel s_out; non-PWL path wraps '
                             'silu with an in-fp EqActivation. '
                             'Mutually exclusive with --eq and --ud_eq.')

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
    parser.add_argument('--acc_bits', type=int, default=16)
    parser.add_argument('--acc_block_k', type=int, default=32)
    parser.add_argument('--acc_wrap', action='store_true',
                        help='Wrap-around instead of saturation')
    parser.add_argument('--acc_dtype', type=str, default='float',
                        help='Tier-2 accumulator dtype (e.g. fp16, int24, int24a0). '
                             'Prefix with "w" for T2 wraparound (wint24a0).')
    parser.add_argument('--lsb_mac_shift', type=int, default=0,
                        help='Right-shift tl.dot by N bits in LSB int16 kernel (default: 0)')
    parser.add_argument('--t1_msb_scan', action='store_true',
                        help='Diagnostic: detect tier-1 accumulator overflow per layer (aborts on mismatch)')

    # Softmax Output Quantization
    parser.add_argument('--smq', type=int, default=0,
                        help='Softmax output quantization bits (0=disabled)')

    # GGUF imitation (per-layer bit-width matching)
    parser.add_argument('--imitate_gguf', type=str, default=None,
                        help='Match per-layer weight bit-widths from a GGUF file. '
                             'Pass a quant scheme name (e.g. Q4_K_S, Q4_K_M, Q4_K_L) '
                             'to lookup <ModelName>-<scheme>.gguf, or an explicit .gguf path. '
                             'Does NOT load actual GGUF weights (use --gguf for that).')

    # GGUF pre-quantized weights
    parser.add_argument('--gguf', type=str, default=None,
                        help='Use GGUF pre-quantized weights. Pass a quant scheme name '
                             '(e.g. Q4_K_S, Q4_K_M, Q4_K_L) to lookup '
                             '<ModelName>-<scheme>.gguf from quantized_models/, '
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
    parser.add_argument('--hwscale', type=str, default=None,
                        help='Merged (a_gscale*w_gscale) per-group scale snapped '
                             'to M<m>S<s>[b<z>|l<z>|bmin] at model load. Inference-only '
                             '(untouched: GPTQ/cal caches). bmin = per-layer auto bias.')

    # FP16 / scalewise calibration mode
    parser.add_argument('--fp16_calib', action='store_true', default=False,
                        help='Calibrate activation scales on the FP16 (pre-GPTQ) model '
                             'and use them at inference (replaces post-GPTQ calibration). '
                             'Cache key strips post-quantization-only flags so the same '
                             'FP16 cal file is shared across w_bits/groupsize/hwscale variants.')
    parser.add_argument('--scalewise', action='store_true', default=False,
                        help='Round the merged hwscale (group_scale * act_scale) to its '
                             'M<m>S<s> representation INSIDE the GPTQ loop, so '
                             'Hessian-aware compensation absorbs scale-rounding error. '
                             'Forces an FP16 calibration pre-pass; requires --hwscale.')
    parser.add_argument('--force_recalib', action='store_true', default=False,
                        help='Recompute activation calibration files even if a cached '
                             'copy already exists on disk.')

    # AdaQuant (alternative to GPTQ)
    parser.add_argument('--adaquant', type=str, nargs='?', const='default', default=None,
                        help='Use AdaQuant instead of GPTQ. No value = defaults. '
                             'Inline params string to customise '
                             '(e.g., "lr.0.001_ep.20_optWSX_adam_cos")')

    # GPTAQ — closed-form FP-target variant of GPTQ
    parser.add_argument('--gptaq', action='store_true', default=False,
                        help='Closed-form GPTAQ: pre-shift W by W (C - H) H^-1 from '
                             'the FP-vs-Q activation gap, then run GPTQ on the shifted '
                             'W. Same per-Linear iteration as GPTQ; reuses the GPTQ '
                             'Cholesky path. Requires --fp16_calib (for the FP forward '
                             'trajectory).')

    # Simulation version (for A/B comparisons, does not affect the run)
    parser.add_argument('--sim_version', type=int, default=0,
                        help='Simulation version tag for A/B comparisons (0=omitted from tag)')

    # FP32 model precision (isolate float16 bottleneck)
    parser.add_argument('--fp32', action='store_true',
                        help='Run model in float32 instead of float16 (isolate precision effects)')

    # Real integer quantization (bypass the 16-bit passthrough)
    parser.add_argument('--realint', action='store_true',
                        help='Force real integer quantize/dequantize even at 16 bits')

    # Output quantization
    parser.add_argument('--stochastic_quant', action='store_true', default=False,
                        help='Use stochastic rounding for all activation quantizers (unbiased)')
    parser.add_argument('--ig_compare', action='store_true', default=False,
                        help='Compare int_gemm vs fake-quant per layer (uses fake-quant for PPL)')
    parser.add_argument('--semi_int_gemm', type=str, default=None,
                        help='Diagnostic GEMM with toggleable precision stages (bitmask or keyword)')
    parser.add_argument('--hw_align', action='store_true', default=False,
                        help='Hardware-aligned activation scales (per-group instead of per-column)')
    parser.add_argument('--hw_accurate', action='store_true', default=False,
                        help='Hardware-accurate activation scales: collapse all static scales '
                             'to per-tensor. Combine with --eq for per-channel down_proj '
                             '(full hardware match). Mutually exclusive with --hw_align.')
    parser.add_argument('--quant_out', type=str, default='none',
                        choices=['none', 'up', 'mlp', 'spec', 'speco', 'all', 'r4', 'res', 'mm', 'ex'],
                        help='Output quantization: none (default), up (up_proj only), '
                             'mlp (gate+up+down_proj), spec (all except q/k/v_proj), '
                             'speco (all except q/k/v/o_proj), all (all layers), '
                             'r4 (pre-rotation on down_proj), res (residual adds), '
                             'mm (Q in attention), ex (all+res+mm). '
                             'Configures 16-bit symmetric quantizer on matching layers.')

    # Preset bundle: 0 = today's behavior, 1 = "production-shaped" auto-fills.
    # dest=preset to avoid shadowing the built-in `set` in attribute access.
    parser.add_argument('--set', dest='preset', type=int, default=0, choices=[0, 1],
                        help='Preset bundle. 0 (default) = no auto-fill. '
                             '1 = if w<16 add gptaq+fp16_calib+hw_accurate; '
                             'if a<16 add static_act+realint+down_bits=16+kv_ex=8+k=v=8; '
                             'if int_gemm add acc_block_k=G+acc_wrap; '
                             'if hwscale add scalewise. Explicit user flags win.')


def apply_set_preset(args):
    """Expand --set N into the equivalent set of explicit flags.

    Set 0 (default): no-op.
    Set 1: auto-fills the production-shaped bundle based on what's already on
    the line. Idempotent. Detection of "user didn't set this" is by value-
    equals-default — passing e.g. ``-k 4`` (the parser default) along with
    ``--set 1`` will still get upgraded to 8. Acceptable for our use cases.

    Keep in sync with the bash apply_set_preset block in
    fake_quant/Script/dart_gptq_wxaykvz.sh.
    """
    # int_gemm acc_block_k flip is always-on (not preset-gated): with
    # w_groupsize < 32 the default acc_block_k=32 always fails the
    # args_config_gen assert at inference, so silent-correct it here so cal
    # and inference compute the same tag (otherwise cal saves without `bk*`
    # and inference looks up `bk{groupsize}` → file mismatch).
    if getattr(args, 'int_gemm', False):
        if (getattr(args, 'acc_block_k', 32) == 32
                and getattr(args, 'groupsize', 128) > 0
                and args.groupsize != 32):
            args.acc_block_k = args.groupsize

    if getattr(args, 'preset', 0) != 1:
        return

    # Weight quantization → gptaq + fp16_calib (hw_accurate moved to act branch
    # because it operates on static activation scales — no a-quant, no scales).
    if getattr(args, 'w_bits', 16) < 16:
        if not getattr(args, 'gptaq', False):
            args.gptaq = True
        if not getattr(args, 'fp16_calib', False):
            args.fp16_calib = True

    # Activation quantization → static + realint + hw_accurate + down_bits=16, k=v=kv_ex=8
    if getattr(args, 'a_bits', 16) < 16:
        if getattr(args, 'down_bits', None) is None:
            args.down_bits = 16
        if getattr(args, 'k_bits', 4) == 4:
            args.k_bits = 8
        # v_bits=None resolves to k_bits later via resolve_v_bits, so leave alone
        if getattr(args, 'kv_ex', 0) == 0:
            args.kv_ex = 8
        if not getattr(args, 'static_act', False):
            args.static_act = True
        if not getattr(args, 'realint', False):
            args.realint = True
        if not getattr(args, 'hw_accurate', False):
            args.hw_accurate = True
    else:
        # Weight-only (no activation quant): k=v=kv_ex=16, leave realint/hw_accurate alone.
        if getattr(args, 'k_bits', 4) == 4:
            args.k_bits = 16
        # v_bits=None → resolves to k_bits=16 via resolve_v_bits
        if getattr(args, 'kv_ex', 0) == 0:
            args.kv_ex = 16

    # Integer GEMM → acc_wrap on (acc_block_k flip is now always-on, above)
    if getattr(args, 'int_gemm', False):
        if not getattr(args, 'acc_wrap', False):
            args.acc_wrap = True

    # hwscale → scalewise (gated: scalewise without hwscale crashes init_scalewise)
    if getattr(args, 'hwscale', None) is not None:
        if not getattr(args, 'scalewise', False):
            args.scalewise = True


def resolve_v_bits(args):
    """Default v_bits to k_bits when not explicitly given."""
    if args.v_bits is None:
        args.v_bits = args.k_bits


# ── Quant-tag and path helpers ───────────────────────────────────────────────

def build_quant_tag(args, for_gptq_cache=False, for_cal_cache=False,
                    for_fp16_cal_cache=False):
    """Build the canonical quant-tag string from parsed args.

    This is the SINGLE SOURCE OF TRUTH for tag construction.
    Called by: calibrate_act_scales.py, analyze_scales.py, run_experiments.py,
    and dart_gptq_wxaykvz.sh (via ``python experiment_config.py``).

    If *for_gptq_cache* is True, the gptq_strength suffix is omitted so that
    all strength values share the same GPTQ checkpoint cache.

    If *for_fp16_cal_cache* is True, the tag is restricted to flags that affect
    FP16 (pre-GPTQ) activation calibration output. All post-quantization-only
    flags are stripped so a single FP16 cal file is shared across configs.
    """
    # The FP16 cal cache strips weight-quantization details entirely — only
    # observer-placement flags (a/k/v_bits) and pre-calibration transforms
    # (eq family, fp32) matter for the FP16 act_scales values.
    if for_fp16_cal_cache:
        return _build_fp16_cal_tag(args)
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
    else:
        _late_r4 = getattr(args, 'late_rot4', False)
        if _late_r4:
            if for_cal_cache:
                pass                 # reuse normal R4 calibration (and R4 GPTQ in cal context)
            elif for_gptq_cache:
                tag += "_noR4"       # reuse noR4 GPTQ checkpoint at runtime
            else:
                tag += "_lateR4"     # unique result tag
        elif getattr(args, 'no_r4', False):
            tag += "_noR4"
        if getattr(args, 'down_bits', None) is not None:
            tag += f"_down{args.down_bits}"
    if getattr(args, 'oproj_bits', None) is not None:
        tag += f"_oproj{args.oproj_bits}"
    if getattr(args, 'eq', False):
        tag += "_eq"
    if getattr(args, 'ud_eq', False):
        tag += "_udeq"
    if getattr(args, 'ugd_eq', False):
        tag += "_ugdeq"
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
    # Integer GEMM tag (omitted for GPTQ cache — GPTQ doesn't use the accumulator)
    if getattr(args, 'int_gemm', False) and not for_gptq_cache:
        parts = ["_intgemm"]
        if getattr(args, 'acc_bits', 16) != 16:
            parts.append(f"acc{args.acc_bits}")
        if getattr(args, 'acc_block_k', 32) != 32:
            parts.append(f"bk{args.acc_block_k}")
        if getattr(args, 'acc_wrap', False):
            parts.append("wrap")
        acc_dtype_str = getattr(args, 'acc_dtype', 'float')
        # intXaY is the calibration-driven auto form — its frac_bits is
        # resolved per-layer at inference init from the loaded cal scales,
        # so the cal file itself CANNOT be captured with intXaY active
        # (circular dep). Strip the t2 tag for cal-cache only in this case;
        # preserve historical per-t2 cal files for manual intNpM / fp16 etc.
        _is_auto_t2 = bool(re.match(r'^w?int\d+a\d+$', acc_dtype_str.lower().strip()))
        if (acc_dtype_str.lower().strip() not in ('float', 'fp32')
                and not (for_cal_cache and _is_auto_t2)):
            parts.append(f"t2{acc_dtype_str}")
        _mshift = getattr(args, 'lsb_mac_shift', 0)
        if _mshift > 0:
            parts.append(f"mshift{_mshift}")
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
    # hwscale / scalewise tag.
    # Default behaviour: hwscale is result-only (GPTQ/cal caches unchanged).
    # With --scalewise, the merged scale is rounded INSIDE the GPTQ loop, so
    # the GPTQ checkpoint and the post-GPTQ calibration depend on the hwscale
    # spec; fold it into those caches as `_scalewise-<spec>`. The result tag
    # keeps the existing `_hws-<spec>` plus a `_scalewise` marker.
    _hwscale = getattr(args, 'hwscale', None)
    _scalewise = getattr(args, 'scalewise', False)
    if _hwscale:
        if _scalewise and (for_gptq_cache or for_cal_cache):
            tag += f"_scalewise-{_hwscale}"
        elif not (for_gptq_cache or for_cal_cache):
            tag += f"_hws-{_hwscale}"
            if _scalewise:
                tag += "_scalewise"
    # AdaQuant tag
    _aq = getattr(args, 'adaquant', None)
    if _aq is not None:
        tag += "_adaquant"
        if _aq != 'default':
            tag += f"-{_aq}"
    # GPTAQ tag — distinguishes the closed-form FP-target variant from plain
    # GPTQ. Applied to result/GPTQ-cache/post-GPTQ cal tags. Skipped for the
    # FP16 cal cache (the FP16 cal file is GPTAQ-independent — it's just the
    # FP-trajectory activation scales).
    if getattr(args, 'gptaq', False) and not for_fp16_cal_cache:
        tag += "_gptaq"
    # Simulation version tag (non-gptq only, for A/B comparisons)
    if not for_gptq_cache:
        _sv = getattr(args, 'sim_version', 0)
        if _sv:
            tag += f"_v{_sv}"
    # FP32 model tag
    if getattr(args, 'fp32', False):
        tag += "_FP32"
    # Real integer quantization tag
    if getattr(args, 'realint', False):
        tag += "_RINT"
    # Hardware-aligned activation scales.
    # Default (no --scalewise): hw_align affects calibration+result but not
    # GPTQ; hw_accurate is runtime-only (collapses to per-tensor at int_gemm).
    # Under --scalewise: GPTQ rounds the merged scale at the runtime grouping
    # (per-tensor for hw_accurate non-down_proj, per-K-group otherwise), so
    # the GPTQ output depends on which alignment mode is active. Fold the
    # corresponding markers into the GPTQ + cal cache tags.
    _hw_align_flag = getattr(args, 'hw_align', False)
    _hw_accurate_flag = getattr(args, 'hw_accurate', False)
    _aligned_in_cache = _scalewise and (for_gptq_cache or for_cal_cache)
    if (_hw_align_flag or _hw_accurate_flag) and (
            not for_gptq_cache or _aligned_in_cache):
        tag += "_aligned"
    if _hw_accurate_flag and (
            not (for_gptq_cache or for_cal_cache) or _aligned_in_cache):
        tag += "_hwacc"
    # Per-Linear weight-groupsize policy (--weight_group_mode). Affects GPTQ
    # output (different per-Linear groupsize → different quantized weights),
    # post-GPTQ cal (observers see different weight-quant distortion), and the
    # result tag. Naturally absent from FP16 cal tag (early return at line ~393).
    _wgm = getattr(args, 'weight_group_mode', 'all')
    if _wgm == 'down':
        tag += "_WGQ-DOWN"
    elif _wgm == 'down_o':
        tag += "_WGQ-DOWN-O"
    # Output quantization tag (activation-side, not relevant for GPTQ cache)
    _qo = getattr(args, 'quant_out', 'none')
    if _qo != 'none' and not for_gptq_cache:
        tag += f"_qout-{_qo}"
    # Stochastic quantization tag (result-only, not relevant for GPTQ/calibration cache)
    if getattr(args, 'stochastic_quant', False) and not (for_gptq_cache or for_cal_cache):
        tag += "_stoch"
    # ig_compare uses fake-quant path for PPL — different result, needs separate cache
    if getattr(args, 'ig_compare', False) and not (for_gptq_cache or for_cal_cache):
        tag += "_igcmp"
    # Semi-int GEMM diagnostic (result-only)
    _sig = getattr(args, 'semi_int_gemm', None)
    if _sig and not (for_gptq_cache or for_cal_cache):
        tag += f"_semi-{_sig}"
    # FP4 weights (changes the GPTQ output, post-quant activations, and result —
    # tag in all three modes).
    _fp4 = getattr(args, 'fp4', 'none')
    if _fp4 == 'all':
        tag += "_FP4"
    elif _fp4 == 'down':
        tag += "_FP4-DOWN"
    # Deployment-side choice of activation scales: --fp16_calib loads FP16 cal
    # at inference, otherwise post-GPTQ cal. Both cal artifacts can coexist on
    # disk under the new (decoupled) cal flow, so two inference runs that
    # differ only on this flag must produce different result files. Suffix
    # appears in the result tag only — GPTQ/cal/fp16-cal tags don't depend on
    # which deployment file is selected.
    if (getattr(args, 'fp16_calib', False)
            and not (for_gptq_cache or for_cal_cache or for_fp16_cal_cache)):
        tag += "_fp16dep"
    return tag


def _build_fp16_cal_tag(args):
    """FP16 calibration tag: every flag that affects the contents (and shape)
    of the FP16 act_scales dict.

    Two kinds of contamination to prevent:
      (a) Forward-path differences during FP16 cal — the same observer sees a
          different distribution. Sources: online rotations (R3 via kv_ex,
          R4 via no_r4), pre-cal weight transforms (GGUF, equalization),
          pre-cal forward swaps (PWL replaces silu before observers fire).
      (b) Observer-set differences — the cached file may be missing entries
          that this run needs. Sources: --quant_out (which out/pre/res/mm
          quantizers exist), --smq (softmax/k-cache observers),
          --down_bits=16 (skips the down_proj input observer), --kv_ex /
          --proj_ex (bit-extended quantizer config).

    Encoded via flags listed below. R1/R2 are not toggled in current runs
    (R1 always on for quarot/dart, R2 always offline) so they're skipped per
    convention; if that ever changes they must be added here.

    Dropped (don't run during FP16 cal forward): w_bits, groupsize, sym,
    w_asym, w_clip, w_rtn, late_rot4, oproj_bits, gscaler, hwscale,
    scalewise, hw_align, hw_accurate, gptq_strength, int_gemm/acc_*, fp4,
    adaquant, sim_version, stochastic_quant, ig_compare, semi_int_gemm,
    realint.
    """
    parts = [f"a{args.a_bits}k{args.k_bits}v{args.v_bits}"]
    if getattr(args, 'eq', False):
        parts.append("eq")
    if getattr(args, 'ud_eq', False):
        parts.append("udeq")
    if getattr(args, 'ugd_eq', False):
        parts.append("ugdeq")
    if getattr(args, 'no_r4', False):
        parts.append("noR4")
    # kv_ex carries R3 state — kv_ex>0 disables R3 in main_for_test, so this
    # flag implicitly encodes whether the attention online R3 Hadamard runs
    # during cal forward. Also affects k/v cache quantizer config (observers).
    if getattr(args, 'kv_ex', 0):
        parts.append(f"kvex{args.kv_ex}")
    if getattr(args, 'proj_ex', 0):
        parts.append(f"projex{args.proj_ex}")
    # down_bits=16 skips the down_proj input observer — different observer
    # set than down_bits<16. Encode whenever set explicitly.
    if getattr(args, 'down_bits', None) is not None:
        parts.append(f"down{args.down_bits}")
    # PWL replaces silu before FP16 observers fire (calibrate_act_scales:863-877).
    if getattr(args, 'pwl_act', False):
        pwl = ["pwl"]
        if getattr(args, 'pwl_n_segments', 9) != 9:
            pwl.append(f"{args.pwl_n_segments}p")
        if getattr(args, 'pwl_input_bits', 16) != 16:
            pwl.append(f"in{args.pwl_input_bits}")
        if getattr(args, 'pwl_output_bits', 16) != 16:
            pwl.append(f"out{args.pwl_output_bits}")
        if getattr(args, 'pwl_no_hw_sim', False):
            pwl.append("nohw")
        parts.append("".join(pwl))
    # quant_out determines which out/pre/res/mm observers exist
    # (calibrate_act_scales:921-950 — runs before FP16 cal).
    _qo = getattr(args, 'quant_out', 'none')
    if _qo != 'none':
        parts.append(f"qout-{_qo}")
    # smq adds softmax/k-cache observers (calibrate_act_scales:952-955).
    if getattr(args, 'smq', 0) > 0:
        parts.append(f"smq{args.smq}")
    # GGUF rewrites weights before any cal — different activations through
    # FP16 observers. Mirrors build_quant_tag's encoding.
    if getattr(args, 'gguf', None):
        gguf_path = resolve_gguf_path(args.gguf, args.model)
        basename = os.path.basename(gguf_path).replace('.gguf', '')
        gparts = basename.split('-')
        qtype = '-'.join(p for p in gparts if p.startswith('Q')) or 'gguf'
        parts.append(f"gguf-{qtype.replace('_', '-')}")
    if getattr(args, 'imitate_gguf', None):
        gguf_path = resolve_gguf_path(args.imitate_gguf, args.model)
        basename = os.path.basename(gguf_path).replace('.gguf', '')
        gparts = basename.split('-')
        first_q = next((i for i, p in enumerate(gparts) if p.startswith('Q')), None)
        imit_label = '-'.join(gparts[first_q:]) if first_q is not None else 'gguf'
        parts.append(f"imitate-{imit_label.replace('_', '-')}")
    if getattr(args, 'fp32', False):
        parts.append("FP32")
    return "_".join(parts)


_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def resolve_act_scales_path(model_path, mode, quant_tag):
    """Resolve calibration .pt file path."""
    model_name = model_name_from_path(model_path)
    return os.path.join(_DATA_DIR, 'act_scales', model_name, f'{mode}_{quant_tag}.pt')


def resolve_fp16_act_scales_path(model_path, mode, fp16_cal_tag):
    """Resolve FP16 calibration .pt file path.

    Distinct filename suffix `__fp16` so it never collides with a post-GPTQ
    cal file that happens to have the same tag.
    """
    model_name = model_name_from_path(model_path)
    return os.path.join(
        _DATA_DIR, 'act_scales', model_name,
        f'{mode}_{fp16_cal_tag}__fp16.pt',
    )


def resolve_gptq_checkpoint_dir(model_path, mode, quant_tag, w_bits, imitate_gguf=False):
    """Resolve GPTQ checkpoint directory (containing .pth files)."""
    model_name = model_name_from_path(model_path)
    w_suffix = 'w0' if imitate_gguf else f'w{w_bits}'
    return os.path.join(
        _DATA_DIR, 'gptq_checkpoints',
        f'{mode}_{model_name}_{quant_tag}',
        f'{model_name}_{w_suffix}',
    )


def resolve_adaquant_checkpoint_dir(model_path, mode, quant_tag, w_bits, imitate_gguf=False):
    """Resolve AdaQuant checkpoint directory (containing .pth files)."""
    model_name = model_name_from_path(model_path)
    w_suffix = 'w0' if imitate_gguf else f'w{w_bits}'
    return os.path.join(
        _DATA_DIR, 'adaquant_checkpoints',
        f'{mode}_{model_name}_{quant_tag}',
        f'{model_name}_{w_suffix}',
    )


def resolve_gptaq_checkpoint_dir(model_path, mode, quant_tag, w_bits, imitate_gguf=False):
    """Resolve GPTAQ checkpoint directory (containing .pth files).

    Parallel to resolve_gptq_checkpoint_dir but under data/gptaq_checkpoints/
    so GPTAQ-quantized weights don't collide with GPTQ checkpoints that share
    the same quant tag stem.
    """
    model_name = model_name_from_path(model_path)
    w_suffix = 'w0' if imitate_gguf else f'w{w_bits}'
    return os.path.join(
        _DATA_DIR, 'gptaq_checkpoints',
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
    _wgm = getattr(args, 'weight_group_mode', 'all')
    if _wgm != 'all':
        cmd += ['--weight_group_mode', _wgm]
    if args.sym:
        cmd.append('--sym')
    if getattr(args, 'w_asym', False):
        cmd.append('--w_asym')
    if args.kv_ex:
        cmd += ['--kv_ex', str(args.kv_ex)]
    if args.proj_ex:
        cmd += ['--proj_ex', str(args.proj_ex)]
    else:
        if getattr(args, 'no_r4', False):
            cmd.append('--no_r4')
        if getattr(args, 'down_bits', None) is not None:
            cmd += ['--down_bits', str(args.down_bits)]
    if getattr(args, 'oproj_bits', None) is not None:
        cmd += ['--oproj_bits', str(args.oproj_bits)]
    if getattr(args, 'eq', False):
        cmd.append('--eq')
    if getattr(args, 'ud_eq', False):
        cmd.append('--ud_eq')
    if getattr(args, 'ugd_eq', False):
        cmd.append('--ugd_eq')
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
        acc_dtype_str = getattr(args, 'acc_dtype', 'float')
        if acc_dtype_str.lower().strip() not in ('float', 'fp32'):
            cmd += ['--acc_dtype', acc_dtype_str]
        _mshift = getattr(args, 'lsb_mac_shift', 0)
        if _mshift > 0:
            cmd += ['--lsb_mac_shift', str(_mshift)]
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
    if getattr(args, 'hwscale', None):
        cmd += ['--hwscale', args.hwscale]
    _aq = getattr(args, 'adaquant', None)
    if _aq is not None:
        cmd += ['--adaquant', _aq]
    if getattr(args, 'sim_version', 0):
        cmd += ['--sim_version', str(args.sim_version)]
    if getattr(args, 'fp32', False):
        cmd.append('--fp32')
    if getattr(args, 'realint', False):
        cmd.append('--realint')
    _qo = getattr(args, 'quant_out', 'none')
    if _qo != 'none':
        cmd += ['--quant_out', _qo]
    if getattr(args, 'late_rot4', False):
        cmd.append('--late_rot4')
    if getattr(args, 'stochastic_quant', False):
        cmd.append('--stochastic_quant')
    if getattr(args, 'ig_compare', False):
        cmd.append('--ig_compare')
    if getattr(args, 'semi_int_gemm', None):
        cmd += ['--semi_int_gemm', args.semi_int_gemm]
    if getattr(args, 'hw_align', False):
        cmd.append('--hw_align')
    if getattr(args, 'hw_accurate', False):
        cmd.append('--hw_accurate')
    if getattr(args, 'fp16_calib', False):
        cmd.append('--fp16_calib')
    if getattr(args, 'scalewise', False):
        cmd.append('--scalewise')
    if getattr(args, 'force_recalib', False):
        cmd.append('--force_recalib')
    if getattr(args, 'gptaq', False):
        cmd.append('--gptaq')
    _fp4 = getattr(args, 'fp4', 'none')
    if _fp4 != 'none':
        cmd += ['--fp4', _fp4]
    if getattr(args, 'preset', 0):
        cmd += ['--set', str(args.preset)]
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
    _parser.add_argument('--for_cal_cache', action='store_true',
                         help='Tag for calibration cache (late_rot4 maps to normal R4 tag)')
    _parser.add_argument('--for_fp16_cal_cache', action='store_true',
                         help='Tag for FP16 (pre-GPTQ) calibration cache: only flags '
                              'that affect the FP16 act_scales contents are kept.')
    _args = _parser.parse_args()
    apply_set_preset(_args)
    resolve_v_bits(_args)
    _args.model = resolve_model(_args.model)
    print(build_quant_tag(_args, for_gptq_cache=_args.for_gptq_cache,
                          for_cal_cache=_args.for_cal_cache,
                          for_fp16_cal_cache=_args.for_fp16_cal_cache))
