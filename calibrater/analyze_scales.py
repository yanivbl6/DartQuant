#!/usr/bin/env python3
# Wrapped by skill: .claude/skills/dart-analyze-scales — update SKILL.md if CLI flags / mode behavior change.
"""Analyze quantization scales from calibration files or GPTQ checkpoints.

Modes:
  (default)    Activation scales from a calibration .pt file
  --weights    Weight group scales recovered from a GPTQ checkpoint

Accepts the same quant arguments as the experiment scripts.
"""

import argparse
import math
import os
import sys
import re

import numpy as np
import torch

# Allow imports from parent dir (experiment_config lives one level up)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import experiment_config as cfg


# ── Path resolution (delegates to experiment_config) ──

def _resolve_paths(args, mode):
    """Return (act_scales_path, gptq_checkpoint_dir, display_tag).

    Uses for_cal_cache=True for the cal lookup so the path matches what
    calibrate_act_scales.py actually saves (strips auto-T2 t2intNaM, uses
    _scalewise-<spec> form for scalewise+hwscale combos). GPTQ dir uses both
    flags so result-only segments are stripped. The third return value is
    the display tag (full form) used for figure filenames.
    """
    cal_tag = cfg.build_quant_tag(args, for_cal_cache=True)
    gptq_tag = cfg.build_quant_tag(args, for_gptq_cache=True, for_cal_cache=True)
    display_tag = cfg.build_quant_tag(args)
    act_path = cfg.resolve_act_scales_path(args.model, mode, cal_tag)
    imitate_gguf = bool(getattr(args, 'imitate_gguf', None))
    # GPTAQ checkpoints live under data/gptaq_checkpoints/ (separate from
    # data/gptq_checkpoints/ — see resolve_gptaq_checkpoint_dir).
    if getattr(args, 'gptaq', False):
        gptq_dir = cfg.resolve_gptaq_checkpoint_dir(
            args.model, mode, gptq_tag, args.w_bits, imitate_gguf=imitate_gguf)
    else:
        gptq_dir = cfg.resolve_gptq_checkpoint_dir(
            args.model, mode, gptq_tag, args.w_bits, imitate_gguf=imitate_gguf)
    return act_path, gptq_dir, display_tag


# ── Weight scale extraction ──

def _recover_group_scales(W_fq, maxq):
    """Recover per-row scales from fake-quantized weights (symmetric).

    Mirrors int_acc_gemm._recover_int_and_scale but only returns scales.
    Tries two candidates (maxq and maxq+1) to handle w_clip/MSE-shrunken ranges.
    """
    abs_max = W_fq.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)

    # Candidate A: max integer = maxq
    scale_a = abs_max / maxq
    q_a = torch.clamp(torch.round(W_fq / scale_a), -(maxq + 1), maxq)
    err_a = (q_a * scale_a - W_fq).pow(2).sum(dim=1, keepdim=True)

    # Candidate B: max integer = maxq+1 (w_clip case)
    scale_b = abs_max / (maxq + 1)
    q_b = torch.clamp(torch.round(W_fq / scale_b), -(maxq + 1), maxq)
    err_b = (q_b * scale_b - W_fq).pow(2).sum(dim=1, keepdim=True)

    scale = torch.where(err_b < err_a, scale_b, scale_a)
    return scale.squeeze(1)  # [N_groups]


def _recover_group_scales_asym(W_fq, maxq_unsigned):
    """Recover per-row scales from fake-quantized weights (asymmetric).

    Mirrors int_acc_gemm._recover_int_and_scale_asym: scale = (max - min) / maxq_unsigned.
    maxq_unsigned is the unsigned range (e.g. 15 for 4-bit).
    """
    w_min = W_fq.amin(dim=1, keepdim=True)
    w_max = W_fq.amax(dim=1, keepdim=True)
    scale = ((w_max - w_min) / maxq_unsigned).clamp(min=1e-10)
    return scale.squeeze(1)  # [N_groups]


def extract_weight_scales(checkpoint_dir, w_bits, w_groupsize, w_sym):
    """Load GPTQ checkpoint and extract per-group weight scales.

    Returns dict {category: numpy array of scale values}.
    """
    if w_sym:
        maxq = 2 ** (w_bits - 1) - 1
        recover = lambda w: _recover_group_scales(w, maxq)
    else:
        maxq_unsigned = 2 ** w_bits - 1
        recover = lambda w: _recover_group_scales_asym(w, maxq_unsigned)

    # Load all .pth parts into a single state_dict
    pth_files = sorted(f for f in os.listdir(checkpoint_dir) if f.endswith('.pth'))
    state_dict = {}
    for f in pth_files:
        part = torch.load(os.path.join(checkpoint_dir, f), map_location='cpu', weights_only=True)
        state_dict.update(part)
        del part

    # Extract weight tensors and categorize
    cat_scales = {}
    weight_pattern = re.compile(
        r'model\.layers\.(\d+)\.(self_attn\.(q|k|v|o)_proj|mlp\.(up|gate|down)_proj)\.weight'
    )

    for key, tensor in state_dict.items():
        m = weight_pattern.match(key)
        if not m:
            continue

        layer_idx = int(m.group(1))
        proj_name = m.group(3) or m.group(4)  # q/k/v/o or up/gate/down
        cat = f'{proj_name}_proj'

        W = tensor.float()  # [N, K]
        N, K = W.shape

        if w_groupsize > 0:
            n_groups = math.ceil(K / w_groupsize)
            padded_K = n_groups * w_groupsize
            if padded_K > K:
                W = torch.nn.functional.pad(W, (0, padded_K - K))
            W_grouped = W.reshape(N * n_groups, w_groupsize)
            scales = recover(W_grouped).numpy()
        else:
            # Per-channel
            scales = recover(W).numpy()

        cat_scales.setdefault(cat, []).extend(scales.tolist())

    return cat_scales


# ── Merged per-group scale extraction (--group mode) ──

# GPTQ checkpoints contain both "<...>.weight" and "<...>.module.weight" for
# the same Linear (Linear + ActQuantWrapper), with identical post-GPTQ values.
# Match only the inner ".module.weight" to avoid double-counting.
_PROJ_RE = re.compile(
    r'^(model\.layers\.(\d+)\.(self_attn\.(q|k|v|o)_proj|mlp\.(up|gate|down)_proj))'
    r'\.module\.weight$'
)


def _act_per_group_scale(col_scale, col_zero, G, a_sym, maxq_a):
    """Collapse per-column activation (scale, zero) → per-group scale, mirroring
    main_for_test.py:877-893. Returns float32 [n_groups] tensor."""
    K = col_scale.shape[0]
    assert K % G == 0, f"K={K} not divisible by acc_block_k={G}"
    n_g = K // G
    if a_sym:
        return col_scale.reshape(n_g, G).max(dim=1).values.float()
    # asym: recover group_min/group_max from (col_min, col_max) via stored zero/scale
    col_min = -(col_zero * col_scale)
    col_max = (maxq_a - col_zero) * col_scale
    group_min = col_min.reshape(n_g, G).min(dim=1).values
    group_max = col_max.reshape(n_g, G).max(dim=1).values
    group_scale = (group_max - group_min) / maxq_a
    dead = (group_min == 0) & (group_max == 0)
    group_scale[dead] = 1.0
    return group_scale.float()


def extract_merged_scales(
    cal_path, gptq_dir, w_bits, w_groupsize, acc_block_k, w_sym, a_sym,
    hw_accurate, hwscale_spec_str=None,
):
    """For each per-group layer, return (raw_merged, scaled_post_global, layer_globals).

    raw_merged   : dict {category: 1-D numpy array} — flattened a_gscale * w_gscale values
    scaled       : dict {category: 1-D numpy array} OR None if hwscale_spec_str is None
    layer_globals: list of (layer_idx, proj, n_groups_total, global_scalar)
                   for diagnostics, only when hwscale_spec_str is set
    """
    # Load both inputs
    cal = torch.load(cal_path, map_location='cpu', weights_only=True)
    sd = {}
    for f in sorted(os.listdir(gptq_dir)):
        if f.endswith('.pth'):
            sd.update(torch.load(os.path.join(gptq_dir, f), map_location='cpu',
                                 weights_only=True))

    G = acc_block_k
    maxq_w_sym = 2 ** (w_bits - 1) - 1
    maxq_w_asym = 2 ** w_bits - 1
    # Activation maxq (a_bits=8 by default; cal may store it differently —
    # use the recorded scale/zero unchanged for sym, derive maxq from zero
    # range for asym). Most common case: a_bits=8 asym → maxq=255.
    maxq_a = 255.0  # fallback for 8-bit asym; sym path doesn't use it

    # Optional hwscale spec
    spec = None
    if hwscale_spec_str is not None:
        # Lazy-import quant_utils (lives under fake_quant)
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'fake_quant'))
        from quant_utils import parse_gscaler, hwscale_global_from_spec
        spec = parse_gscaler(hwscale_spec_str)

    raw_per_cat = {}
    scaled_per_cat = {} if spec is not None else None
    layer_globals = []

    for key, W in sd.items():
        m = _PROJ_RE.match(key)
        if not m:
            continue
        layer_path = m.group(1)
        layer_idx = int(m.group(2))
        proj_name = m.group(4) or m.group(5)
        cat = f'{proj_name}_proj'

        # Skip non-down layers under hw_accurate (mirrors main_for_test.py:864)
        if hw_accurate and proj_name != 'down':
            continue

        # Activation per-group scale from cal
        cal_q_key = f'{layer_path}.quantizer'
        if cal_q_key not in cal:
            continue
        a_col_scale = cal[cal_q_key]['scale'].float()
        a_col_zero = cal[cal_q_key].get('zero')
        if a_col_zero is None:
            a_col_zero = torch.zeros_like(a_col_scale)
        else:
            a_col_zero = a_col_zero.float()
        a_gscale = _act_per_group_scale(a_col_scale, a_col_zero, G, a_sym, maxq_a)

        # Weight per-group scale from GPTQ checkpoint
        Wf = W.float()
        N, K = Wf.shape
        n_w_groups = K // w_groupsize
        if w_groupsize * n_w_groups != K:
            n_w_groups = math.ceil(K / w_groupsize)
            padded_K = n_w_groups * w_groupsize
            Wf = torch.nn.functional.pad(Wf, (0, padded_K - K))
        Wg = Wf.reshape(N * n_w_groups, w_groupsize)
        if w_sym:
            w_scale_flat = _recover_group_scales(Wg, maxq_w_sym)
        else:
            w_scale_flat = _recover_group_scales_asym(Wg, maxq_w_asym)
        w_gscale = w_scale_flat.reshape(N, n_w_groups)

        # Merged scale (must align activation groups with weight groups: with
        # acc_block_k == w_groupsize they share the same n_groups, which is the
        # configuration hwscale requires at inference).
        if a_gscale.shape[0] != n_w_groups:
            # Mismatched group geometry — skip rather than guess.
            continue
        raw_merged = (a_gscale.unsqueeze(0) * w_gscale).flatten().numpy()
        raw_per_cat.setdefault(cat, []).append(raw_merged)

        if spec is not None:
            from quant_utils import hwscale_global_from_spec
            gs = hwscale_global_from_spec(
                spec, raw_scale=a_gscale.unsqueeze(0) * w_gscale)
            scaled = raw_merged / max(gs, 1e-300)
            scaled_per_cat.setdefault(cat, []).append(scaled)
            layer_globals.append((layer_idx, cat, raw_merged.size, float(gs)))

    raw_per_cat = {k: np.concatenate(v) for k, v in raw_per_cat.items()}
    if scaled_per_cat is not None:
        scaled_per_cat = {k: np.concatenate(v) for k, v in scaled_per_cat.items()}
    return raw_per_cat, scaled_per_cat, layer_globals


# ── Categorize keys ──

def categorize_key(key):
    """Return (category, layer_idx, sublayer) for a scale dict key."""
    m = re.match(r'layer\.(\d+)\.k_quantizer', key)
    if m:
        return 'k_cache', int(m.group(1)), 'k_cache'

    m = re.match(r'model\.layers\.(\d+)\.(.+?)\.pre_quantizer', key)
    if m:
        sublayer = m.group(2)
        return 'pre_rotation', int(m.group(1)), sublayer

    m = re.match(r'model\.layers\.(\d+)\.(.+?)\.out_quantizer', key)
    if m:
        sublayer = m.group(2)
        if 'v_proj' in sublayer:
            return 'v_cache', int(m.group(1)), sublayer
        elif 'q_proj' in sublayer or 'k_proj' in sublayer:
            return 'attn_qk_output', int(m.group(1)), sublayer
        elif 'o_proj' in sublayer:
            return 'attn_o_output', int(m.group(1)), sublayer
        elif 'gate_proj' in sublayer or 'up_proj' in sublayer:
            return 'mlp_gate_up_output', int(m.group(1)), sublayer
        elif 'down_proj' in sublayer:
            return 'mlp_down_output', int(m.group(1)), sublayer
        else:
            return 'other_output', int(m.group(1)), sublayer

    m = re.match(r'model\.layers\.(\d+)\.(.+?)\.quantizer', key)
    if m:
        sublayer = m.group(2)
        if 'q_proj' in sublayer or 'k_proj' in sublayer:
            return 'attn_qk_input', int(m.group(1)), sublayer
        elif 'v_proj' in sublayer or 'o_proj' in sublayer:
            return 'attn_vo_input', int(m.group(1)), sublayer
        elif 'gate_proj' in sublayer or 'up_proj' in sublayer:
            return 'mlp_gate_up_input', int(m.group(1)), sublayer
        elif 'down_proj' in sublayer:
            return 'mlp_down_input', int(m.group(1)), sublayer
        else:
            return 'other_input', int(m.group(1)), sublayer

    return 'unknown', -1, key


# ── Statistics ──

def compute_stats(values):
    """Compute statistics for a 1-D numpy array."""
    return {
        'count': len(values),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'p2.5': float(np.percentile(values, 2.5)),
        'p97.5': float(np.percentile(values, 97.5)),
        'p5': float(np.percentile(values, 5)),
        'p95': float(np.percentile(values, 95)),
    }


def print_stats(name, stats):
    print(f"\n{'=' * 60}")
    print(f"  {name}  ({stats['count']} values)")
    print(f"{'=' * 60}")
    print(f"  Range:       [{stats['min']:.6g}, {stats['max']:.6g}]")
    print(f"  Mean ± Std:  {stats['mean']:.6g} ± {stats['std']:.6g}")
    print(f"  Median:      {stats['median']:.6g}")
    print(f"  95% range:   [{stats['p2.5']:.6g}, {stats['p97.5']:.6g}]")
    print(f"  90% range:   [{stats['p5']:.6g}, {stats['p95']:.6g}]")

    # mantissa+shift analysis: dynamic range in bits over the 95% range
    if stats['p2.5'] > 0 and stats['p97.5'] > 0:
        log2_range = np.log2(stats['p97.5'] / stats['p2.5'])
        print(f"  log2(p97.5/p2.5): {log2_range:.1f}  (shift bits for 95% range)")


# ── Histogram plotting ──

def plot_histograms(all_scales, all_zeros, output_dir, tag):
    """Create histogram PNGs for scales and zeros."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    for field_name, cat_data in [('scale', all_scales), ('zero', all_zeros)]:
        if not cat_data:
            continue

        n_cats = len(cat_data)
        fig, axes = plt.subplots(n_cats, 1, figsize=(10, 3 * n_cats), squeeze=False)
        fig.suptitle(f'{field_name.upper()} distributions — {tag}', fontsize=14, y=1.0)

        for idx, (cat_name, values) in enumerate(sorted(cat_data.items())):
            ax = axes[idx, 0]
            ax.hist(values, bins=100, alpha=0.8, edgecolor='black', linewidth=0.3)
            ax.set_title(f'{cat_name} ({len(values)} values)')
            ax.set_xlabel(field_name)
            ax.set_ylabel('count')
            ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=1, label=f'mean={np.mean(values):.4g}')
            ax.axvline(np.median(values), color='green', linestyle='--', linewidth=1, label=f'median={np.median(values):.4g}')
            ax.legend(fontsize=8)

        plt.tight_layout()
        out_path = os.path.join(output_dir, f'{tag}_{field_name}_hist.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved: {out_path}")

    # Also plot all scales combined on a single axis with log scale
    if all_scales:
        all_vals = np.concatenate(list(all_scales.values()))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        fig.suptitle(f'All scales combined — {tag}', fontsize=13)

        ax1.hist(all_vals, bins=200, alpha=0.8, edgecolor='black', linewidth=0.2)
        ax1.set_xlabel('scale')
        ax1.set_ylabel('count')
        ax1.set_title('Linear scale')

        # Log-scale view
        pos = all_vals[all_vals > 0]
        if len(pos) > 0:
            ax2.hist(np.log2(pos), bins=200, alpha=0.8, edgecolor='black', linewidth=0.2, color='orange')
            ax2.set_xlabel('log2(scale)')
            ax2.set_ylabel('count')
            ax2.set_title('Log2 scale')

        plt.tight_layout()
        out_path = os.path.join(output_dir, f'{tag}_scale_combined.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")


# ── Main ──

def print_scale_report(cat_scales, cat_zeros=None, tag=''):
    """Print stats and optionally plot histograms for categorized scales."""
    print(f"\n{'#' * 60}")
    print(f"  SCALE statistics")
    print(f"{'#' * 60}")
    for cat in sorted(cat_scales.keys()):
        vals = np.array(cat_scales[cat])
        print_stats(f"scale / {cat}", compute_stats(vals))

    # Print zero stats only for categories with non-trivial zeros
    if cat_zeros:
        has_nonzero = {cat: np.any(np.array(v) != 0) for cat, v in cat_zeros.items()}
        if any(has_nonzero.values()):
            print(f"\n{'#' * 60}")
            print(f"  ZERO-POINT statistics")
            print(f"{'#' * 60}")
            for cat in sorted(cat_zeros.keys()):
                if has_nonzero.get(cat, False):
                    vals = np.array(cat_zeros[cat])
                    print_stats(f"zero / {cat}", compute_stats(vals))

    # Overall summary
    all_scale_vals = np.concatenate([np.array(v) for v in cat_scales.values()])
    print(f"\n{'#' * 60}")
    print(f"  OVERALL ({len(all_scale_vals)} scale values)")
    print(f"{'#' * 60}")
    print_stats("all scales", compute_stats(all_scale_vals))

    return all_scale_vals


def main():
    parser = argparse.ArgumentParser(
        description='Analyze quantization scales (activation or weight)',
        epilog="""Examples:
  python analyze_scales.py quarot -m 1b -w 4 -a 8 -k 8 -G 128 --sym --kv_ex 8 --proj_ex 15
  python analyze_scales.py --weights baseline -m 1b -w 4 -a 8 -k 8 -G 128 --sym --kv_ex 8 --proj_ex 15""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('mode', choices=['baseline', 'quarot', 'dart'],
                        help='Rotation mode')
    cfg.add_model_arg(parser)
    cfg.add_quant_args(parser)
    parser.add_argument('--weights', action='store_true',
                        help='Analyze weight group scales from GPTQ checkpoint (instead of activation scales)')
    parser.add_argument('--group', action='store_true',
                        help='Analyze MERGED per-group scales (a_gscale * w_gscale) for layers '
                             'that use per-group scaling — mirrors hwscale\'s actual input. '
                             'Scope follows --hw_accurate (down_proj only) / --hw_align (all int_gemm '
                             'layers). Mutually exclusive with --weights. '
                             'When combined with --hwscale <spec>, plots a second histogram of '
                             'merged_scale / hwscale_global (continuous, NO snap applied).')
    parser.add_argument('--pt_path', type=str, default=None,
                        help='Explicit path to .pt file (activation mode) or GPTQ checkpoint dir (weight mode)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip histogram generation')

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Note: ignoring unrecognized args (likely inference-only): {unknown}",
              file=sys.stderr)
    # Apply preset flips (e.g. acc_block_k follows groupsize when not explicitly
    # set) so the tag matches what calibrate_act_scales.py / run_experiments.py
    # actually wrote. Without this, default-valued args produce wrong paths.
    cfg.apply_set_preset(args)
    cfg.resolve_v_bits(args)
    args.model = cfg.resolve_model(args.model)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'figures')

    act_path, gptq_dir, quant_tag = _resolve_paths(args, args.mode)

    if args.group and args.weights:
        print("Error: --group and --weights are mutually exclusive.", file=sys.stderr)
        sys.exit(1)
    if args.fp16_calib and (args.weights or args.group):
        print("Error: --fp16_calib (analyze FP16 cal) is mutually exclusive "
              "with --weights and --group.", file=sys.stderr)
        sys.exit(1)
    if args.hwscale and not args.group:
        # --hwscale is needed for cal/GPTQ path resolution under --scalewise
        # (the cal tag becomes _scalewise-<spec>). Allow it through; the
        # post-global histogram is still gated on --group below.
        print(f"Note: --hwscale={args.hwscale} accepted for path resolution; "
              f"post-global histogram only generated under --group.",
              file=sys.stderr)

    if args.group:
        # ── Merged per-group scale analysis (a_gscale × w_gscale) ──
        # Cal and GPTQ caches strip result-only tag segments (hwscale/hwacc/etc),
        # so the file lookups need their own tag forms instead of `quant_tag`.
        _cal_tag = cfg.build_quant_tag(args, for_cal_cache=True)
        _gptq_tag = cfg.build_quant_tag(args, for_gptq_cache=True, for_cal_cache=True)
        cal_pt = (args.pt_path
                  if (args.pt_path and os.path.isfile(args.pt_path))
                  else cfg.resolve_act_scales_path(args.model, args.mode, _cal_tag))
        _resolve_ckpt = (cfg.resolve_gptaq_checkpoint_dir
                         if getattr(args, 'gptaq', False)
                         else cfg.resolve_gptq_checkpoint_dir)
        ckpt_dir = (args.pt_path
                    if (args.pt_path and os.path.isdir(args.pt_path))
                    else _resolve_ckpt(
                        args.model, args.mode, _gptq_tag, args.w_bits,
                        imitate_gguf=bool(getattr(args, 'imitate_gguf', None))))

        if not os.path.isfile(cal_pt):
            print(f"Error: calibration file not found: {cal_pt}")
            sys.exit(1)
        if not os.path.isdir(ckpt_dir):
            print(f"Error: GPTQ checkpoint dir not found: {ckpt_dir}")
            sys.exit(1)

        w_sym = not getattr(args, 'w_asym', False) and not args.sym is False
        # Above is awkward — reuse the same convention as build_quant_tag:
        # w_asym is True only when --w_asym is passed AND not --sym.
        w_asym = getattr(args, 'w_asym', False) and not args.sym
        w_sym = not w_asym
        # Activation symmetry: by convention activations are asym (a_asym=True
        # is the default in calibrate_act_scales.py). --sym makes everything sym.
        a_sym = bool(args.sym)
        hw_accurate = bool(getattr(args, 'hw_accurate', False))

        print(f"Cal:  {cal_pt}")
        print(f"GPTQ: {ckpt_dir}")
        print(f"w_sym={w_sym}  a_sym={a_sym}  hw_accurate={hw_accurate}  "
              f"acc_block_k={args.acc_block_k}  w_groupsize={args.groupsize}")
        if args.hwscale:
            print(f"hwscale spec: {args.hwscale} (continuous post-global, NO snap)")

        raw_per_cat, scaled_per_cat, layer_globals = extract_merged_scales(
            cal_path=cal_pt,
            gptq_dir=ckpt_dir,
            w_bits=args.w_bits,
            w_groupsize=args.groupsize,
            acc_block_k=args.acc_block_k,
            w_sym=w_sym,
            a_sym=a_sym,
            hw_accurate=hw_accurate,
            hwscale_spec_str=args.hwscale,
        )

        if not raw_per_cat:
            print("No per-group layers in scope — check --hw_accurate / --hw_align flags "
                  "and that the cal/GPTQ files match the requested config.")
            sys.exit(1)

        # Tag (raw merged scales)
        scope_tag = 'hwacc' if hw_accurate else 'all'
        base_tag = f'merged-{scope_tag}_{args.mode}_{quant_tag}'
        print(f"\n=== Raw merged scales (a_gscale * w_gscale) — {scope_tag} scope ===")
        print_scale_report(raw_per_cat, tag=base_tag)
        if not args.no_plot:
            plot_histograms(raw_per_cat, {}, output_dir, base_tag)

        # Tag (post-global scales)
        if scaled_per_cat is not None:
            postg_tag = f'{base_tag}_postglobal-{args.hwscale}'
            # Per-layer global summary
            print(f"\n=== Per-layer hwscale_global (spec={args.hwscale}) ===")
            print(f"{'layer':>6s}  {'proj':>10s}  {'#groups':>8s}  {'global':>12s}  {'log2':>8s}")
            for li, c, n, gs in layer_globals:
                lg = math.log2(gs) if gs > 0 else float('-inf')
                print(f"{li:>6d}  {c:>10s}  {n:>8d}  {gs:>12.3e}  {lg:>8.2f}")
            globals_arr = np.array([gs for *_, gs in layer_globals])
            if globals_arr.size > 0 and (globals_arr > 0).all():
                lg = np.log2(globals_arr)
                print(f"\nglobal_scalar log2: min={lg.min():.2f} median={np.median(lg):.2f} "
                      f"max={lg.max():.2f}  (fp32 normal range: log2 in [-127, 127])")
                if lg.max() > 100 or lg.min() < -100:
                    print("WARNING: global_scalar exceeds safe fp32 range — snap precision will degrade.")

            print(f"\n=== Post-global merged scales (raw / global, NO snap applied) ===")
            print_scale_report(scaled_per_cat, tag=postg_tag)
            if not args.no_plot:
                plot_histograms(scaled_per_cat, {}, output_dir, postg_tag)
        return

    if args.weights:
        # ── Weight group scale analysis ──
        ckpt_dir = args.pt_path or gptq_dir

        if not os.path.isdir(ckpt_dir):
            print(f"Error: GPTQ checkpoint not found: {ckpt_dir}")
            # List available checkpoints
            grandparent = os.path.dirname(os.path.dirname(ckpt_dir))
            if os.path.isdir(grandparent):
                prefix = f'{args.mode}_'
                available = [d for d in os.listdir(grandparent) if d.startswith(prefix)]
                if available:
                    print(f"\nAvailable {args.mode} checkpoints:")
                    for d in sorted(available):
                        print(f"  {d}")
            sys.exit(1)

        print(f"Loading GPTQ checkpoint: {ckpt_dir}")
        cat_scales = extract_weight_scales(
            ckpt_dir, args.w_bits, args.groupsize, args.sym)

        tag = f'weights_{args.mode}_{quant_tag}'

        print_scale_report(cat_scales, tag=tag)

        if not args.no_plot:
            cat_scales_np = {k: np.array(v) for k, v in cat_scales.items()}
            plot_histograms(cat_scales_np, {}, output_dir, tag)

    else:
        # ── Activation scale analysis ──
        # When --fp16_calib is set, analyze the FP16 (pre-GPTQ) cal artifact
        # instead of the post-GPTQ cal. The FP16 cal is what this config
        # actually deploys with, so it's the right artifact to inspect.
        if args.fp16_calib:
            fp16_tag = cfg.build_quant_tag(args, for_fp16_cal_cache=True)
            resolved_path = cfg.resolve_fp16_act_scales_path(
                args.model, args.mode, fp16_tag)
        else:
            resolved_path = act_path
        pt_path = args.pt_path or resolved_path

        if not os.path.isfile(pt_path):
            print(f"Error: calibration file not found: {pt_path}")
            scales_dir = os.path.dirname(resolved_path)
            if os.path.isdir(scales_dir):
                prefix = f'{args.mode}_'
                available = [f for f in os.listdir(scales_dir) if f.startswith(prefix) and f.endswith('.pt')]
                if args.fp16_calib:
                    available = [f for f in available if f.endswith('__fp16.pt')]
                if available:
                    kind = 'FP16 calibrations' if args.fp16_calib else 'calibrations'
                    print(f"\nAvailable {args.mode} {kind}:")
                    for f in sorted(available):
                        print(f"  {f}")
            sys.exit(1)

        print(f"Loading: {pt_path}")
        data = torch.load(pt_path, map_location='cpu', weights_only=True)
        print(f"Keys: {len(data)}")

        cat_scales = {}
        cat_zeros = {}
        for key, entry in data.items():
            if not isinstance(entry, dict) or 'scale' not in entry:
                continue  # skip non-scale entries (e.g. __eq_factors__)
            cat, layer_idx, sublayer = categorize_key(key)
            scale = entry['scale'].numpy().flatten()
            zero = entry['zero'].numpy().flatten()
            cat_scales.setdefault(cat, []).extend(scale.tolist())
            cat_zeros.setdefault(cat, []).extend(zero.tolist())

        tag = os.path.basename(pt_path).replace('.pt', '')
        print_scale_report(cat_scales, cat_zeros, tag=tag)

        if not args.no_plot:
            cat_scales_np = {k: np.array(v) for k, v in cat_scales.items()}
            has_nonzero = {cat: np.any(np.array(v) != 0) for cat, v in cat_zeros.items()}
            cat_zeros_np = {k: np.array(v) for k, v in cat_zeros.items() if has_nonzero.get(k, False)}
            plot_histograms(cat_scales_np, cat_zeros_np, output_dir, tag)


if __name__ == '__main__':
    main()
