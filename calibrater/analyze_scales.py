#!/usr/bin/env python3
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
    """Return (act_scales_path, gptq_checkpoint_dir, quant_tag)."""
    quant_tag = cfg.build_quant_tag(args)
    act_path = cfg.resolve_act_scales_path(args.model, mode, quant_tag)
    gptq_dir = cfg.resolve_gptq_checkpoint_dir(args.model, mode, quant_tag, args.w_bits)
    return act_path, gptq_dir, quant_tag


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


def extract_weight_scales(checkpoint_dir, w_bits, w_groupsize, w_sym):
    """Load GPTQ checkpoint and extract per-group weight scales.

    Returns dict {category: numpy array of scale values}.
    """
    assert w_sym, "Weight scale analysis currently requires symmetric quantization"
    maxq = 2 ** (w_bits - 1) - 1

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
            scales = _recover_group_scales(W_grouped, maxq).numpy()
        else:
            # Per-channel
            scales = _recover_group_scales(W, maxq).numpy()

        cat_scales.setdefault(cat, []).extend(scales.tolist())

    return cat_scales


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
    parser.add_argument('--pt_path', type=str, default=None,
                        help='Explicit path to .pt file (activation mode) or GPTQ checkpoint dir (weight mode)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip histogram generation')

    args = parser.parse_args()
    cfg.resolve_v_bits(args)
    args.model = cfg.resolve_model(args.model)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'figures')

    act_path, gptq_dir, quant_tag = _resolve_paths(args, args.mode)

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
        pt_path = args.pt_path or act_path

        if not os.path.isfile(pt_path):
            print(f"Error: calibration file not found: {pt_path}")
            scales_dir = os.path.dirname(act_path)
            if os.path.isdir(scales_dir):
                prefix = f'{args.mode}_'
                available = [f for f in os.listdir(scales_dir) if f.startswith(prefix) and f.endswith('.pt')]
                if available:
                    print(f"\nAvailable {args.mode} calibrations:")
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
