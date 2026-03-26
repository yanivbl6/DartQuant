#!/usr/bin/env python3
"""Diagnostic: integer right-shift vs float multiply for undoing frac bits.

The >> frac at the end quantizes to integer steps → destroys fractional precision.
The fix: acc.float() * 2^(-(frac+bias)) preserves it.
"""

import torch

torch.manual_seed(42)

N = 512
n_blocks = 16

def accumulate(contribs, acc_bits, frac_bits, w_shift_bias, undo_mode):
    """
    undo_mode:
      'shift'   — >> frac (integer), then float * 2^(-bias)   [CURRENT]
      'float'   — to_float, then * 2^(-(frac+bias))           [PROPOSED FIX]
    """
    n_blocks, N = contribs.shape
    t2_max = 2 ** (acc_bits - 1) - 1
    t2_min = -t2_max - 1
    frac_scale = float(1 << frac_bits) if frac_bits > 0 else 1.0

    acc = torch.zeros(N, dtype=torch.int64)
    for i in range(n_blocks):
        c = contribs[i].double()
        if frac_bits > 0:
            acc += (c * frac_scale).round().long()
        else:
            acc += c.round().long()
        acc = acc.clamp(t2_min, t2_max)

    if undo_mode == 'shift':
        # Current: integer right-shift, then float multiply
        if frac_bits > 0:
            acc = (acc + (1 << (frac_bits - 1))) >> frac_bits
        result = acc.double()
        if w_shift_bias > 0:
            result *= 2.0 ** (-w_shift_bias)
    elif undo_mode == 'float':
        # Proposed: convert to float, single multiply
        result = acc.double() * (2.0 ** (-(frac_bits + w_shift_bias)))
    return result.float()


def ground_truth(contribs, w_shift_bias):
    return (contribs.double().sum(dim=0) * 2.0 ** (-w_shift_bias)).float()


bias = 10

print(f"{'min_contrib':>12s} | {'Config':<22s} | {'Mode':<7s} | {'MeanErr%':>9s} | {'MaxErr%':>9s}")
print('-' * 80)

for min_contrib in [125.0, 10.0, 2.5, 0.84, 0.3, 0.1, 0.01]:
    contribs = torch.rand(n_blocks, N) * min_contrib * 9 + min_contrib
    gt = ground_truth(contribs, bias)
    gt_abs = gt.abs().clamp(min=1e-15)

    for total_bits, frac in [(32, 0), (32, 4), (32, 8), (36, 12), (40, 16), (48, 24)]:
        for mode in ['shift', 'float']:
            result = accumulate(contribs, total_bits, frac, bias, mode)
            rel = ((result - gt).abs() / gt_abs)
            mean_e = rel.mean().item() * 100
            max_e = rel.max().item() * 100
            # Skip boring rows
            if mean_e < 0.001 and max_e < 0.01:
                continue
            label = f"int{total_bits}p{frac}"
            print(f"{min_contrib:>12.2f} | {label:<22s} | {mode:<7s} | {mean_e:>8.4f}% | {max_e:>8.4f}%")
    print()
