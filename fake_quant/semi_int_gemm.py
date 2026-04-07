"""Diagnostic GEMM with toggleable precision stages.

Mirrors int_gemm_capped_reference logic in float, with each precision-loss
stage independently toggleable via a bitmask.

Usage:
    --semi_int_gemm <mask>    e.g. "111111" (=int_gemm), "000000" (=float), "100000" (act quant only)

Bitmask stages:
    Bit 0: Act quantization (round + clamp + cast)
    Bit 1: Int16 hi/lo decomposition (split into two int8 GEMMs)
    Bit 2: Dot product rounding (partial.round())
    Bit 3: Tier-1 accumulator cap (clamp/wrap to acc_bits)
    Bit 4: Weight ZP correction rounding + re-cap
    Bit 5: Tier-2 integer accumulation (round + clamp)
"""

import logging
import math
from typing import Optional

import torch

from int_acc_gemm import (
    parse_acc_dtype,
    _decompose_int16_to_int8,
)


def _t2_acc_dtype_info(acc_dtype_str):
    """Return (torch_dtype, int_clamp_max, int_clamp_min) for tier-2 accumulator."""
    result = parse_acc_dtype(acc_dtype_str)
    kind, bits = result[0], result[1]
    if kind == 'int':
        max_val = 2 ** (bits - 1) - 1
        return torch.int64, max_val, -max_val - 1
    elif kind == 'bfloat':
        return torch.bfloat16, None, None
    elif kind == 'float' and bits == 16:
        return torch.float16, None, None
    return torch.float32, None, None


# ---------------------------------------------------------------------------
# Bitmask constants
# ---------------------------------------------------------------------------

MASK_ACT_QUANT = 0       # Activation quantization (round+clamp+cast)
MASK_INT16_DECOMP = 1    # Int16 hi/lo decomposition
MASK_DOT_ROUND = 2       # Dot product rounding
MASK_ACC_CAP = 3          # Tier-1 accumulator cap
MASK_WZP_ROUND_CAP = 4   # Weight ZP correction round + re-cap
MASK_T2_INT = 5           # Tier-2 integer accumulation

NUM_BITS = 6

_KEYWORDS = {
    'full': '1' * NUM_BITS,
    'none': '0' * NUM_BITS,
    'fake': '0' * NUM_BITS,
}

_BIT_NAMES = [
    'act_quant', 'int16_decomp', 'dot_round',
    'acc_cap', 'wzp_round_cap', 't2_int',
]


def parse_mask(mask_str: str) -> str:
    """Parse mask string: keyword or binary string. Returns NUM_BITS-char string of '0'/'1'."""
    s = mask_str.strip().lower()
    if s in _KEYWORDS:
        return _KEYWORDS[s]
    if len(s) != NUM_BITS or not all(c in '01' for c in s):
        raise ValueError(
            f"Invalid semi_int_gemm mask: '{mask_str}'. "
            f"Must be {NUM_BITS}-char binary string or keyword: {list(_KEYWORDS.keys())}")
    return s


def describe_mask(mask: str) -> str:
    """Human-readable description of which stages are enabled."""
    parts = []
    for i, name in enumerate(_BIT_NAMES):
        if mask[i] == '1':
            parts.append(name)
    return '+'.join(parts) if parts else 'none'


def _bit(mask, idx):
    return mask[idx] == '1'


# ---------------------------------------------------------------------------
# Main kernel
# ---------------------------------------------------------------------------

def semi_int_gemm(
    x_float: torch.Tensor,
    w_int: torch.Tensor,
    w_scale: torch.Tensor,
    act_quantizer,
    mask: str,
    acc_bits: int = 32,
    block_k: int = 32,
    w_group_size: int = -1,
    bias: Optional[torch.Tensor] = None,
    acc_wrap: bool = False,
    w_zp: Optional[torch.Tensor] = None,
    acc_dtype: str = 'float',
    w_shift_bias: int = 0,
    w_zp_correction: Optional[torch.Tensor] = None,
    w_zp_cross: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Diagnostic GEMM with toggleable precision stages."""
    orig_shape = x_float.shape
    K = orig_shape[-1]
    x_2d = x_float.reshape(-1, K)
    M = x_2d.shape[0]
    N = w_int.shape[0]

    # ------------------------------------------------------------------
    # Stage 0: Activation quantization
    # ------------------------------------------------------------------
    if not _bit(mask, MASK_ACT_QUANT):
        # No act quantization — dequantize weights and do plain float matmul.
        # Bits 1-5 are meaningless without integer activations.
        if w_group_size > 0 and w_scale.dim() == 2:
            n_groups = w_scale.shape[1]
            G = w_group_size
            w_float = (w_int.float().reshape(N, n_groups, G)
                       * w_scale.unsqueeze(2)).reshape(N, -1)[:, :K]
        else:
            w_float = w_int.float() * w_scale.unsqueeze(1)
        output = x_2d.float() @ w_float.t()
        if bias is not None:
            output = output + bias.unsqueeze(0)
        return output.reshape(*orig_shape[:-1], N)

    use_act_groups = (getattr(act_quantizer, 'groupsize', -1) > 0
                      and act_quantizer.static)

    a_int, a_scale, a_zp_correction = act_quantizer.quantize_to_int(x_2d)
    is_int16 = (a_int.dtype == torch.int16)

    # Bit 0 is on (guaranteed by early return above)
    a_work = a_int.float()

    # Resolve scales for post-loop
    if use_act_groups:
        a_group_scale = a_scale
        a_group_size = act_quantizer.groupsize
        a_token_scale = torch.empty(0, device=x_float.device)
    else:
        a_group_scale = None
        a_group_size = -1
        a_token_scale = a_scale  # [M]

    # ------------------------------------------------------------------
    # Stage 1: Int16 hi/lo decomposition
    # ------------------------------------------------------------------
    if _bit(mask, MASK_INT16_DECOMP) and is_int16 and _bit(mask, MASK_ACT_QUANT):
        # Decompose int16 → two int8 halves, run separately, combine
        a_hi, a_lo = _decompose_int16_to_int8(a_int)

        out_hi = _semi_int_kloop(
            a_hi.float(), w_int, w_scale, mask, acc_bits, block_k,
            w_group_size, acc_wrap, w_zp, acc_dtype, w_shift_bias,
            a_group_scale=a_group_scale, a_group_size=a_group_size,
            M=M, N=N, K=K)
        out_lo = _semi_int_kloop(
            a_lo.float(), w_int, w_scale, mask, acc_bits, block_k,
            w_group_size, acc_wrap, w_zp, acc_dtype, w_shift_bias,
            a_group_scale=a_group_scale, a_group_size=a_group_size,
            M=M, N=N, K=K)

        output = out_hi * 256.0 + out_lo
    else:
        # Single pass (no decomposition)
        output = _semi_int_kloop(
            a_work, w_int, w_scale, mask, acc_bits, block_k,
            w_group_size, acc_wrap, w_zp, acc_dtype, w_shift_bias,
            a_group_scale=a_group_scale, a_group_size=a_group_size,
            M=M, N=N, K=K)

    # ------------------------------------------------------------------
    # Post-loop: apply remaining scales (always float, same as int_gemm)
    # ------------------------------------------------------------------
    if a_group_size <= 0:
        output = output * a_token_scale.unsqueeze(1)

    if w_group_size <= 0:
        output = output * w_scale.unsqueeze(0)

    if bias is not None:
        output = output + bias.unsqueeze(0)

    # ------------------------------------------------------------------
    # Post-kernel ZP corrections (always float, same as int_gemm_capped)
    # ------------------------------------------------------------------
    if a_zp_correction is not None and w_zp_correction is not None:
        output = output + a_zp_correction.unsqueeze(1) * w_zp_correction.unsqueeze(0)

    if a_zp_correction is not None and w_zp_cross is not None:
        output = output + a_zp_correction.unsqueeze(1) * w_zp_cross.unsqueeze(0)

    # Debug comparison disabled — was verified identical per-layer

    return output.reshape(*orig_shape[:-1], N)


# ---------------------------------------------------------------------------
# K-block loop (core of the reference kernel)
# ---------------------------------------------------------------------------

def _semi_int_kloop(
    a_work: torch.Tensor,      # [M, K] float (quantized-as-float or raw scaled)
    w_int: torch.Tensor,       # [N, K] int8
    w_scale: torch.Tensor,     # [N] or [N, n_groups]
    mask: str,
    acc_bits: int,
    block_k: int,
    w_group_size: int,
    acc_wrap: bool,
    w_zp: Optional[torch.Tensor],
    acc_dtype: str,
    w_shift_bias: int,
    a_group_scale: Optional[torch.Tensor] = None,
    a_group_size: int = -1,
    M: int = 0, N: int = 0, K: int = 0,
) -> torch.Tensor:
    """K-block loop mirroring int_gemm_capped_reference with bitmask toggles."""

    # Tier-1 accumulator range
    acc_max = 2 ** (acc_bits - 1) - 1
    acc_min = -acc_max - 1
    acc_range = acc_max - acc_min + 1

    # Tier-2 accumulator setup
    t2_dtype, t2_clamp_max, t2_clamp_min = _t2_acc_dtype_info(acc_dtype)
    t2_is_int = t2_clamp_max is not None
    _, _, t2_frac_bits = parse_acc_dtype(acc_dtype)
    t2_frac_scale = float(1 << t2_frac_bits) if t2_frac_bits > 0 else 1.0

    if _bit(mask, MASK_T2_INT) and t2_is_int:
        output = torch.zeros(M, N, dtype=torch.int64, device=a_work.device)
    else:
        output = torch.zeros(M, N, dtype=torch.float32, device=a_work.device)

    for k_start in range(0, K, block_k):
        k_end = min(k_start + block_k, K)

        a_block = a_work[:, k_start:k_end]
        w_block = w_int[:, k_start:k_end]

        # Dot product
        partial = a_block.float() @ w_block.float().t()

        # Stage 2: Dot product rounding
        if _bit(mask, MASK_DOT_ROUND):
            partial = partial.round()

        # Stage 3: Tier-1 accumulator cap
        if _bit(mask, MASK_ACC_CAP):
            if acc_wrap:
                partial = ((partial.long() - acc_min) % acc_range + acc_min).float()
            else:
                partial = partial.clamp(acc_min, acc_max)

        # Stage 4: Weight ZP correction
        if w_zp is not None:
            a_block_sum = a_block.float().sum(dim=1)  # [M]
            if _bit(mask, MASK_ACT_QUANT):
                a_block_sum = a_block_sum.round()

            if w_group_size > 0 and w_zp.dim() == 2:
                g_idx = k_start // w_group_size
                w_zp_g = w_zp[:, g_idx]
            else:
                w_zp_g = w_zp

            correction = a_block_sum.unsqueeze(1) * w_zp_g.unsqueeze(0)
            if _bit(mask, MASK_WZP_ROUND_CAP):
                correction = correction.round()
            partial = partial + correction

            # Re-cap after correction
            if _bit(mask, MASK_WZP_ROUND_CAP) and _bit(mask, MASK_ACC_CAP):
                if acc_wrap:
                    partial = ((partial.long() - acc_min) % acc_range + acc_min).float()
                else:
                    partial = partial.clamp(acc_min, acc_max)

        # Per-group activation scale
        if a_group_size > 0 and a_group_scale is not None:
            a_g_idx = k_start // a_group_size
            partial = partial * a_group_scale[a_g_idx]

        # Weight group scale
        if w_group_size > 0:
            g_idx = k_start // w_group_size
            contrib = partial.float() * w_scale[:, g_idx].unsqueeze(0)
        else:
            contrib = partial.float()

        # Stage 5: Tier-2 accumulation
        if _bit(mask, MASK_T2_INT) and t2_is_int:
            if t2_frac_bits > 0:
                output += (contrib * t2_frac_scale).round().long()
            else:
                output += contrib.round().long()
            output = output.clamp(t2_clamp_min, t2_clamp_max)
        else:
            output += contrib.to(output.dtype)

    # Convert to float, undo shifts
    total_shift = t2_frac_bits + w_shift_bias
    output = output.float()
    if total_shift > 0:
        output *= 2.0 ** (-total_shift)

    return output
