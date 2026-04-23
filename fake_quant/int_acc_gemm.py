"""
Integer GEMM with capped accumulator — Triton kernel + PyTorch reference.

Simulates hardware integer GEMM where the accumulator has limited bit-width
(e.g., int16/int20 instead of full int32).  Activations and weights are
quantized to int8, multiplied via tensor-core int8 dot products, and the
partial sums are clamped to the accumulator range after every BLOCK_K
multiply-accumulate steps.

Usage (from ActQuantWrapper):
    from int_acc_gemm import int_gemm_capped, prepare_int_weights
"""

import math
import logging
import os
import re
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def parse_acc_dtype(s):
    """Parse tier-2 accumulator dtype string.

    Returns (kind, bits, frac_bits) where:
      kind      : 'int', 'float', or 'bfloat'
      bits      : total bit-width
      frac_bits : number of fractional bits (0 for float types)

    Format: int<N>[p<M>] — e.g. int32, int32p4, int26p8.
    Auto:   int<N>a<Y>  — frac_bits resolved from calibration at init time;
            returns ('int', N, 0) as a placeholder. Callers must run
            resolve_auto_acc_dtype(...) before the first forward so every
            qlayer.acc_dtype is rewritten to a concrete int<N>p<M>.
    Also:   float/fp32, half/fp16, bfloat/bf16.
    """
    s = s.lower().strip()
    aliases = {
        'float': ('float', 32, 0), 'fp32': ('float', 32, 0),
        'half': ('float', 16, 0), 'fp16': ('float', 16, 0),
        'bfloat': ('bfloat', 16, 0), 'bf16': ('bfloat', 16, 0),
    }
    if s in aliases:
        return aliases[s]
    m = re.match(r'^int(\d+)(?:p(\d+))?$', s)
    if m:
        bits = int(m.group(1))
        frac = int(m.group(2)) if m.group(2) else 0
        return ('int', bits, frac)
    m = re.match(r'^int(\d+)a(\d+)$', s)
    if m:
        return ('int', int(m.group(1)), 0)
    raise ValueError(
        f"Unknown acc_dtype: '{s}'. Must be int<N>[p<M>], int<N>a<Y> "
        "(e.g. int25, int32p4, int26p8, int24a1), "
        "or float/fp32, half/fp16, bfloat/bf16"
    )


def parse_acc_dtype_auto(s):
    """Return (total_bits, safety_bits) for 'int<N>a<Y>', else None.

    The auto form indicates per-layer T1→T2 shift resolution from
    calibration. `safety_bits` is reserved headroom beyond the observed
    output MSB. `total_bits` is the full T2 accumulator width (the X-bit
    clamp remains in force exactly as for any 'int<N>p<M>' run).
    """
    if not isinstance(s, str):
        return None
    m = re.match(r'^int(\d+)a(\d+)$', s.lower().strip())
    return (int(m.group(1)), int(m.group(2))) if m else None


def resolve_auto_acc_dtype(qlayers, total_bits, safety_bits):
    """Resolve per-layer acc_dtype for int<X>a<Y> mode.

    Walks every ActQuantWrapper using int_gemm, derives max|output| from
    the loaded out_quantizer calibration, computes a per-layer frac_bits,
    and rewrites qlayer.acc_dtype to 'int<total_bits>p<frac_bits>'.

    Raises RuntimeError if any eligible layer lacks calibrated
    out_quantizer scales, or if the bit budget is insufficient.

    The T2 clamp at 2^(total_bits-1)-1 is unchanged — this is purely a
    per-layer shift selection, not a clamp relaxation.
    """
    frac_per_layer = {}
    for name, qlayer in qlayers.items():
        if not getattr(qlayer, 'use_int_gemm', False):
            continue
        if 'lm_head' in name:
            continue

        oq = qlayer.out_quantizer
        # scale may be [N] (per-channel from calibration) or [1] (per-tensor
        # after hw_accurate collapse) — both are valid. Uninitialized scale
        # is zeros(1) with static=False, which fails the .static check.
        if not (getattr(oq, 'static', False) and oq.scale is not None
                and oq.scale.numel() >= 1):
            raise RuntimeError(
                f"int{total_bits}a{safety_bits} mode requires calibrated "
                f"out_quantizer for {name}, but none was found. Re-run with "
                f"--quant_out ex --realint --static-act (so out_quantizer "
                f"scales are captured and loaded), or use explicit "
                f"int{total_bits}p<M>.")
        if not getattr(oq, 'sym', False):
            raise RuntimeError(
                f"int{total_bits}a{safety_bits} mode expects symmetric "
                f"out_quantizer, but {name}.out_quantizer.sym is False. "
                f"Output quantizers are symmetric by convention — check "
                f"calibration config.")

        maxq = float(oq.maxq.item()) if torch.is_tensor(oq.maxq) else float(oq.maxq)
        abs_max = float((oq.scale.abs() * maxq).max().item())
        if abs_max <= 0 or not math.isfinite(abs_max):
            raise RuntimeError(
                f"int{total_bits}a{safety_bits}: {name} has non-positive "
                f"or non-finite abs_max={abs_max} from calibration.")

        I = max(0, math.ceil(math.log2(max(abs_max, 1e-30))))
        total_shift = (total_bits - 1) - I - safety_bits
        if total_shift < 0:
            raise RuntimeError(
                f"int{total_bits}a{safety_bits}: {name} output MSB={I} "
                f"(abs_max={abs_max:.3g}) exceeds the accumulator range. "
                f"Increase total bits or reduce safety.")

        wsb = int(getattr(qlayer, 'w_shift_bias', 0))
        frac_bits = total_shift - wsb
        if frac_bits < 0:
            raise RuntimeError(
                f"int{total_bits}a{safety_bits}: {name} has "
                f"w_shift_bias={wsb}, output MSB={I}, safety={safety_bits} "
                f"— leaves no room for frac_bits (got {frac_bits}). "
                f"Disable hwscale/gscaler or increase total bits.")

        qlayer.acc_dtype = f'int{total_bits}p{frac_bits}'
        frac_per_layer[name] = frac_bits
        logging.info("[int_acc_auto] %s: abs_max=%.3g I=%d w_shift_bias=%d "
                     "safety=%d -> int%dp%d",
                     name, abs_max, I, wsb, safety_bits, total_bits, frac_bits)

    if frac_per_layer:
        vals = sorted(frac_per_layer.values())
        logging.info("[int_acc_auto] summary: %d layers resolved, "
                     "frac_bits min=%d median=%d max=%d",
                     len(vals), vals[0], vals[len(vals) // 2], vals[-1])
    else:
        logging.warning("[int_acc_auto] no int_gemm layers found to resolve")

# Triton is optional — fall back to the PyTorch reference if unavailable.
_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# A. Triton kernel
# ---------------------------------------------------------------------------
if _HAS_TRITON:
    @triton.jit
    def _int_gemm_capped_acc_kernel(
        # Pointers
        A_ptr, B_ptr, C_ptr,
        A_scale_ptr,   # [M]  per-token activation scale  (used when A_GROUP_SIZE <= 0)
        B_scale_ptr,   # [N]  per-channel weight scale  (used when GROUP_SIZE <= 0)
        bias_ptr,
        # Dimensions
        M, N, K,
        # Strides — A is [M, K] row-major, B is [N, K] row-major
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        # Flags
        HAS_BIAS: tl.constexpr,
        GROUP_SIZE: tl.constexpr,       # weight group size (-1 = per-channel)
        # Grouped weight scales — [N, K // GROUP_SIZE] (only when GROUP_SIZE > 0)
        B_gscale_ptr,
        stride_bgs_n, stride_bgs_g,
        # Per-group activation scales — [K // A_GROUP_SIZE] (only when A_GROUP_SIZE > 0)
        A_GROUP_SIZE: tl.constexpr,     # activation group size (-1 = per-token)
        A_gscale_ptr,
        # Tier-1 accumulator cap
        ACC_MAX: tl.constexpr,
        ACC_MIN: tl.constexpr,
        ACC_WRAP: tl.constexpr,     # True = two's-complement wrap-around; False = saturation
        ACC_RANGE: tl.constexpr,    # = ACC_MAX - ACC_MIN + 1 (= 2^acc_bits)
        # Tier-2 accumulator dtype control
        T2_IS_INT: tl.constexpr,    # True = integer tier-2 (simulated via clamp)
        T2_MAX: tl.constexpr,       # tier-2 int clamp max (ignored if T2_IS_INT=False)
        T2_MIN: tl.constexpr,       # tier-2 int clamp min (ignored if T2_IS_INT=False)
        T2_IS_FP16: tl.constexpr,   # True = fp16 tier-2
        T2_IS_BF16: tl.constexpr,   # True = bfloat16 tier-2
        # Fixed-point fractional bits for integer tier-2 (int32p4 → 4)
        T2_FRAC_BITS: tl.constexpr,
        T2_FRAC_SCALE: tl.constexpr,  # = 2^T2_FRAC_BITS as float (precomputed)
        # Gscaler shift bias — w_scale was prescaled by 2^bias; undo after K-loop
        W_SHIFT_BIAS: tl.constexpr,
        # hwscale per-layer global FP32 scale — applied once at T2 post-loop.
        # NOT constexpr: per-layer varying values would trigger a Triton
        # recompile each time. Passed as a runtime scalar; multiply-by-1.0
        # is a no-op so the default path pays effectively zero cost.
        GLOBAL_SCALE_FP32,
        # Weight zero-point correction (asymmetric weights)
        HAS_W_ZP: tl.constexpr,    # True = apply w_zp correction inside accumulator
        W_zp_ptr,                   # [N, n_groups] or [N] int32 — weight zero points
        stride_wzp_n,               # stride for N dimension of w_zp
        stride_wzp_g,               # stride for group dimension of w_zp
        # Tile sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ACC_BLOCK_K: tl.constexpr,  # logical accumulation block size (may be < BLOCK_K for padding)
        MAC_SHIFT: tl.constexpr,    # right-shift tl.dot partial by N bits (0 = disabled)
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Pointers for the first K-block
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

        # Tier-2 accumulator — dtype depends on acc_dtype parameter
        if T2_IS_INT:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        elif T2_IS_FP16:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
        elif T2_IS_BF16:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.bfloat16)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, ACC_BLOCK_K):
            k_mask = (offs_k < ACC_BLOCK_K) & ((k_start + offs_k) < K)

            # Load int8 tiles
            a_mask = (offs_m[:, None] < M) & (k_mask[None, :])
            b_mask = (offs_n[:, None] < N) & (k_mask[None, :])
            a_tile = tl.load(a_ptrs, mask=a_mask, other=0)
            b_tile = tl.load(b_ptrs, mask=b_mask, other=0)

            # int8 × int8 → int32 dot product
            # tl.dot expects [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
            # b_tile is [BLOCK_N, BLOCK_K] so we transpose it
            partial = tl.dot(a_tile, tl.trans(b_tile))  # [BLOCK_M, BLOCK_N] int32

            # LSB MAC shift: reduce magnitude before tier-1 cap to prevent wrap
            if MAC_SHIFT > 0:
                partial = partial >> MAC_SHIFT

            # Cap to tier-1 accumulator range
            if ACC_WRAP:
                partial = ((partial - ACC_MIN) % ACC_RANGE + ACC_RANGE) % ACC_RANGE + ACC_MIN
            else:
                partial = tl.minimum(tl.maximum(partial, ACC_MIN), ACC_MAX)

            # Weight zero-point correction: partial += sum(a_block) * w_zp
            # Applied after dot-product cap, then capped again.
            if HAS_W_ZP:
                a_block_sum = tl.sum(a_tile.to(tl.int32), axis=1)  # [BLOCK_M]
                if GROUP_SIZE > 0:
                    g_idx_zp = k_start // GROUP_SIZE
                    w_zp_g = tl.load(W_zp_ptr + offs_n * stride_wzp_n + g_idx_zp * stride_wzp_g,
                                     mask=offs_n < N, other=0)     # [BLOCK_N] int32
                else:
                    w_zp_g = tl.load(W_zp_ptr + offs_n * stride_wzp_n,
                                     mask=offs_n < N, other=0)     # [BLOCK_N] int32
                correction = a_block_sum[:, None] * w_zp_g[None, :]
                if MAC_SHIFT > 0:
                    correction = correction >> MAC_SHIFT
                partial = partial + correction
                # Cap/wrap again after correction
                if ACC_WRAP:
                    partial = ((partial - ACC_MIN) % ACC_RANGE + ACC_RANGE) % ACC_RANGE + ACC_MIN
                else:
                    partial = tl.minimum(tl.maximum(partial, ACC_MIN), ACC_MAX)

            partial_f = partial.to(tl.float32)

            # Undo LSB MAC shift in float domain
            if MAC_SHIFT > 0:
                partial_f = partial_f * (1 << MAC_SHIFT)

            # Per-group activation scale (static mode)
            if A_GROUP_SIZE > 0:
                a_g_idx = k_start // A_GROUP_SIZE
                a_g_scale = tl.load(A_gscale_ptr + a_g_idx)
                partial_f = partial_f * a_g_scale

            # Compute contribution (partial * weight group scale)
            if GROUP_SIZE > 0:
                # Grouped weight scales — one scale per (N, group_index)
                g_idx = k_start // GROUP_SIZE
                g_scale = tl.load(
                    B_gscale_ptr + offs_n * stride_bgs_n + g_idx * stride_bgs_g,
                    mask=offs_n < N, other=1.0)
                contrib = partial_f * g_scale[None, :]
            else:
                contrib = partial_f

            # Accumulate in tier-2 dtype
            if T2_IS_INT:
                if T2_FRAC_BITS > 0:
                    scaled = contrib * T2_FRAC_SCALE
                else:
                    scaled = contrib
                # Round-to-nearest (not truncate) for faithful fixed-point
                acc += (scaled + tl.where(scaled >= 0, 0.5, -0.5)).to(tl.int32)
                # Clamp to tier-2 accumulator range
                acc = tl.minimum(tl.maximum(acc, T2_MIN), T2_MAX)
            elif T2_IS_FP16:
                acc += contrib.to(tl.float16)
            elif T2_IS_BF16:
                acc += contrib.to(tl.bfloat16)
            else:
                acc += contrib

            # Advance pointers
            a_ptrs += ACC_BLOCK_K * stride_ak
            b_ptrs += ACC_BLOCK_K * stride_bk

        # Convert to float, then undo frac bits and gscaler prescaling together
        acc = acc.to(tl.float32)
        if T2_FRAC_BITS > 0:
            if W_SHIFT_BIAS > 0:
                acc = acc * (2.0 ** (-(T2_FRAC_BITS + W_SHIFT_BIAS)))
            else:
                acc = acc * (2.0 ** (-T2_FRAC_BITS))
        elif W_SHIFT_BIAS > 0:
            acc = acc * (2.0 ** (-W_SHIFT_BIAS))

        # hwscale per-layer global FP32 scale — merges the "scale-the-scales"
        # correction that replaces the gscaler exponent bias for --hwscale.
        # Unconditional multiply (runtime scalar, no branch, no-op at 1.0).
        acc = acc * GLOBAL_SCALE_FP32

        # Apply scales that weren't folded in during the K-block loop
        if A_GROUP_SIZE <= 0:
            # Per-token activation scale (dynamic mode)
            a_scale = tl.load(A_scale_ptr + offs_m, mask=offs_m < M, other=1.0)
            acc = acc * a_scale[:, None]

        if GROUP_SIZE <= 0:
            # Per-channel weight scale
            b_scale = tl.load(B_scale_ptr + offs_n, mask=offs_n < N, other=1.0)
            acc = acc * b_scale[None, :]

        # Add bias
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
            acc += bias[None, :]

        # Store
        c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=mask)


# ---------------------------------------------------------------------------
# B. PyTorch reference implementation (no Triton needed)
# ---------------------------------------------------------------------------
def _t2_acc_dtype_info(acc_dtype_str):
    """Return (torch_dtype, int_clamp_max, int_clamp_min) for tier-2 accumulator.

    For float types: returns (torch_dtype, None, None).
    For int types: returns (torch.int64, max_val, min_val) — int64 storage with clamping.
    """
    kind, bits, _frac = parse_acc_dtype(acc_dtype_str)
    if kind == 'int':
        max_val = 2 ** (bits - 1) - 1
        min_val = -max_val - 1
        return torch.int64, max_val, min_val
    elif kind == 'bfloat':
        return torch.bfloat16, None, None
    elif kind == 'float' and bits == 16:
        return torch.float16, None, None
    else:  # float32
        return torch.float32, None, None


def _int16_gemm_reference(
    a_int16: torch.Tensor,      # [M, K] int16
    a_scale: torch.Tensor,
    w_int: torch.Tensor,
    w_scale: torch.Tensor,
    acc_bits: int,
    block_k: int,
    w_group_size: int = -1,
    bias: Optional[torch.Tensor] = None,
    acc_wrap: bool = False,
    a_group_scale: Optional[torch.Tensor] = None,
    a_group_size: int = -1,
    acc_dtype: str = 'float',
    w_shift_bias: int = 0,
    w_zp: Optional[torch.Tensor] = None,
    lsb_mac_shift: int = 0,
    global_scale_fp32: float = 1.0,
) -> torch.Tensor:
    """Int16 reference GEMM via two int8 reference calls with carry decomposition."""
    M, K = a_int16.shape
    N = w_int.shape[0]
    a_hi, a_lo = _decompose_int16_to_int8(a_int16)

    # Both halves are int8 — MSB keeps original acc_bits, LSB gets extra bits
    out_hi = int_gemm_capped_reference(
        a_hi, a_scale, w_int, w_scale, acc_bits, block_k,
        w_group_size, None, acc_wrap,
        a_group_scale=a_group_scale, a_group_size=a_group_size,
        acc_dtype=acc_dtype, w_shift_bias=w_shift_bias, w_zp=w_zp,
        global_scale_fp32=global_scale_fp32)
    out_lo = int_gemm_capped_reference(
        a_lo, a_scale, w_int, w_scale, acc_bits + lsb_mac_shift, block_k,
        w_group_size, None, acc_wrap,
        a_group_scale=a_group_scale, a_group_size=a_group_size,
        acc_dtype=acc_dtype, w_shift_bias=w_shift_bias, w_zp=w_zp,
        global_scale_fp32=global_scale_fp32)

    output = out_hi * 256.0 + out_lo
    if bias is not None:
        output += bias.unsqueeze(0)
    return output


def int_gemm_capped_reference(
    a_int: torch.Tensor,        # [M, K] int8 or int16
    a_scale: torch.Tensor,      # [M] float  (used when a_group_size <= 0)
    w_int: torch.Tensor,        # [N, K] int8
    w_scale: torch.Tensor,      # [N] or [N, n_groups] float
    acc_bits: int,
    block_k: int,
    w_group_size: int = -1,
    bias: Optional[torch.Tensor] = None,
    acc_wrap: bool = False,
    a_group_scale: Optional[torch.Tensor] = None,  # [n_a_groups] float
    a_group_size: int = -1,
    acc_dtype: str = 'float',
    w_shift_bias: int = 0,
    w_zp: Optional[torch.Tensor] = None,  # [N] or [N, n_groups] — weight zero-point correction
    mac_shift: int = 0,
    lsb_mac_shift: int = 0,
    global_scale_fp32: float = 1.0,
) -> torch.Tensor:
    """Pure-PyTorch reference for capped-accumulator integer GEMM."""
    M, K = a_int.shape
    N = w_int.shape[0]

    # --- int16: delegate to decomposition wrapper ---
    if a_int.dtype == torch.int16:
        return _int16_gemm_reference(
            a_int, a_scale, w_int, w_scale, acc_bits, block_k,
            w_group_size, bias, acc_wrap,
            a_group_scale=a_group_scale, a_group_size=a_group_size,
            acc_dtype=acc_dtype, w_shift_bias=w_shift_bias, w_zp=w_zp,
            lsb_mac_shift=lsb_mac_shift,
            global_scale_fp32=global_scale_fp32)

    acc_max = 2 ** (acc_bits - 1) - 1
    acc_min = -acc_max - 1
    acc_range = acc_max - acc_min + 1  # = 2^effective_acc_bits

    # Tier-2 accumulator setup
    t2_dtype, t2_clamp_max, t2_clamp_min = _t2_acc_dtype_info(acc_dtype)
    t2_is_int = t2_clamp_max is not None
    _, _, t2_frac_bits = parse_acc_dtype(acc_dtype)
    t2_frac_scale = float(1 << t2_frac_bits) if t2_frac_bits > 0 else 1.0
    output = torch.zeros(M, N, dtype=t2_dtype, device=a_int.device)

    for k_start in range(0, K, block_k):
        k_end = min(k_start + block_k, K)
        # Compute dot product in float (PyTorch lacks int matmul on CUDA),
        # then round to int semantics and cap to accumulator range.
        partial = a_int[:, k_start:k_end].float() @ w_int[:, k_start:k_end].float().t()
        partial = partial.round()
        # LSB MAC shift: reduce magnitude before tier-1 cap
        if mac_shift > 0:
            partial = (partial.long() >> mac_shift).float()
        if acc_wrap:
            partial = ((partial.long() - acc_min) % acc_range + acc_min).float()
        else:
            partial = partial.clamp(acc_min, acc_max)

        # Weight zero-point correction: partial += sum(a_block) * w_zp
        if w_zp is not None:
            a_block_sum = a_int[:, k_start:k_end].long().sum(dim=1)  # [M]
            if w_group_size > 0 and w_zp.dim() == 2:
                g_idx = k_start // w_group_size
                w_zp_g = w_zp[:, g_idx]                               # [N]
            else:
                w_zp_g = w_zp                                          # [N]
            correction = (a_block_sum.unsqueeze(1) * w_zp_g.unsqueeze(0)).round()
            if mac_shift > 0:
                correction = (correction.long() >> mac_shift).float()
            partial = partial + correction
            # Cap/wrap again after correction
            if acc_wrap:
                partial = ((partial.long() - acc_min) % acc_range + acc_min).float()
            else:
                partial = partial.clamp(acc_min, acc_max)

        # Undo LSB MAC shift in float domain
        if mac_shift > 0:
            partial = partial * float(1 << mac_shift)

        # Per-group activation scale (static mode)
        if a_group_size > 0 and a_group_scale is not None:
            a_g_idx = k_start // a_group_size
            partial = partial * a_group_scale[a_g_idx]

        # Scale by weight group scale, then cast to tier-2 dtype and accumulate
        if w_group_size > 0:
            g_idx = k_start // w_group_size
            contrib = partial.float() * w_scale[:, g_idx].unsqueeze(0)
        else:
            contrib = partial.float()

        if t2_is_int:
            if t2_frac_bits > 0:
                output += (contrib * t2_frac_scale).round().long()
            else:
                output += contrib.round().long()
            output = output.clamp(t2_clamp_min, t2_clamp_max)
        else:
            output += contrib.to(t2_dtype)

    # Convert to float, then undo frac bits and gscaler prescaling together
    total_shift = t2_frac_bits + w_shift_bias
    output = output.float()
    if total_shift > 0:
        output *= 2.0 ** (-total_shift)

    # hwscale per-layer global FP32 scale (mirror of the Triton kernel)
    if global_scale_fp32 != 1.0:
        output *= global_scale_fp32

    # Apply scales that weren't folded in during the K-block loop
    if a_group_size <= 0:
        output *= a_scale.unsqueeze(1)

    if w_group_size <= 0:
        output *= w_scale.unsqueeze(0)

    if bias is not None:
        output += bias.unsqueeze(0)

    return output


# ---------------------------------------------------------------------------
# C. Main wrapper — dispatches to Triton or reference
# ---------------------------------------------------------------------------
def int_gemm_capped(
    x_float: torch.Tensor,
    w_int: torch.Tensor,
    w_scale: torch.Tensor,
    act_quantizer,               # ActQuantizer instance
    acc_bits: int = 32,
    block_k: int = 32,
    w_group_size: int = -1,
    bias: Optional[torch.Tensor] = None,
    use_triton: bool = True,
    acc_wrap: bool = False,
    w_zp_correction: Optional[torch.Tensor] = None,
    w_zp: Optional[torch.Tensor] = None,
    w_zp_cross: Optional[torch.Tensor] = None,
    acc_dtype: str = 'float',
    w_shift_bias: int = 0,
    lsb_mac_shift: int = 0,
    global_scale_fp32: float = 1.0,
    _t1_scan_label: str = '',
) -> torch.Tensor:
    """
    Quantize activations → run integer GEMM with capped accumulator → float output.

    Parameters
    ----------
    x_float : [..., K] float activations (pre-rotation, post-Hadamard)
    w_int   : [N, K] int8 weight matrix
    w_scale : [N] or [N, n_groups] float weight scales (prescaled by 2^w_shift_bias
              when gscaler is active)
    act_quantizer : ActQuantizer — used to quantize x_float to int8 + scale
    acc_bits : accumulator bit-width (32 = no capping)
    block_k  : K-block size for capping granularity
    w_group_size : weight group size (-1 = per-channel)
    bias    : optional [N] float bias
    use_triton : use Triton kernel if available
    acc_wrap : use two's-complement wrap-around instead of saturation
    w_zp_correction : optional [N] float — precomputed Σ_k s_w(j,k)·q_w(j,k),
        used with asymmetric activations to correct for the zero-point shift.
    w_zp : optional [N] or [N, n_w_groups] float — weight zero points for
        asymmetric weight quantization.  The correction w_zp * sum(a_int) is
        applied in float outside the capped accumulator.
    w_zp_cross : optional [N] float — precomputed cross-term
        Σ_g(w_zp[j,g] * w_scale[j,g] * G) for dynamic activation + weight
        asymmetry.  Combined with per-token a_zp_correction at runtime.
    w_shift_bias : gscaler bias — tier-2 accumulates prescaled values,
        then multiplies by 2^(-w_shift_bias) after the K-loop.

    Returns
    -------
    [..., N] float output
    """
    orig_shape = x_float.shape
    K = orig_shape[-1]
    x_2d = x_float.reshape(-1, K)
    M = x_2d.shape[0]
    N = w_int.shape[0]

    # Quantize activations to int8
    a_int, a_scale, a_zp_correction = act_quantizer.quantize_to_int(x_2d)

    # Detect per-group activation mode: quantize_to_int returns [n_groups]
    # instead of [M] when using static per-group scales.
    use_act_groups = (getattr(act_quantizer, 'groupsize', -1) > 0
                      and act_quantizer.static)
    if use_act_groups:
        a_group_scale = a_scale      # [n_groups]
        a_group_size = act_quantizer.groupsize
        a_token_scale = torch.empty(0, device=x_float.device)
    else:
        a_group_scale = None
        a_group_size = -1
        a_token_scale = a_scale      # [M]

    # Dispatch — w_zp correction is now inside the kernel (before accumulator cap)
    is_int16 = (a_int.dtype == torch.int16)

    if use_triton and _HAS_TRITON and x_float.is_cuda:
        if is_int16:
            output = _triton_int16_gemm(
                a_int, a_token_scale, w_int, w_scale,
                acc_bits, block_k, w_group_size, bias,
                M, N, K, acc_wrap,
                a_gscale=a_group_scale,
                a_group_size=a_group_size,
                acc_dtype=acc_dtype,
                w_shift_bias=w_shift_bias,
                w_zp=w_zp,
                lsb_mac_shift=lsb_mac_shift,
                global_scale_fp32=global_scale_fp32,
                _t1_scan_label=_t1_scan_label)
        else:
            output = _triton_int_gemm(
                a_int, a_token_scale, w_int, w_scale,
                acc_bits, block_k, w_group_size, bias,
                M, N, K, acc_wrap,
                a_gscale=a_group_scale,
                a_group_size=a_group_size,
                acc_dtype=acc_dtype,
                w_shift_bias=w_shift_bias,
                w_zp=w_zp,
                global_scale_fp32=global_scale_fp32)
    else:
        # Reference impl handles int16 internally via _int16_gemm_reference
        output = int_gemm_capped_reference(a_int, a_token_scale, w_int, w_scale,
                                           acc_bits, block_k, w_group_size,
                                           bias, acc_wrap,
                                           a_group_scale=a_group_scale,
                                           a_group_size=a_group_size,
                                           acc_dtype=acc_dtype,
                                           w_shift_bias=w_shift_bias,
                                           w_zp=w_zp,
                                           lsb_mac_shift=lsb_mac_shift,
                                           global_scale_fp32=global_scale_fp32)

    # Zero-point correction for asymmetric activations (applied in float,
    # outside the capped accumulator).
    # Term: a_zp * Σ_k(s_w * q_w_signed)  per output channel
    if a_zp_correction is not None and w_zp_correction is not None:
        output += a_zp_correction.unsqueeze(1) * w_zp_correction.unsqueeze(0)

    # NOTE: Weight zero-point correction (old Term C) is now inside the kernel,
    # applied before the accumulator cap/wrap. No post-kernel correction needed.

    # Cross-term for combined activation + weight asymmetry (dynamic mode).
    # Term: a_zp_correction[m] * Σ_g(w_zp[j,g] * w_scale[j,g] * G)
    # For static mode this is folded into static_zp_bias instead.
    if a_zp_correction is not None and w_zp_cross is not None:
        output += a_zp_correction.unsqueeze(1) * w_zp_cross.unsqueeze(0)

    return output.reshape(*orig_shape[:-1], N)


def _apply_weight_zp_correction(
    output: torch.Tensor,       # [M, N] — modified in-place
    a_int: torch.Tensor,        # [M, K] int8
    a_token_scale: torch.Tensor,  # [M] or empty
    a_group_scale: Optional[torch.Tensor],  # [n_act_groups] or None
    a_group_size: int,
    w_zp: torch.Tensor,         # [N] or [N, n_w_groups]
    w_scale: torch.Tensor,      # [N] or [N, n_w_groups]
    w_group_size: int,
    M: int, K: int,
    use_act_groups: bool,
):
    """Compute and add the weight zero-point correction to *output* in-place.

    correction[m, j] = Σ over weight-groups g of:
        w_zp[j,g] * ( Σ_{k in group g} a_scale(k) * a_int[m,k] ) * w_scale[j,g]

    All arithmetic is in float, outside the capped accumulator.
    """
    a_f = a_int.float()

    if w_group_size > 0 and w_zp.dim() == 2:
        # --- Per-group weights ---
        n_w_groups = w_zp.shape[1]
        G = w_group_size
        padded_K = n_w_groups * G
        if padded_K > K:
            a_padded = F.pad(a_f, (0, padded_K - K))
        else:
            a_padded = a_f
        # [M, n_w_groups, G] -> sum over G -> [M, n_w_groups]
        a_group_sums = a_padded.reshape(M, n_w_groups, G).sum(dim=2)

        # w_zp_scaled[j, g] = w_zp[j,g] * w_scale[j,g]
        w_zp_scaled = w_zp * w_scale  # [N, n_w_groups]

        if use_act_groups:
            # Each activation group has its own scale.  We need to weight
            # each k-element's contribution by its activation group scale.
            # a_group_size == acc_block_k by design.
            n_act_groups = a_group_scale.shape[0]
            # Scale each activation element by its group scale, then re-sum
            # per weight group.
            a_scaled = a_f * a_group_scale.repeat_interleave(a_group_size).unsqueeze(0)[:, :K]
            if padded_K > K:
                a_scaled = F.pad(a_scaled, (0, padded_K - K))
            a_scaled_sums = a_scaled.reshape(M, n_w_groups, G).sum(dim=2)  # [M, n_w_groups]
            correction = a_scaled_sums @ w_zp_scaled.T  # [M, N]
        else:
            # Per-token scale: factor out after the matmul
            correction = (a_group_sums @ w_zp_scaled.T) * a_token_scale.unsqueeze(1)
    else:
        # --- Per-channel weights ---
        a_sum = a_f.sum(dim=1)  # [M]
        w_zp_scaled = w_zp * w_scale  # [N]

        if use_act_groups:
            # Weight each k by its activation group scale, then sum
            n_act_groups = a_group_scale.shape[0]
            a_scaled = a_f * a_group_scale.repeat_interleave(a_group_size).unsqueeze(0)[:, :K]
            a_scaled_sum = a_scaled.sum(dim=1)  # [M]
            correction = a_scaled_sum.unsqueeze(1) * w_zp_scaled.unsqueeze(0)
        else:
            correction = (a_token_scale * a_sum).unsqueeze(1) * w_zp_scaled.unsqueeze(0)

    output += correction


def _decompose_int16_to_int8(a_int16):
    """Decompose int16 activations into (a_hi, a_lo) both signed int8.

    Identity (with signed reinterpretation of the low byte):
        a_int16 = a_hi_adj * 256 + a_lo_signed
    where a_lo_signed = int8 reinterpret of (a_int16 & 0xFF), and
    a_hi_adj = (a_int16 >> 8) + carry, carry = 1 if unsigned_lo >= 128.

    Overflow note: a_hi_adj can reach 128 when a_int16 is in [32640, 32767],
    which overflows int8.  We pre-clamp a_int16 to [-32768, 32639] so the
    decomposition is exact.  For bits <= 15 (maxq <= 16383) this never
    triggers.  For bits=16 it clips 128 out of 65536 values (0.2%).

    Returns (a_hi, a_lo) both int8.
    """
    a32 = a_int16.to(torch.int32).clamp(-32768, 32639)
    a_lo = (a32 & 0xFF).to(torch.int8)             # reinterpret as signed
    carry = (a32 & 0x80) >> 7                       # 1 if unsigned lo >= 128 (int32)
    a_hi = ((a32 >> 8) + carry).to(torch.int8)     # safe: max is 127 after clamp
    return a_hi, a_lo


def _triton_int16_gemm(
    a_int16, a_scale, w_int, w_scale,
    acc_bits, block_k, w_group_size, bias,
    M, N, K, acc_wrap=False,
    a_gscale=None, a_group_size=-1,
    acc_dtype='float',
    w_shift_bias=0,
    w_zp=None,
    lsb_mac_shift=0,
    global_scale_fp32=1.0,
    _t1_scan_label='',
):
    """Int16 GEMM via two int8 kernel calls.

    Decomposes int16 activations into signed (a_hi, a_lo) int8 bytes
    using carry absorption so that:  a_int16 = a_hi * 256 + a_lo  (exact).

    Result = kernel(a_hi, w) * 256 + kernel(a_lo, w).

    Both calls use the unmodified int8 Triton kernel with the same acc_bits.
    lsb_mac_shift: widen the LSB accumulator by N extra bits
    (e.g. lsb_mac_shift=2 with acc_bits=16 → LSB uses acc_bits=18).
    MSB keeps original acc_bits. No shifting applied.
    """
    a_hi, a_lo = _decompose_int16_to_int8(a_int16)

    _common = dict(
        a_scale=a_scale, w_int=w_int, w_scale=w_scale,
        block_k=block_k, w_group_size=w_group_size, bias=None,
        M=M, N=N, K=K, acc_wrap=acc_wrap,
        a_gscale=a_gscale, a_group_size=a_group_size,
        acc_dtype=acc_dtype, w_shift_bias=w_shift_bias, w_zp=w_zp,
        global_scale_fp32=global_scale_fp32,
    )

    out_hi = _triton_int_gemm(a_hi, acc_bits=acc_bits, **_common)
    out_lo = _triton_int_gemm(a_lo, acc_bits=acc_bits + lsb_mac_shift, **_common)

    # t1_msb_scan: compare MSB and LSB individually against acc32
    if _t1_scan_label:
        _ref_common = dict(_common, acc_wrap=False)
        ref_hi = _triton_int_gemm(a_hi, acc_bits=32, **_ref_common)
        ref_lo = _triton_int_gemm(a_lo, acc_bits=32, **_ref_common)
        results = {}
        for tag, out, ref in [('MSB', out_hi, ref_hi), ('LSB', out_lo, ref_lo)]:
            d = (out.float() - ref.float())
            ma = d.abs().max().item()
            rms = (ref.float() ** 2).mean().sqrt().item() + 1e-12
            nrmse = (d ** 2).mean().sqrt().item() / rms
            nd = int((d.abs() > 1e-6).sum().item())
            results[tag] = (nrmse, ma, nd)
        # Only print when the outer scan will also flag a mismatch
        if any(r[1] > 1e-6 for r in results.values()):
            msb_n, lsb_n = results['MSB'][0], results['LSB'][0]
            culprit = 'MSB' if msb_n > lsb_n else 'LSB' if lsb_n > msb_n else 'BOTH'
            for tag in ['MSB', 'LSB']:
                nrmse, ma, nd = results[tag]
                marker = ' <<<' if tag == culprit or culprit == 'BOTH' else ''
                print(f"[t1_msb_scan] {_t1_scan_label} {tag}: "
                      f"NRMSE={nrmse:.4e}, max_abs={ma:.4e}, n_diff={nd}{marker}",
                      flush=True)

    output = out_hi * 256.0 + out_lo
    if bias is not None:
        output += bias.unsqueeze(0)
    return output


def _triton_int_gemm(
    a_int, a_scale, w_int, w_scale,
    acc_bits, block_k, w_group_size, bias,
    M, N, K, acc_wrap=False,
    a_gscale=None, a_group_size=-1,
    acc_dtype='float',
    w_shift_bias=0,
    w_zp=None,
    mac_shift=0,
    global_scale_fp32=1.0,
):
    """Launch the Triton int8 kernel.  a_int must be int8."""
    assert a_int.dtype == torch.int8, \
        f"_triton_int_gemm requires int8 activations (got {a_int.dtype})"

    BLOCK_M = 32
    BLOCK_N = 64
    ACC_BLOCK_K = block_k
    BLOCK_K = max(32, block_k)  # Triton tl.dot requires K >= 32; pad if needed

    # float32 can't represent INT32_MAX (2^31-1) exactly — it rounds up to
    # 2^31 which overflows int32.  Cap clamp boundaries at 2^30 so
    # tl.minimum/tl.maximum constexprs survive float32 promotion in Triton.
    _F32_SAFE = 2 ** 30

    acc_max = min(2 ** (acc_bits - 1) - 1, _F32_SAFE)
    acc_min = -acc_max
    acc_range = acc_max - acc_min + 1

    # Tier-2 accumulator constexprs
    t2_kind, t2_bits, t2_frac_bits = parse_acc_dtype(acc_dtype)
    t2_is_int = (t2_kind == 'int')
    t2_max = min(2 ** (t2_bits - 1) - 1, _F32_SAFE) if t2_is_int else 0
    t2_min = -t2_max if t2_is_int else 0
    t2_is_fp16 = (t2_kind == 'float' and t2_bits == 16)
    t2_is_bf16 = (t2_kind == 'bfloat')
    output = torch.empty(M, N, dtype=torch.float32, device=a_int.device)

    has_bias = bias is not None
    bias_ptr = bias if has_bias else torch.empty(0, device=a_int.device)

    grouped = w_group_size > 0
    if grouped:
        b_gscale = w_scale  # [N, n_groups]
        b_scale_dummy = torch.empty(0, device=a_int.device)
        stride_bgs_n = b_gscale.stride(0)
        stride_bgs_g = b_gscale.stride(1)
    else:
        b_gscale = torch.empty(0, device=a_int.device)
        b_scale_dummy = w_scale  # [N]
        stride_bgs_n = 0
        stride_bgs_g = 0

    # Per-group activation scales
    if a_gscale is not None and a_group_size > 0:
        a_gscale_ptr = a_gscale  # [n_a_groups]
    else:
        a_gscale_ptr = torch.empty(0, device=a_int.device)

    # Weight zero-point (asymmetric weights)
    has_w_zp = w_zp is not None
    if has_w_zp:
        # Store as int32 for exact integer arithmetic inside kernel
        w_zp_int = w_zp.round().to(torch.int32).contiguous()
        if w_zp_int.dim() == 2:
            stride_wzp_n = w_zp_int.stride(0)
            stride_wzp_g = w_zp_int.stride(1)
        else:
            stride_wzp_n = w_zp_int.stride(0)
            stride_wzp_g = 0
    else:
        w_zp_int = torch.empty(0, dtype=torch.int32, device=a_int.device)
        stride_wzp_n = 0
        stride_wzp_g = 0

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _int_gemm_capped_acc_kernel[grid](
        a_int, w_int, output,
        a_scale,
        b_scale_dummy,
        bias_ptr,
        M, N, K,
        a_int.stride(0), a_int.stride(1),
        w_int.stride(0), w_int.stride(1),
        output.stride(0), output.stride(1),
        HAS_BIAS=has_bias,
        GROUP_SIZE=w_group_size if grouped else -1,
        B_gscale_ptr=b_gscale,
        stride_bgs_n=stride_bgs_n,
        stride_bgs_g=stride_bgs_g,
        A_GROUP_SIZE=a_group_size,
        A_gscale_ptr=a_gscale_ptr,
        ACC_MAX=acc_max,
        ACC_MIN=acc_min,
        ACC_WRAP=acc_wrap,
        ACC_RANGE=acc_range,
        T2_IS_INT=t2_is_int,
        T2_MAX=t2_max,
        T2_MIN=t2_min,
        T2_IS_FP16=t2_is_fp16,
        T2_IS_BF16=t2_is_bf16,
        T2_FRAC_BITS=t2_frac_bits,
        T2_FRAC_SCALE=float(1 << t2_frac_bits) if t2_frac_bits > 0 else 1.0,
        W_SHIFT_BIAS=w_shift_bias,
        GLOBAL_SCALE_FP32=float(global_scale_fp32),
        HAS_W_ZP=has_w_zp,
        W_zp_ptr=w_zp_int,
        stride_wzp_n=stride_wzp_n,
        stride_wzp_g=stride_wzp_g,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        ACC_BLOCK_K=ACC_BLOCK_K,
        MAC_SHIFT=mac_shift,
    )

    return output


# ---------------------------------------------------------------------------
# D. Weight preparation — extract int8 + scales from fake-quantized weights
# ---------------------------------------------------------------------------
def _recover_int_and_scale(W_fq: torch.Tensor, maxq: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Recover integer representation and scale from fake-quantized weights.

    Symmetric quantization maps integers in ``[-(maxq+1), maxq]`` (e.g.
    ``[-8, 7]`` for 4-bit).  When the GPTQ scale was shrunk by MSE search
    (``--w_clip``), some weights clamp to ``-(maxq+1)``, whose absolute value
    exceeds ``maxq``.  A naïve ``abs_max / maxq`` then overestimates the scale
    and corrupts all recovered integers.

    We resolve this by trying two candidate scales — ``abs_max / maxq`` and
    ``abs_max / (maxq + 1)`` — and keeping the one with smaller round-trip
    reconstruction error per row.

    Parameters
    ----------
    W_fq : [N, G] fake-quantized weight slice (float)
    maxq : positive half of the symmetric range (e.g. 7 for 4-bit)

    Returns
    -------
    q : [N, G] int8 integers
    scale : [N, 1] float32 per-row scale
    """
    abs_max = W_fq.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)

    # Candidate A: assume max absolute integer is maxq (common case)
    scale_a = abs_max / maxq
    q_a = torch.clamp(torch.round(W_fq / scale_a), -(maxq + 1), maxq)
    err_a = (q_a * scale_a - W_fq).pow(2).sum(dim=1, keepdim=True)

    # Candidate B: assume max absolute integer is maxq+1 (w_clip / -8 case)
    scale_b = abs_max / (maxq + 1)
    q_b = torch.clamp(torch.round(W_fq / scale_b), -(maxq + 1), maxq)
    err_b = (q_b * scale_b - W_fq).pow(2).sum(dim=1, keepdim=True)

    use_b = err_b < err_a
    scale = torch.where(use_b, scale_b, scale_a)
    q = torch.where(use_b.expand_as(q_a), q_b, q_a)

    return q.to(torch.int8), scale


def _compute_adaptive_midpoint(zero: torch.Tensor, maxq: int) -> int:
    """Pick midpoint minimizing |midpoint - zero| while keeping w_int in int8."""
    lo = max(0, maxq - 127)   # ensures maxq - midpoint <= 127
    hi = min(maxq, 128)       # ensures 0 - midpoint >= -128
    raw = round(zero.float().mean().item())
    return max(lo, min(hi, raw))


def _recover_int_and_scale_asym(
    W_fq: torch.Tensor, maxq: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover midpoint-centred integer, scale, and zp offset from asymmetric
    fake-quantized weights.

    Asymmetric quantization maps integers in ``[0, maxq]`` (e.g. ``[0, 15]``
    for 4-bit).  The fake-quantized value is ``scale * (q_unsigned - zero)``.

    We decompose this as::

        w = scale * (q_centered + w_zp)

    where ``midpoint = (maxq + 1) // 2`` (fixed), ``q_centered = q_unsigned -
    midpoint`` (always in true int4/intN range), and ``w_zp = midpoint - zero``
    (per-row correction applied inside the kernel before accumulator cap).

    Parameters
    ----------
    W_fq : [N, G] fake-quantized weight slice (float)
    maxq : full unsigned range (e.g. 15 for 4-bit, i.e. ``2**bits - 1``)

    Returns
    -------
    q_centered : [N, G] int8 — midpoint-centred integer (true intN range)
    scale      : [N, 1] float32 per-row scale
    w_zp       : [N, 1] float32 per-row ``midpoint - zero`` (correction factor)
    """
    midpoint = (maxq + 1) // 2
    w_min = W_fq.amin(dim=1, keepdim=True)
    w_max = W_fq.amax(dim=1, keepdim=True)
    scale = ((w_max - w_min) / maxq).clamp(min=1e-10)
    zero = torch.round(-w_min / scale)
    q_unsigned = torch.clamp(torch.round(W_fq / scale) + zero, 0, maxq)
    q_centered = (q_unsigned - midpoint).to(torch.int8)
    w_zp = midpoint - zero                      # [N, 1] per-row correction
    return q_centered, scale, w_zp


def _requantize_with_gptq_params(
    W: torch.Tensor,
    gptq_scale: torch.Tensor,
    gptq_zero: torch.Tensor,
    maxq: int,
    w_group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Re-quantize fake-quantized weights using GPTQ's original scale/zero.

    Instead of re-deriving scale/zero from the fake-quantized values (which
    fails when not all quantization levels [0, maxq] are used in a group),
    this uses the exact scale/zero that GPTQ computed during quantization.

    Parameters
    ----------
    W          : [N, K] fake-quantized weight (float)
    gptq_scale : [N, n_groups] or [N, 1] GPTQ's per-group scale
    gptq_zero  : [N, n_groups] or [N, 1] GPTQ's per-group zero
    maxq       : full unsigned range (e.g. 15 for 4-bit)
    w_group_size : group size (-1 for per-channel)

    Returns
    -------
    w_int  : [N, K] int8 (midpoint-centred)
    w_scale : [N, n_groups] or [N] float32
    w_zp   : [N, n_groups] or [N] float32 (midpoint - zero)
    """
    midpoint = (maxq + 1) // 2
    N, K = W.shape

    # Verify gptq_zero is integer-valued (from torch.round in GPTQ)
    zp_frac = (gptq_zero - gptq_zero.round()).abs().max().item()
    assert zp_frac < 0.01, \
        f"gptq_zero has fractional part {zp_frac:.4e} — cannot use as integer zp"

    if w_group_size > 0:
        n_groups = gptq_scale.shape[1]
        padded_K = n_groups * w_group_size
        if padded_K > K:
            W_padded = F.pad(W, (0, padded_K - K))
        else:
            W_padded = W
        # Expand scale/zero to match weight columns: [N, n_groups] -> [N, K]
        scale_expanded = gptq_scale.repeat_interleave(w_group_size, dim=1)[:, :padded_K]
        zero_expanded = gptq_zero.repeat_interleave(w_group_size, dim=1)[:, :padded_K]
        q_unsigned = torch.clamp(
            torch.round(W_padded / scale_expanded.clamp(min=1e-10)) + zero_expanded,
            0, maxq,
        )
        q_centered = (q_unsigned - midpoint).to(torch.int8)
        w_int = q_centered[:, :K].contiguous()
        w_zp = midpoint - gptq_zero              # [N, n_groups]
        return w_int, gptq_scale, w_zp
    else:
        # Per-channel: gptq_scale/gptq_zero are [N, 1]
        q_unsigned = torch.clamp(
            torch.round(W / gptq_scale.clamp(min=1e-10)) + gptq_zero,
            0, maxq,
        )
        q_centered = (q_unsigned - midpoint).to(torch.int8)
        w_zp = midpoint - gptq_zero.squeeze(1)   # [N]
        return q_centered, gptq_scale.squeeze(1), w_zp


def prepare_int_weights(
    linear: torch.nn.Linear,
    w_bits: int,
    w_sym: bool = True,
    w_group_size: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Re-quantize fake-quantized (post-GPTQ) float weights back to (w_int8, w_scale, w_zp).

    If the linear layer has ``_gptq_w_scale`` and ``_gptq_w_zero`` buffers
    (stored by GPTQ's ``fasterquant``), those exact parameters are reused.
    Otherwise, scale/zero are re-derived from the fake-quantized values.

    Returns
    -------
    w_int  : [N, K] int8
    w_scale : [N] or [N, K // w_group_size] float32
    w_zp   : [N] or [N, K // w_group_size] float32, or None for symmetric
    """
    W = linear.weight.data.float()  # [N, K]
    N, K = W.shape

    # --- Fast path: reuse GPTQ's original scale/zero (asymmetric only) ---
    if (not w_sym
            and hasattr(linear, '_gptq_w_scale')
            and linear._gptq_w_scale is not None):
        maxq = 2 ** w_bits - 1
        dev = W.device
        w_int, w_scale, w_zp = _requantize_with_gptq_params(
            W, linear._gptq_w_scale.to(dev).float(),
            linear._gptq_w_zero.to(dev).float(),
            maxq, w_group_size,
        )
        # Diagnostic: measure reconstruction error
        if w_group_size > 0 and w_scale.dim() == 2:
            w_zp_exp = w_zp.repeat_interleave(w_group_size, dim=1)[:, :K] if w_zp is not None else 0
            recon = w_scale.repeat_interleave(w_group_size, dim=1)[:, :K] * (w_int.float() + w_zp_exp)
        else:
            w_zp_col = w_zp.unsqueeze(1) if (w_zp is not None and w_zp.dim() == 1) else (w_zp if w_zp is not None else 0)
            recon = w_scale.unsqueeze(1) * (w_int.float() + w_zp_col)
        recon_err = (recon - W).abs().max().item()
        logging.info("  prepare_int_weights: STORED GPTQ params (w_bits=%d, maxq=%d, "
                     "scale=%s, recon_err=%.3e)", w_bits, maxq,
                     list(linear._gptq_w_scale.shape), recon_err)
        # Verify true intN range
        midpoint = (maxq + 1) // 2
        w_min_val, w_max_val = w_int.min().item(), w_int.max().item()
        assert w_min_val >= -midpoint, \
            f"w_int below true int{w_bits} min: {w_min_val} < {-midpoint}"
        assert w_max_val <= maxq - midpoint, \
            f"w_int above true int{w_bits} max: {w_max_val} > {maxq - midpoint}"
        return w_int, w_scale, w_zp

    # --- Fallback: re-derive scale/zero from fake-quantized values ---
    logging.info("  prepare_int_weights: RECOVERY path (w_sym=%s, w_bits=%d)", w_sym, w_bits)
    if w_sym:
        maxq = 2 ** (w_bits - 1) - 1
        recover_fn = _recover_int_and_scale
    else:
        maxq = 2 ** w_bits - 1
        recover_fn = _recover_int_and_scale_asym

    if w_group_size > 0:
        n_groups = math.ceil(K / w_group_size)
        # Pad K so it's divisible by group_size, then batch all groups at once
        padded_K = n_groups * w_group_size
        if padded_K > K:
            W_padded = F.pad(W, (0, padded_K - K))
        else:
            W_padded = W
        # Reshape [N, K] -> [N * n_groups, group_size] so each row is one group
        W_grouped = W_padded.reshape(N * n_groups, w_group_size)
        result = recover_fn(W_grouped, maxq)
        q_flat, scale_flat = result[0], result[1]
        # Reshape back and trim padding
        w_int = q_flat.reshape(N, padded_K)[:, :K].contiguous()
        w_scale = scale_flat.squeeze(1).reshape(N, n_groups)
        if not w_sym:
            w_zp = result[2].squeeze(1).reshape(N, n_groups)
        else:
            w_zp = None
    else:
        result = recover_fn(W, maxq)
        w_int = result[0]
        w_scale = result[1].squeeze(1)                # [N]
        w_zp = result[2].squeeze(1) if not w_sym else None

    # Verify true intN range for asymmetric weights
    if not w_sym:
        midpoint = (2 ** w_bits) // 2
        w_min_val, w_max_val = w_int.min().item(), w_int.max().item()
        assert w_min_val >= -midpoint, \
            f"w_int below true int{w_bits} min: {w_min_val} < {-midpoint}"
        assert w_max_val <= maxq - midpoint, \
            f"w_int above true int{w_bits} max: {w_max_val} > {maxq - midpoint}"

    return w_int, w_scale, w_zp


# ---------------------------------------------------------------------------
# E. GPTQ weight params save/load — separate file to keep main checkpoint unchanged
# ---------------------------------------------------------------------------

_GPTQ_W_PARAMS_FILENAME = '_gptq_w_params.pt'


def save_gptq_w_params(model, ckpt_dir):
    """Save per-group GPTQ scale/zero from model layers to a separate file.

    Only saves layers that have _gptq_w_scale/_gptq_w_zero attributes
    (set by fasterquant for asymmetric weight quantization).
    """
    params = {}
    for name, module in model.named_modules():
        if hasattr(module, '_gptq_w_scale') and module._gptq_w_scale is not None:
            params[name] = {
                'scale': module._gptq_w_scale.cpu(),
                'zero': module._gptq_w_zero.cpu(),
            }
    if params:
        path = os.path.join(ckpt_dir, _GPTQ_W_PARAMS_FILENAME)
        torch.save(params, path)
        logging.info("Saved GPTQ weight params for %d layers to %s", len(params), path)


def load_gptq_w_params(model, ckpt_dir):
    """Load per-group GPTQ scale/zero from separate file and attach to model layers.

    Sets _gptq_w_scale/_gptq_w_zero as plain attributes on matching nn.Linear layers.
    No-op if the params file doesn't exist (backwards compatible with old checkpoints).
    """
    path = os.path.join(ckpt_dir, _GPTQ_W_PARAMS_FILENAME)
    if not os.path.isfile(path):
        return
    params = torch.load(path, map_location='cpu')
    modules = dict(model.named_modules())
    n_loaded = 0
    for name, p in params.items():
        if name in modules:
            modules[name]._gptq_w_scale = p['scale']
            modules[name]._gptq_w_zero = p['zero']
            n_loaded += 1
    logging.info("Loaded GPTQ weight params for %d layers from %s", n_loaded, path)


# ---------------------------------------------------------------------------
# F. Tag helper — consistent filename-safe tag for int GEMM configuration
# ---------------------------------------------------------------------------
_INT_GEMM_DEFAULTS = dict(
    acc_bits=32,
    acc_block_k=32,
    acc_wrap=False,
)


def int_gemm_tag(acc_bits=32, acc_block_k=32, acc_wrap=False):
    """Build a filename-safe tag string for int GEMM configuration.

    Returns e.g. ``"_intgemm"`` with defaults,
    ``"_intgemm_acc16"`` for 16-bit accumulator,
    ``"_intgemm_acc20_bk64_wrap"`` when everything differs.
    Only non-default values are included.
    """
    parts = ["_intgemm"]
    if acc_bits != _INT_GEMM_DEFAULTS['acc_bits']:
        parts.append(f"acc{acc_bits}")
    if acc_block_k != _INT_GEMM_DEFAULTS['acc_block_k']:
        parts.append(f"bk{acc_block_k}")
    if acc_wrap and acc_wrap != _INT_GEMM_DEFAULTS['acc_wrap']:
        parts.append("wrap")
    return "_".join(parts)
