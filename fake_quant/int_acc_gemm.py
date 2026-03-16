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
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

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
        # Accumulator cap
        ACC_MAX: tl.constexpr,
        ACC_MIN: tl.constexpr,
        ACC_WRAP: tl.constexpr,     # True = two's-complement wrap-around; False = saturation
        ACC_RANGE: tl.constexpr,    # = ACC_MAX - ACC_MIN + 1 (= 2^acc_bits)
        # Tile sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Pointers for the first K-block
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

        # Float accumulator for the outer (post-clamp) sum
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k_mask = (k_start + offs_k) < K

            # Load int8 tiles
            a_mask = (offs_m[:, None] < M) & (k_mask[None, :])
            b_mask = (offs_n[:, None] < N) & (k_mask[None, :])
            a_tile = tl.load(a_ptrs, mask=a_mask, other=0)
            b_tile = tl.load(b_ptrs, mask=b_mask, other=0)

            # int8 x int8 → int32 dot product (uses tensor cores on Ampere+)
            # tl.dot expects [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
            # b_tile is [BLOCK_N, BLOCK_K] so we transpose it
            partial = tl.dot(a_tile, tl.trans(b_tile))  # [BLOCK_M, BLOCK_N] int32

            # Cap to accumulator range
            if ACC_WRAP:
                # Two's-complement wrap-around (double-modulo handles C-style remainder)
                partial = ((partial - ACC_MIN) % ACC_RANGE + ACC_RANGE) % ACC_RANGE + ACC_MIN
            else:
                # Saturation (clamp)
                partial = tl.minimum(tl.maximum(partial, ACC_MIN), ACC_MAX)

            partial_f = partial.to(tl.float32)

            # Per-group activation scale (static mode)
            if A_GROUP_SIZE > 0:
                a_g_idx = k_start // A_GROUP_SIZE
                a_g_scale = tl.load(A_gscale_ptr + a_g_idx)
                partial_f = partial_f * a_g_scale

            if GROUP_SIZE > 0:
                # Grouped weight scales — one scale per (N, group_index)
                g_idx = k_start // GROUP_SIZE
                g_scale = tl.load(
                    B_gscale_ptr + offs_n * stride_bgs_n + g_idx * stride_bgs_g,
                    mask=offs_n < N, other=1.0)
                acc += partial_f * g_scale[None, :]
            else:
                acc += partial_f

            # Advance pointers
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

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
def int_gemm_capped_reference(
    a_int: torch.Tensor,        # [M, K] int8
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
) -> torch.Tensor:
    """Pure-PyTorch reference for capped-accumulator integer GEMM."""
    M, K = a_int.shape
    N = w_int.shape[0]

    acc_max = 2 ** (acc_bits - 1) - 1
    acc_min = -acc_max - 1
    acc_range = acc_max - acc_min + 1  # = 2^acc_bits

    output = torch.zeros(M, N, dtype=torch.float32, device=a_int.device)

    for k_start in range(0, K, block_k):
        k_end = min(k_start + block_k, K)
        # Compute dot product in float (PyTorch lacks int matmul on CUDA),
        # then round to int semantics and cap to accumulator range.
        partial = a_int[:, k_start:k_end].float() @ w_int[:, k_start:k_end].float().t()
        partial = partial.round()
        if acc_wrap:
            # Two's-complement wrap-around (use int64 to avoid float32 precision loss)
            partial = ((partial.long() - acc_min) % acc_range + acc_min).float()
        else:
            # Saturation (clamp)
            partial = partial.clamp(acc_min, acc_max)

        # Per-group activation scale (static mode)
        if a_group_size > 0 and a_group_scale is not None:
            a_g_idx = k_start // a_group_size
            partial = partial * a_group_scale[a_g_idx]

        if w_group_size > 0:
            g_idx = k_start // w_group_size
            output += partial.float() * w_scale[:, g_idx].unsqueeze(0)
        else:
            output += partial.float()

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
) -> torch.Tensor:
    """
    Quantize activations → run integer GEMM with capped accumulator → float output.

    Parameters
    ----------
    x_float : [..., K] float activations (pre-rotation, post-Hadamard)
    w_int   : [N, K] int8 weight matrix
    w_scale : [N] or [N, n_groups] float weight scales
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

    # Dispatch
    if use_triton and _HAS_TRITON and x_float.is_cuda:
        output = _triton_int_gemm(a_int, a_token_scale, w_int, w_scale,
                                  acc_bits, block_k, w_group_size, bias,
                                  M, N, K, acc_wrap,
                                  a_gscale=a_group_scale,
                                  a_group_size=a_group_size)
    else:
        output = int_gemm_capped_reference(a_int, a_token_scale, w_int, w_scale,
                                           acc_bits, block_k, w_group_size,
                                           bias, acc_wrap,
                                           a_group_scale=a_group_scale,
                                           a_group_size=a_group_size)

    # Zero-point correction for asymmetric activations (applied in float,
    # outside the capped accumulator).
    # Term: a_zp * Σ_k(s_w * q_w_signed)  per output channel
    if a_zp_correction is not None and w_zp_correction is not None:
        output += a_zp_correction.unsqueeze(1) * w_zp_correction.unsqueeze(0)

    # Zero-point correction for asymmetric weights (applied in float,
    # outside the capped accumulator).
    # Term: w_zp * Σ_k(a_int) * a_scale * w_scale  per output channel
    if w_zp is not None:
        _apply_weight_zp_correction(
            output, a_int, a_token_scale, a_group_scale, a_group_size,
            w_zp, w_scale, w_group_size, M, K, use_act_groups)

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


def _triton_int_gemm(
    a_int, a_scale, w_int, w_scale,
    acc_bits, block_k, w_group_size, bias,
    M, N, K, acc_wrap=False,
    a_gscale=None, a_group_size=-1,
):
    """Launch the Triton kernel."""
    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = block_k

    acc_max = 2 ** (acc_bits - 1) - 1
    acc_min = -acc_max - 1
    acc_range = acc_max - acc_min + 1  # = 2^acc_bits

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
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
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


def _recover_int_and_scale_asym(
    W_fq: torch.Tensor, maxq: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover midpoint-centred integer, scale, and zp offset from asymmetric
    fake-quantized weights.

    Asymmetric quantization maps integers in ``[0, maxq]`` (e.g. ``[0, 15]``
    for 4-bit).  The fake-quantized value is ``scale * (q_unsigned - zero)``.

    We decompose this as::

        w = scale * (q_centered + midpoint - zero)
          = scale * q_centered  +  scale * zp_offset

    where ``midpoint = (maxq + 1) // 2``, ``q_centered = q_unsigned - midpoint``
    (always in ``[-midpoint, maxq - midpoint]`` — the standard signed int
    range), and ``zp_offset = midpoint - zero``.

    ``q_centered`` is stored as int8 for the kernel (same range as symmetric).
    The ``zp_offset`` correction is applied in float outside the accumulator.

    Parameters
    ----------
    W_fq : [N, G] fake-quantized weight slice (float)
    maxq : full unsigned range (e.g. 15 for 4-bit, i.e. ``2**bits - 1``)

    Returns
    -------
    q_centered : [N, G] int8 — midpoint-centred integer (same range as symmetric)
    scale      : [N, 1] float32 per-row scale
    zp_offset  : [N, 1] float32 per-row ``midpoint - zero`` (correction factor)
    """
    midpoint = (maxq + 1) // 2
    w_min = W_fq.amin(dim=1, keepdim=True)
    w_max = W_fq.amax(dim=1, keepdim=True)
    scale = ((w_max - w_min) / maxq).clamp(min=1e-10)
    zero = torch.round(-w_min / scale)
    q_unsigned = torch.clamp(torch.round(W_fq / scale) + zero, 0, maxq)
    q_centered = q_unsigned - midpoint  # always in [-midpoint, maxq - midpoint]
    zp_offset = midpoint - zero         # correction factor
    return q_centered.to(torch.int8), scale, zp_offset


def prepare_int_weights(
    linear: torch.nn.Linear,
    w_bits: int,
    w_sym: bool = True,
    w_group_size: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Re-quantize fake-quantized (post-GPTQ) float weights back to (w_int8, w_scale, w_zp).

    After GPTQ, weights are stored as float tensors whose values lie on the
    quantization grid.  We recover the integer representation by recomputing
    per-channel (or per-group) scales and rounding.

    Returns
    -------
    w_int  : [N, K] int8
    w_scale : [N] or [N, K // w_group_size] float32
    w_zp   : [N] or [N, K // w_group_size] float32, or None for symmetric
    """
    W = linear.weight.data.float()  # [N, K]
    N, K = W.shape

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
        w_zp = result[2].squeeze(1) if not w_sym else None  # [N] or None

    return w_int, w_scale, w_zp


# ---------------------------------------------------------------------------
# E. Tag helper — consistent filename-safe tag for int GEMM configuration
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
