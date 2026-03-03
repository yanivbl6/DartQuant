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
        A_scale_ptr,   # [M]  per-token activation scale
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

            if GROUP_SIZE > 0:
                # Grouped weight scales — one scale per (N, group_index)
                g_idx = k_start // GROUP_SIZE
                g_scale = tl.load(
                    B_gscale_ptr + offs_n * stride_bgs_n + g_idx * stride_bgs_g,
                    mask=offs_n < N, other=1.0)
                acc += partial.to(tl.float32) * g_scale[None, :]
            else:
                acc += partial.to(tl.float32)

            # Advance pointers
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        # Apply scales: per-token activation scale * per-channel weight scale
        a_scale = tl.load(A_scale_ptr + offs_m, mask=offs_m < M, other=1.0)
        if GROUP_SIZE > 0:
            # Grouped: weight scales already folded in above
            acc = acc * a_scale[:, None]
        else:
            b_scale = tl.load(B_scale_ptr + offs_n, mask=offs_n < N, other=1.0)
            acc = acc * a_scale[:, None] * b_scale[None, :]

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
    a_scale: torch.Tensor,      # [M] float
    w_int: torch.Tensor,        # [N, K] int8
    w_scale: torch.Tensor,      # [N] or [N, n_groups] float
    acc_bits: int,
    block_k: int,
    w_group_size: int = -1,
    bias: Optional[torch.Tensor] = None,
    acc_wrap: bool = False,
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

        if w_group_size > 0:
            g_idx = k_start // w_group_size
            output += partial.float() * w_scale[:, g_idx].unsqueeze(0)
        else:
            output += partial.float()

    # Apply scales
    if w_group_size > 0:
        output *= a_scale.unsqueeze(1)
    else:
        output *= a_scale.unsqueeze(1) * w_scale.unsqueeze(0)

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
    a_int, a_scale = act_quantizer.quantize_to_int(x_2d)

    # Dispatch
    if use_triton and _HAS_TRITON and x_float.is_cuda:
        output = _triton_int_gemm(a_int, a_scale, w_int, w_scale,
                                  acc_bits, block_k, w_group_size, bias,
                                  M, N, K, acc_wrap)
    else:
        output = int_gemm_capped_reference(a_int, a_scale, w_int, w_scale,
                                           acc_bits, block_k, w_group_size,
                                           bias, acc_wrap)

    return output.reshape(*orig_shape[:-1], N)


def _triton_int_gemm(
    a_int, a_scale, w_int, w_scale,
    acc_bits, block_k, w_group_size, bias,
    M, N, K, acc_wrap=False,
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


def prepare_int_weights(
    linear: torch.nn.Linear,
    w_bits: int,
    w_sym: bool = True,
    w_group_size: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Re-quantize fake-quantized (post-GPTQ) float weights back to (w_int8, w_scale).

    After GPTQ, weights are stored as float tensors whose values lie on the
    quantization grid.  We recover the integer representation by recomputing
    per-channel (or per-group) scales and rounding.

    Returns
    -------
    w_int : [N, K] int8
    w_scale : [N] or [N, K // w_group_size] float32
    """
    assert w_sym, "Integer GEMM currently requires symmetric weight quantization"

    W = linear.weight.data.float()  # [N, K]
    N, K = W.shape

    maxq = 2 ** (w_bits - 1) - 1

    if w_group_size > 0:
        n_groups = math.ceil(K / w_group_size)
        w_int = torch.zeros_like(W, dtype=torch.int8)
        w_scale = torch.zeros(N, n_groups, dtype=torch.float32, device=W.device)

        for g in range(n_groups):
            k_start = g * w_group_size
            k_end = min(k_start + w_group_size, K)
            group = W[:, k_start:k_end]

            q, scale = _recover_int_and_scale(group, maxq)
            w_int[:, k_start:k_end] = q
            w_scale[:, g] = scale.squeeze(1)
    else:
        q, scale = _recover_int_and_scale(W, maxq)
        w_int = q
        w_scale = scale.squeeze(1)                    # [N]

    return w_int, w_scale


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
