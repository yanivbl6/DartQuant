"""Quick smoke-test for the integer GEMM with capped accumulator."""

import sys
import torch
import math

# ---------------------------------------------------------------------------
# 0. Check environment
# ---------------------------------------------------------------------------
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute capability: {cap[0]}.{cap[1]}")

try:
    import triton
    print(f"Triton: {triton.__version__}")
except ImportError:
    print("Triton: NOT INSTALLED")

from int_acc_gemm import (
    _HAS_TRITON,
    _compute_adaptive_midpoint,
    int_gemm_capped_reference,
    prepare_int_weights,
)
if _HAS_TRITON:
    from int_acc_gemm import _triton_int_gemm

print(f"_HAS_TRITON = {_HAS_TRITON}")
print()

# ---------------------------------------------------------------------------
# 1. Helpers
# ---------------------------------------------------------------------------
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_test_data(M, N, K, w_bits=8, device=None):
    """Create random int8 activations, int8 weights, and per-token/per-channel scales."""
    if device is None:
        device = DEV
    maxq = 2 ** (w_bits - 1) - 1
    a_int = torch.randint(-maxq - 1, maxq + 1, (M, K), dtype=torch.int8, device=device)
    w_int = torch.randint(-maxq - 1, maxq + 1, (N, K), dtype=torch.int8, device=device)
    a_scale = torch.rand(M, device=device, dtype=torch.float32) * 0.1 + 0.01
    w_scale = torch.rand(N, device=device, dtype=torch.float32) * 0.1 + 0.01
    return a_int, a_scale, w_int, w_scale


def reference_float_gemm(a_int, a_scale, w_int, w_scale, bias=None):
    """Unquantized float GEMM for comparison (no capping)."""
    out = (a_int.float() @ w_int.float().t()) * a_scale[:, None] * w_scale[None, :]
    if bias is not None:
        out += bias[None, :]
    return out


# ---------------------------------------------------------------------------
# 2. Test: reference implementation (no Triton needed)
# ---------------------------------------------------------------------------
def test_reference_no_cap():
    """acc_bits=32 reference should match float GEMM exactly."""
    print("TEST: reference (acc_bits=32, no capping) vs float GEMM")
    M, N, K = 64, 128, 256
    a_int, a_scale, w_int, w_scale = make_test_data(M, N, K)

    out_ref = int_gemm_capped_reference(a_int, a_scale, w_int, w_scale,
                                        acc_bits=32, block_k=32)
    out_float = reference_float_gemm(a_int, a_scale, w_int, w_scale)

    diff = (out_ref - out_float).abs()
    max_err = diff.max().item()
    rel_err = (diff / (out_float.abs() + 1e-10)).max().item()
    print(f"  max abs error: {max_err:.2e}")
    print(f"  max rel error: {rel_err:.2e}")
    assert max_err < 1e-3, f"FAIL: max_err={max_err}"
    print("  PASS\n")


def test_reference_capping_active():
    """acc_bits=16 should produce different results from acc_bits=32."""
    print("TEST: reference capping active (acc_bits=16 vs 32)")
    M, N, K = 32, 64, 512  # large K → more overflow
    a_int, a_scale, w_int, w_scale = make_test_data(M, N, K)

    out_32 = int_gemm_capped_reference(a_int, a_scale, w_int, w_scale,
                                       acc_bits=32, block_k=64)
    out_16 = int_gemm_capped_reference(a_int, a_scale, w_int, w_scale,
                                       acc_bits=16, block_k=64)

    diff = (out_32 - out_16).abs().max().item()
    print(f"  max diff between acc_bits=32 and acc_bits=16: {diff:.4f}")
    assert diff > 0.0, "FAIL: capping had no effect (outputs identical)"
    print("  PASS (capping changes the output)\n")


def test_reference_wrap_vs_saturate():
    """Wrap-around and saturation should produce different results for acc_bits=16."""
    print("TEST: reference wrap-around vs saturation (acc_bits=16)")
    M, N, K = 32, 64, 512  # large K → overflow likely
    a_int, a_scale, w_int, w_scale = make_test_data(M, N, K)

    out_sat = int_gemm_capped_reference(a_int, a_scale, w_int, w_scale,
                                        acc_bits=16, block_k=64, acc_wrap=False)
    out_wrap = int_gemm_capped_reference(a_int, a_scale, w_int, w_scale,
                                         acc_bits=16, block_k=64, acc_wrap=True)

    diff = (out_sat - out_wrap).abs().max().item()
    print(f"  max diff between saturate and wrap: {diff:.4f}")
    assert diff > 0.0, "FAIL: wrap and saturate produced identical outputs"
    print("  PASS (wrap-around differs from saturation)\n")


def test_reference_wrap_no_effect_32bit():
    """With acc_bits=32, wrap-around should match saturation (no overflow)."""
    print("TEST: reference wrap vs saturate with acc_bits=32 (should match)")
    M, N, K = 32, 64, 256
    a_int, a_scale, w_int, w_scale = make_test_data(M, N, K)

    out_sat = int_gemm_capped_reference(a_int, a_scale, w_int, w_scale,
                                        acc_bits=32, block_k=32, acc_wrap=False)
    out_wrap = int_gemm_capped_reference(a_int, a_scale, w_int, w_scale,
                                         acc_bits=32, block_k=32, acc_wrap=True)

    diff = (out_sat - out_wrap).abs().max().item()
    print(f"  max diff: {diff:.2e}")
    assert diff < 1e-3, f"FAIL: wrap and saturate differ at 32-bit: {diff}"
    print("  PASS\n")


def test_reference_grouped_weights():
    """Grouped weight quantization (w_group_size > 0)."""
    print("TEST: reference with grouped weight scales (group_size=64)")
    M, N, K = 32, 64, 256
    a_int, a_scale, w_int, _ = make_test_data(M, N, K)
    n_groups = K // 64
    w_scale_grouped = torch.rand(N, n_groups, device=DEV, dtype=torch.float32) * 0.1 + 0.01

    out = int_gemm_capped_reference(a_int, a_scale, w_int, w_scale_grouped,
                                    acc_bits=32, block_k=32, w_group_size=64)
    print(f"  output shape: {out.shape}")
    assert out.shape == (M, N), f"FAIL: wrong shape {out.shape}"
    assert not torch.isnan(out).any(), "FAIL: NaN in output"
    print("  PASS\n")


def test_reference_with_bias():
    """Bias is correctly added."""
    print("TEST: reference with bias")
    M, N, K = 16, 32, 128
    a_int, a_scale, w_int, w_scale = make_test_data(M, N, K)
    bias = torch.randn(N, device=DEV, dtype=torch.float32)

    out_no_bias = int_gemm_capped_reference(a_int, a_scale, w_int, w_scale,
                                            acc_bits=32, block_k=32)
    out_with_bias = int_gemm_capped_reference(a_int, a_scale, w_int, w_scale,
                                              acc_bits=32, block_k=32, bias=bias)

    diff = (out_with_bias - out_no_bias - bias[None, :]).abs().max().item()
    print(f"  max bias error: {diff:.2e}")
    assert diff < 1e-3, f"FAIL: bias not applied correctly"
    print("  PASS\n")


# ---------------------------------------------------------------------------
# 3. Test: Triton kernel
# ---------------------------------------------------------------------------
def test_triton_vs_reference():
    """Triton kernel should match PyTorch reference."""
    print("TEST: Triton kernel vs reference (acc_bits=32)")
    M, N, K = 64, 128, 256
    a_int, a_scale, w_int, w_scale = make_test_data(M, N, K)

    out_ref = int_gemm_capped_reference(a_int, a_scale, w_int, w_scale,
                                        acc_bits=32, block_k=32)
    out_triton = _triton_int_gemm(a_int, a_scale, w_int, w_scale,
                                  acc_bits=32, block_k=32, w_group_size=-1,
                                  bias=None, M=M, N=N, K=K)

    diff = (out_ref - out_triton).abs()
    max_err = diff.max().item()
    rel_err = (diff / (out_ref.abs() + 1e-10)).max().item()
    print(f"  max abs error: {max_err:.2e}")
    print(f"  max rel error: {rel_err:.2e}")
    assert max_err < 1e-2, f"FAIL: max_err={max_err}"
    print("  PASS\n")


def test_triton_capping():
    """Triton capping should match reference capping."""
    print("TEST: Triton kernel capping (acc_bits=16) vs reference")
    M, N, K = 32, 64, 512
    a_int, a_scale, w_int, w_scale = make_test_data(M, N, K)

    out_ref = int_gemm_capped_reference(a_int, a_scale, w_int, w_scale,
                                        acc_bits=16, block_k=32)
    out_triton = _triton_int_gemm(a_int, a_scale, w_int, w_scale,
                                  acc_bits=16, block_k=32, w_group_size=-1,
                                  bias=None, M=M, N=N, K=K)

    diff = (out_ref - out_triton).abs()
    max_err = diff.max().item()
    print(f"  max abs error: {max_err:.2e}")
    assert max_err < 1e-2, f"FAIL: max_err={max_err}"
    print("  PASS\n")


def test_triton_with_bias():
    """Triton kernel handles bias correctly."""
    print("TEST: Triton kernel with bias")
    M, N, K = 32, 64, 128
    a_int, a_scale, w_int, w_scale = make_test_data(M, N, K)
    bias = torch.randn(N, device=DEV, dtype=torch.float32)

    out_ref = int_gemm_capped_reference(a_int, a_scale, w_int, w_scale,
                                        acc_bits=32, block_k=32, bias=bias)
    out_triton = _triton_int_gemm(a_int, a_scale, w_int, w_scale,
                                  acc_bits=32, block_k=32, w_group_size=-1,
                                  bias=bias, M=M, N=N, K=K)

    diff = (out_ref - out_triton).abs().max().item()
    print(f"  max abs error: {diff:.2e}")
    assert diff < 1e-2, f"FAIL: diff={diff}"
    print("  PASS\n")


def test_triton_grouped():
    """Triton kernel with grouped weight scales."""
    print("TEST: Triton kernel with grouped weights (group_size=64)")
    M, N, K = 32, 64, 256
    GROUP = 64
    a_int, a_scale, w_int, _ = make_test_data(M, N, K)
    n_groups = K // GROUP
    w_scale_g = torch.rand(N, n_groups, device=DEV, dtype=torch.float32) * 0.1 + 0.01

    out_ref = int_gemm_capped_reference(a_int, a_scale, w_int, w_scale_g,
                                        acc_bits=32, block_k=32, w_group_size=GROUP)
    out_triton = _triton_int_gemm(a_int, a_scale, w_int, w_scale_g,
                                  acc_bits=32, block_k=32, w_group_size=GROUP,
                                  bias=None, M=M, N=N, K=K)

    diff = (out_ref - out_triton).abs().max().item()
    print(f"  max abs error: {diff:.2e}")
    assert diff < 1e-2, f"FAIL: diff={diff}"
    print("  PASS\n")


def test_triton_wrap_vs_reference():
    """Triton wrap-around should match reference wrap-around."""
    print("TEST: Triton kernel wrap-around (acc_bits=16) vs reference")
    M, N, K = 32, 64, 512
    a_int, a_scale, w_int, w_scale = make_test_data(M, N, K)

    out_ref = int_gemm_capped_reference(a_int, a_scale, w_int, w_scale,
                                        acc_bits=16, block_k=32, acc_wrap=True)
    out_triton = _triton_int_gemm(a_int, a_scale, w_int, w_scale,
                                  acc_bits=16, block_k=32, w_group_size=-1,
                                  bias=None, M=M, N=N, K=K, acc_wrap=True)

    diff = (out_ref - out_triton).abs().max().item()
    print(f"  max abs error: {diff:.2e}")
    assert diff < 1e-2, f"FAIL: max_err={diff}"
    print("  PASS\n")


# ---------------------------------------------------------------------------
# 4. Test: prepare_int_weights
# ---------------------------------------------------------------------------
def test_prepare_int_weights():
    """Verify weight preparation recovers correct integers."""
    print("TEST: prepare_int_weights (4-bit, per-channel)")
    N, K = 64, 256
    linear = torch.nn.Linear(K, N, bias=False).to(DEV)
    # Simulate fake-quantized weights: quantize then dequantize
    W = linear.weight.data.float()
    maxq = 7  # 4-bit
    abs_max = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    scale = abs_max / maxq
    q = torch.clamp(torch.round(W / scale), -8, 7)
    linear.weight.data = (q * scale).to(linear.weight.dtype)

    w_int, w_scale, _ = prepare_int_weights(linear, w_bits=4, w_sym=True, w_group_size=-1)

    # Verify round-trip: w_int * w_scale ≈ linear.weight
    reconstructed = w_int.float() * w_scale[:, None]
    diff = (reconstructed - linear.weight.data.float()).abs().max().item()
    print(f"  round-trip max error: {diff:.2e}")
    assert diff < 1e-4, f"FAIL: round-trip error too large"

    # Verify integers are in range
    assert w_int.min() >= -8, f"FAIL: min={w_int.min()}"
    assert w_int.max() <= 7, f"FAIL: max={w_int.max()}"
    print(f"  int range: [{w_int.min()}, {w_int.max()}]")
    print("  PASS\n")


def test_prepare_int_weights_with_w_clip():
    """Verify weight preparation works when MSE scale search produces -8 values.

    This is the key regression test: with --w_clip the GPTQ quantizer shrinks
    the scale, causing some weights to clamp to -(maxq+1) = -8.  The old code
    used abs_max / maxq which overestimated the scale by 8/7, corrupting all
    recovered integers in that group.
    """
    print("TEST: prepare_int_weights with w_clip (scale shrunk, -8 values present)")
    N, K = 64, 256
    linear = torch.nn.Linear(K, N, bias=False).to(DEV)
    W = linear.weight.data.float()
    maxq = 7  # 4-bit

    # Simulate MSE-shrunken scale (p=0.9): scale = 0.9 * abs_max / maxq
    abs_max = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    scale = 0.9 * abs_max / maxq   # smaller scale → more values hit -8
    q = torch.clamp(torch.round(W / scale), -8, 7)
    linear.weight.data = (q * scale).to(linear.weight.dtype)

    # Verify -8 values actually exist (sanity check for the test)
    n_neg8 = (q == -8).sum().item()
    print(f"  number of -8 values: {n_neg8}")
    assert n_neg8 > 0, "FAIL: test setup did not produce -8 values"

    w_int, w_scale, _ = prepare_int_weights(linear, w_bits=4, w_sym=True, w_group_size=-1)

    # Verify round-trip: w_int * w_scale ≈ linear.weight
    reconstructed = w_int.float() * w_scale[:, None]
    diff = (reconstructed - linear.weight.data.float()).abs().max().item()
    print(f"  round-trip max error: {diff:.2e}")
    assert diff < 1e-4, f"FAIL: round-trip error too large: {diff}"

    # Verify -8 values are recovered correctly
    orig_neg8_mask = (q == -8)
    assert (w_int[orig_neg8_mask] == -8).all(), "FAIL: -8 values not recovered"
    print("  PASS\n")


def test_prepare_int_weights_grouped():
    """Verify grouped weight preparation."""
    print("TEST: prepare_int_weights (4-bit, grouped, group_size=64)")
    N, K = 64, 256
    linear = torch.nn.Linear(K, N, bias=False).to(DEV)

    w_int, w_scale, _ = prepare_int_weights(linear, w_bits=4, w_sym=True, w_group_size=64)

    assert w_int.shape == (N, K), f"FAIL: w_int shape {w_int.shape}"
    assert w_scale.shape == (N, K // 64), f"FAIL: w_scale shape {w_scale.shape}"
    assert w_int.dtype == torch.int8
    print(f"  w_int shape: {w_int.shape}, w_scale shape: {w_scale.shape}")
    print("  PASS\n")


def test_prepare_int_weights_grouped_w_clip():
    """Verify grouped weight prep with MSE-shrunken scales (-8 values)."""
    print("TEST: prepare_int_weights (4-bit, grouped, group_size=64, w_clip)")
    N, K = 64, 256
    GROUP = 64
    linear = torch.nn.Linear(K, N, bias=False).to(DEV)
    W = linear.weight.data.float()
    maxq = 7

    # Simulate per-group MSE-shrunken scales
    n_groups = K // GROUP
    for g in range(n_groups):
        ks, ke = g * GROUP, (g + 1) * GROUP
        grp = W[:, ks:ke]
        abs_max = grp.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
        scale = 0.9 * abs_max / maxq
        q = torch.clamp(torch.round(grp / scale), -8, 7)
        W[:, ks:ke] = q * scale
    linear.weight.data = W.to(linear.weight.dtype)

    w_int, w_scale, _ = prepare_int_weights(linear, w_bits=4, w_sym=True, w_group_size=GROUP)

    # Verify per-group round-trip
    for g in range(n_groups):
        ks, ke = g * GROUP, (g + 1) * GROUP
        recon = w_int[:, ks:ke].float() * w_scale[:, g:g+1]
        diff = (recon - linear.weight.data[:, ks:ke].float()).abs().max().item()
        assert diff < 1e-4, f"FAIL: group {g} round-trip error {diff}"

    print(f"  all {n_groups} groups pass round-trip check")
    print("  PASS\n")


# ---------------------------------------------------------------------------
# 5. Test: quantize_to_int contiguity and end-to-end int_gemm_capped
# ---------------------------------------------------------------------------
def test_quantize_to_int_contiguous_scale():
    """per_token_scale from quantize_to_int must be contiguous (stride 1).

    The Triton kernel indexes A_scale_ptr with stride 1.  If the scale
    tensor has stride K (from slicing a [M, K] broadcasted tensor), the
    kernel reads wrong values and dequantizes with garbage scales.

    This was the root cause of PPL ~1.4M with --int_gemm.
    """
    print("TEST: quantize_to_int returns contiguous per_token_scale")
    from quant_utils import ActQuantizer
    M, K = 64, 2048
    x = torch.randn(M, K, device=DEV, dtype=torch.float32)

    quantizer = ActQuantizer()
    quantizer.configure(bits=8, groupsize=-1, sym=True, clip_ratio=0.9)

    q_int8, per_token_scale, _zp = quantizer.quantize_to_int(x)

    assert per_token_scale.is_contiguous(), \
        f"FAIL: per_token_scale stride={per_token_scale.stride()}, expected contiguous (stride 1)"
    assert per_token_scale.shape == (M,), f"FAIL: shape={per_token_scale.shape}"
    print(f"  stride: {per_token_scale.stride()}, shape: {per_token_scale.shape}")
    print("  PASS\n")


def test_int_gemm_capped_end_to_end():
    """End-to-end: int_gemm_capped (Triton) should match normal fake-quant path.

    This verifies that quantize_to_int + Triton kernel produces the same
    result as ActQuantizer.forward() + nn.Linear.forward().
    """
    print("TEST: int_gemm_capped (Triton) vs normal fake-quant path")
    from quant_utils import ActQuantizer
    from int_acc_gemm import int_gemm_capped, prepare_int_weights

    M, N, K = 64, 128, 256
    torch.manual_seed(42)

    # Create fake-quantized weights (simulate post-GPTQ)
    linear = torch.nn.Linear(K, N, bias=False).to(DEV)
    W = linear.weight.data.float()
    maxq = 7  # 4-bit
    abs_max = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
    scale = abs_max / maxq
    q = torch.clamp(torch.round(W / scale), -8, 7)
    linear.weight.data = (q * scale).to(linear.weight.dtype)

    # Prepare int weights
    w_int, w_scale, _ = prepare_int_weights(linear, w_bits=4, w_sym=True, w_group_size=-1)

    # Configure activation quantizer
    quantizer_normal = ActQuantizer()
    quantizer_normal.configure(bits=8, groupsize=-1, sym=True, clip_ratio=0.9)
    quantizer_int = ActQuantizer()
    quantizer_int.configure(bits=8, groupsize=-1, sym=True, clip_ratio=0.9)

    # Input
    x = torch.randn(M, K, device=DEV, dtype=torch.float32)

    # Normal path: fake-quant + float matmul
    x_fq = quantizer_normal(x)
    quantizer_normal.free()
    out_normal = x_fq @ linear.weight.data.float().t()

    # Int GEMM path
    out_int = int_gemm_capped(
        x_float=x, w_int=w_int, w_scale=w_scale,
        act_quantizer=quantizer_int,
        acc_bits=32, block_k=32, w_group_size=-1,
        use_triton=(_HAS_TRITON and torch.cuda.is_available()),
    )

    diff = (out_normal - out_int).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    rel_err = (diff / (out_normal.abs() + 1e-10)).max().item()
    print(f"  max abs error: {max_err:.2e}")
    print(f"  mean abs error: {mean_err:.2e}")
    print(f"  max rel error: {rel_err:.2e}")
    # Tolerance: the paths are not bit-identical (float16 vs int8 precision)
    # but they should be very close for acc_bits=32
    assert max_err < 0.5, f"FAIL: max_err={max_err} (int_gemm output diverges from normal path)"
    print("  PASS\n")


# ---------------------------------------------------------------------------
# 6. Asymmetric weight tests (w_asym)
# ---------------------------------------------------------------------------

def _fake_quant_asym(W, w_bits, group_size=-1):
    """Fake-quantize weights using the same asymmetric method as WeightQuantizer.

    Mimics WeightQuantizer.find_params (clamp xmin<=0, xmax>=0) + quantize.
    Returns the fake-quantized weight tensor.
    """
    maxq = 2 ** w_bits - 1
    N, K = W.shape

    if group_size > 0:
        n_groups = K // group_size
        W_out = W.clone()
        for g in range(n_groups):
            ks, ke = g * group_size, (g + 1) * group_size
            grp = W[:, ks:ke].float()
            xmin = torch.minimum(grp.min(dim=1, keepdim=True)[0],
                                 torch.zeros(N, 1, device=W.device))
            xmax = torch.maximum(grp.max(dim=1, keepdim=True)[0],
                                 torch.zeros(N, 1, device=W.device))
            scale = ((xmax - xmin).clamp(min=1e-5)) / maxq
            zero = torch.round(-xmin / scale)
            q = torch.clamp(torch.round(grp / scale) + zero, 0, maxq)
            W_out[:, ks:ke] = scale * (q - zero)
        return W_out
    else:
        W_f = W.float()
        xmin = torch.minimum(W_f.min(dim=1, keepdim=True)[0],
                             torch.zeros(N, 1, device=W.device))
        xmax = torch.maximum(W_f.max(dim=1, keepdim=True)[0],
                             torch.zeros(N, 1, device=W.device))
        scale = ((xmax - xmin).clamp(min=1e-5)) / maxq
        zero = torch.round(-xmin / scale)
        q = torch.clamp(torch.round(W_f / scale) + zero, 0, maxq)
        return scale * (q - zero)


def test_prepare_int_weights_asym(device=None):
    """Verify asymmetric weight preparation round-trips correctly."""
    if device is None:
        device = DEV
    print("TEST: prepare_int_weights (4-bit, asymmetric, per-channel)")
    N, K = 64, 256
    linear = torch.nn.Linear(K, N, bias=False).to(device)
    linear.weight.data = _fake_quant_asym(linear.weight.data, w_bits=4).to(linear.weight.dtype)

    w_int, w_scale, w_zp = prepare_int_weights(linear, w_bits=4, w_sym=False, w_group_size=-1)

    assert w_zp is None, "FAIL: w_zp should be None (folded into w_int)"

    # Round-trip: w_scale * w_int ≈ original weight (w_zp folded in)
    reconstructed = w_scale[:, None] * w_int.float()
    diff = (reconstructed - linear.weight.data.float()).abs().max().item()
    print(f"  round-trip max error: {diff:.2e}")
    assert diff < 1e-4, f"FAIL: round-trip error too large: {diff}"
    print("  PASS\n")


def test_prepare_int_weights_asym_grouped():
    """Verify asymmetric grouped weight preparation round-trips correctly."""
    print("TEST: prepare_int_weights (4-bit, asymmetric, group_size=128)")
    N, K = 64, 512
    GROUP = 128
    linear = torch.nn.Linear(K, N, bias=False).to(DEV)
    linear.weight.data = _fake_quant_asym(linear.weight.data, w_bits=4,
                                           group_size=GROUP).to(linear.weight.dtype)

    w_int, w_scale, w_zp = prepare_int_weights(linear, w_bits=4, w_sym=False,
                                                w_group_size=GROUP)

    assert w_zp is None, "FAIL: w_zp should be None (folded into w_int)"
    assert w_scale.shape == (N, K // GROUP), f"FAIL: w_scale shape {w_scale.shape}"

    # Per-group round-trip (w_zp folded in)
    n_groups = K // GROUP
    for g in range(n_groups):
        ks, ke = g * GROUP, (g + 1) * GROUP
        recon = w_scale[:, g:g+1] * w_int[:, ks:ke].float()
        diff = (recon - linear.weight.data[:, ks:ke].float()).abs().max().item()
        assert diff < 1e-4, f"FAIL: group {g} round-trip error {diff}"

    print(f"  all {n_groups} groups pass round-trip check")
    print("  PASS\n")


def test_int_gemm_capped_asym_weights_sym_act():
    """End-to-end: asymmetric weights + symmetric activations.

    int_gemm_capped (acc_bits=32) should match plain float matmul of the
    fake-quantized values, since acc_bits=32 means no capping.
    """
    print("TEST: int_gemm_capped — asym weights, sym activations, acc_bits=32")
    from quant_utils import ActQuantizer
    from int_acc_gemm import int_gemm_capped, prepare_int_weights

    M, N, K = 64, 128, 256
    GROUP = 128
    torch.manual_seed(42)

    # Create fake-quantized asymmetric weights
    linear = torch.nn.Linear(K, N, bias=False).to(DEV)
    linear.weight.data = _fake_quant_asym(linear.weight.data, w_bits=4,
                                           group_size=GROUP).to(linear.weight.dtype)

    # Prepare int weights (asymmetric, w_zp folded into w_int)
    w_int, w_scale, w_zp = prepare_int_weights(linear, w_bits=4, w_sym=False,
                                                w_group_size=GROUP)
    assert w_zp is None, "w_zp should be None (folded)"
    w_int = w_int.to(DEV)
    w_scale = w_scale.to(DEV)

    # Precompute w_zp_correction (for asymmetric activations)
    n_groups = w_scale.shape[1]
    group_sums = w_int.float().reshape(N, n_groups, GROUP).sum(dim=2)
    w_zp_correction = (w_scale * group_sums).sum(dim=1)

    # Symmetric activation quantizer
    quantizer = ActQuantizer()
    quantizer.configure(bits=8, groupsize=-1, sym=True, clip_ratio=1.0)

    x = torch.randn(M, K, device=DEV, dtype=torch.float32)

    # Normal path: quantize activations, float matmul with fake-quantized weights
    quantizer.find_params(x)
    x_fq = quantizer(x)
    quantizer.free()
    out_normal = x_fq @ linear.weight.data.float().t()

    # Int GEMM path (no w_zp, no w_zp_cross — folded)
    quantizer2 = ActQuantizer()
    quantizer2.configure(bits=8, groupsize=-1, sym=True, clip_ratio=1.0)
    out_int = int_gemm_capped(
        x_float=x, w_int=w_int, w_scale=w_scale,
        act_quantizer=quantizer2,
        acc_bits=32, block_k=32, w_group_size=GROUP,
        use_triton=(_HAS_TRITON and torch.cuda.is_available()),
        w_zp_correction=w_zp_correction,
    )

    diff = (out_normal - out_int).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    rel_err = (diff / (out_normal.abs() + 1e-10)).max().item()
    print(f"  max abs error: {max_err:.2e}")
    print(f"  mean abs error: {mean_err:.2e}")
    print(f"  max rel error: {rel_err:.2e}")
    assert max_err < 0.5, f"FAIL: max_err={max_err}"
    print("  PASS\n")


def test_int_gemm_capped_asym_weights_asym_act():
    """End-to-end: asymmetric weights + asymmetric activations (dynamic).

    This exercises all 4 correction terms (A, B, C, D).
    """
    print("TEST: int_gemm_capped — asym weights, asym activations, acc_bits=32")
    from quant_utils import ActQuantizer
    from int_acc_gemm import int_gemm_capped, prepare_int_weights

    M, N, K = 64, 128, 256
    GROUP = 128
    torch.manual_seed(42)

    # Create fake-quantized asymmetric weights
    linear = torch.nn.Linear(K, N, bias=False).to(DEV)
    linear.weight.data = _fake_quant_asym(linear.weight.data, w_bits=4,
                                           group_size=GROUP).to(linear.weight.dtype)

    # Prepare int weights (w_zp folded)
    w_int, w_scale, w_zp = prepare_int_weights(linear, w_bits=4, w_sym=False,
                                                w_group_size=GROUP)
    assert w_zp is None
    w_int = w_int.to(DEV)
    w_scale = w_scale.to(DEV)

    n_groups = w_scale.shape[1]
    group_sums = w_int.float().reshape(N, n_groups, GROUP).sum(dim=2)
    w_zp_correction = (w_scale * group_sums).sum(dim=1)

    # Asymmetric activation quantizer
    quantizer = ActQuantizer()
    quantizer.configure(bits=8, groupsize=-1, sym=False, clip_ratio=1.0)

    x = torch.randn(M, K, device=DEV, dtype=torch.float32)

    # Normal path
    quantizer.find_params(x)
    x_fq = quantizer(x)
    quantizer.free()
    out_normal = x_fq @ linear.weight.data.float().t()

    # Int GEMM path (no w_zp, no w_zp_cross — folded)
    quantizer2 = ActQuantizer()
    quantizer2.configure(bits=8, groupsize=-1, sym=False, clip_ratio=1.0)
    out_int = int_gemm_capped(
        x_float=x, w_int=w_int, w_scale=w_scale,
        act_quantizer=quantizer2,
        acc_bits=32, block_k=32, w_group_size=GROUP,
        use_triton=(_HAS_TRITON and torch.cuda.is_available()),
        w_zp_correction=w_zp_correction,
    )

    diff = (out_normal - out_int).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    rel_err = (diff / (out_normal.abs() + 1e-10)).max().item()
    print(f"  max abs error: {max_err:.2e}")
    print(f"  mean abs error: {mean_err:.2e}")
    print(f"  max rel error: {rel_err:.2e}")
    assert max_err < 0.5, f"FAIL: max_err={max_err}"
    print("  PASS\n")


def test_int_gemm_capped_asym_5bit():
    """Asymmetric 5-bit weights — the configuration that triggered the w_bits bug."""
    print("TEST: int_gemm_capped — 5-bit asym weights, sym activations, acc_bits=32")
    from quant_utils import ActQuantizer
    from int_acc_gemm import int_gemm_capped, prepare_int_weights

    M, N, K = 32, 64, 256
    GROUP = 128
    torch.manual_seed(123)

    linear = torch.nn.Linear(K, N, bias=False).to(DEV)
    linear.weight.data = _fake_quant_asym(linear.weight.data, w_bits=5,
                                           group_size=GROUP).to(linear.weight.dtype)

    w_int, w_scale, w_zp = prepare_int_weights(linear, w_bits=5, w_sym=False,
                                                w_group_size=GROUP)
    assert w_zp is None
    w_int = w_int.to(DEV)
    w_scale = w_scale.to(DEV)

    n_groups = w_scale.shape[1]
    group_sums = w_int.float().reshape(N, n_groups, GROUP).sum(dim=2)
    w_zp_correction = (w_scale * group_sums).sum(dim=1)

    quantizer = ActQuantizer()
    quantizer.configure(bits=8, groupsize=-1, sym=True, clip_ratio=1.0)

    x = torch.randn(M, K, device=DEV, dtype=torch.float32)

    # Normal path
    quantizer.find_params(x)
    x_fq = quantizer(x)
    quantizer.free()
    out_normal = x_fq @ linear.weight.data.float().t()

    # Int GEMM path
    quantizer2 = ActQuantizer()
    quantizer2.configure(bits=8, groupsize=-1, sym=True, clip_ratio=1.0)
    out_int = int_gemm_capped(
        x_float=x, w_int=w_int, w_scale=w_scale,
        act_quantizer=quantizer2,
        acc_bits=32, block_k=32, w_group_size=GROUP,
        use_triton=(_HAS_TRITON and torch.cuda.is_available()),
        w_zp_correction=w_zp_correction,
    )

    diff = (out_normal - out_int).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    rel_err = (diff / (out_normal.abs() + 1e-10)).max().item()
    print(f"  max abs error: {max_err:.2e}")
    print(f"  mean abs error: {mean_err:.2e}")
    print(f"  max rel error: {rel_err:.2e}")
    assert max_err < 0.5, f"FAIL: max_err={max_err}"
    print("  PASS\n")


def test_int_gemm_capped_asym_6bit():
    """Asymmetric 6-bit weights (Q6_K layers in GGUF)."""
    print("TEST: int_gemm_capped — 6-bit asym weights, sym activations, acc_bits=32")
    from quant_utils import ActQuantizer
    from int_acc_gemm import int_gemm_capped, prepare_int_weights

    M, N, K = 32, 64, 256
    GROUP = 128
    torch.manual_seed(123)

    linear = torch.nn.Linear(K, N, bias=False).to(DEV)
    linear.weight.data = _fake_quant_asym(linear.weight.data, w_bits=6,
                                           group_size=GROUP).to(linear.weight.dtype)

    w_int, w_scale, w_zp = prepare_int_weights(linear, w_bits=6, w_sym=False,
                                                w_group_size=GROUP)
    assert w_zp is None
    w_int = w_int.to(DEV)
    w_scale = w_scale.to(DEV)

    n_groups = w_scale.shape[1]
    group_sums = w_int.float().reshape(N, n_groups, GROUP).sum(dim=2)
    w_zp_correction = (w_scale * group_sums).sum(dim=1)

    quantizer = ActQuantizer()
    quantizer.configure(bits=8, groupsize=-1, sym=True, clip_ratio=1.0)

    x = torch.randn(M, K, device=DEV, dtype=torch.float32)

    quantizer.find_params(x)
    x_fq = quantizer(x)
    quantizer.free()
    out_normal = x_fq @ linear.weight.data.float().t()

    quantizer2 = ActQuantizer()
    quantizer2.configure(bits=8, groupsize=-1, sym=True, clip_ratio=1.0)
    out_int = int_gemm_capped(
        x_float=x, w_int=w_int, w_scale=w_scale,
        act_quantizer=quantizer2,
        acc_bits=32, block_k=32, w_group_size=GROUP,
        use_triton=(_HAS_TRITON and torch.cuda.is_available()),
        w_zp_correction=w_zp_correction,
    )

    diff = (out_normal - out_int).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    rel_err = (diff / (out_normal.abs() + 1e-10)).max().item()
    print(f"  max abs error: {max_err:.2e}")
    print(f"  mean abs error: {mean_err:.2e}")
    print(f"  max rel error: {rel_err:.2e}")
    assert max_err < 0.5, f"FAIL: max_err={max_err}"
    print("  PASS\n")


# ---------------------------------------------------------------------------
# 6b. Recovery vs GPTQ WeightQuantizer — does recovery match original params?
# ---------------------------------------------------------------------------

def test_recovery_vs_weight_quantizer():
    """Check that _recover_int_and_scale_asym recovers the same integers
    as WeightQuantizer.find_params + asym_quant.

    WeightQuantizer clamps xmin<=0, xmax>=0 (ensuring 0 is representable).
    The recovery function uses raw min/max of the fake-quantized values.
    If a group has no weight that maps to q=0, the recovery computes
    different scale/zero → different integers.
    """
    print("TEST: recovery vs WeightQuantizer — per-channel, 4-bit asym")
    from quant_utils import WeightQuantizer, asym_quant, asym_dequant
    from int_acc_gemm import _recover_int_and_scale_asym

    torch.manual_seed(42)
    N, K = 64, 256

    # Use WeightQuantizer to fake-quantize (same as GPTQ)
    W = torch.randn(N, K, device=DEV)
    wq = WeightQuantizer()
    wq.configure(bits=4, perchannel=True, sym=False, mse=False)
    wq.find_params(W)
    W_fq = wq.quantize(W)  # fake-quantized

    # Also get the original integers from GPTQ's quantizer
    maxq = int(wq.maxq.item())
    q_orig, _, _ = asym_quant(W, wq.scale, wq.zero, wq.maxq)  # [N, K] in [0, maxq]
    # True centered: q_unsigned - gptq_zero (w_zp folded in)
    q_centered_orig = (q_orig.long() - wq.zero.long()).clamp(-128, 127).to(torch.int8)

    # Now recover using _recover_int_and_scale_asym (returns w_zp=None, folded)
    q_centered_rec, scale_rec, zp_rec = _recover_int_and_scale_asym(W_fq, maxq)
    assert zp_rec is None, "Expected w_zp=None after folding"

    # Compare integers
    n_mismatch = (q_centered_rec != q_centered_orig).sum().item()
    n_total = N * K
    pct = 100.0 * n_mismatch / n_total
    print(f"  integer mismatches: {n_mismatch}/{n_total} ({pct:.2f}%)")

    # Compare reconstructed values (w_zp folded → w_int IS the centered integer)
    recon_orig = wq.scale * (q_orig - wq.zero)
    recon_rec = scale_rec * q_centered_rec.float()
    val_diff = (recon_rec - recon_orig.float()).abs().max().item()
    print(f"  max value reconstruction diff: {val_diff:.2e}")

    if n_mismatch > 0:
        row_mismatches = (q_centered_rec != q_centered_orig).sum(dim=1)
        bad_rows = (row_mismatches > 0).nonzero(as_tuple=True)[0]
        print(f"  rows with mismatches: {bad_rows.tolist()[:10]}...")

    assert n_mismatch == 0, f"FAIL: {n_mismatch} integer mismatches"
    print("  PASS\n")


def test_recovery_vs_weight_quantizer_grouped():
    """Same test but with per-group quantization (group_size=128)."""
    print("TEST: recovery vs WeightQuantizer — grouped, 4-bit asym")
    from quant_utils import WeightQuantizer, asym_quant
    from int_acc_gemm import _recover_int_and_scale_asym

    torch.manual_seed(42)
    N, K = 64, 512
    GROUP = 128

    W = torch.randn(N, K, device=DEV)
    maxq_val = 2 ** 4 - 1  # 15

    # Per-group fake-quantize using WeightQuantizer (mimicking GPTQ)
    W_fq = W.clone()
    q_centered_orig = torch.zeros_like(W, dtype=torch.int8)
    for g in range(K // GROUP):
        ks, ke = g * GROUP, (g + 1) * GROUP
        wq = WeightQuantizer()
        wq.configure(bits=4, perchannel=True, sym=False, mse=False)
        wq.find_params(W[:, ks:ke])
        W_fq[:, ks:ke] = wq.quantize(W[:, ks:ke])
        q_g, _, _ = asym_quant(W[:, ks:ke], wq.scale, wq.zero, wq.maxq)
        # True centered: q_unsigned - gptq_zero (w_zp folded)
        q_centered_orig[:, ks:ke] = (q_g.long() - wq.zero.long()).clamp(-128, 127).to(torch.int8)

    # Recovery: reshape into groups (same as prepare_int_weights)
    n_groups = K // GROUP
    W_grouped = W_fq.reshape(N * n_groups, GROUP)
    q_rec, scale_rec, zp_rec = _recover_int_and_scale_asym(W_grouped, maxq_val)
    q_centered_rec = q_rec.reshape(N, K)

    n_mismatch = (q_centered_rec != q_centered_orig).sum().item()
    n_total = N * K
    pct = 100.0 * n_mismatch / n_total
    print(f"  integer mismatches: {n_mismatch}/{n_total} ({pct:.2f}%)")

    if n_mismatch > 0:
        # Find which groups diverge
        for g in range(n_groups):
            ks, ke = g * GROUP, (g + 1) * GROUP
            gm = (q_centered_rec[:, ks:ke] != q_centered_orig[:, ks:ke]).sum().item()
            if gm > 0:
                print(f"    group {g} (cols {ks}-{ke}): {gm} mismatches")
                # Show a sample row
                row_mm = (q_centered_rec[:, ks:ke] != q_centered_orig[:, ks:ke]).any(dim=1)
                r = row_mm.nonzero(as_tuple=True)[0][0].item()
                print(f"      sample row {r}: orig ints {q_centered_orig[r, ks:ks+8].tolist()}"
                      f" vs rec {q_centered_rec[r, ks:ks+8].tolist()}")
                if g >= 3:
                    print(f"      ... (skipping remaining groups)")
                    break

    assert n_mismatch == 0, f"FAIL: {n_mismatch} integer mismatches"
    print("  PASS\n")


def test_recovery_vs_weight_quantizer_5bit():
    """5-bit grouped asymmetric — the GGUF Q5_K layers."""
    print("TEST: recovery vs WeightQuantizer — grouped, 5-bit asym")
    from quant_utils import WeightQuantizer, asym_quant
    from int_acc_gemm import _recover_int_and_scale_asym

    torch.manual_seed(42)
    N, K = 64, 512
    GROUP = 128
    BITS = 5
    maxq_val = 2 ** BITS - 1  # 31

    W = torch.randn(N, K, device=DEV)

    W_fq = W.clone()
    q_centered_orig = torch.zeros_like(W, dtype=torch.int8)
    for g in range(K // GROUP):
        ks, ke = g * GROUP, (g + 1) * GROUP
        wq = WeightQuantizer()
        wq.configure(bits=BITS, perchannel=True, sym=False, mse=False)
        wq.find_params(W[:, ks:ke])
        W_fq[:, ks:ke] = wq.quantize(W[:, ks:ke])
        q_g, _, _ = asym_quant(W[:, ks:ke], wq.scale, wq.zero, wq.maxq)
        q_centered_orig[:, ks:ke] = (q_g.long() - wq.zero.long()).clamp(-128, 127).to(torch.int8)

    n_groups = K // GROUP
    W_grouped = W_fq.reshape(N * n_groups, GROUP)
    q_rec, scale_rec, zp_rec = _recover_int_and_scale_asym(W_grouped, maxq_val)
    q_centered_rec = q_rec.reshape(N, K)

    n_mismatch = (q_centered_rec != q_centered_orig).sum().item()
    n_total = N * K
    pct = 100.0 * n_mismatch / n_total
    print(f"  integer mismatches: {n_mismatch}/{n_total} ({pct:.2f}%)")

    if n_mismatch > 0:
        for g in range(min(n_groups, 4)):
            ks, ke = g * GROUP, (g + 1) * GROUP
            gm = (q_centered_rec[:, ks:ke] != q_centered_orig[:, ks:ke]).sum().item()
            if gm > 0:
                print(f"    group {g}: {gm} mismatches")

    assert n_mismatch == 0, f"FAIL: {n_mismatch} integer mismatches"
    print("  PASS\n")


def test_recovery_vs_weight_quantizer_6bit():
    """6-bit grouped asymmetric — the GGUF Q6_K layers."""
    print("TEST: recovery vs WeightQuantizer — grouped, 6-bit asym")
    from quant_utils import WeightQuantizer, asym_quant
    from int_acc_gemm import _recover_int_and_scale_asym

    torch.manual_seed(42)
    N, K = 64, 512
    GROUP = 128
    BITS = 6
    maxq_val = 2 ** BITS - 1  # 63

    W = torch.randn(N, K, device=DEV)

    W_fq = W.clone()
    q_centered_orig = torch.zeros_like(W, dtype=torch.int8)
    for g in range(K // GROUP):
        ks, ke = g * GROUP, (g + 1) * GROUP
        wq = WeightQuantizer()
        wq.configure(bits=BITS, perchannel=True, sym=False, mse=False)
        wq.find_params(W[:, ks:ke])
        W_fq[:, ks:ke] = wq.quantize(W[:, ks:ke])
        q_g, _, _ = asym_quant(W[:, ks:ke], wq.scale, wq.zero, wq.maxq)
        q_centered_orig[:, ks:ke] = (q_g.long() - wq.zero.long()).clamp(-128, 127).to(torch.int8)

    n_groups = K // GROUP
    W_grouped = W_fq.reshape(N * n_groups, GROUP)
    q_rec, scale_rec, zp_rec = _recover_int_and_scale_asym(W_grouped, maxq_val)
    q_centered_rec = q_rec.reshape(N, K)

    n_mismatch = (q_centered_rec != q_centered_orig).sum().item()
    n_total = N * K
    pct = 100.0 * n_mismatch / n_total
    print(f"  integer mismatches: {n_mismatch}/{n_total} ({pct:.2f}%)")

    if n_mismatch > 0:
        for g in range(min(n_groups, 4)):
            ks, ke = g * GROUP, (g + 1) * GROUP
            gm = (q_centered_rec[:, ks:ke] != q_centered_orig[:, ks:ke]).sum().item()
            if gm > 0:
                print(f"    group {g}: {gm} mismatches")

    assert n_mismatch == 0, f"FAIL: {n_mismatch} integer mismatches"
    print("  PASS\n")


def test_gptq_stored_params_with_error_compensation():
    """Test the full GPTQ flow: fasterquant stores per-group scale/zero on the
    layer, and prepare_int_weights reuses them to get exact integers.

    Uses the real GPTQ class with Hessian-based error compensation, which shifts
    columns within a group so that not all q levels [0, maxq] are used. This is
    the actual failure mode that causes recovery to produce wrong scale/zero.
    """
    print("TEST: GPTQ fasterquant stores params → prepare_int_weights reuses them")
    from gptq_utils import GPTQ
    from quant_utils import WeightQuantizer, asym_quant
    from int_acc_gemm import prepare_int_weights

    torch.manual_seed(42)
    N, K = 64, 256
    GROUP = 128
    BITS = 4
    maxq = 2 ** BITS - 1
    # Create a linear layer and run real GPTQ on it
    linear = torch.nn.Linear(K, N, bias=False).to(DEV)

    # Set up GPTQ with a realistic Hessian (from random activations)
    gptq = GPTQ(linear)
    for _ in range(8):
        x = torch.randn(32, K, device=DEV)
        gptq.add_batch(x, None)

    # Configure asymmetric quantizer
    gptq.quantizer = WeightQuantizer()
    gptq.quantizer.configure(bits=BITS, perchannel=True, sym=False, mse=False)

    # Run GPTQ — this stores _gptq_w_scale/_gptq_w_zero on the layer
    loss = gptq.fasterquant(groupsize=GROUP)
    print(f"  GPTQ loss: {loss:.6f}")

    # Verify buffers were stored
    assert hasattr(linear, '_gptq_w_scale'), "FAIL: _gptq_w_scale not stored"
    assert hasattr(linear, '_gptq_w_zero'), "FAIL: _gptq_w_zero not stored"
    n_groups = K // GROUP
    assert linear._gptq_w_scale.shape == (N, n_groups), \
        f"FAIL: scale shape {linear._gptq_w_scale.shape} != ({N}, {n_groups})"
    print(f"  Stored params: scale {linear._gptq_w_scale.shape}, zero {linear._gptq_w_zero.shape}")

    # Use stored params path (has _gptq_w_scale/_gptq_w_zero)
    w_int_stored, w_scale_stored, w_zp_stored = prepare_int_weights(
        linear, BITS, w_sym=False, w_group_size=GROUP)

    # Use recovery path (strip the buffers)
    linear_no_params = torch.nn.Linear(K, N, bias=False).to(DEV)
    linear_no_params.weight.data = linear.weight.data.clone()
    w_int_recov, w_scale_recov, w_zp_recov = prepare_int_weights(
        linear_no_params, BITS, w_sym=False, w_group_size=GROUP)

    # Both should reconstruct the same fake-quantized values
    # w_zp is folded into w_int, so reconstruction is just w_scale * w_int
    assert w_zp_stored is None, "Expected w_zp=None (folded) for stored-params path"
    assert w_zp_recov is None, "Expected w_zp=None (folded) for recovery path"
    W_fq = linear.weight.data.float()
    recon_stored = w_scale_stored.repeat_interleave(GROUP, dim=1) * w_int_stored.float()
    recon_recov = w_scale_recov.repeat_interleave(GROUP, dim=1) * w_int_recov.float()

    err_stored = (recon_stored - W_fq).abs().max().item()
    err_recov = (recon_recov - W_fq).abs().max().item()
    print(f"  Stored-params reconstruction error: {err_stored:.2e}")
    print(f"  Recovery reconstruction error:      {err_recov:.2e}")

    n_int_diff = (w_int_stored != w_int_recov).sum().item()
    print(f"  Integer differences (stored vs recovery): {n_int_diff}/{N*K}")

    # The stored-params path must reconstruct exactly (within float rounding)
    assert err_stored < 1e-4, \
        f"FAIL: stored-params reconstruction error {err_stored:.2e} too large"
    print("  PASS\n")


# ---------------------------------------------------------------------------
# 6c. Full ActQuantWrapper end-to-end test
# ---------------------------------------------------------------------------

def _test_actquantwrapper_asym(w_bits, w_group_size, static_act, a_sym,
                                acc_block_k=128, label=""):
    """End-to-end test through ActQuantWrapper.forward().

    Compares int_gemm path vs fake-quant path for asymmetric weights.
    This mimics the exact pipeline in main_for_test.py.
    """
    from quant_utils import ActQuantWrapper, WeightQuantizer, asym_quant_dequant

    M, K, N = 32, 256, 64
    torch.manual_seed(42)

    # 1. Create linear layer + fake-quantize weights using WeightQuantizer
    linear = torch.nn.Linear(K, N, bias=False).to(DEV)
    W = linear.weight.data.float()
    maxq = 2 ** w_bits - 1

    if w_group_size > 0:
        for g in range(K // w_group_size):
            ks, ke = g * w_group_size, (g + 1) * w_group_size
            wq = WeightQuantizer()
            wq.configure(bits=w_bits, perchannel=True, sym=False, mse=False)
            wq.find_params(W[:, ks:ke])
            W[:, ks:ke] = wq.quantize(W[:, ks:ke])
    else:
        wq = WeightQuantizer()
        wq.configure(bits=w_bits, perchannel=True, sym=False, mse=False)
        wq.find_params(W)
        W = wq.quantize(W)

    linear.weight.data = W.to(linear.weight.dtype)

    # 2. Create two ActQuantWrappers: one for int_gemm, one for fake-quant reference
    aqw_int = ActQuantWrapper(torch.nn.Linear(K, N, bias=False).to(DEV))
    aqw_int.module.weight.data = linear.weight.data.clone()
    aqw_ref = ActQuantWrapper(torch.nn.Linear(K, N, bias=False).to(DEV))
    aqw_ref.module.weight.data = linear.weight.data.clone()

    # 3. Configure activation quantizers
    a_bits = 8
    aqw_int.quantizer.configure(bits=a_bits, groupsize=-1, sym=a_sym, clip_ratio=1.0)
    aqw_ref.quantizer.configure(bits=a_bits, groupsize=-1, sym=a_sym, clip_ratio=1.0)

    # 4. Call prepare_int_gemm on the int_gemm wrapper
    aqw_int.prepare_int_gemm(
        w_bits=w_bits, w_sym=False, w_group_size=w_group_size,
        acc_bits=32, acc_block_k=acc_block_k, use_triton=True, acc_wrap=False)

    # 5. For static mode: simulate calibration (per-column scales)
    if static_act:
        # Simulate calibration: compute per-column scales from calibration data
        x_cal = torch.randn(256, K, device=DEV)
        maxq_a = aqw_int.quantizer.maxq.float().to(DEV)

        if a_sym:
            col_abs_max = x_cal.abs().max(dim=0)[0]  # [K]
            col_scale = col_abs_max / maxq_a
            col_scale = col_scale.clamp(min=1e-10)
            col_zero = torch.zeros(K, device=DEV)
        else:
            col_min = torch.minimum(x_cal.min(dim=0)[0], torch.zeros(K, device=DEV))
            col_max = torch.maximum(x_cal.max(dim=0)[0], torch.zeros(K, device=DEV))
            col_scale = ((col_max - col_min) / maxq_a).clamp(min=1e-10)
            col_zero = torch.round(-col_min / col_scale)

        # Set per-column scales on both quantizers (ref uses per-column directly)
        aqw_ref.quantizer.maxq = aqw_ref.quantizer.maxq.to(DEV)
        aqw_ref.quantizer.scale = col_scale.clone()
        aqw_ref.quantizer.zero = col_zero.clone()
        aqw_ref.quantizer.static = True

        # For int_gemm: convert per-column to per-group (same as main_for_test.py)
        G = acc_block_k
        q = aqw_int.quantizer
        n_groups = K // G

        if a_sym:
            q.scale = col_scale.reshape(n_groups, G).max(dim=1)[0]
            q.zero = torch.zeros(n_groups, device=DEV)
        else:
            col_min_repr = -(col_zero * col_scale)
            col_max_repr = (maxq_a - col_zero) * col_scale
            group_min = col_min_repr.reshape(n_groups, G).min(dim=1)[0]
            group_max = col_max_repr.reshape(n_groups, G).max(dim=1)[0]
            group_scale = (group_max - group_min) / maxq_a
            group_zero = torch.round(-group_min / group_scale)
            dead = (group_min == 0) & (group_max == 0)
            group_scale[dead] = 1.0
            group_zero[dead] = 0.0
            q.scale = group_scale
            q.zero = group_zero

        q.static = True
        q.groupsize = G

        # Precompute static zp bias
        aqw_int.compute_static_zp_bias()

    # 6. Run forward
    x = torch.randn(M, K, device=DEV)
    out_int = aqw_int(x)
    out_ref = aqw_ref(x)

    diff = (out_int - out_ref).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    out_mag = out_ref.abs().mean().item()
    rel_err = max_err / (out_mag + 1e-10)
    print(f"  {label}: max_err={max_err:.2e}, mean_err={mean_err:.2e}, "
          f"rel(max/mean_out)={rel_err:.2e}, out_magnitude={out_mag:.2e}")
    return max_err, rel_err


def test_actquantwrapper_asym_suite():
    """Run ActQuantWrapper end-to-end for all relevant configurations."""
    print("TEST: ActQuantWrapper end-to-end — asymmetric weights")
    results = []

    configs = [
        # (w_bits, w_group_size, static_act, a_sym, label)
        (4, 128, False, True,  "4bit g128 dynamic sym_act"),
        (4, 128, False, False, "4bit g128 dynamic asym_act"),
        (4, 128, True,  True,  "4bit g128 static  sym_act"),
        (4, 128, True,  False, "4bit g128 static  asym_act"),
        (5, 128, False, True,  "5bit g128 dynamic sym_act"),
        (5, 128, True,  True,  "5bit g128 static  sym_act"),
        (6, 128, False, True,  "6bit g128 dynamic sym_act"),
        (6, 128, True,  True,  "6bit g128 static  sym_act"),
        (4, -1,  False, True,  "4bit perchan dynamic sym_act"),
        (4, -1,  False, False, "4bit perchan dynamic asym_act"),
    ]

    any_fail = False
    for w_bits, w_gs, static, a_sym, label in configs:
        max_err, rel_err = _test_actquantwrapper_asym(
            w_bits, w_gs, static, a_sym, label=label)
        if max_err > 1.0:
            print(f"    *** FAIL: max_err={max_err:.2e} ***")
            any_fail = True

    assert not any_fail, "FAIL: some configurations had large errors"
    print("  PASS\n")


# ---------------------------------------------------------------------------
# 7. Run all
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: No CUDA — only reference tests will run\n")

    print("=" * 60)
    print("Reference implementation tests")
    print("=" * 60)
    test_reference_no_cap()
    test_reference_capping_active()
    test_reference_wrap_vs_saturate()
    test_reference_wrap_no_effect_32bit()
    test_reference_grouped_weights()
    test_reference_with_bias()

    if torch.cuda.is_available():
        print("=" * 60)
        print("Weight preparation tests")
        print("=" * 60)
        test_prepare_int_weights()
        test_prepare_int_weights_with_w_clip()
        test_prepare_int_weights_grouped()
        test_prepare_int_weights_grouped_w_clip()

    if torch.cuda.is_available():
        print("=" * 60)
        print("quantize_to_int + end-to-end tests")
        print("=" * 60)
        test_quantize_to_int_contiguous_scale()
        test_int_gemm_capped_end_to_end()

    if _HAS_TRITON and torch.cuda.is_available():
        print("=" * 60)
        print("Triton kernel tests")
        print("=" * 60)
        test_triton_vs_reference()
        test_triton_capping()
        test_triton_with_bias()
        test_triton_grouped()
        test_triton_wrap_vs_reference()
    elif not _HAS_TRITON:
        print("SKIPPED: Triton tests (triton not installed)")
    else:
        print("SKIPPED: Triton tests (no CUDA)")

    if torch.cuda.is_available():
        print("=" * 60)
        print("Asymmetric weight (w_asym) tests")
        print("=" * 60)
        test_prepare_int_weights_asym()
        test_prepare_int_weights_asym_grouped()
        test_int_gemm_capped_asym_weights_sym_act()
        test_int_gemm_capped_asym_weights_asym_act()
        test_int_gemm_capped_asym_5bit()
        test_int_gemm_capped_asym_6bit()

        print("=" * 60)
        print("Recovery vs WeightQuantizer tests")
        print("=" * 60)
        test_recovery_vs_weight_quantizer()
        test_recovery_vs_weight_quantizer_grouped()
        test_recovery_vs_weight_quantizer_5bit()
        test_recovery_vs_weight_quantizer_6bit()

        print("=" * 60)
        print("Stored GPTQ params tests")
        print("=" * 60)
        test_gptq_stored_params_with_error_compensation()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
