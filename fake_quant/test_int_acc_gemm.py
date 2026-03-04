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

    w_int, w_scale = prepare_int_weights(linear, w_bits=4, w_sym=True, w_group_size=-1)

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

    w_int, w_scale = prepare_int_weights(linear, w_bits=4, w_sym=True, w_group_size=-1)

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

    w_int, w_scale = prepare_int_weights(linear, w_bits=4, w_sym=True, w_group_size=64)

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

    w_int, w_scale = prepare_int_weights(linear, w_bits=4, w_sym=True, w_group_size=GROUP)

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
    w_int, w_scale = prepare_int_weights(linear, w_bits=4, w_sym=True, w_group_size=-1)

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
# 6. Run all
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

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
