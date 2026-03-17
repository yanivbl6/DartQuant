# Weight Asymmetry Causes Accumulator Drift

## Summary

When using asymmetric weight quantization (`--w_asym`) with the capped-accumulator integer GEMM (`--int_gemm --acc_bits 16`), partial sums inside the kernel overflow the limited-width accumulator, producing garbage outputs and catastrophically bad perplexity. Symmetric weight quantization works correctly under the same settings.

## Background: Capped Accumulator Integer GEMM

Real hardware integer GEMM units have finite-width accumulators. Our `int_gemm_capped` simulates this: it multiplies integer activations by integer weights and accumulates the dot product in blocks of size `acc_block_k`, clamping partial sums to fit within `acc_bits` (e.g., int16 range [-32768, 32767]).

The full computation for a single output element is:

```
y = kernel(a_int, w_int)       # Term A: capped integer dot product
  + a_zp_correction             # Term B: activation zero-point × weight sum
  + w_zp_correction             # Term C: activation sum × weight zero-point
  + cross_zp_correction          # Term D: activation zp × weight zp × K
```

Terms B, C, and D are correction terms computed in exact floating point, outside the kernel. They compensate for the zero-point offsets so that the result matches the true floating-point GEMM.

## The Problem

### Symmetric quantization (works)

With symmetric quantization, integer weights are centered around zero:

```
w_int ∈ [-8, 7]    (4-bit example)
```

When the kernel computes `sum(a_int × w_int)` over a block of K elements, positive and negative products partially cancel. The partial sums stay moderate and well within int16 range.

**Example** (K=4, a_int=[3, 5, 2, 4], w_int=[-2, 3, -1, 4]):
```
partial sums: -6, +9, +7, +23
```
All comfortably within [-32768, 32767]. No overflow, no information loss.

### Asymmetric quantization (broken)

With asymmetric quantization, the weight zero-point shifts all integer weights to be non-negative:

```
w_int ∈ [0, 15]    (4-bit example, w_zp ≈ -8)
```

Every `w_int` value is inflated by `|w_zp|` compared to its symmetric counterpart. The true floating-point weight is recovered by `w_float = scale × (w_int + w_zp)`, where `w_zp` is negative. But inside the kernel, we accumulate `a_int × w_int` (the inflated values) and only subtract the `w_zp` contribution afterward, in exact float (Term C).

This creates a systematic positive bias in the kernel's partial sums.

**Example** (same true weights, asymmetric representation with w_zp = -8):
```
w_int = [6, 11, 7, 12]    (= symmetric w_int + 8)
a_int = [3, 5, 2, 4]

partial sums: 18, 73, 87, 135
```

With `acc_block_k=128` and 8-bit activations (a_int up to 127), worst case for a single block:
```
max partial sum = 128 × 127 × 15 = 241,920   >> 32,767 (int16 max)
```

The accumulator saturates, clamping the sum. **Information is permanently lost.**

Term C then subtracts `sum(a_int) × w_zp × scale` in exact float — but it's compensating for a kernel result that has already been corrupted by saturation. You cannot recover bits that were clamped away.

## Numeric Walkthrough

Consider K=2, acc_max=7 (3-bit accumulator for clarity), true weights = [-1, -1]:

### Symmetric path
```
w_int = [-1, -1],  w_zp = 0
a_int = [2, 2]

kernel: 2×(-1) + 2×(-1) = -4        (no overflow, -4 ∈ [-8, 7])
term C: 0                             (w_zp = 0)
result: -4                            ✓ correct
```

### Asymmetric path
```
w_int = [3, 3],  w_zp = -4            (same true weight: scale×(3 + (-4)) = scale×(-1))
a_int = [2, 2]

kernel: 2×3 + 2×3 = 12 → clamped to 7   (overflow! 12 > 7)
term C: (2+2) × (-4) × scale = -16
result: 7 + (-16) = -9                ✗ should be -4, error = 5
```

The kernel lost 5 units of information to saturation. Term C faithfully subtracts the zero-point correction, but it's subtracting from a corrupted base.

## Observed Impact

Using the `--ig_compare` diagnostic mode, which runs both the int_gemm and normal float GEMM paths and compares per-layer:

### Symmetric weights (`--w_asym` omitted)
```
[ig_cmp] model.layers.0.self_attn.q_proj: full=2.34e-02   (2.3% relative error)
[ig_cmp] model.layers.0.self_attn.k_proj: full=3.91e-02   (3.9% relative error)
```
All layers show 2-4% Linf relative error — acceptable quantization noise.

### Asymmetric weights (`--w_asym`)
```
[ig_cmp] model.layers.1.self_attn.o_proj: A_only=9.36e+01 A+B+D=9.36e+01 full=1.14e+01
    |bias|=5.91e+00  |termC|=6.65e+01  |ref|=7.11e-01
```

Breaking this down:
- **A_only = 93.6×**: The kernel output alone is 93× off from the correct answer
- **A+B+D = 93.6×**: Adding the static bias (terms B and D) doesn't help — the kernel is the problem
- **|termC| = 66.5**: Term C provides a massive correction of magnitude 66.5...
- **|ref| = 0.711**: ...but the correct answer has magnitude only 0.711
- **full = 11.4×**: After all corrections, still 11× relative error

This is the classic signature of catastrophic cancellation: two large quantities (kernel ≈ 93 × ref, termC ≈ 93 × ref) are subtracted to produce a small result (≈ 1 × ref), but the kernel has lost precision to saturation, so the residual error dominates.

## Root Cause

The fundamental issue is architectural: the capped accumulator integer GEMM assumes that partial sums of `a_int × w_int` remain within the accumulator's range. Symmetric quantization satisfies this assumption because `w_int` values are balanced around zero, keeping partial sums moderate through natural cancellation. Asymmetric quantization violates it by shifting all `w_int` values positive, creating systematically large partial sums that overflow the accumulator.

The zero-point correction terms (B, C, D) are mathematically correct — they would perfectly recover the true result if the kernel computed `sum(a_int × w_int)` exactly. But the capped accumulator introduces a non-linear saturation that destroys the algebraic identity these corrections rely on.

## Potential Fixes

### Option 1: Subtract w_zp inside the kernel (correct fix)

Modify the Triton kernel to compute `a_int × (w_int - w_zp)` instead of `a_int × w_int`. This keeps partial sums centered around zero, matching the symmetric case. Term C becomes unnecessary.

**Pros**: Enables asymmetric + int_gemm correctly.
**Cons**: Requires kernel changes; per-group `w_zp` must be loaded and broadcast inside the inner loop; slightly increases register pressure.

### Option 2: Disallow `--w_asym` with `--int_gemm` (pragmatic fix)

Add argument validation that rejects the combination. Symmetric quantization works correctly (2-4% error) and is the expected use case for hardware integer GEMM.

**Pros**: Zero code risk, immediate fix.
**Cons**: Limits flexibility.

### Option 3: Reduce acc_block_k (workaround)

Use very small block sizes so partial sums can't overflow even with asymmetric bias. Required block size: `acc_block_k ≤ acc_max / (a_max × w_max)` = `32767 / (127 × 15)` ≈ 17.

**Pros**: No kernel changes.
**Cons**: Tiny blocks destroy performance; defeats the purpose of block accumulation.

## Files Involved

- `fake_quant/int_acc_gemm.py` — Triton kernel and correction terms
  - `int_gemm_capped()` (line ~214): dispatches kernel + applies corrections
  - `_apply_weight_zp_correction()` (line ~316): computes Term C
  - Triton kernel (line ~36): the capped accumulator inner loop
  - `prepare_int_weights()` (line ~599): computes `w_zp` from GPTQ params
- `fake_quant/quant_utils.py` — `ActQuantWrapper.forward()` orchestrates the call
- `fake_quant/gptq_utils.py` — `fasterquant()` stores per-group scale/zero used by int_gemm

## Diagnostic Tool

The `--ig_compare` flag (added during this investigation) runs both paths side-by-side and prints per-term error breakdown. Usage:

```bash
python main_for_test.py [usual args] --w_asym --int_gemm --acc_block_k 128 --acc_bits 16 --ig_compare
```

It returns the float GEMM result for correct PPL while printing int_gemm divergence per layer. Throws `RuntimeError` if any layer exceeds 100% relative error.
