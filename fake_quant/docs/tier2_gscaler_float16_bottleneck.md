# Tier-2 Accumulator Precision is Bottlenecked by Float16 Cast

## Finding

When using integer tier-2 accumulators with gscaler prescaling (e.g., `--acc_dtype int22 --gscaler M8S0b10`), reducing the tier-2 bit-width below ~22 bits has no effect on accuracy. The reason is that the int_gemm output is cast to float16 (the model's working dtype) immediately after the kernel, and float16's 10-bit mantissa is the true precision bottleneck.

## The Pipeline

```
Triton kernel (float32 internally)
  └─ tier-2 accumulator: int<N> with clamp
  └─ undo gscaler prescaling: acc *= 2^(-bias)
  └─ return float32 output

ActQuantWrapper.forward() [quant_utils.py:708]
  └─ x = int_gemm_capped(...).to(x_dtype)    ← x_dtype is float16
  └─ x = x + static_zp_bias                  ← still float16
  └─ return x → next layer
```

The `.to(x_dtype)` at line 708 of `quant_utils.py` converts the float32 kernel output to float16 (the model's working dtype, set at line 663 as `x_dtype = x.dtype`). Float16 has a 10-bit mantissa (~11 bits of precision).

## Effective Precision Analysis

With gscaler `M8S0b10` (bias=10), the weight scales are prescaled by `2^10` before the kernel. Inside the kernel, the tier-2 accumulator stores integer values that are `2^10` larger than the true output. After the K-loop, the kernel multiplies by `2^(-10)` to undo this.

Effective precision of the int_gemm output (before float16 cast):

| acc_dtype | Tier-2 bits | - Gscaler bias | = Effective bits |
|-----------|-------------|----------------|------------------|
| int32     | 32          | 10             | 22               |
| int26     | 26          | 10             | 16               |
| int24     | 24          | 10             | 14               |
| int22     | 22          | 10             | 12               |
| int20     | 20          | 10             | 10               |
| int18     | 18          | 10             | 8                |

After the float16 cast, anything above ~11 bits is truncated to float16's mantissa precision. So:

- **int22 (12 effective bits) → float16 (11 bits)**: ~1 bit lost by float16, not tier-2
- **int32 (22 effective bits) → float16 (11 bits)**: ~11 bits lost by float16, not tier-2
- Both produce identical float16 outputs

The tier-2 only becomes the bottleneck when its effective precision drops below float16's ~11 bits, i.e., around `int20` (10 effective bits) or below.

## Observed Behavior

Experiments with `--gscaler M8S0b10 --acc_bits 16 --acc_block_k 128 --static-act`:
- int22 through int32: **identical** accuracy across all metrics
- Below int22: accuracy starts to degrade (tier-2 becomes the bottleneck)

This matches the analysis: int22 gives 12 effective bits > 11 bits (float16), so float16 is the bottleneck. At int20 (10 effective bits < 11 bits), the tier-2 takes over.

## Implications

1. For float16 models with gscaler bias=10, there is no benefit to tier-2 accumulators wider than ~int22. The float16 cast discards the extra precision.
2. If the model ran in bfloat16 (7-bit mantissa, ~8 bits precision), the threshold would be even lower: ~int18.
3. To actually test tier-2 precision effects, either:
   - Run the model in float32 (remove the `.to(x_dtype)` bottleneck)
   - Use a smaller gscaler bias so effective bits are closer to float16's mantissa
   - Reduce tier-2 below int20 where it becomes the true bottleneck

## Files

- `fake_quant/quant_utils.py:708` — the `.to(x_dtype)` cast that bottlenecks precision
- `fake_quant/quant_utils.py:663` — `x_dtype = x.dtype` (float16 for half-precision models)
- `fake_quant/int_acc_gemm.py:200-205` — tier-2 to float32 conversion and gscaler undo
- `fake_quant/quant_utils.py:576-579` — gscaler prescaling setup
