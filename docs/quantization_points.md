# Quantization Points in a Transformer Block

This document maps every quantization point in a single Llama decoder layer
as simulated by DartQuant.  Integer-GEMM accumulator details (`--int_gemm`,
`--acc_bits`, `--acc_wrap`) are out of scope here -- they affect the
*precision of the matmul* rather than adding a distinct quantization point.

Rotations (R1-R4) and equalization are not quantization points -- they are
linear transforms that redistribute activation energy to improve
quantization quality.  They are noted in the diagram for context but
documented separately at the end.

---

## Block Diagram

```
                     hidden_states (from previous block or embedding)
                            |
                      [RES_QUANT]  ── --quant_out res/ex  (*)
                            |
                       LayerNorm
                            |
               ┌────────────┼────────────┐
               v            v            v
           ┌───────┐   ┌───────┐   ┌───────┐
           │q_proj │   │k_proj │   │v_proj │
           │       │   │       │   │       │
           │ Q_IN  │   │ Q_IN  │   │ Q_IN  │  ── input quantizer (--a_bits)
           │ matmul│   │ matmul│   │ matmul│
           │ Q_OUT │   │ Q_OUT │   │ V_OUT │  ── --quant_out all/mm/ex (16-bit)
           └───┬───┘   └───┬───┘   └───┬───┘     v_proj uses --v_bits for V-cache
               |            |            |
               v            v            |
        ┌──────────────────────┐         |
        │  RoPE (pos. embed.)  │         |
        │                      │         |
        │  (R3 Hadamard Q,K)   │         |
        │                      │         |
        │  [Q_QUANT] Q quant   │  ── --quant_out mm/ex (16-bit)
        │  [K_QUANT] K quant   │  ── --k_bits, --k_groupsize (K-cache)
        └──────┬───────────────┘         |
               |                         |
               v                         v
        ┌─────────────────────────────────────┐
        │         Attention Mechanism          │
        │                                      │
        │   Q @ K^T / sqrt(d)                  │
        │   + causal mask                      │
        │   softmax                            │
        │   [SMQ] softmax output quant  ── --smq (unsigned, N-bit)
        │   attn_weights @ V                   │
        │                                      │
        └──────────────┬───────────────────────┘
                       |
                       v
                 ┌───────────┐
                 │  o_proj    │
                 │            │
                 │  (R2)      │  (rotation, not a quantization point)
                 │  Q_IN      │  ── input quantizer (--a_bits, or --o_per_head groupsize)
                 │  matmul    │
                 │  Q_OUT     │  ── --quant_out all/mm/ex (16-bit)
                 └─────┬──────┘
                       |
                       + ─── residual add (skip connection)
                       |
                 [RES_QUANT]  ── --quant_out res/ex  (*)
                       |
                  LayerNorm
                       |
          ┌────────────┴────────────┐
          v                         v
    ┌───────────┐            ┌───────────┐
    │ gate_proj  │            │  up_proj   │
    │            │            │            │
    │  Q_IN      │            │  Q_IN      │  ── input quantizer (--a_bits)
    │  matmul    │            │  matmul    │
    │  Q_OUT     │            │  Q_OUT     │  ── --quant_out all/mm/ex (16-bit)
    └─────┬──────┘            └─────┬──────┘
          |                         |
        SiLU / [PWL]                |     ── --pwl_act (piecewise-linear approx)
          |                         |
          └────────── * ────────────┘     element-wise multiply (float)
                      |
                [PRE_QUANT]               ── --quant_out r4/ex (16-bit, before R4)
                      |
               ┌──────────────┐
               │  down_proj    │
               │               │
               │  (R4 Hadamard)│  (rotation, not a quantization point)
               │  (eq scaling) │  (linear rescaling, not a quantization point)
               │  Q_IN         │  ── input quantizer (--down_bits or --a_bits)
               │  matmul       │
               │  Q_OUT        │  ── --quant_out all/mm/ex (16-bit)
               └───────┬───────┘
                       |
                       + ─── residual add (skip connection)
                       |
                 [RES_QUANT]  ── --quant_out res/ex  (*)
                       |
                       v
              output (to next block)

(*) RES_QUANT note: at 16-bit, the residual quantizer is strictly coarser
    than the downstream input quantizers (which re-quantize after LayerNorm).
    Its purpose is to assert that the residual stream fits in 16-bit integer
    representation (hardware constraint), not to add meaningful quantization
    noise.
```

---

## Quantization Point Reference

### Input Quantizers (Q_IN)

Applied at the input of every `nn.Linear` layer, inside `ActQuantWrapper`.

| Layer     | Flag             | Default | Notes |
|-----------|------------------|---------|-------|
| q/k/v_proj| `--a_bits`       | 16 (off)| Per-token, asymmetric by default |
| o_proj    | `--a_bits`       | 16 (off)| `--o_per_head` sets groupsize = head_dim |
| gate/up   | `--a_bits`       | 16 (off)| Same as q/k/v |
| down_proj | `--down_bits` or `--a_bits` | 16 | Often set independently (e.g. 16-bit when R4 is active) |
| lm_head   | (always 16)      | 16      | Never quantized |

All input quantizers support static (pre-calibrated) or dynamic (per-forward)
scales.  Asymmetric quantization is the default for activations (`--a_asym`
is implicit).

### Output Quantizers (Q_OUT) -- `--quant_out`

Applied after the matmul output of `ActQuantWrapper`, before the result
leaves the wrapper.

| Mode    | Layers affected | Notes |
|---------|----------------|-------|
| `none`  | (nothing)      | Default |
| `up`    | up_proj        | |
| `mlp`   | gate + up + down_proj | |
| `spec`  | all except q/k/v_proj | |
| `speco` | all except q/k/v/o_proj | |
| `all`   | all layers (except lm_head) | |
| `mm`    | all layers + Q quantizer in attention | |
| `r4`    | (none -- uses pre_quantizer on down_proj instead) | |
| `res`   | (none -- uses residual quantizers instead) | |
| `ex`    | all layers + Q quantizer + residual + pre-R4 | Full HW simulation |

Output quantizers are 16-bit symmetric, per-token.  They do not override
the v_proj V-cache out_quantizer which is configured separately via
`--v_bits`.

### Pre-Rotation Quantizer (PRE_QUANT) -- `--quant_out r4/ex`

Applied on down_proj input *before* the R4 Hadamard rotation.  This is a
separate quantizer (`pre_quantizer`) from the input quantizer.  It ensures
the data entering R4 is in integer format.

### K-Cache Quantizer (K_QUANT)

Applied to K after RoPE (and R3 if enabled), inside `QKRotationWrapper`.

| Flag           | Effect |
|----------------|--------|
| `--k_bits`     | Bit-width (default: 16 = off) |
| `--k_groupsize`| -1 = per-token, >= head_dim = per-head, < head_dim = sub-head groups |
| `--k_asym`     | Asymmetric K quantization |
| `--k_clip_ratio`| Clipping ratio for scale computation |

### Q Quantizer (Q_QUANT) -- `--quant_out mm/ex`

Applied to Q after RoPE (and R3 if enabled), inside `QKRotationWrapper`.
Parallel to K-cache quantizer but for query states.  16-bit symmetric,
per-token.  Ensures Q is in integer format before Q @ K^T.

### V-Cache Quantizer (V_OUT)

Applied as the out_quantizer on v_proj.  Configured separately from
`--quant_out`:

| Flag           | Effect |
|----------------|--------|
| `--v_bits`     | Bit-width (default: same as --k_bits) |
| `--v_groupsize`| Group size (default: -1 = per-token) |
| `--v_asym`     | Asymmetric V quantization |

### Softmax Output Quantizer (SMQ) -- `--smq`

Applied to attention weights after softmax, before the attn_weights @ V
matmul.  Unsigned quantization (softmax output is in [0,1]).

| Flag   | Effect |
|--------|--------|
| `--smq N` | N-bit unsigned quantization (0 = disabled) |

### Residual Quantizers (RES_QUANT) -- `--quant_out res/ex`

Applied after each residual add in the decoder layer (2 per block):
1. After attention output + skip connection
2. After MLP output + skip connection

16-bit symmetric, per-token.  At 16-bit these are strictly coarser than
the downstream input quantizers (which re-quantize after LayerNorm).
Their purpose is to verify the residual stream fits in 16-bit integer
representation on hardware, not to introduce meaningful quantization noise.

---

## Rotations (not quantization points)

Rotations are linear transforms that redistribute activation energy across
dimensions, reducing outliers and improving quantization quality.  They do
not themselves quantize data.

| Rotation | Where Applied | Mode | Flag |
|----------|--------------|------|------|
| **R1** | Embedding, q/k/v/o_proj weights, gate/up_proj weights, lm_head | Offline (fused into weights) | `--use_r1`, `--r1_path` |
| **R2** | v_proj output + o_proj input (per-head) | Offline (fused) or Online (Hadamard) | `--use_r2 offline/online/none`, `--r2_path` |
| **R3** | Q and K after RoPE (per-head Hadamard) | Online | `--use_r3` (disabled by `--kv_ex`) |
| **R4** | down_proj input (full-dim Hadamard) | Online | `--use_r4` (disabled by `--no_r4` / `--proj_ex`) |

**R1** and **R2** (offline) are fused into weights at model load time and
have zero runtime cost.  **R2** (online), **R3**, and **R4** are applied
at runtime as Hadamard transforms.

### Rotation-Quantization Interactions

- **R3 + K-cache**: R3 is applied to Q and K *before* K-cache quantization.
  When `--kv_ex N` is used, R3 is disabled and K-cache is quantized to N
  bits without rotation.

- **R4 + down_proj**: R4 is applied *before* the down_proj input quantizer.
  The `--quant_out r4/ex` pre_quantizer sits *before* R4, quantizing the
  element-wise multiply result before the Hadamard spreads it.

- **R2 + o_proj**: In online mode, R2 is applied as a partial Hadamard at
  the start of o_proj's forward, before the input quantizer.

---

## Other Operations (not quantization points)

### Equalization -- `--eq`

Per-channel linear rescaling on down_proj input (after R4, before input
quantizer).  Mathematically transparent: `W_eq @ x_eq = W @ x`.  Not a
quantization point -- it reshapes the activation distribution to improve
the downstream quantizer's effectiveness.

### PWL Activation -- `--pwl_act`

Replaces SiLU with a piecewise-linear approximation (from Hailo SDK).
Has its own input/output quantizers:

| Flag                | Effect |
|---------------------|--------|
| `--pwl_n_segments`  | Number of linear segments (default: 9) |
| `--pwl_input_bits`  | Input quantizer bits (default: 16) |
| `--pwl_output_bits` | Output quantizer bits (default: 16) |
| `--pwl_no_hw_sim`   | Disable HW precision simulation |
