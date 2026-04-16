# Inside the MAC Unit — Integer GEMM Kernel

Companion to [quantization_points.md](quantization_points.md).  That document
treats every `matmul` box as a single opaque step; here we crack that box
open and show what actually happens when `--int_gemm` is active.  The
diagrams below trace one `nn.Linear` forward through
[int_acc_gemm.py](../fake_quant/int_acc_gemm.py) and the Triton kernel
`_int_gemm_capped_acc_kernel`.

The goal is to make the **real vs. simulated** split explicit at every
stage — what the GPU is actually computing, versus what is being *modeled*
to match a target NPU datapath.

---

## Legend

| Symbol        | Meaning                                                     |
|---------------|-------------------------------------------------------------|
| **REAL**      | Computed on the GPU at that precision — hardware-faithful   |
| **SIM**       | Stored in a wider type but *constrained* to model target HW |
| `[shape]`     | Tensor shape — `M`=tokens, `N`=out-ch, `K`=in-ch             |
| `dtype`       | Actual PyTorch/Triton dtype in use                          |
| `A_GS`, `GS`  | Shorthand for `A_GROUP_SIZE`, `GROUP_SIZE` in the kernel     |

---

## Top-level MAC block

```
  SETUP-TIME   (prepare_int_weights — once at model load)
  ┌──────────────────────────────────────────────┐
  │  W_int   [N,K]  int8                   REAL  │──► w_int     ───┐
  │  w_scale fp32   SIM: gscaler snapped         │──► w_scale   ───┤
  │          (×2^w_shift_bias prescale, optional)│                 │
  │  w_zp    int32  (asym W only)                │──► w_zp      ───┤
  └──────────────────────────────────────────────┘                 │
                                                                   ▼
  RUNTIME                  ┌──────────────────┐                 ┌───────────┐    ┌───────────┐    ┌─────────────────┐
   x_float [..., K]  ────► │  ActQuantizer    │ ─► a_int     ─► │           │    │           │    │ + zp corrections│
   fp16/bf16               │ .quantize_to_int │ ─► a_scale   ─► │  K-LOOP   │ ─► │ POST-LOOP │ ─► │   (3 paths →)   │ ──► y [..., N]
   post-LN / R / eq        │                  │ ─► a_zp_corr ─► │  K-chunks │    │ (post-K)  │    │                 │     fp16/bf16
                           └──────────────────┘                 └───────────┘    └───────────┘    └─────────────────┘     (cast)
                                                                      ▲                                   │
                                                          (see next figure)                               │
                                                                                                          ▼
                                                          + a_zp_corr ⊗ w_zp_corr    (asym a, sym  W)
                                                          + a_zp_corr ⊗ w_zp_cross   (asym a, asym W — dynamic)
                                                          + static_zp_bias           (asym a, asym W — static)
```

---

## Inside the K-loop (one `ACC_BLOCK_K` iteration)

This is the heart of the simulation — the block that models one NPU MAC
cycle.  The outer loop runs ⌈K / ACC_BLOCK_K⌉ times per output tile.

```
  ◄═══  L O O P   O V E R   K - C H U N K S :   k_start = 0, ACC_BLOCK_K, 2·ACC_BLOCK_K, …   (⌈K / ACC_BLOCK_K⌉ iterations)  ═══►

  a_tile ─┐
  int8    │
  [BM,BK] │    ┌──────┐   ┌─────────┐   ┌───────────┐   ┌─────────┐   ┌────────┐   ┌─────────────┐   ┌─────────────┐   ┌───────────┐
          ├──► │tl.dot│─► │ tier-1  │─► │+ Σa·w_zp  │─► │ tier-1  │─► │.to fp32│─► │ × a_gscale  │─► │ × w_gscale  │─► │  tier-2   │──► acc
  b_tile ─┘    │int8× │   │   cap   │   │ (in-loop) │   │  cap²   │   │ promote│   │ (per-group  │   │ (per-group  │   │ accumulate│
  int8         │ int8 │   │         │   │           │   │         │   │        │   │  activation │   │  weight)    │   │           │
  [BN,BK]      │→int32│   │         │   │           │   │         │   │        │   │  static)    │   │             │   │           │
               └──────┘   └─────────┘   └───────────┘   └─────────┘   └────────┘   └─────────────┘   └─────────────┘   └───────────┘
                REAL        SIM           REAL            SIM           REAL          REAL              REAL              REAL / SIM
                int8×int8   ±2^(A-1)      int32 add       same as       int→fp32      fp32 × SIM-       fp32 × SIM-       acc_dtype:
                → int32     cap / wrap    HAS_W_ZP        cap¹          (no-op on     scale             scale             int<N>[pM] clamp
                tensor      --acc_bits      only          (re-capped)   HW)           (A_GS>0 only)     (GS>0 only)       fp16 / bf16
                core int    --acc_wrap                    same flags                                                       bf16 / fp32
                                                                                                                           --acc_dtype

   acc persists across K-chunks — it is the tier-2 state carried into the next iteration.
```

### K-loop stage details — what each box really does

The K-loop runs `⌈K / ACC_BLOCK_K⌉` times per output tile.  Every box in
the diagram fires *once per chunk* — **one chunk = `ACC_BLOCK_K` products
summed, capped, and pushed into tier-2**.  The tier-1 cap is **NOT**
applied after every multiply; only at the end of each chunk.

**① `tl.dot`  — int8 × int8 → int32**
 - `partial[m, n] = Σ_{k=0..BK−1}  a_tile[m, k] · b_tile[n, k]`
 - REAL tensor-core int8 matmul — accumulates into the GPU's native int32.
 - **No capping during this inner sum.** The int32 accumulator is wide
   enough that intra-chunk overflow cannot occur for int8 operands.

**② Tier-1 accumulator cap  — `--acc_bits A`, `--acc_wrap`**
 - **Frequency: once per K-chunk** (i.e. once per `ACC_BLOCK_K` products),
   *not* after every multiply.  This is the single answer to "when does
   acc16 capping happen?"
 - Saturate (default):   `partial = clamp(partial, −(2^(A−1)−1), 2^(A−1)−1)`
 - Wrap  (`--acc_wrap`):  `partial = ((partial − ACC_MIN) mod 2^A) + ACC_MIN`
 - **Approximation for saturation:**  real HW with a narrow accumulator
   saturates *mid-chunk* (after each add).  Here we sum in int32 and
   clip at chunk end — so mid-chunk overflows that would have saturated
   on HW are invisible.  **Wrap mode is exact**: modular arithmetic
   commutes with addition, so end-of-chunk mod equals per-step mod.
 - Smaller `ACC_BLOCK_K` → more frequent caps on smaller partial sums
   (closer to real mid-chunk HW behavior); larger `ACC_BLOCK_K` → fewer
   caps on larger partial sums (coarser simulation).

**③ `+ Σ_k(a) · w_zp_g`  — in-loop weight zero-point correction**
 - Fires only when `HAS_W_ZP` (asymmetric weights).
 - Integer add in int32:  `a_block_sum[M] · w_zp_g[N] → [M, N]`.
 - Performed *inside* the narrow-accumulator domain so the next tier-1
   cap applies to the corrected value — this faithfully models an NPU
   that folds the w_zp correction into the same MAC block.

**④ Tier-1 cap²  — re-cap after the w_zp add**
 - Same formula, same `--acc_bits`, same `--acc_wrap` as ②.
 - Models HW committing the w_zp-corrected value back to the narrow
   accumulator before moving on.

**⑤ `.to(fp32)`  — promote out of integer domain**
 - Pure dtype conversion in Triton; no value change (int32 fits exactly
   in fp32 up to 2^24, capped values are well below that).
 - On real HW there is no cast — the int value is the direct input to a
   fixed-point rescaler.

**⑥ `× a_gscale[g]`  — per-group activation scale (if `A_GROUP_SIZE > 0`)**
 - One scalar per K-chunk, applied to `partial_f` in fp32.
 - Folds *inside* the loop because the scale is constant across the
   chunk (per-group scale groupsize == `ACC_BLOCK_K` by design).
 - **Dynamic per-token `a_scale` is NOT applied here** — it is one
   scalar per row (not per chunk), so it factors out of the full K-sum
   and is applied after the loop (see post-K-loop ③).

**⑦ `× w_gscale[N, g]`  — per-group weight scale (if `GROUP_SIZE > 0`)**
 - One scalar per (output-channel, K-chunk).
 - `w_gscale` is fp32 in memory but was pre-quantized to the gscaler
   `M<m>S<s>` mantissa+shift format at setup — this is the **SIM** step:
   the multiply is a real fp32 op, but the operand is a gscaler-snapped
   value.  Optionally prescaled by `2^w_shift_bias` (see corner case 2).
 - **Per-channel weight scale (`GROUP_SIZE ≤ 0`) is NOT applied here** —
   same reason as per-token activation scale; applied after the loop.

**⑧ Tier-2 accumulate  — `--acc_dtype <kind>`**
 - Four modes, selected by `--acc_dtype`:

   | Mode             | Operation                                                                  | REAL / SIM |
   |------------------|----------------------------------------------------------------------------|------------|
   | `float` / `fp32` | `acc_fp32 += contrib`                                                       | REAL       |
   | `fp16`           | `acc_fp16 += contrib.to(fp16)`                                              | REAL (HW cast) |
   | `bf16`           | `acc_bf16 += contrib.to(bf16)`                                              | REAL       |
   | `int<N>[p<M>]`   | `acc_i32 += round(contrib · 2^M).to(int32)`;  `acc_i32 = clamp(±(2^(N−1)−1))` | **SIM**    |

 - **Integer tier-2 is clamped every K-chunk** — same frequency as
   tier-1.  `fp16`/`bf16` accumulate natively and are *not* re-capped
   per chunk; their accuracy loss comes from the reduced mantissa on
   each add, not from clamping.
 - The `round(· 2^M)` step models fixed-point `int<N>p<M>` storage:
   `N` total bits, `M` fractional.  E.g. `int26p8` = 26-bit signed with
   8 fractional bits (representable range ±2^17, resolution 2^−8).
 - `acc` is the only state that **persists** across loop iterations —
   it is the running tier-2 sum carried into the next chunk.

---

## Post-K-loop (tier-2 → output, once per `[BM, BN]` tile)

```
  (runs ONCE per output tile, after K-loop completes)

               ┌─────────┐   ┌────────────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────┐   ┌──────────┐
  acc ───────► │.to fp32 │─► │× 2^-(frac+w_shift) │─► │× a_scale[M]  │─► │× w_scale[N]  │─► │+ bias  │─► │ tl.store │──► y  [M, N] fp32
  (tier-2)     │ promote │   │ undo frac_bits +   │   │ dynamic only │   │ per-channel  │   │(if any)│   │          │
               │ REAL    │   │  gscaler prescale  │   │ if A_GS ≤ 0  │   │ if GS ≤ 0    │   │  REAL  │   │          │
               └─────────┘   │ (single fp32 mul)  │   │  REAL        │   │  REAL        │   └────────┘   └──────────┘
                             │  REAL              │   │ (factors out │   │ (factors out
                             └────────────────────┘   │  of K-sum)   │   │  of K-sum)
                                                      └──────────────┘   └──────────────┘
```

### Post-K-loop stage details

Runs **ONCE per output tile** after the K-loop completes — i.e. after
the last K-chunk has been accumulated into tier-2.

**① `acc.to(fp32)`** — Promote the tier-2 accumulator to fp32 for the
final rescale.  For integer tier-2 this is a real int→fp cast; for
fp16/bf16 it is a native widening cast; for fp32 it is a no-op.

**② `× 2^(−(T2_FRAC_BITS + W_SHIFT_BIAS))`** — Single fp32 multiply
that *simultaneously* undoes **two** prescale tricks performed inside
the loop:
 - `T2_FRAC_BITS` — the fractional-bit multiplier from `int<N>p<M>`
   tier-2.  Each `contrib` was multiplied by `2^M` before adding into
   `acc_i32` (so the fixed-point representation had M fractional bits).
   Undone here as `× 2^(−M)`.
 - `W_SHIFT_BIAS` — the gscaler-bias prescale on `w_scale`.  If the
   gscaler had `bias ≥ 1` and tier-2 is integer, `w_scale` was
   pre-multiplied by `2^bias` at setup (giving extra integer headroom).
   The inverse `× 2^(−bias)` is applied here.
 - Merged into one fp32 multiply to avoid two passes.  When both are
   zero this step is skipped entirely.

**③ `× a_scale[M]`  — only when `A_GROUP_SIZE ≤ 0`**
 - Dynamic per-token scale: one fp32 scalar per row `m`.
 - Static per-tensor fallback: same `[M]` shape, all entries equal to
   `col_scale.max()`.
 - Couldn't be folded into the loop because one scalar per row ≠ one
   scalar per K-chunk.

**④ `× w_scale[N]`  — only when `GROUP_SIZE ≤ 0`**
 - Per-channel weight scale (one fp32 scalar per output channel `n`).
 - Same reasoning as ③: factors out of the full K-sum only, not per
   K-chunk.  Snapped to gscaler `M<m>S<s>` format at setup (SIM).

**⑤ `+ bias[N]`  — only when the `nn.Linear` has a bias**
 - Plain fp32 add, broadcast across `M`.

**⑥ `tl.store`** — Write the `[M, N]` fp32 output tile to global memory.
The caller (`ActQuantWrapper.forward`) may then add the external
zero-point correction terms and cast back to the original `x_dtype`.

---

## Format summary per stage

| Stage                            | Shape            | Actual dtype   | Real / Sim | Notes |
|----------------------------------|------------------|----------------|------------|-------|
| `x_float` input                  | `[…, K]`         | fp16 / bf16    | REAL       | post-LN, post-rotation, post-eq |
| `a_int`                          | `[M, K]`         | int8 / int16   | REAL       | int16 decomposed into 2 int8 passes (`_decompose_int16_to_int8`) |
| `a_scale` (dynamic)              | `[M]`            | fp32           | REAL       | per-token, computed at runtime |
| `a_scale` (static per-group)     | `[n_agroups]`    | fp32           | REAL       | groupsize = `acc_block_k` by design |
| `a_zp_corr`                      | `[M]`            | fp32           | REAL       | asym act only; `scale·(shift−zero)` |
| `w_int`                          | `[N, K]`         | int8           | REAL       | midpoint-centred when asym |
| `w_scale`                        | `[N]` / `[N, gW]`| fp32           | **SIM**    | snapped to gscaler `M<m>S<s>` mantissa+shift format; optionally prescaled by `2^w_shift_bias` (only when tier-2 is integer and `gscaler.bias ≥ 1`) |
| `w_zp`                           | `[N]` / `[N, gW]`| int32          | REAL       | asym W only |
| `partial = tl.dot(a,bᵀ)`         | `[BM, BN]`       | int32          | REAL       | tensor-core int8 dot product |
| partial after tier-1 cap         | `[BM, BN]`       | int32          | **SIM**    | clamped/wrapped to `±2^(acc_bits−1)` — models narrow NPU accumulator |
| partial after w_zp correction    | `[BM, BN]`       | int32          | REAL+SIM   | integer add (real), then tier-1 cap (sim) |
| `contrib`                        | `[BM, BN]`       | fp32           | REAL       | scales applied in fp32 |
| `acc` (tier-2, int)              | `[BM, BN]`       | int32 storage  | **SIM**    | clamped to `±2^(T2_bits−1)` each step; frac_bits models fixed-point |
| `acc` (tier-2, fp16/bf16)        | `[BM, BN]`       | fp16 / bf16    | REAL       | native HW reduced-precision accumulate |
| output `y`                       | `[…, N]`         | fp32 → x_dtype | REAL       | cast back at wrapper boundary |

---

## Argument → effect map

Args are the ones on `--int_gemm` enabled layers (`ActQuantWrapper`).

| Flag                                | Controls                                       | When it matters                                     |
|-------------------------------------|------------------------------------------------|-----------------------------------------------------|
| `--int_gemm`                        | Master switch: take this path at all           | Always required for this kernel                     |
| `--a_bits N`                        | int8 (N≤8) or int16 (8<N≤16) activation path   | Chooses `_triton_int_gemm` vs `_triton_int16_gemm` (two passes + `·256` + add) |
| `--w_bits N`                        | Range of `w_int` integers                      | Drives `prepare_int_weights`; `w_int` is still int8-typed regardless |
| `--a_asym` / symmetric              | Whether `a_zp_corr` is produced                | Adds post-kernel term `a_zp_corr ⊗ w_zp_corr` (or `⊗ w_zp_cross` for asym W) |
| `--w_sym` / `--w_asym`              | Whether `w_zp` exists                          | Activates in-kernel `HAS_W_ZP` branch + re-cap      |
| `--a_groupsize G` (static)          | Per-group activation scales                    | Folds `a_gscale` *inside* the K-loop; must equal `acc_block_k` |
| `--w_groupsize gW`                  | Per-group weight scales                        | Folds `w_gscale` *inside* the K-loop when `>0`      |
| static vs dynamic                   | Source of `a_scale`                            | Dynamic → per-token scale applied *after* the K-loop. Static per-group → per-K-block scale applied *inside*. Static per-tensor → single scalar outside. |
| `--acc_bits N`                      | Tier-1 accumulator width                       | Sets `ACC_MAX/MIN` — one cap applied per `ACC_BLOCK_K` products, NOT per multiply |
| `--acc_block_k N`                   | K-chunk size = # products between caps         | Also sets the static-mode activation groupsize. Smaller N ⇒ more frequent cap on smaller partial sums (closer to real mid-chunk HW); larger N ⇒ fewer caps on larger sums |
| `--acc_wrap`                        | Saturate (default) vs two's-complement wrap    | Flips tier-1 cap formula. Wrap mode is exact vs real HW (mod commutes with add); saturate mode is an approximation |
| `--acc_dtype <kind>`                | Tier-2 accumulator format                      | `int<N>[p<M>]` → integer/fixed-point sim; `fp16`/`bf16` → real reduced precision; `fp32` → no-op baseline |
| `--gscaler M<m>S<s>[b<k>\|l<k>]`    | Snaps `w_scale` (and weight quantizer scale) to mantissa+shift | Changes stored `w_scale` values *before* the kernel; when `bias≥1` and tier-2 is int, also triggers `w_shift_bias` prescaling (integer headroom trick) |
| `--smq` / softmax quantizer         | *Outside* this MAC unit                        | Noted for completeness — affects input of the *next* MAC |

---

## Corner cases worth calling out

**1. int16 activation path.**  `tl.dot` is int8-only, so int16 activations
are decomposed into signed `(a_hi, a_lo)` int8 bytes via carry absorption
and the kernel is called twice.  The results are combined as
`out_hi · 256 + out_lo`.  Both passes use the *same* `acc_bits` — the cap
simulates the NPU accumulator for each half independently.  Bias is added
once, after combining.

**2. gscaler prescaling (`w_shift_bias`).**  When the gscaler has
`bias ≥ 1` and tier-2 is integer, `w_scale` is multiplied by `2^bias`
before the kernel runs.  The kernel accumulates larger products into its
integer tier-2 (giving headroom for the fixed-point representation), and
the inverse `2^(−bias)` is folded into the single post-loop fp32 multiply
together with the fractional-bit undo (`2^(−frac_bits)`).  For fp16/bf16
tier-2, this prescaling is skipped — fp16 would overflow with `bias ≥ 1`.

**3. Why per-group scales *can* fold in, per-channel scales *cannot*.**
A per-group scale is one scalar per `K`-block, so it factors out of that
block's partial sum and can be multiplied once on `partial_f`.  A per-token
or per-channel scale is one scalar per whole K-sum, so it factors out only
after the full reduction — hence the post-loop multiplies.  Per-column
scales cannot factor out at all and are not representable in this kernel
(this is why static mode requires `groupsize = acc_block_k`).

**4. Three places zero-point corrections happen.**  Asymmetric
quantization introduces cross-terms `(q − zp)(q − zp)` that, when
expanded, produce four terms.  The kernel handles them in three separate
locations:

 - **`w_zp` term** (symmetric a, asym W):  applied *inside* the K-loop
   before the tier-1 cap, as an integer add `Σ_k(a) · w_zp`.  Capped again.
 - **`a_zp × w_corr` term** (asym a, sym W, or asym-W dynamic):  applied
   in fp32 *after* the kernel returns, as `a_zp_corr[M] ⊗ w_zp_corr[N]`.
 - **Static fully-precomputed term** (asym a, asym W, static mode):
   collapsed into a single `[N]` vector `static_zp_bias` at setup time and
   added once in `ActQuantWrapper.forward`, *before* any bf16 cast.

**5. Tier-1 cap is applied twice when `HAS_W_ZP`.**  Once after the
`tl.dot`, then again after the `w_zp` integer correction.  Both caps use
the same `ACC_MAX/MIN` and the same `ACC_WRAP` mode — this faithfully
models an NPU that commits to the narrow accumulator between both ops.
