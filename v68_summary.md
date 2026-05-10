Hi,
Sending a summary of the latest quantization campaign on Llama-3.2-1B-Instruct.
Group-size 32 is still running; everything below is for group size 16, which
covers the production-shape decisions. Calibration is done via wikitext.

CONTEXT

The goal of recent experiments was to validate a fully-integerized w4a8 stack with GSU. 

The simulation is intended to simulate Helium.

This version differs from previous iterations on several key points:
1. QuaRot's R4 is removed completely, replaced by 3-way equalization (Up&Gate vs Down).
2. Calibration is done before weight quantization, and relied upon for GPTQ. Second calibration is optional.
3. GPTQ is replaced by default with GPTaQ to match current SDK standard.
4. For down layer only, group scales are calculated from both activation and weight stats (no float values).


BOTTOM LINE

At group size 16, the production-shaped stack (4-bit weights, 8-bit
activations, 8-bit KV-cache, integer accumulator, hardware-faithful softmax (16bit integer),
and merged hardware scale) reaches:

    wikitext2 PPL      14.25   (FP16 reference: 13.16)
    9-task average   51.66    (FP16 reference: 54.3)
    MMLU                    43.24    (FP16 reference: 48.2)

MMLU is the stress benchmark and drops by ~5 points.
The other eight evaluation tasks (Piqa, Hellaswag, ARC-Easy,
ARC-Challenge, Winogrande, Lambada, SocialIQA, OpenBookQA)
move by ~2 points on average, with Lambada the outlier at ~6.

FINDINGS AND RECOMMENDATIONS

1. Tier 2 (outer) accumulator floor is 24 bits.
   Sweeping the integer width on the production stack, 28b / 26b / 24b are
   within ~0.3 PPL of one another. At 22b the model collapses (+40 PPL,
   MMLU back to chance). At 20b it is unrunnable (Lambada = 0).
Specify the T2 accumulator at 24+ bits.

   ![T2 accumulator sweep — wikitext2 PPL, MMLU, 9-task average vs T2 width](figures/v68_finding1_t2_accumulator.png)

2. Hardware merged-scale format: 4 mantissa + 4 shift, mid-anchored, is the
   production target.
   Of the four formats compared, only 4-mantissa+4-shift and the
   over-provisioned 8-mantissa+4-shift maintain accuracy. The
   5-mantissa+3-shift variant loses ~10 MMLU; the 8-mantissa+0-shift variant
   collapses entirely.
 Use the 4+4 format. It matches the 8+4 reference within noise at 50%
       less bits. Anything narrower on the shift field is unsafe at G=16.
       M4S4 also closely resembles NVFP4 specs.

   ![HW scale format comparison — M4S4 vs M5S3 vs M8S0 vs M8S4](figures/v68_finding2_hw_scale_format.png)

3. Scale-rounding must be folded into the GPTQ loop.
   The merged per-group scale is rounded to its hardware representation.
   Doing that rounding inside the GPTQ loop (so the Hessian-aware
   compensation absorbs the rounding error jointly with weight rounding) is
   what makes the 4+4 format competitive at all.
Always pair hardware scale rounding with in-loop scale rounding
       during calibration. This is partially done in the SDK already  but will
       gain extra importance with M4S4 scales.

4. Asymmetric weight quantisation's benefit is not conclusive
   Switching weights from symmetric to asymmetric at production shape gains
   ~0.4 on the 9-task average and ~1.1 on MMLU at the same wikitext2 PPL.
Since those are noisy benchmarks, and with the feature being costly,
       more targetted research must be done to determine cost-benefit.

5. GPTAQ is helpful for ppl, inconclusive for reasoning.
stick with GPTaQ since it has no direct cost.

6. GGUF-shape (Q4_K_S) imitation is also inconclusive
   Matching the per-layer bit allocation of llama.cpp's Q4_K_S buys ~0.1
   wikitext2 PPL but costs ~1.2 MMLU vs the same stack with uniform
   4-bit weights.
Keep track on GGUF as alternative but stick with int4 for main results.

7. Calibration deploy mode (FP16 vs post-GPTQ) is within noise on every
   benchmark at production shape.
Unquantized calibration is good enough. (matches SDK)

8. FP4 worked out-of-the-box for those specific settings.
     In general, FP4 requires changing the optimization flow, but with nvfp4 it is not
     necessary. Results show better MMLU despite worse ptb ppl.  
Keep FP4 implementation but adjust expectations.



HEADLINE TABLE

    Configuration                                                              wiki PPL  Avg   MMLU
    -----------------------------------------------------------------------
    FP16 (reference)                                                            13.16   54.3   48.2
    llama.cpp Q4_K_S (reference)                                13.72   53.6   47.3
    Weight-only 4-bit (no activation quant)              13.42   53.8   47.0
    Weight and Activations                                               13.92   52.0   44.1
    Production: w4a8 + 4+4 scale + 24b acc            14.25   51.7   43.3



NEXT STEPS

Currently running G=32




Happy to walk through any of these in more detail.

Best,
Yaniv