---
name: dart-analyze-scales
description: Inspect quantization scale distributions (activations, weights, or merged a×w per-group) by invoking calibrater/analyze_scales.py. Use as a FIRST go-to when debugging why a quantization feature is not working — e.g. unexpected PPL, suspected outliers, hwscale path issues, weight asymmetry questions, K-cache anomalies, or any "why is this distribution shaped wrong" question. Also use when the user asks to "look at the scales", "histogram the activations", "merged scale analysis", "why is down_proj weird", "compare scale distributions". Args: a config description (the user's current debugging target) and which mode (activations, weights, or merged).
---

# dart-analyze-scales

Wrap [calibrater/analyze_scales.py](calibrater/analyze_scales.py). Four modes — pick by user intent:

| Mode | Flag | Reads | Use when |
|---|---|---|---|
| **Activations** (default) | (no flag) | post-GPTQ cal `.pt` | "look at activation scales", suspected per-channel outliers, K-cache or out_quantizer anomalies, post-cal sanity |
| **FP16 cal** | `--fp16_calib` | FP16 (pre-GPTQ) cal `.pt` | "the deployment uses fp16dep cal", "compare FP16 cal vs post-GPTQ cal", investigate scalewise/gptaq inputs |
| **Weights** | `--weights` | GPTQ checkpoint | "are the weight groups uniform", w_asym vs w_sym investigations, per-projection weight stats |
| **Merged a×w per-group** | `--group` | both | "is hwscale getting reasonable input", debugging `--hw_accurate` / `--hwscale` / `--scalewise` paths, T2 frac-bits analysis |

`--fp16_calib`, `--weights`, and `--group` are mutually exclusive. Add `--hwscale <spec>` to also report **post-global** scaled distribution under `--group` — what hwscale's snap operates on. (Outside `--group`, `--hwscale` is accepted for path resolution but doesn't add a histogram — see "Mode-conditional args" below.)

## Always-do procedure

1. **Determine the config under investigation.** The script needs the FULL quant-arg surface to resolve cal/GPTQ paths. Grab args from the user's stated config or from the most recent discussed run. Wrong args → "calibration file not found" error and a list of available alternatives.
2. **cd to `calibrater/`.** The script lives there and its path resolution depends on cwd.
3. **Pick the mode** per the table above.
4. **Run with `--no-plot` first** if you only need the stats (faster, no matplotlib import). Add plots only if the user wants figures.
5. **Read stdout, summarize**. Don't dump raw output — extract the insight: "down_proj scales p95/p5 ratio is 4096× — log2 dynamic range 12 bits — that's why X". Point at figure paths if generated.

## Invocation pattern

```bash
cd /workspace/DartQuant/calibrater && python analyze_scales.py <MODE> -m 1b <full quant args>
```

The `<MODE>` is the rotation mode (`baseline` / `quarot` / `dart`) — first positional. Quant args mirror the experiment under debugging.

**Args that DO affect path resolution (post-GPTQ cal):**
`-w/-a/-k/-v/-G --kv_ex --down_bits --quant_out --smq --pwl_act --sim_version --int_gemm --acc_bits --acc_wrap --scalewise --realint --hw_accurate --gptaq --w_asym --imitate_gguf`

**Args that don't matter for path resolution but are HARMLESS to pass:**
- `--static-act`, `--fast`, `--very-fast`, `--overwrite`, `--gptq` (the run_experiments.py flag, NOT the `--gptaq` quant flag) — analyze_scales.py uses `parse_known_args()`, so unknown flags are echoed and ignored. Pass-through from runfile lines is safe; no need to strip.
- `--acc_dtype` — passed to the parser but ignored for cal-path resolution. The cal tag uses `for_cal_cache=True` which strips the auto-T2 `t2intNaM` segment (matches calibrate_act_scales.py's save logic). Don't go looking for a `_t2int24a0_` tagged cal file; one is never written by current code (see [project_acc_dtype_cal_bleed.md](file:///home/yanivbl/.claude/projects/-workspace-DartQuant/memory/project_acc_dtype_cal_bleed.md)).

**Mode-conditional args:**
- `--hwscale <spec>` — used for path resolution always (cal tag becomes `_scalewise-<spec>` under `--scalewise`). Outside `--group`, it prints a soft note and is otherwise ignored — no error, no abort. Pass it freely from the runfile config.

**FP16 cal: use the `--fp16_calib` flag.**
The `--fp16_calib` deployment path writes to a separate file at `data/act_scales/<model>/<mode>_<short_tag>__fp16.pt` (e.g. `quarot_a8k8v8_ugdeq_noR4_kvex8_down16_pwl_qout-ex_smq16__fp16.pt`). To analyze it, just add `--fp16_calib` to the analyze_scales.py invocation — the script handles tag construction via `_build_fp16_cal_tag()` automatically. The FP16 cal tag **excludes** GPTQ-related flags (`-w`, `--scalewise`, `--hwscale`, `--gptaq`, `--int_gemm`, `--acc_*`, `--realint`, `--sim_version`); only forward-affecting flags are encoded (a/k/v bits, eq mode, no_r4, kv_ex, proj_ex, down_bits, pwl_act, quant_out, smq, gguf). You can pass the full quant-arg set without thinking about which flags are FP16-relevant — the tag builder ignores the irrelevant ones.

If the user mentions a recent run by short name (e.g. "v67 sym", "M4S4 t2int24 wasym"), translate to the full arg set from `data/runs/runs_v6.ini`. Pass-through is safe — no inference-only stripping required.

## Escape hatch: `--pt_path` (for non-canonical paths only)

For canonical post-GPTQ cal: just pass the right quant args; path resolution works.
For canonical FP16 cal: use `--fp16_calib`.

`--pt_path` is reserved for genuinely non-canonical lookups:
- Cross-version comparisons (e.g. read v63 cal while running args for v67).
- Manually-copied artifacts (forensic / debugging snapshots).
- Stale cal files saved before tag-logic changes.

```bash
ls /workspace/DartQuant/data/act_scales/Llama-3.2-1B-Instruct/ | grep <pattern>
# pick the right .pt, then:
python analyze_scales.py <MODE> -m 1b --pt_path /full/path/to.pt --no-plot
```

`--pt_path` accepts either a `.pt` file (activation/`--fp16_calib` modes) or a GPTQ checkpoint directory (`--weights` mode). Quant args are still required for the analysis math (e.g. `--w_bits` for weight scale recovery), but the path lookup is skipped.

## Mode-specific notes

### Activations (default)

Reads `data/act_scales/.../<tag>.pt`. Output is per-category stats:

- **Inputs:** `attn_qk_input`, `attn_vo_input`, `mlp_gate_up_input`, `mlp_down_input`
- **Outputs:** `attn_qk_output`, `attn_o_output`, `mlp_gate_up_output`, `mlp_down_output`
- **Caches:** `v_cache`, `k_cache`
- **Pre-rotation:** `pre_rotation`

For each: count, min/max, mean±std, median, 95%/90% percentile range, and `log2(p97.5/p2.5)` (the dynamic range in bits — directly relevant for shift/scale hardware).

Zeros only printed for categories with non-trivial zeros (asym).

This is the FIRST thing to run when debugging cal contamination, K-quantizer issues, or any "scales look wrong". Yesterday's `--acc_dtype` cal-bleed bug would have been visible immediately as `mlp_down_input` zero distribution being abnormal.

### Weights (`--weights`)

Reads GPTQ checkpoint dir. Reports per-projection (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `gate_proj`, `down_proj`) group-scale stats. Useful for spotting:
- Per-projection scale magnitude differences (e.g. v_proj 100× wider than k_proj)
- w_sym vs w_asym recovery anomalies
- Specific projections with unusually wide dynamic range (calibration / clip_ratio candidates)

Recovers scales from fake-quantized weights; tries both `maxq` and `maxq+1` candidates to handle `--w_clip` MSE-shrunken ranges.

### Merged a×w (`--group`)

The most informative mode for hwscale debugging. Computes `merged = a_gscale × w_gscale` per group — exactly what hwscale snaps on. Scope:
- **`--hw_accurate`** → down_proj only (matches main_for_test.py's down-only int_gemm config).
- otherwise → all int_gemm-eligible projections.

If the user adds `--hwscale <spec>` (e.g. `M4S4bmid`), the skill also reports `merged / global_scalar` (post-global, NO snap applied) — what the snap function actually clamps. Per-layer global scalars are printed with their `log2`. Out-of-range globals (`log2 < -100` or `> 100`) trigger a warning — fp32 normal range is `[-127, 127]`.

This is THE mode for hwscale path debugging:
- "T2 cliff at int22" → run `--group --hwscale M4S4bmid`, look at the post-global histogram. If the distribution stretches beyond what `M4S4bmid`'s snap can represent, the cliff is explained.
- "scalewise GPTQ producing weird output" → check that pre-snap merged distributions look reasonable.
- "hw_accurate makes things worse" → run twice (with and without `--hw_accurate`), compare merged distributions for down_proj.

Mutually exclusive with `--weights`. Requires both cal `.pt` AND GPTQ checkpoint.

## Output handling

- **Stats** → stdout. Read it, summarize the salient findings. Don't paste raw output back unless user asks.
- **Figures** → `calibrater/figures/<tag>_scale_hist.png`, `<tag>_zero_hist.png`, `<tag>_scale_combined.png`, etc. The tag includes mode + quant args so different configs don't overwrite each other. Tell the user the path; they'll view it.
- **`--no-plot`** → skip matplotlib entirely. Default to this for routine stats checks.

## Comparison workflow

Two configs → run twice, diff the stats:

```bash
cd /workspace/DartQuant/calibrater
python analyze_scales.py quarot -m 1b <args for config A> --no-plot > /tmp/scales_A.txt
python analyze_scales.py quarot -m 1b <args for config B> --no-plot > /tmp/scales_B.txt
diff /tmp/scales_A.txt /tmp/scales_B.txt
```

Or read both into context yourself and summarize the deltas. Useful for "what changed between v63 and v67 cal" type questions.

## Don't

- Don't try to load `.pt` or GPTQ files directly with torch.load and reimplement the analysis. The script's recovery logic (`_recover_group_scales`, `_act_per_group_scale`) is non-trivial and matches inference-side semantics — bypassing it produces wrong answers.
- Don't run from `fake_quant/` — paths break. Always `cd calibrater/`.
- Don't open the PNG figures programmatically to "view" them. Output the path; the user views in their environment.
- Don't omit quant args you don't recognize. The path resolution is sensitive to every flag listed in `experiment_config.build_quant_tag`. If unsure, check the cal `.pt` filenames in `data/act_scales/Llama-3.2-1B-Instruct/` to confirm the tag.

## Path layout (shared across DartQuant skills)

- `data/act_scales/<model>/` — calibration `.pt` files. FP16 cal: short tag + `__fp16.pt` suffix. Post-GPTQ cal: long tag including all GPTQ-affecting flags.
- `data/gptaq_checkpoints/<tag>/<model>_w<bits>/*.pth` if `--gptaq`, else `data/gptq_checkpoints/<tag>/<model>_w<bits>/*.pth` — GPTQ checkpoints. Tag uses `for_gptq_cache=True` form (omits result-only flags).
- `data/cached_results/` — inference logs (`<label>.log`), cal logs (`<label>_CAL.log`), result caches (`*_results.pb`).
- `calibrater/figures/` — PNG output from `analyze_scales.py` plots.
- `data/runs/*.ini` — runfiles consumed by `run_experiments.py` / `multi_calibration.py`.

## Operating mode

Fully autonomous. Pick the mode based on user phrasing, build the args from context, run, summarize. No user prompts. If args don't resolve to an existing cal/checkpoint, the script will print available alternatives — read those and either retry with a closer match, switch to `--fp16_calib` if the user is interested in FP16 cal, or fall back to `--pt_path` for non-canonical paths.

## Examples

User: "why is wasym 4 PPL worse than sym?" (config: v67 M4S4 t2int24)
You:
1. `cd /workspace/DartQuant/calibrater && python analyze_scales.py --weights quarot -m 1b -w 4 -a 8 -k 8 -v 8 -G 16 --kv_ex 8 --down_bits 16 --no_r4 --ugd_eq --realint --hw_accurate --gptaq --sim_version 67 --int_gemm --acc_bits 16 --acc_wrap --hwscale M4S4bmid --scalewise --quant_out ex --smq 16 --pwl_act --w_asym --no-plot`
2. Same command with `--sym` instead of `--w_asym`.
3. Diff the two: which projection's scale distribution shifted? Report the finding (e.g. "v_proj group scales 2× wider under w_asym").

User: "look at the merged scales for the M4S4 t2int24 sym config"
You: `cd /workspace/DartQuant/calibrater && python analyze_scales.py --group --hwscale M4S4bmid quarot -m 1b ... <full args> --no-plot`
Read post-global stats, identify any out-of-range globals.

User: "first pass — why does down_proj look weird in v67 sym"
You: activation mode (default), filter the stdout to `mlp_down_input` and `mlp_down_output` categories, report the stats.

User: "compare FP16 cal vs post-GPTQ cal for the v67 sym config"
You:
1. `cd /workspace/DartQuant/calibrater && python analyze_scales.py quarot -m 1b -w 4 -a 8 -k 8 -v 8 -G 16 --kv_ex 8 --down_bits 16 --no_r4 --ugd_eq --realint --hw_accurate --gptaq --sim_version 67 --int_gemm --acc_bits 16 --acc_wrap --hwscale M4S4bmid --scalewise --quant_out ex --smq 16 --pwl_act --no-plot > /tmp/postgptq_cal.txt`
2. Same command with `--fp16_calib` added → `/tmp/fp16_cal.txt`.
3. `diff /tmp/fp16_cal.txt /tmp/postgptq_cal.txt`. Report which categories differ (FP16 cal won't have GPTQ-induced post-quant shifts — useful for separating cal-bleed effects from GPTQ artifacts).
