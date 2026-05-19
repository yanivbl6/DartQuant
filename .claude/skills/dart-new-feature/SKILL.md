---
name: dart-new-feature
description: Add a new feature/flag/knob to DartQuant — a hardware-fidelity simulation toggle that flows through the rotation → equalization → fp16-cal → gptq → quant-cal → eval pipeline. Use when the user asks to "add a feature", "add a flag for X", "implement Y as a knob", "let's experiment with Z", "simulate <hardware behavior>", or otherwise wants a new dimension introduced into the quant tag space. Enforces the project's backward-compatibility, tag-scoping, and hardware-fidelity invariants, and runs the user-required validation sweep as part of the implementation. Args: one-line description of the feature (or just invoke and ask).
---

# dart-new-feature

Add a new feature to DartQuant. The skill exists because new flags are easy to write but easy to write *wrong* — silent cal contamination, broken backward compat, or a "pretty PPL" that came from an fp32 cheat are the recurring failure modes. Follow the steps below in order; don't skip the validation pass.

## Ground rules (non-negotiable)

1. **Backward compatible.** Default-off, default behavior bit-identical to before. Any caller that doesn't pass the new flag must produce the exact same tag, the exact same cached artifact paths, and the exact same numerics as the pre-change codebase. If your change cannot be made backward compatible, STOP and ask the user — that's a major-version bump (74 → 75/76/8), and per [VERSIONS.md](../../../VERSIONS.md) those are user-requested only.

2. **Off-by-default = no tag.** When the feature is inactive, `build_quant_tag` MUST produce the exact same string as before. No `_<feature>0`, no `_off`, no `_default` — those break cache reuse for every prior run. The active branch adds a tag; the inactive branch adds nothing.

3. **Tag-scope follows the pipeline.** The pipeline is:

   ```
   rotation → equalization → fp16-cal → gptq → quant-cal → eval (results)
   ```

   Each artifact is keyed by a tag built via `build_quant_tag(args, for_gptq_cache=…, for_cal_cache=…, for_fp16_cal_cache=…)`. Pick the earliest stage your feature touches; every downstream artifact must include the tag, every upstream artifact must NOT (so it stays shareable).

   | Feature touches… | `for_fp16_cal_cache` | `for_cal_cache` (post-GPTQ) | `for_gptq_cache` | result tag |
   |---|---|---|---|---|
   | Rotation / equalization / FP16-cal observer | ✓ | ✓ | ✓ | ✓ |
   | GPTQ algorithm itself | ✗ | ✓ | ✓ | ✓ |
   | Post-GPTQ quantized cal only | ✗ | ✓ | ✗ | ✓ |
   | Eval-time only (runtime numerics, no weight change) | ✗ | ✗ | ✗ | ✓ |

   "Touches" = changes the numerics of the artifact written at that stage. If unsure, include it — a path miss is recoverable; silent cal contamination is the recurring class of bug (`project_tag_mismatch_pattern.md` in auto-memory).

4. **Tag format.** `keyword-value` (or bare `keyword` for a boolean), short, filename-safe, **no underscores inside the tag** — the `_` is the inter-tag separator. Examples that exist today: `_hws-128`, `_gptqs50`, `_intgemm_acc16_bk32_t2int16a4`, `_gguf-Q4-K-M`. Examples that are wrong: `_my_new_flag` (underscores inside), `_myFeature=true` (`=` is fine-ish but inconsistent), `_my_feature_with_a_very_long_descriptive_name` (too long, will blow path limits on sweeps).

5. **Computationally demanding features get a heads-up flag.** If turning the feature on multiplies wall-clock by >2x, or substantially increases GPU memory, say so explicitly when the user asks to enable it (and in the argparse help string). Examples: a finer cal grid, more shots, exhaustive search across rotation seeds.

6. **Hardware fidelity is the goal. No cheating.** PPL going to garbage is acceptable — the feature exists to *measure* a hardware constraint, not work around it. Forbidden shortcuts:

   - Using **dynamic** activation quantization in a code path that's meant to simulate **static** scales (real silicon doesn't get to recompute scales per token).
   - Keeping K-cache / V-cache / accumulator in **fp32** when the target hardware is integer.
   - Falling back to fp16 for any tensor the feature is supposed to constrain.
   - Skipping a quantization step when the feature is "on but degenerate" — let the bad PPL surface. Silent fallbacks turn the next person's debugging into archaeology.

   If you can't make the hardware-faithful version work and want to ship a relaxed version, flag it as TODO in the code and tell the user *before* validation.

## Workflow

### 1. Scope it (before writing code)

Answer in writing (a few lines to the user, not a doc file):

- **What hardware behavior** does this simulate? (One sentence.)
- **Pipeline stage(s) touched** — rotation / eq / fp16-cal / GPTQ / quant-cal / eval. This determines the tag-scope row above.
- **Tag string** — what does the active branch add? E.g. `_calds-c4` for a c4 calibration dataset, `_nshots-256` for an alternate shot count.
- **Default value** — confirm the off/default branch produces the unchanged tag.
- **Computational cost** — same / >2x / GPU-memory-heavy.
- **Validation plan** — see step 4. State it now so the user can correct the design before you write code.

Don't proceed until this is settled. If the user gave a vague request, write a short proposal and ask for confirmation.

### 2. Wire the flag

Single canonical site: [experiment_config.py](../../../experiment_config.py).

a. **Argparse**: add the flag in the argparse section (around lines 260–280, alongside `--sim_version`). Use a `default=` value that produces the unchanged behavior. Help string mentions the cost flag (rule 5) if applicable.

b. **`build_quant_tag`**: add the tag-emission block at the position matching the pipeline stage your feature lives in (rotation tags go near eq, GPTQ tags go near GPTAQ, etc.). Gate it on the correct `for_*_cache` booleans per the table in rule 3. Pattern:

   ```python
   _myflag = getattr(args, 'myflag', <default>)
   if _myflag != <default>:        # active branch only — off = no tag
       tag += f"_kw-{_myflag}"     # or just "_kw" for a boolean
   ```

c. **CLI fanout for the shell driver**: if the feature is reachable via `dart_gptq_wxaykvz.sh`, add a `TAG_ARGS` line at the bottom of the `--sim_version` block (~[line 584](../../../fake_quant/Script/dart_gptq_wxaykvz.sh#L584)) so the shell forwards the flag into `experiment_config.py` when computing tags. Skip this if the feature is calibration-side only.

d. **Subprocess propagation**: if `experiment_config.py`'s subprocess-builder helpers (the `cmd += ['--sim_version', …]` block around [line 815](../../../experiment_config.py#L815)) need the flag, add it there too.

e. **Apply-lockstep**: if the feature has an `apply_<feature>(args)` helper that derives downstream attrs from the new flag (the pattern used by `--calib_set`'s `apply_calib_set` resolving `args.cal_dataset` etc.), grep for **every** call to `apply_set_preset` and add the new apply call right after it. Sites to check: `calibrate_act_scales.py`, `multi_calibration.py` (two spots — runfile group dedupe AND the cal-only entry), `run_experiments.py`, `analyze_scales.py`, and the `__main__` block in `experiment_config.py`. Missing any one of these silently produces wrong tags on that path because `build_quant_tag` reads the un-resolved attrs from raw `line_args`. Symptom: `--dry` shows fewer cal groups than runfile rows.

### 3. Implement the behavior

Standard project conventions — see the top-level [CLAUDE.md](../../../CLAUDE.md) and the canonical-modules table. Things to remember while implementing:

- Hardware fidelity (rule 6). Re-read it before writing the numerics.
- If the feature affects GPTQ-side quantization, the `_vN` GPTQ versioning is already in place — but the feature still needs its own tag on top, because a sibling without the feature must reuse its checkpoint, and we don't want to bump the major just for a feature.
- If the feature affects a parallel artifact path (e.g. AdaQuant or GPTAQ checkpoint via `.replace('gptq_checkpoints', …)` in `main_for_test.py` / `calibrate_act_scales.py`), no extra work — those derive from `args.gptq_checkpoint_path` and inherit the tag automatically.

### 4. Validate

Validation is part of the feature. Don't hand off to the user "for testing" — running the sweep is your job. Use the [dart-run-experiments](../dart-run-experiments/SKILL.md) skill.

**Default validation sweep — 3–5 runs, `-F` (fast) or `--very-fast`:**

1. **Off-baseline**: feature off, on a config you already have a `[DONE]` result for. PPL must match the existing cached number bit-for-bit (within fp noise). If it doesn't, your default branch isn't actually a no-op — fix that before anything else.

2. **On, mild setting**: feature on with the smallest/least-aggressive value. PPL should be close to baseline (small perturbation).

3. **On, aggressive setting**: feature on with the value that should clearly stress the system. PPL should show the expected degradation direction. If you can't predict the direction, write down your prediction *before* running, then check.

4. **On + interaction with another active feature** (optional but recommended): combine with `--scalewise`, `--gptaq`, or `--int_gemm` depending on what's plausible to combine. Catches tag-collision bugs early.

**Runfile naming**: `data/runs/runs_v<current-version>_<feature-name>.ini`. Read the current version from the bottom row of [VERSIONS.md](../../../VERSIONS.md) (no leading `v` in the version number — so v74 → `runs_v74_calds.ini`). `<feature-name>` is the short tag-keyword you picked in step 1 (e.g. `calds`, `nshots`), kept terse since it's already namespaced by version. Copy from a sibling runfile in `data/runs/` to inherit the right base config.

Use `--very-fast` if the feature is unrelated to long-context behavior, `-F` otherwise. Launch in background via the `dart-run-experiments` skill (it handles GPU selection, `--dry` preview, etc.).

**Don't pause before launching.** Once the runfile parses cleanly, kick off the validation sweep immediately. The only reason to stop is **no free GPUs** — in which case surface that (and `nvidia-smi`'s view of who's using them) and wait for the user. Don't make the user say "now go run it" — running it is part of finishing the feature, not a separate phase to confirm. If rows beyond #1 will need cal regeneration (their tag differs from any existing cache), add `--recalib` or hand off to [dart-shepherd-runfile](../dart-shepherd-runfile/SKILL.md) without asking.

**Watch the sweep while it runs — don't just wait for completion.** Failures often happen in the first 1–5 minutes (cal-time bugs, missing data files, OOM at model-load) and waiting 30 min for the completion notification wastes a sweep's worth of GPU time. Check at minutes ~2, ~5, ~15, then on completion. Either set up an `inotify`/`tail -f` Monitor on the cal logs filtered to error keywords (`Traceback|Error|RuntimeError|AssertionError|exit 1|OOM`), or just snapshot status every few minutes with `ps -p <pid>`, `nvidia-smi`, and `tail -30 /tmp/<launch>.log`. A failed row leaves its GPU idle (memory drops to ~11 MiB) — easy tell that something died early. When a failure is spotted mid-sweep, decide immediately whether to (a) let the rest finish and analyze, (b) kill and restart with a fix, or (c) kill and revise the runfile. Don't sit on a known-bad run for 30 more minutes.

**If validation surfaces a bug**: don't overwrite the runfile and re-run silently. Per the bump rules in [CLAUDE.md](../../../CLAUDE.md) → "Sim versions", a behavior-changing fix is a minor-version bump (74 → 741 → 742 → …). Procedure:

1. Fix the bug.
2. Append a row to [VERSIONS.md](../../../VERSIONS.md) with the new minor version + commit hash + one-line note (what the fix was).
3. Create a NEW runfile `runs_v<new-version>_<feature-name>.ini` (e.g. `runs_v741_calds.ini`) with `--sim_version <new-version>` set on each line. Keep the old runfile around — it's the historical record of the broken version's PPLs.
4. Re-run the validation sweep against the new runfile.

This keeps each iteration's results isolated in the cache, so you can diff "feature on at v74" against "feature on at v741" and see exactly what the fix changed. Major bumps (74 → 75) remain user-requested only — don't bump major just because the feature took several iterations to land.

### 5. Sanity-check the results

After the sweep completes (use [dart-ppl-status](../dart-ppl-status/SKILL.md) to read the numbers):

- Does the **off-baseline** match the prior cached run? If no → tag is wrong somewhere; diff `Quant tag:` lines in the new and old `_CAL.log`s. This is the `project_tag_mismatch_pattern.md` playbook.
- Does the **mild setting** show a small perturbation? If it's identical to off, the flag probably isn't taking effect — verify by grepping the log for the new tag string and by checking the args got propagated.
- Does the **aggressive setting** match the predicted direction? If yes → ship it. If no → STOP and report to the user; don't ship a feature with backwards numerics on a hunch.

When reporting back, give absolute PPL numbers (per `feedback_ppl_comparison.md` in auto-memory — no % deltas on already-degraded configs, and PPL is `exp(loss)` so deltas don't compose).

### 6. When in doubt, ask

If the results don't make sense and you don't have a clean hypothesis, surface to the user. Better one extra round-trip than shipping a feature whose validation you talked yourself into.

## File-path quick reference (touch these in order)

For an additive scalar/string feature flag like `--calib_set`, the wiring spans **5 files** in a fixed order. Knowing this map up front would have saved an hour of grepping:

| # | File | Block | What goes here |
|---|------|-------|----------------|
| 1 | [experiment_config.py](../../../experiment_config.py) | `add_quant_args` (~line 268) | `parser.add_argument('--<flag>', ...)` — the canonical argparse entry |
| 2 | [experiment_config.py](../../../experiment_config.py) | near `apply_set_preset` (~line 310) | resolver + `apply_<feature>(args)` helper that materializes any derived attrs |
| 3 | [experiment_config.py](../../../experiment_config.py) | `build_quant_tag` (~line 533) | the `if ... != default: tag += "_<kw>-..."` block |
| 4 | [experiment_config.py](../../../experiment_config.py) | `_build_fp16_cal_tag` (~line 670) | **separate** tag builder for the FP16 cal cache — same emission, easy to miss |
| 5 | [experiment_config.py](../../../experiment_config.py) | `build_quant_args` (~line 817) | `if getattr(args, '<flag>', ...): cmd += ['--<flag>', val]` — subprocess CLI fanout for cal dispatch |
| 6 | [calibrater/calibrate_act_scales.py](../../../calibrater/calibrate_act_scales.py) | argparse in `parse_args()` + `apply_<feature>(args)` after `apply_set_preset(args)` in `main()` | mirror the argparse entry, call the apply helper |
| 7 | [fake_quant/args_config_gen.py](../../../fake_quant/args_config_gen.py) | argparse in `parser_gen()` | mirror the argparse entry |
| 8 | [fake_quant/main_for_test.py](../../../fake_quant/main_for_test.py) | top of `main()` after `args = args_config_gen.parser_gen()` | call `apply_<feature>(args)` (lazy `sys.path` + `import experiment_config`) |
| 9 | [fake_quant/Script/dart_gptq_wxaykvz.sh](../../../fake_quant/Script/dart_gptq_wxaykvz.sh) | variable init (~165), CLI parser (~230), `TAG_ARGS` (~586), `${FLAG}` declaration (~468), python invocation list (~727) | **5 separate touch sites in one file** — every one of them is required |
| 10 | [calibrater/multi_calibration.py](../../../calibrater/multi_calibration.py) | both `apply_set_preset` call sites (~257, ~407) | call `apply_<feature>` right after, **per apply-lockstep rule below** |
| 11 | [fake_quant/run_experiments.py](../../../fake_quant/run_experiments.py) | `apply_set_preset` call site (~370) | same |
| 12 | [calibrater/analyze_scales.py](../../../calibrater/analyze_scales.py) | `apply_set_preset` call site (~482) | same |

If your feature also changes the data loader / numerics path, you'll also touch:
- `fake_quant/data_utils.py` (loaders), `fake_quant/gptq_utils.py` / `gptaq_utils.py` (Hessian), etc.

But the wiring above is the **necessary skeleton** — anything else is feature-specific.

## What would have saved time (observations from `--calib_set`)

These are non-obvious things that cost real minutes to discover. Read them before starting:

- **The shell driver has FIVE touch sites in one file.** Not three. Init the var (~165), parse `--<flag>) X="$2";` in the `case` (~230), `TAG_ARGS` line so the Python tag-printer sees the value (~586), separate `${X_FLAG}` declaration block (~468), and the line in the `python main_for_test.py \` invocation list (~727). Miss any one and you get a silent shell-side failure (cache collision or "flag ignored").
- **`apply_set_preset` is called from FIVE places**, not one — `calibrate_act_scales.py`, `multi_calibration.py` (twice), `run_experiments.py`, `analyze_scales.py`. Every site that builds a `quant_tag` from raw `line_args` needs your `apply_<feature>` next to it. Missing this is the **apply-lockstep bug**: `--dry` shows fewer cal groups than runfile rows, and rows silently share caches.
- **`_build_fp16_cal_tag` is a completely separate function.** If your feature changes FP16 observer output (almost any cal-side feature does), both `build_quant_tag` AND `_build_fp16_cal_tag` need the tag emission. Easy to fix the first and forget the second.
- **The CLI tag printer at the bottom of `experiment_config.py`** (`__main__` block, lines ~918–935) is invoked by the shell driver to compute tags. It calls `apply_set_preset` but you need to add `apply_<feature>` here too — otherwise the shell-driven tag and the Python-driven tag diverge.
- **`args_config_gen.py` doesn't import `experiment_config`.** Call your `apply_<feature>` from `main_for_test.py` (where the lazy `sys.path` insert is already used for `resolve_gguf_path`), not from inside the argparser.
- **Default `getattr(args, '<flag>', None)`** in your tag-emission and propagation code — it's safe across all callers regardless of whether their argparser defines the flag. Used everywhere; copy the `sim_version` pattern.
- **The runner's flag table is narrow**: only `[DONE]` and `[ERROR-CALIBRATE]` actually skip; `[FAST]`, `[CAL]`, `[ERROR]` all get picked up in default mode (and re-stamped to `[DONE]` on success). If you want to keep a broken row from being retried forever, mark it `[ERROR-CALIBRATE]` not `[ERROR]`.
- **The validation off-baseline cache hit is verifiable from the inference log**: look for `WIKITEXT2 PPL: <n> (cached)` (and similar) — the literal `(cached)` annotation proves the result came from the `.pb` cache, not a recomputation. This is your bit-perfect no-op proof.
- **`--recalib` re-runs cal for every line in the runfile.** If only one row needs new cal (because the others reuse cached artifacts), use the single-line cal pattern from [dart-run-experiments](../dart-run-experiments/SKILL.md) — pull the cal command from the `--recalib --dry` output, run it directly via `nohup`, then launch inference normally. Saves ~5–15 min per unaffected row.
- **`show_results.py` groups by tag and dedupes by file.** When a tagless run hits the same cache as a pre-existing `[DONE]` run, they collapse into a single row in the output table. To prove byte-identity, grep the inference log for `(cached)` rather than diffing PPL columns.
- **The data loaders have pre-existing edge cases** that only surface at unusual `nsamples`. Encountered: `get_c4_new` at `nsamples=1024` hits a `random.randint(0, -1)` "empty range" ValueError when a sampled doc has exactly `seqlen` tokens. Loaders for wiki/ptb concatenate-then-window so they're safe; c4 picks-then-samples so it's not. If your feature exposes a new code path through these loaders, expect to discover (and report, not silently work around) one of these.
- **Watch the sweep at minutes 2, 5, 15** — failures that happen at cal pre-pass crash within seconds, and waiting 30 min for the completion notification is a waste. Idle GPU memory (~11 MiB) on a row that should be calibrating is a strong "this row already died" signal.
- **Subprocess args propagation has TWO paths**: (a) Python→Python via `build_quant_args` in `experiment_config.py:817` (consumed by `multi_calibration.py` and `run_experiments.py` when dispatching cal), and (b) Shell→Python via the `${FLAG}` insert in the python invocation list in `dart_gptq_wxaykvz.sh:727`. Both need your flag or one of cal/inference will run without it.

## Common pitfalls

- **Tag emitted in default branch** → breaks cache reuse for every previously-run config. Always gate on `!= default`.
- **Tag missing on `for_gptq_cache=True`** when the feature actually changes GPTQ → silent contamination across feature values. Re-check the table in rule 3.
- **Underscore in the tag** → splits the tag into two on the parser side (`show_results.py` and tag-diffing tooling tokenize on `_`).
- **Forgetting the shell `TAG_ARGS` line** → Python-side runs are versioned but shell-driven runs share caches across feature values. Easy to miss because it only bites in the eval path.
- **"It works on my one config"** → run the 3–5-run sweep. Especially the off-baseline.
- **Cheating to make the demo PPL look nice** — see rule 6. The feature exists to measure hardware constraints. Garbage PPL is fine; a misleading number is not.

## Don't

- Don't bump `--sim_version` for a new feature. Versions are for bug fixes that change behavior, not for additive features (which are tagged independently). The version is for "this config but with a bug fixed"; the feature tag is for "this config but with feature X on". They compose.
- Don't create a new docs file describing the feature. The flag's argparse help string and a one-line entry near the feature in `CLAUDE.md` (if the feature is a permanent capability worth flagging in the orientation) are sufficient.
- Don't run the validation sweep without `--dry` first (per the `dart-run-experiments` skill's invariants).
- Don't merge / commit without the user's go-ahead after the validation report.

## Example invocation

User: "let's add a feature to vary the calibration dataset — c4 instead of wikitext, and configurable shot count."

Skill response (concise, following step 1):

> **Scope**
> - Hardware behavior: not hardware — methodology knob for cal-data robustness.
> - Pipeline stage: FP16-cal and post-GPTQ cal (both observe activations on the cal corpus).
> - Tag scope: ✓ fp16-cal, ✓ post-GPTQ cal, ✓ gptq (different cal → different GPTQ Hessian), ✓ result.
> - Tag string proposal: `_calds-c4` (omitted when c4 not selected, i.e. default = wikitext = no tag); `_ncal-256` for non-default shot counts.
> - Default: wikitext, 128 shots — matches current behavior, no tag emitted.
> - Cost: same wall-clock, possibly different memory if c4 sequences are longer.
> - Validation: 4 runs on a known M4S4 config in `data/runs/runs_v74_calds.ini` (current version per VERSIONS.md) — (a) wikitext/128 default off-baseline must match cached PPL, (b) c4/128, (c) wikitext/256, (d) c4/256. `--very-fast`.
>
> OK to proceed?

After user confirms → step 2 (wire), step 3 (implement), step 4 (validation runfile + dart-run-experiments), step 5 (sanity-check), report.
