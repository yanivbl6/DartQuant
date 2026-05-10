---
name: dart-run-experiments
description: Drive fake_quant/run_experiments.py to launch DartQuant calibrations or inference batches over a runfile (default ../data/runs/runs_v6.ini). Use when the user asks to "run the experiments", "kick off the runfile", "rerun X", "regenerate cal", "run the ablations", or otherwise wants to execute or re-execute experiment batches. Always run a GPU availability check first. Always preview with --dry before any --recalib invocation. Args may be: a runfile path, a substring filter, or empty (use default runfile).
---

# dart-run-experiments

Wrap `fake_quant/run_experiments.py`. Two responsibilities: (a) pick the right flag combination based on user intent, (b) avoid known footguns (especially the `--gptq + no --recalib` cal mismatch from 2026-05-04).

## Always-do steps (in order)

1. **GPU check** — show what's free before suggesting a command.
   ```bash
   nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
   ```
   "Free" = `memory.used < ~500 MiB`. Note which GPUs are free.

   **NVML init failure → user-gate.** If `nvidia-smi` returns `Failed to initialize NVML`, stop and ask the user before launching anything. Don't try to work around it; CUDA subprocesses fail unpredictably even when `/dev/nvidia*` is present.

2. **Decide flag combination** from user intent (see flag table below).

3. **`--dry` first, always — interpret the output yourself.** Run the dry preview to get the numbered queue and GPU assignment. Use it to:
   - Verify the queue matches the user's stated intent (which lines, how many).
   - Verify the queue matches user intent. Default mode runs anything that isn't `[DONE]` or `[ERROR-CALIBRATE]` — that includes unflagged, `[FAST]`, `[CAL]`, and `[ERROR]`. Fast mode (`-F`) additionally skips `[FAST]`. **Common surprise:** `[FAST]` lines DO run in default mode (and get upgraded to `[DONE]`); `[CAL]` lines always need inference. See "Flag semantics" below for the full table.
   - Catch obvious mistakes (wrong runfile path, no lines selected, GPUs that aren't free).

   **Two entry points — pick by intent:**
   - **Inference batch** (or combined cal+inference): run from `fake_quant/`, runfile path `../data/runs/runs_v6.ini`.
     ```bash
     cd /workspace/DartQuant/fake_quant && python run_experiments.py --runfile ../data/runs/runs_v6.ini -g <GPUS> --dry [other flags]
     ```
   - **Cal-only** (e.g. user wants to regenerate cal artifacts without launching inference): run from `calibrater/`, runfile path `../data/runs/runs_v6.ini`.
     ```bash
     cd /workspace/DartQuant/calibrater && python multi_calibration.py --runfile ../data/runs/runs_v6.ini -g <GPUS> --dry [other flags]
     ```
   `multi_calibration.py` shares MOST of the flag surface (`--runfile`, `--recalib`, `--gptq`, `-g`, `--dry`) but **NOT** `--skip` — it forwards `--skip` as a pass-through arg to inner `calibrate_act_scales.py` calls instead of filtering its own queue. Don't rely on `multi_calibration.py --skip` to restrict cal scope. Dry output paths are relative to the cwd you ran from — don't get confused if you `--dry` `run_experiments.py` from `fake_quant/` and see cal subprocess paths that look "wrong"; they execute with cwd=calibrater/.

   No user-confirmation step. If the dry output matches intent, proceed. If it doesn't match, pick the closest reasonable interpretation, log what you chose and why, and proceed. ABORT (with a clear log) only if the queue is empty or no GPUs are free.

4. **`--recalib` is whole-file-only.** It re-runs EVERY line in the runfile. Use it ONLY when you genuinely want to regenerate cal for the whole file (e.g. cal-code change affecting all lines). For partial scope, do NOT pair `--recalib` with `--skip` — see "Partial-scope re-runs" below. The right partial pattern is to unflag the lines you want re-run and let the runner's normal mode pick them up.

5. **Launch** in background. Multi-hour cal+inference batches MUST run in background (`run_in_background: true`) — never block the conversation on them.

6. **Operating mode.** This skill is designed to run in autonomous/overnight loops. NEVER block on user input. When intent is ambiguous, pick the safest default (see "Default-on-ambiguity" rules below), log the choice, and proceed.

## Flag decision table

User says ... | Intent | Entry point + flags | Notes
---|---|---|---
"rerun the inference for the v6 runs" | results cache invalid, cal+GPTQ OK | `run_experiments.py` from `fake_quant/`, `--overwrite` | Tells the inference subprocess to ignore `*_results.pb` cache and recompute. Does NOT bypass the runfile flag filter — `[DONE]` lines are still skipped (strip `[DONE]` first if you want them re-run).
"the cal code changed, regenerate" | cal must be redone WHOLE FILE | `run_experiments.py` from `fake_quant/`, `--recalib` | **Re-runs EVERY line.** For partial scope, unflag instead — see "Partial-scope re-runs" below.
"just regenerate cal, don't run inference" | cal-only refresh | `multi_calibration.py` from `calibrater/`, `--recalib` | Cal-only entry point. Note: `--skip` is NOT supported for cal-queue trimming here — see line 34.
"the GPTQ code changed" | GPTQ ckpt invalid | `run_experiments.py` from `fake_quant/`, `--recalib --gptq` | NEVER `--gptq` alone — see footgun. Same `--recalib` scope warning applies.
"preview which lines would run" | scoping | add `--dry` (to whichever entry point) | This is mandatory before any real run.
"run only the unflagged lines" | runfile in normal mode | `run_experiments.py` from `fake_quant/` (no extra flags) | Default mode skips `[DONE]` and `[ERROR-CALIBRATE]`; runs everything else (unflagged, `[FAST]`, `[CAL]`, `[ERROR]`). `[FAST]` lines get upgraded to `[DONE]` after full eval completes.
"upgrade [FAST] lines to [DONE] (full eval)" | already have fast PPL, want lm_eval | `run_experiments.py` from `fake_quant/` (no `-F`, no other flags) | DON'T strip `[FAST]` — default mode picks them up automatically (`needs_inference([FAST], fast=False) → True`).
"re-cal only these specific lines" | partial recalib | unflag the lines in the runfile, run normally | If cal artifact already exists, only inference reruns. If cal is missing/stale, use single-line cal recipe below.
"quick PPL check" | speed | add `--very-fast` | wt2 only, ~3 min/run.
"normal eval, no MMLU" | speed | add `-F` / `--fast` | wt2/ptb/c4, ~7-10 min/run. Most common default.
"thorough eval with lm_eval" | full results | omit fast flags | Hours/run.

## Default-on-ambiguity rules (no user prompts)

These replace footgun "ask the user" patterns with autonomous safe defaults. Always log the rule that fired.

**`--gptq` requested without `--recalib`** ⚠️ — auto-add `--recalib`. New GPTQ regen against an old cal artifact produces bad PPL (this bit us on 2026-05-04). Adding `--recalib` does the right work; the cost is one extra cal pass. Log: "auto-added --recalib to avoid GPTQ-vs-cal mismatch".

**`--gptq` when GPTQ code didn't change** — auto-drop `--gptq`. GPTQ is deterministic from (FP16 cal + weights + args), so regenerating identical checkpoints wastes ~3 min/run. Heuristic: if `git log --since="<last cal time>" -- fake_quant/gptq_utils.py fake_quant/gptaq_utils.py` returns nothing, drop the flag. Log: "dropped --gptq, GPTQ code unchanged since last cal".

**Specified GPUs not free** — auto-substitute with the largest set of free GPUs (memory < 500 MiB) of the same count. If fewer free GPUs than requested, use whatever's free. If zero free, ABORT with a log message identifying the busy GPUs and their owning processes (`nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv`). Log: "GPUs 0-3 requested but 0,2 busy; running on 4,5,6,7 instead".

**Forgetting `--runfile` for runfile-mode runs** — auto-add `--runfile ../data/runs/runs_v6.ini` (the default). Log: "added default --runfile". Only fall back to legacy 7-experiment mode if the user explicitly asked for it.

**Empty queue after unflagging / `--skip`** — abort with log explaining no lines matched intent. This is a hard error; better to surface than silently launch nothing.

**Multiple plausible label-match interpretations** — pick the most specific (longest substring match against user phrasing). If two are equally specific, run BOTH (the queue is just the union). Log: "intent matched 'M4S4 ablations' AND 'gguf' subsets; running union".

## GPU spec format

- Single number `1` → expands to `1,2,3,4,5,6,7` (7 experiments fan-out, only used in legacy mode without `--runfile`)
- Comma list `0,1,2,3` → exactly those GPUs, one per concurrent run
- For runfile mode: pick a comma-list with as many GPUs as you have free; `run_experiments.py` schedules across them.

## Default runfile

`../data/runs/runs_v6.ini` (relative to fake_quant/). Use this unless the user specifies a different path.

## Partial-scope re-runs: unflag, don't `--recalib --skip`

For partial scope (one or a few specific lines), the right pattern is to **unflag those lines in the runfile** and run normally. The runner's default mode skips `[DONE]` (and `[ERROR-CALIBRATE]`) and runs everything else, including `[FAST]`/`[CAL]`/`[ERROR]` — that's the entire mechanism. No `--recalib`, no `--skip` complement-of-intent computation.

**Important:** you only need to strip `[DONE]` to re-run a finished line. Stripping `[FAST]` or `[CAL]` is unnecessary in default mode (they're already picked up); strip them only if you specifically want a from-scratch state in the runfile.

Why not `--recalib --skip <complement>`:
- `--skip` restricts INFERENCE scope only; cal scope is independent.
- `--recalib --skip X` re-cals every line including the skipped ones — wasted GPU on already-good cals.
- It's also fragile (1-indexed line numbers shift if the runfile is edited).

Standard partial-scope pattern:
1. Edit the runfile: remove `[DONE]` from the lines you want re-run (or `[FAST]` if running with `-F` and you want them re-included). `[CAL]/[ERROR]` are picked up automatically — no need to strip.
2. Run normally: `cd /workspace/DartQuant/fake_quant && python run_experiments.py --runfile ../data/runs/runs_v6.ini -g <GPUS> -F` (or no `-F` for full eval).
3. The runner picks up everything that needs work per the flag table below; on success it stamps `[FAST]` or `[DONE]` back.

### Flag semantics (the actual rules from `runfile_flags.needs_inference`)

| Flag | Default mode (no `-F`) | Fast mode (`-F`) |
|---|---|---|
| (unflagged) | run | run |
| `[CAL]` | run (cal done, inference still pending) | run |
| `[FAST]` | **run** (upgrades to `[DONE]`) | skip |
| `[ERROR]` | run (retries the failure) | run |
| `[DONE]` | skip | skip |
| `[ERROR-CALIBRATE]` | skip (cal-side failure, won't auto-recover) | skip |

So in plain English: **`[DONE]` and `[ERROR-CALIBRATE]` are the only flags that actually skip a line**. The others are state markers that don't block re-running.

What if cal is missing for an unflagged line? See "Single-line cal for a brand-new line" below — the runner won't auto-cal; you cal once directly, then unflag and run inference.

`--skip` exists for the rare case where the runner's queue includes lines you specifically want excluded (e.g. one of the unflagged lines is broken and you don't want to retry it tonight). It's a simple ordinal trimmer, not a partial-recalib mechanism.

⚠️ **`multi_calibration.py` does NOT support `--skip` for cal-queue trimming.** Despite sharing most of `run_experiments.py`'s flag surface, `multi_calibration.py` forwards `--skip <list>` as a pass-through arg to each inner `calibrate_act_scales.py` invocation rather than filtering its own queue. If you need single-line cal, use the direct-cal pattern below.

## Single-line cal for a brand-new line (the right way)

When a new line is added to the runfile and inference fails with `"Static act scales not found"` → a `[ERROR]` mark gets stamped (the wrapper conflates "cal missing" with "execution error"; see `[ERROR]` recovery section). The temptation is `run_experiments.py --recalib --skip <complement>`, but that re-cals every other line too.

Cleanest path: pull the exact cal command from the dry preview and run `calibrate_act_scales.py` directly.

```bash
# 1. Get the cal command for the target line from the dry preview's cal section:
cd /workspace/DartQuant/fake_quant && python run_experiments.py --runfile ../data/runs/runs_v6.ini -g <gpu> --recalib --dry
# (look at the cal section; each cal job's full command is printed)

# 2. Run that ONE cal command directly (in background):
cd /workspace/DartQuant/calibrater && nohup env CUDA_VISIBLE_DEVICES=<gpu> python calibrate_act_scales.py <args from dry> > /tmp/<label>_cal.log 2>&1 &

# 3. After cal finishes, remove [ERROR] from the runfile line if needed, then run inference normally:
cd /workspace/DartQuant/fake_quant && python run_experiments.py --runfile ../data/runs/runs_v6.ini -g <gpu> -F
```

The runner's normal mode picks up the now-unflagged line and skips `[DONE]` ones. Cost: one cal pass on one GPU (~5–10 min) + inference (~3 min) instead of N cals × N lines.

## Calibration GPU

`--calibrate_gpu` defaults to the first GPU in `--gpus`. Cal runs sequentially on one GPU before the parallel inference batch. If the user wants cal on a specific GPU (e.g. to leave the bigger GPUs free for parallel inference), pass `--calibrate_gpu <N>`.

## Logging convention

Each run's log lands at `data/cached_results/<label>.log` (per the runfile entry name) and `<label>_CAL.log` for cal phase. After launching in background, point the user at these paths so they can tail.

## Path layout (shared across DartQuant skills)

- `data/act_scales/<model>/` — calibration `.pt` files (consumed by `dart-analyze-scales` skill).
- `data/gptaq_checkpoints/<tag>/<model>_w<bits>/*.pth` if `--gptaq`, else `data/gptq_checkpoints/<tag>/<model>_w<bits>/*.pth` — GPTQ checkpoints. `--gptq` runner flag deletes the matching dir.
- `data/cached_results/` — inference logs, cal logs, `*_results.pb` (consumed by `dart-ppl-status` skill).
- `data/runs/*.ini` — runfiles. `runs_v6.ini` is the default.

## Out-of-scope: ad-hoc runs not in any runfile

This skill drives `run_experiments.py` / `multi_calibration.py` over a runfile. If the user asks to launch a one-off run that isn't represented in `runs_v6.ini` (e.g. the `s1_no_fp16cal_full_*` PPL-only debug runs from 2026-05-04), the right tool is `fake_quant/Script/dart_gptq_wxaykvz.sh` directly. Don't try to force the request through `run_experiments.py`. Either:
- Surface to the user: "this run isn't in any runfile; use `bash Script/dart_gptq_wxaykvz.sh ...` directly, or add it to `runs_v6.ini` first."
- OR (if intent is unambiguous) invoke `dart_gptq_wxaykvz.sh` directly, but log clearly that you're going outside this skill's scope.

## `[ERROR]` / `[ERROR-CALIBRATE]` recovery

When a runfile line fails, `run_experiments.py` writes back `[ERROR]` (inference) or `[ERROR-CALIBRATE]` (cal phase) to that line. The skill's response when the user asks "why did <label> fail":
1. Grep the label's logs:
   - `data/cached_results/<label>.log` (inference output)
   - `data/cached_results/<label>_CAL.log` (cal phase output)
2. Look for tracebacks (`Traceback (most recent call last)`), `RuntimeError`, `AssertionError`, or `Error:` lines.
3. Surface the failing line + the relevant traceback excerpt — don't dump the whole log.

⚠️ **`[ERROR]` ≠ "this line is broken".** The runner stamps `[ERROR]` on any non-zero exit, including the case where the wrapper bails with `"Static act scales not found"` because cal was never run for a brand-new line. Common signature in the inference log: a single message saying `Static act scales not found at ...` followed by `Run calibration first: ...`. That's not a real failure — the line just needs a cal pass first. Use the "Single-line cal for a brand-new line" recipe above; don't chase tracebacks that aren't there.

Don't auto-retry failed lines without understanding the failure first; the same failure will likely repeat.

## Don't

- Don't block on user input. Pick the safe default per "Default-on-ambiguity rules" and proceed. Log the choice.
- Don't tail running jobs in a loop — kick them off in the background and use `dart-ppl-status` for status checks.
- Don't restructure the runfile (reorder lines, rewrite labels, change configs). The user owns runs_v6.ini's content; if the queue is wrong, ABORT and log.
- DO feel free to unflag lines when the user wants them re-run, but know which flag actually blocks: only `[DONE]` (and `[ERROR-CALIBRATE]`) skip in default mode. Stripping `[FAST]/[CAL]/[ERROR]` is usually unnecessary — those re-run automatically. The runner re-stamps on the next successful run.

## Example: autonomous workflow

User: "the cal-fix is committed, rerun the v6 ablations with the new cal."
You:
1. `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader` → identify free GPUs.
2. Substring-match "v6 ablations" against runfile labels → `s1_no_fp16cal_full_M4S4_t2int24a0_*` lines. Strip their `[DONE]` markers (any `[FAST]/[CAL]/[ERROR]` will be picked up automatically; only `[DONE]` blocks default-mode runs).
3. `cd /workspace/DartQuant/fake_quant && python run_experiments.py --runfile ../data/runs/runs_v6.ini -g <free-gpus> -F --dry` → confirm only the unflagged lines are in the queue.
4. Drop `--dry`, launch in background. Output: log paths + a one-line summary of what was launched + a pointer to `dart-ppl-status` for monitoring.

Cal: if the unflagged lines have stale cal that needs regenerating too, run the matching cal commands directly first (single-line cal recipe) before the inference pass — don't try to do it through `--recalib --skip`.

Queue empty → ABORT with log. Queue contains unexpected lines → log a warning but proceed.

## Example: footgun caught and auto-fixed

User: "rerun with `--overwrite --gptq`"
You:
1. Detect: `--gptq` without `--recalib`. Per default-on-ambiguity rules, auto-add `--recalib`.
2. Log: "auto-added --recalib to avoid GPTQ-vs-cal mismatch (the 2026-05-04 bug)".
3. Continue with the standard workflow (GPU check → dry → launch). Whole-file scope under `--recalib` — if narrower scope is intended, the unflag-pattern is wrong here (cal regen is global by definition); surface to the user instead.
4. Final command logged: `... --recalib --gptq --overwrite -F`.

## Example: the footgun the user actually hit

User: "rerun with `--overwrite --gptq`"
You: "That combo regenerates GPTQ but reuses the old cal — they'll mismatch and PPL will be bad (this is the bug we fixed yesterday). You probably want `--recalib` instead. With `--recalib --gptq` if GPTQ code also changed, or just `--recalib` if only cal code changed. Want me to do the latter?"
